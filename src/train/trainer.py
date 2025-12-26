"""
VLM Trainer Module

VLM 파인튜닝을 위한 학습 로직
"""

import json
import os
from pathlib import Path
from typing import Literal

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModel, Trainer, TrainingArguments

from src.data.collator import DeepSeekOCRDataCollator
from src.train.layers import get_target_modules, print_trainable_params


class VLMTrainer:
    """VLM 파인튜닝 트레이너"""

    def __init__(
        self,
        config_path: str | Path | None = None,
        env_path: str | Path | None = None,
    ):
        # 환경 변수 로드
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # 설정 로드
        self.config = self._load_config(config_path)
        self.training_mode = os.getenv("TRAINING_MODE", "vision")

        # 모델 설정
        model_config = self.config.get("model", {})
        self.base_model_path = os.getenv(
            "BASE_MODEL_PATH", model_config.get("base_model_path", "./deepseek_ocr")
        )
        self.load_in_4bit = model_config.get("load_in_4bit", False)
        self.use_gradient_checkpointing = model_config.get(
            "use_gradient_checkpointing", "unsloth"
        )

        # LoRA 설정
        lora_config = self.config.get("lora", {})
        self.lora_r = lora_config.get("r", 16)
        self.lora_alpha = lora_config.get("lora_alpha", 16)
        self.lora_dropout = lora_config.get("lora_dropout", 0)

        # 학습 설정
        train_config = self.config.get("training", {})
        self.train_config = train_config

        # 이미지 설정
        image_config = self.config.get("image", {})
        self.image_size = image_config.get("image_size", 640)
        self.base_size = image_config.get("base_size", 1024)
        self.crop_mode = image_config.get("crop_mode", True)

        # 출력 설정
        output_config = self.config.get("output", {})
        self.output_dir = os.getenv(
            "OUTPUT_MODEL_PATH", output_config.get("output_dir", "./outputs")
        )
        self.checkpoint_dir = os.getenv(
            "CHECKPOINT_DIR", output_config.get("checkpoint_dir", "./checkpoints")
        )

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _load_config(self, config_path: str | Path | None) -> dict:
        """설정 파일을 로드합니다."""
        if config_path is None:
            default_paths = [
                Path("config/train_config.yaml"),
                Path("train_config.yaml"),
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def load_model(self):
        """모델과 토크나이저를 로드합니다."""
        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                "unsloth is required. Install it with: pip install unsloth"
            ) from e

        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

        print(f"Loading model from {self.base_model_path}...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.base_model_path,
            load_in_4bit=self.load_in_4bit,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        print("Model loaded successfully!")

        return self.model, self.tokenizer

    def setup_lora(
        self,
        mode: Literal["vision", "llm", "both"] | None = None,
        custom_target_modules: list[str] | None = None,
    ):
        """LoRA 어댑터를 설정합니다."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                "unsloth is required. Install it with: pip install unsloth"
            ) from e

        if mode is None:
            mode = self.training_mode

        target_modules = get_target_modules(
            mode=mode,
            model=self.model,
            config_path="config/train_config.yaml",
            custom_modules=custom_target_modules,
        )

        print(f"Setting up LoRA for {mode} mode...")
        print(f"Target modules: {target_modules}")

        self.model = FastVisionModel.get_peft_model(
            self.model,
            target_modules=target_modules,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            random_state=self.train_config.get("seed", 3407),
            use_rslora=False,
            loftq_config=None,
        )

        print_trainable_params(self.model)
        return self.model

    def prepare_dataset(self, dataset_path: str | Path) -> list[dict]:
        """데이터셋을 로드하고 준비합니다."""
        dataset_path = Path(dataset_path)

        if dataset_path.suffix == ".jsonl":
            data = []
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif dataset_path.suffix == ".json":
            with open(dataset_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

        print(f"Loaded {len(data)} samples from {dataset_path}")
        return data

    def train(
        self,
        dataset: list[dict] | str | Path,
        eval_dataset: list[dict] | str | Path | None = None,
        resume_from_checkpoint: str | None = None,
    ):
        """학습을 시작합니다."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() and setup_lora() first.")

        try:
            from unsloth import FastVisionModel, is_bf16_supported
        except ImportError as e:
            raise ImportError(
                "unsloth is required. Install it with: pip install unsloth"
            ) from e

        # 데이터셋 준비
        if isinstance(dataset, (str, Path)):
            dataset = self.prepare_dataset(dataset)
        if isinstance(eval_dataset, (str, Path)):
            eval_dataset = self.prepare_dataset(eval_dataset)

        # 학습 모드 활성화
        FastVisionModel.for_training(self.model)

        # Data Collator 생성
        data_collator = DeepSeekOCRDataCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            image_size=self.image_size,
            base_size=self.base_size,
            crop_mode=self.crop_mode,
            train_on_responses_only=True,
        )

        # TrainingArguments 설정
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.train_config.get("gradient_accumulation_steps", 4),
            warmup_steps=self.train_config.get("warmup_steps", 5),
            max_steps=self.train_config.get("max_steps", 60),
            learning_rate=float(self.train_config.get("learning_rate", 2e-4)),
            logging_steps=self.train_config.get("logging_steps", 1),
            save_steps=self.train_config.get("save_steps", 100),
            eval_steps=self.train_config.get("eval_steps", 50) if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            optim=self.train_config.get("optim", "adamw_8bit"),
            weight_decay=self.train_config.get("weight_decay", 0.001),
            lr_scheduler_type=self.train_config.get("lr_scheduler_type", "linear"),
            seed=self.train_config.get("seed", 3407),
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            report_to="none",
            dataloader_num_workers=self.train_config.get("dataloader_num_workers", 2),
            remove_unused_columns=False,
        )

        # Trainer 생성
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        # 학습 시작
        print("Starting training...")
        trainer_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 메모리 통계 출력
        self._print_memory_stats(trainer_stats)

        return trainer_stats

    def _print_memory_stats(self, trainer_stats):
        """학습 후 메모리 통계를 출력합니다."""
        gpu_stats = torch.cuda.get_device_properties(0)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
        print(f"Training time: {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes")
        print(f"Peak GPU memory: {used_memory} GB / {max_memory} GB ({used_memory / max_memory * 100:.1f}%)")
        print("=" * 50)

    def save_model(
        self,
        output_path: str | Path | None = None,
        save_merged: bool = False,
    ):
        """모델을 저장합니다."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded.")

        output_path = Path(output_path or self.output_dir) / "lora_model"
        output_path.mkdir(parents=True, exist_ok=True)

        if save_merged:
            print(f"Saving merged model to {output_path}...")
            self.model.save_pretrained_merged(str(output_path), self.tokenizer)
        else:
            print(f"Saving LoRA adapters to {output_path}...")
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))

        print(f"Model saved to {output_path}")
        return output_path

    def inference(
        self,
        image_path: str | Path,
        prompt: str = "<image>\nFree OCR. ",
        save_results: bool = False,
        output_path: str = "./output",
    ) -> str:
        """모델 추론을 수행합니다."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                "unsloth is required. Install it with: pip install unsloth"
            ) from e

        FastVisionModel.for_inference(self.model)

        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=output_path,
            image_size=self.image_size,
            base_size=self.base_size,
            crop_mode=self.crop_mode,
            save_results=save_results,
            test_compress=False,
        )

        return result
