"""
VLM Trainer Module

VLM 파인튜닝을 위한 학습 로직
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModel, Trainer, TrainingArguments

# 학습 메타데이터 파일명
TRAINING_META_FILE = "training_meta.json"

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
        self._config_path = config_path
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

        # 출력 설정 (나중에 set_output_dir로 오버라이드 가능)
        output_config = self.config.get("output", {})
        self._default_output_dir = os.getenv(
            "OUTPUT_MODEL_PATH", output_config.get("output_dir", "./outputs")
        )
        self.output_dir = self._default_output_dir
        self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")
        self.save_merged = output_config.get("save_merged_16bit", False)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def set_output_dir(self, output_dir: str | Path):
        """출력 디렉토리를 설정합니다. checkpoint_dir은 자동으로 output_dir/checkpoints로 설정됩니다."""
        self.output_dir = str(output_dir)
        self.checkpoint_dir = str(Path(output_dir) / "checkpoints")

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

        logger.info(f"Loading model from {self.base_model_path}...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.base_model_path,
            load_in_4bit=self.load_in_4bit,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        logger.success("Model loaded successfully!")

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
            config_path=self._config_path,
            custom_modules=custom_target_modules,
        )

        logger.info(f"Setting up LoRA for {mode} mode...")
        logger.debug(f"Target modules: {target_modules}")

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

        logger.info(f"Loaded {len(data)} samples from {dataset_path}")
        return data

    def save_training_meta(
        self,
        dataset_path: str | Path,
        eval_dataset_path: str | Path | None = None,
        mode: str | None = None,
    ):
        """학습 메타데이터를 저장합니다."""
        meta_path = Path(self.output_dir) / TRAINING_META_FILE
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "dataset": str(Path(dataset_path).resolve()),
            "eval_dataset": str(Path(eval_dataset_path).resolve()) if eval_dataset_path else None,
            "config_path": str(Path(self._config_path).resolve()) if self._config_path else None,
            "mode": mode or self.training_mode,
            "output_dir": str(Path(self.output_dir).resolve()),
            "checkpoint_dir": str(Path(self.checkpoint_dir).resolve()),
            "base_model_path": self.base_model_path,
            "created_at": datetime.now().isoformat(),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.debug(f"Training metadata saved to {meta_path}")
        return meta_path

    @staticmethod
    def load_training_meta(checkpoint_path: str | Path) -> dict | None:
        """
        체크포인트에서 학습 메타데이터를 로드합니다.

        Args:
            checkpoint_path: 체크포인트 디렉토리 경로 (예: ./output/checkpoints/checkpoint-100)

        Returns:
            메타데이터 딕셔너리 또는 None
        """
        checkpoint_path = Path(checkpoint_path)

        # checkpoint_path가 checkpoint-XXX 형태면 상위 디렉토리에서 메타데이터 찾기
        # ./output/checkpoints/checkpoint-100 -> ./output/training_meta.json
        if checkpoint_path.name.startswith("checkpoint-"):
            meta_path = checkpoint_path.parent.parent / TRAINING_META_FILE
        else:
            # ./output/checkpoints -> ./output/training_meta.json
            meta_path = checkpoint_path.parent / TRAINING_META_FILE

        if not meta_path.exists():
            # 직접 경로도 시도
            alt_meta_path = checkpoint_path / TRAINING_META_FILE
            if alt_meta_path.exists():
                meta_path = alt_meta_path
            else:
                return None

        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load training metadata: {e}")
            return None

    @staticmethod
    def validate_training_meta(
        meta: dict,
        dataset_path: str | Path | None = None,
        config_path: str | Path | None = None,
        mode: str | None = None,
    ) -> tuple[bool, list[str]]:
        """
        학습 메타데이터와 현재 설정을 비교 검증합니다.

        Returns:
            (is_valid, warnings): 검증 결과와 경고 메시지 리스트
        """
        warnings = []

        if dataset_path:
            saved_dataset = meta.get("dataset", "")
            current_dataset = str(Path(dataset_path).resolve())
            if saved_dataset != current_dataset:
                warnings.append(
                    f"Dataset mismatch:\n"
                    f"  - Saved: {saved_dataset}\n"
                    f"  - Current: {current_dataset}"
                )

        if config_path:
            saved_config = meta.get("config_path", "")
            current_config = str(Path(config_path).resolve())
            if saved_config != current_config:
                warnings.append(
                    f"Config mismatch:\n"
                    f"  - Saved: {saved_config}\n"
                    f"  - Current: {current_config}"
                )

        if mode:
            saved_mode = meta.get("mode", "")
            if saved_mode != mode:
                warnings.append(
                    f"Mode mismatch:\n"
                    f"  - Saved: {saved_mode}\n"
                    f"  - Current: {mode}"
                )

        return len(warnings) == 0, warnings

    def train(
        self,
        dataset: list[dict] | str | Path,
        eval_dataset: list[dict] | str | Path | None = None,
        resume_from_checkpoint: str | None = None,
        mode: str | None = None,
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

        # 원본 경로 저장 (메타데이터용)
        dataset_path = dataset if isinstance(dataset, (str, Path)) else None
        eval_dataset_path = eval_dataset if isinstance(eval_dataset, (str, Path)) else None

        # 데이터셋 준비
        if isinstance(dataset, (str, Path)):
            dataset = self.prepare_dataset(dataset)
        if isinstance(eval_dataset, (str, Path)):
            eval_dataset = self.prepare_dataset(eval_dataset)

        # 학습 메타데이터 저장 (resume이 아닐 때만)
        if not resume_from_checkpoint and dataset_path:
            self.save_training_meta(
                dataset_path=dataset_path,
                eval_dataset_path=eval_dataset_path,
                mode=mode or self.training_mode,
            )

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

        # TrainingArguments 설정 (체크포인트는 checkpoint_dir에 저장)
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            per_device_train_batch_size=self.train_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.train_config.get("gradient_accumulation_steps", 4),
            warmup_steps=self.train_config.get("warmup_steps", 5),
            max_steps=self.train_config.get("max_steps", 60),
            learning_rate=float(self.train_config.get("learning_rate", 2e-4)),
            logging_steps=self.train_config.get("logging_steps", 1),
            save_steps=self.train_config.get("save_steps", 100),
            eval_steps=self.train_config.get("eval_steps", 50) if eval_dataset else None,
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
        logger.info("Starting training...")
        trainer_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 메모리 통계 출력
        self._print_memory_stats(trainer_stats)

        return trainer_stats

    def _print_memory_stats(self, trainer_stats):
        """학습 후 메모리 통계를 출력합니다."""
        logger.success("Training Complete!")
        logger.info(
            f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds "
            f"({trainer_stats.metrics['train_runtime'] / 60:.2f} minutes)"
        )

        # CUDA 사용 가능 여부 확인
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping GPU memory stats")
            return

        try:
            gpu_stats = torch.cuda.get_device_properties(0)
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(
                f"Peak GPU memory: {used_memory} GB / {max_memory} GB "
                f"({used_memory / max_memory * 100:.1f}%)"
            )
        except Exception as e:
            logger.warning(f"Could not get GPU memory stats: {e}")

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
            logger.info(f"Saving merged model to {output_path}...")
            self.model.save_pretrained_merged(str(output_path), self.tokenizer)
        else:
            logger.info(f"Saving LoRA adapters to {output_path}...")
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))

        # 학습 설정 저장
        self._save_train_config(output_path)

        logger.success(f"Model saved to {output_path}")
        return output_path

    def _save_train_config(self, output_path: Path):
        """학습에 사용된 설정을 저장합니다."""
        config_path = output_path / "train_config.yaml"

        # 현재 학습 설정 구성
        train_config = {
            "model": {
                "base_model_path": self.base_model_path,
                "load_in_4bit": self.load_in_4bit,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
            },
            "lora": {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            },
            "training": self.train_config,
            "image": {
                "image_size": self.image_size,
                "base_size": self.base_size,
                "crop_mode": self.crop_mode,
            },
            "training_mode": self.training_mode,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f, default_flow_style=False, allow_unicode=True)

        logger.debug(f"Training config saved to {config_path}")

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
