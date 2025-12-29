#!/usr/bin/env python3
"""
VLM Finetuning CLI

PDF → 이미지 변환, Teacher 모델 추론, 학습을 위한 CLI 도구
"""

import argparse
import datetime
import signal
import sys
from pathlib import Path

from loguru import logger


class GracefulShutdown:
    """Graceful shutdown handler for training"""

    shutdown_requested = False
    trainer = None

    @classmethod
    def request_shutdown(cls, signum, frame):
        """Signal handler for graceful shutdown"""
        if cls.shutdown_requested:
            logger.warning("Force shutdown requested, exiting immediately...")
            sys.exit(1)

        cls.shutdown_requested = True
        logger.warning("Shutdown requested, finishing current step...")

        if cls.trainer is not None:
            try:
                logger.info("Saving emergency checkpoint...")
                cls.trainer.save_model(save_merged=False)
                logger.success("Emergency checkpoint saved")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

# 기본 모델 경로 상수
DEFAULT_MODEL_DIR = Path("models/deepseek_ocr")
DEFAULT_MODEL_REPO = "unsloth/DeepSeek-OCR"


def ensure_model_exists():
    """모델 디렉토리가 없으면 Hugging Face에서 다운로드"""
    if DEFAULT_MODEL_DIR.exists():
        logger.debug(f"Model found at {DEFAULT_MODEL_DIR}")
        return

    logger.info(f"Model not found at {DEFAULT_MODEL_DIR}")
    logger.info(f"Downloading {DEFAULT_MODEL_REPO} from Hugging Face...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            DEFAULT_MODEL_REPO,
            local_dir=str(DEFAULT_MODEL_DIR),
        )
        logger.success(f"Model downloaded to {DEFAULT_MODEL_DIR}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def cmd_pdf2img(args):
    """PDF를 이미지로 변환"""
    from src.data.pdf2img import pdf2img

    logger.info(f"Source: {args.source}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"DPI: {args.dpi}, Format: {args.format}")

    image_paths = pdf2img(
        source=args.source,
        output_dir=args.output,
        dpi=args.dpi,
        image_format=args.format,
        start_page=args.start_page,
        end_page=args.end_page,
    )

    logger.success(f"Converted {len(image_paths)} pages -> {args.output}")


def cmd_infer(args):
    """이미지에서 Teacher 모델로 추론하여 데이터셋 생성"""
    import tempfile

    import yaml

    from src.data.infer import (
        DatasetInferencer,
        list_available_prompts,
        run_inference,
        run_inference_markdown,
    )

    # prompt key 목록 보기
    if args.list_prompts:
        prompts = list_available_prompts(args.config)
        logger.info(f"Available tasks in {args.config}:")
        for key in prompts:
            logger.info(f"  - {key}")
        return

    # 체크포인트 재개 모드 처리
    config_path = args.config
    task = args.task
    images = args.img
    output = args.output
    checkpoint_path = None

    if args.resume:
        # --resume가 파일 경로로 제공된 경우
        if args.resume != "auto" and Path(args.resume).exists():
            checkpoint_path = Path(args.resume)
            checkpoint_data = DatasetInferencer.load_inference_checkpoint(checkpoint_path)

            if checkpoint_data:
                meta = checkpoint_data.get("meta", {})
                logger.info("Found inference checkpoint metadata")

                # 인자가 없으면 메타데이터에서 자동 로드
                if not args.config:
                    config_path = meta.get("config_path")
                    logger.info(f"Auto-loaded config from checkpoint: {config_path}")

                if args.task == "document" and meta.get("task"):  # 기본값이면 덮어쓰기
                    task = meta.get("task")
                    logger.info(f"Auto-loaded task from checkpoint: {task}")

                if meta.get("image_source"):
                    images = meta.get("image_source")
                    logger.info(f"Auto-loaded image_source from checkpoint: {images}")

                if not args.output and meta.get("output_path"):
                    output = meta.get("output_path")
                    logger.info(f"Auto-loaded output from checkpoint: {output}")

                # 인자가 제공된 경우 검증
                is_valid, warnings = DatasetInferencer.validate_inference_checkpoint(
                    checkpoint_data,
                    image_source=args.img if args.img != f"./data/img/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}" else None,
                    config_path=args.config if args.config else None,
                    task=args.task if args.task != "document" else None,
                )

                if not is_valid:
                    logger.warning("Inference configuration mismatch detected:")
                    for warning in warnings:
                        logger.warning(warning)
                    logger.warning("Proceeding with current arguments (not checkpoint metadata)")
            else:
                logger.warning(f"Could not load checkpoint metadata from {checkpoint_path}")

    # config 필수 체크
    if not config_path:
        logger.error("Config is required. Use --config or resume from a checkpoint with metadata.")
        sys.exit(1)

    # config에서 output format 읽기
    with open(config_path) as f:
        config = yaml.safe_load(f)
    output_format = config.get("output", {}).get("format", "jsonl")
    is_markdown = output_format == "md"

    # output 경로 결정
    if output is None:
        if is_markdown:
            output = f"./data/markdown/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        else:
            output = f"./data/json/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

    # checkpoint 경로 결정 (--resume만 사용된 경우)
    if args.resume == "auto":
        checkpoint_path = Path(output).with_suffix(".checkpoint.json")

    # PDF 입력 처리: PDF → 이미지 변환 후 추론
    temp_dir_context = None
    try:
        if args.pdf:
            from src.data.pdf2img import pdf2img

            # TemporaryDirectory를 사용하여 자동 정리
            temp_dir_context = tempfile.TemporaryDirectory(prefix="ocr_infer_")
            temp_dir = temp_dir_context.name
            logger.info(f"PDF source: {args.pdf}")
            logger.info(f"Converting PDF to images in: {temp_dir}")

            image_paths = pdf2img(
                source=args.pdf,
                output_dir=temp_dir,
                dpi=args.dpi,
                image_format="png",
            )
            logger.info(f"Converted {len(image_paths)} pages")
            image_source = temp_dir
        else:
            image_source = images

        logger.info(f"Image source: {image_source}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Task: {task}")
        logger.info(f"Output: {output}")
        logger.info(f"Format: {'markdown' if is_markdown else output_format}")
        if args.resume:
            logger.info(f"Resume mode: enabled")

        if is_markdown:
            # 마크다운 파일로 저장 (같은 PDF 페이지는 자동 병합)
            output_dir = run_inference_markdown(
                image_source=image_source,
                output_dir=output,
                config_path=config_path,
                task=task,
            )
            logger.success(f"Markdown files saved to: {output_dir}")
        elif args.resume:
            # 체크포인트 모드로 데이터셋 저장
            inferencer = DatasetInferencer(config_path, task)
            output_path = inferencer.run_with_checkpoint(
                image_source=image_source,
                output_path=output,
                checkpoint_path=checkpoint_path,
            )
            logger.success(f"Dataset saved to: {output_path}")
        else:
            # 일반 모드로 데이터셋 저장
            output_path = run_inference(
                image_source=image_source,
                output_path=output,
                config_path=config_path,
                task=task,
            )
            logger.success(f"Dataset saved to: {output_path}")
    finally:
        # 임시 디렉토리 정리
        if temp_dir_context is not None:
            temp_dir_context.cleanup()
            logger.debug("Cleaned up temporary directory")


def cmd_train(args):
    """모델 학습"""
    from src.train.trainer import VLMTrainer

    # Graceful shutdown 설정
    signal.signal(signal.SIGINT, GracefulShutdown.request_shutdown)
    signal.signal(signal.SIGTERM, GracefulShutdown.request_shutdown)

    # resume 시 메타데이터 로드 및 검증/자동로드
    dataset = args.dataset
    eval_dataset = args.eval_dataset
    config = args.config
    mode = args.mode
    output = args.output

    if args.resume:
        meta = VLMTrainer.load_training_meta(args.resume)
        if meta:
            logger.info("Found training metadata from checkpoint")

            # 인자가 없으면 메타데이터에서 자동 로드
            if not dataset:
                dataset = meta.get("dataset")
                logger.info(f"Auto-loaded dataset from metadata: {dataset}")

            if not eval_dataset and meta.get("eval_dataset"):
                eval_dataset = meta.get("eval_dataset")
                logger.info(f"Auto-loaded eval_dataset from metadata: {eval_dataset}")

            if not config and meta.get("config_path"):
                config = meta.get("config_path")
                logger.info(f"Auto-loaded config from metadata: {config}")

            if not mode and meta.get("mode"):
                mode = meta.get("mode")
                logger.info(f"Auto-loaded mode from metadata: {mode}")

            if not output and meta.get("output_dir"):
                output = meta.get("output_dir")
                logger.info(f"Auto-loaded output_dir from metadata: {output}")

            # 인자가 제공된 경우 검증
            is_valid, warnings = VLMTrainer.validate_training_meta(
                meta,
                dataset_path=args.dataset if args.dataset else None,
                config_path=args.config if args.config != "config/train_config.yaml" else None,
                mode=args.mode,
            )

            if not is_valid:
                logger.warning("Training configuration mismatch detected:")
                for warning in warnings:
                    logger.warning(warning)
                logger.warning("Proceeding with current arguments (not metadata)")
        else:
            logger.warning("No training metadata found for checkpoint")

    # dataset이 여전히 없으면 에러
    if not dataset:
        logger.error("Dataset is required. Use --dataset or resume from a checkpoint with metadata.")
        sys.exit(1)

    trainer = VLMTrainer(
        config_path=config,
        env_path=args.env,
    )

    # GracefulShutdown에 trainer 등록
    GracefulShutdown.trainer = trainer

    # CLI에서 output이 주어지면 output_dir 설정 (checkpoint_dir도 자동 설정)
    if output:
        trainer.set_output_dir(output)

    effective_mode = mode or trainer.training_mode
    logger.info(f"Layer mode: {effective_mode}")
    logger.info(f"Base model: {trainer.base_model_path}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output dir: {trainer.output_dir}")
    logger.info(f"Checkpoint dir: {trainer.checkpoint_dir}")

    # 모델 로드
    trainer.load_model()

    # LoRA 설정 (레이어 선택)
    trainer.setup_lora(mode=effective_mode)

    # 학습 (shutdown 요청 시 조기 종료)
    if GracefulShutdown.shutdown_requested:
        logger.warning("Shutdown requested before training started")
        return

    # 학습
    trainer.train(
        dataset=dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume,
        mode=effective_mode,
    )

    # 모델 저장 (CLI --save-merged가 있으면 덮어쓰기, 없으면 config 사용)
    save_merged = args.save_merged or trainer.save_merged
    trainer.save_model(
        output_path=trainer.output_dir,
        save_merged=save_merged,
    )


def cmd_evaluate(args):
    """모델 평가"""
    import csv
    import json

    from src.data.infer import get_student_instruction
    from src.eval.metrics import evaluate_model
    from src.train.trainer import VLMTrainer

    trainer = VLMTrainer(config_path=args.train_config)

    # Task에서 prompt 가져오기
    prompt = get_student_instruction(args.task)
    logger.info(f"Task: {args.task}")
    logger.info(f"Prompt: {prompt}")

    # 모델 로드
    trainer.load_model()

    # 데이터셋 로드
    eval_data = trainer.prepare_dataset(args.dataset)

    # 평가
    results = evaluate_model(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        eval_data=eval_data,
        image_size=trainer.image_size,
        base_size=trainer.base_size,
        crop_mode=trainer.crop_mode,
        prompt=prompt,
        verbose=True,
    )

    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON 저장 (메트릭만)
        with open(output_path, "w", encoding="utf-8") as f:
            save_results = {k: v for k, v in results.items() if k != "detailed_results"}
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        logger.success(f"Metrics saved to {output_path}")

        # CSV 저장 (상세 결과)
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "ground_truth", "prediction", "cer", "wer"])
            for item in results.get("detailed_results", []):
                writer.writerow([
                    item.get("image_path", ""),
                    item.get("reference", ""),
                    item.get("hypothesis", ""),
                    f"{item.get('cer', 0):.4f}",
                    f"{item.get('wer', 0):.4f}",
                ])
        logger.success(f"Detailed results saved to {csv_path}")


def cmd_inspect(args):
    """모델 레이어 검사"""
    from src.train.layers import inspect_model_layers
    from src.train.trainer import VLMTrainer

    trainer = VLMTrainer(config_path=args.train_config)

    # 모델 로드
    trainer.load_model()

    # 레이어 검사
    layers = inspect_model_layers(trainer.model, pattern=args.pattern)

    logger.info(f"Found {len(layers)} layers")
    if args.pattern:
        logger.info(f"Pattern: {args.pattern}")

    for layer in layers[: args.limit]:
        logger.info(layer)

    if len(layers) > args.limit:
        logger.info(f"... and {len(layers) - args.limit} more")


def main():
    parser = argparse.ArgumentParser(
        description="VLM Finetuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PDF to images (optional, can use --pdf directly in infer)
  uv run main.py pdf2img -s document.pdf

  # List available tasks
  uv run main.py infer -c config/teacher_api.yaml -l

  # Run inference from images
  uv run main.py infer -i ./images -c config/teacher_api.yaml -t document

  # Run inference directly from PDF (converts internally)
  uv run main.py infer -p document.pdf -c config/teacher_api.yaml -t document

  # Resume inference from checkpoint
  uv run main.py infer -r ./data/result.checkpoint.json

  # Train model (-m selects layers: vision/llm/both)
  uv run main.py train -d dataset.jsonl -m vision

  # Resume training from checkpoint
  uv run main.py train -r ./output/checkpoints/checkpoint-100

  # Evaluate model
  uv run main.py evaluate -d eval.jsonl -t document -o results.json

  # Inspect model layers
  uv run main.py inspect -p "vision"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =============================================
    # pdf2img command
    # =============================================
    pdf2img_parser = subparsers.add_parser(
        "pdf2img",
        help="Convert PDF to images",
    )
    pdf2img_parser.add_argument(
        "-s", "--source",
        type=str,
        required=True,
        help="PDF file or directory containing PDFs",
    )
    pdf2img_parser.add_argument(
        "-o", "--output",
        type=str,
        default=f"./data/img/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Output directory for images",
    )
    pdf2img_parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=200,
        help="Image resolution (default: 200)",
    )
    pdf2img_parser.add_argument(
        "-f", "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image format (default: png)",
    )
    pdf2img_parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Start page (1-indexed, for single PDF)",
    )
    pdf2img_parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="End page (1-indexed, for single PDF)",
    )

    # =============================================
    # infer command
    # =============================================
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on images or PDF using Teacher model",
    )
    infer_parser.add_argument(
        "-i", "--img",
        type=str,
        default=f"./data/img/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Image file or directory containing images",
    )
    infer_parser.add_argument(
        "-p", "--pdf",
        type=str,
        default=None,
        help="PDF file or directory containing PDFs (converts to images internally)",
    )
    infer_parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)",
    )
    infer_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Teacher model config YAML (e.g., config/teacher_api.yaml). "
             "Optional if resuming with --resume.",
    )
    infer_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (auto-determined from config if not specified)",
    )
    infer_parser.add_argument(
        "-t", "--task",
        type=str,
        default="document",
        help="Task name from prompts.yaml (e.g., document, without layouts)",
    )
    infer_parser.add_argument(
        "-l", "--list-prompts",
        action="store_true",
        help="List available tasks in prompts.yaml",
    )
    infer_parser.add_argument(
        "-r", "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Enable checkpoint mode for resumable inference. "
             "Optionally provide checkpoint file path to resume from.",
    )

    # =============================================
    # train command
    # =============================================
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Path to training dataset (JSONL). Optional if resuming with metadata.",
    )
    train_parser.add_argument(
        "-e", "--eval-dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset",
    )
    train_parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["vision", "llm", "both"],
        default=None,
        help="Layer selection mode: vision (encoder), llm (decoder), both",
    )
    train_parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/train_config.yaml",
        help="Training config YAML",
    )
    train_parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file",
    )
    train_parser.add_argument(
        "-o", "--output",
        type=str,
        default=f"./models/finetuned/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Output directory for saved model",
    )
    train_parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    train_parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save merged model instead of LoRA only",
    )

    # =============================================
    # evaluate command
    # =============================================
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset",
    )
    eval_parser.add_argument(
        "-c", "--train-config",
        type=str,
        default="config/train_config.yaml",
        help="Training config YAML (for model settings)",
    )
    eval_parser.add_argument(
        "-t", "--task",
        type=str,
        default="document",
        help="Task name from prompts.yaml (e.g., document, without layouts)",
    )
    eval_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for results JSON (CSV will also be generated)",
    )

    # =============================================
    # inspect command
    # =============================================
    inspect_parser = subparsers.add_parser("inspect", help="Inspect model layers")
    inspect_parser.add_argument(
        "-c", "--train-config",
        type=str,
        default="config/train_config.yaml",
        help="Training config YAML",
    )
    inspect_parser.add_argument(
        "-p", "--pattern",
        type=str,
        default=None,
        help="Regex pattern to filter layers",
    )
    inspect_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=100,
        help="Maximum number of layers to show",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 모델 체크 (train, evaluate, inspect, infer 명령에서 필요)
    if args.command in ("train", "evaluate", "inspect", "infer"):
        ensure_model_exists()

    # 명령 실행
    commands = {
        "pdf2img": cmd_pdf2img,
        "infer": cmd_infer,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "inspect": cmd_inspect,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
