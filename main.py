#!/usr/bin/env python3
"""
VLM Finetuning CLI

PDF → 이미지 변환, Teacher 모델 추론, 학습을 위한 CLI 도구
"""

import argparse
import datetime
import sys
from pathlib import Path


def cmd_pdf2img(args):
    """PDF를 이미지로 변환"""
    from src.data.pdf2img import pdf2img

    print(f"Source: {args.source}")
    print(f"Output directory: {args.output}")
    print(f"DPI: {args.dpi}")
    print(f"Format: {args.format}")

    image_paths = pdf2img(
        source=args.source,
        output_dir=args.output,
        dpi=args.dpi,
        image_format=args.format,
        start_page=args.start_page,
        end_page=args.end_page,
    )

    print(f"\nConverted {len(image_paths)} pages")
    print(f"Images saved to: {args.output}")


def cmd_infer(args):
    """이미지에서 Teacher 모델로 추론하여 데이터셋 생성"""
    from src.data.infer import list_available_prompts, run_inference

    # prompt key 목록 보기
    if args.list_prompts:
        prompts = list_available_prompts(args.config)
        print(f"Available prompt keys in {args.config}:")
        for key in prompts:
            print(f"  - {key}")
        return

    print(f"Image source: {args.images}")
    print(f"Config: {args.config}")
    print(f"Prompt key: {args.prompt_key}")
    print(f"Output: {args.output}")

    output_path = run_inference(
        image_source=args.images,
        output_path=args.output,
        config_path=args.config,
        prompt_key=args.prompt_key,
    )

    print(f"\nDataset saved to: {output_path}")


def cmd_train(args):
    """모델 학습"""
    from src.train.trainer import VLMTrainer

    trainer = VLMTrainer(
        config_path=args.config,
        env_path=args.env,
    )

    print(f"Layer mode: {args.mode or trainer.training_mode}")
    print(f"Base model: {trainer.base_model_path}")
    print(f"Dataset: {args.dataset}")

    # 모델 로드
    trainer.load_model()

    # LoRA 설정 (레이어 선택)
    trainer.setup_lora(mode=args.mode or trainer.training_mode)

    # 학습
    trainer.train(
        dataset=args.dataset,
        eval_dataset=args.eval_dataset,
        resume_from_checkpoint=args.resume,
    )

    # 모델 저장
    trainer.save_model(
        output_path=args.output,
        save_merged=args.save_merged,
    )


def cmd_evaluate(args):
    """모델 평가"""
    import json

    from src.eval.metrics import evaluate_model
    from src.train.trainer import VLMTrainer

    trainer = VLMTrainer(config_path=args.train_config)

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
        prompt=args.prompt or "<image>\nFree OCR. ",
        verbose=True,
    )

    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            save_results = {k: v for k, v in results.items() if k != "detailed_results"}
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")


def cmd_inspect(args):
    """모델 레이어 검사"""
    from src.train.layers import inspect_model_layers
    from src.train.trainer import VLMTrainer

    trainer = VLMTrainer(config_path=args.train_config)

    # 모델 로드
    trainer.load_model()

    # 레이어 검사
    layers = inspect_model_layers(trainer.model, pattern=args.pattern)

    print(f"\nFound {len(layers)} layers")
    if args.pattern:
        print(f"Pattern: {args.pattern}")
    print("=" * 50)

    for layer in layers[: args.limit]:
        print(layer)

    if len(layers) > args.limit:
        print(f"... and {len(layers) - args.limit} more")


def main():
    parser = argparse.ArgumentParser(
        description="VLM Finetuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Convert PDF to images
  uv run main.py pdf2img --source document.pdf

  # Step 2: List available prompt keys
  uv run main.py infer --config config/teacher_api.yaml --list-prompts

  # Step 2: Run inference with Teacher model (API)
  uv run main.py infer --images ./images --config config/teacher_api.yaml --prompt-key ocr_markdown

  # Step 2 (alt): Run inference with local model
  uv run main.py infer --images ./images --config config/teacher_local.yaml --prompt-key ocr_simple

  # Step 3: Train model (--mode selects layers: vision/llm/both)
  uv run main.py train --dataset dataset.jsonl --mode vision

  # Evaluate model
  uv run main.py evaluate --dataset eval.jsonl

  # Inspect model layers
  uv run main.py inspect --pattern "vision"
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
        "--source",
        type=str,
        required=True,
        help="PDF file or directory containing PDFs",
    )
    pdf2img_parser.add_argument(
        "--output",
        type=str,
        default=f"./dataset/img/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Output directory for images",
    )
    pdf2img_parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image resolution (default: 200)",
    )
    pdf2img_parser.add_argument(
        "--format",
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
        help="Run inference on images using Teacher model",
    )
    infer_parser.add_argument(
        "--images",
        type=str,
        default=f"./dataset/img/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Image file or directory containing images",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Teacher model config YAML (e.g., config/teacher_api.yaml)",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default=f"./dataset/json/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
        help="Output dataset file (e.g., dataset.jsonl)",
    )
    infer_parser.add_argument(
        "--prompt-key",
        type=str,
        default="ocr_simple",
        help="Prompt key from config YAML (e.g., ocr_markdown, ocr_simple, layout)",
    )
    infer_parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt keys in the config file",
    )

    # =============================================
    # train command
    # =============================================
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL)",
    )
    train_parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset",
    )
    train_parser.add_argument(
        "--mode",
        type=str,
        choices=["vision", "llm", "both"],
        default=None,
        help="Layer selection mode: vision (encoder), llm (decoder), both",
    )
    train_parser.add_argument(
        "--config",
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
        "--output",
        type=str,
        default=None,
        help="Output directory for saved model",
    )
    train_parser.add_argument(
        "--resume",
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
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset",
    )
    eval_parser.add_argument(
        "--train-config",
        type=str,
        default="config/train_config.yaml",
        help="Training config YAML (for model settings)",
    )
    eval_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Inference prompt",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )

    # =============================================
    # inspect command
    # =============================================
    inspect_parser = subparsers.add_parser("inspect", help="Inspect model layers")
    inspect_parser.add_argument(
        "--train-config",
        type=str,
        default="config/train_config.yaml",
        help="Training config YAML",
    )
    inspect_parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex pattern to filter layers",
    )
    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of layers to show",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

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
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
