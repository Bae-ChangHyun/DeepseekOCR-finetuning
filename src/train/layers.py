"""
Layer Selection Module

비전 인코더 또는 LLM 학습을 위한 타겟 레이어 선택
"""

import re
from typing import Literal

import yaml
from loguru import logger


def get_target_modules(
    mode: Literal["vision", "llm", "both"],
    model=None,
    config_path: str | None = None,
    custom_modules: list[str] | None = None,
) -> list[str]:
    """
    학습 모드에 따라 타겟 모듈을 반환합니다.

    Args:
        mode: 학습 모드 ("vision", "llm", "both")
        model: 모델 인스턴스 (레이어 검증용, optional)
        config_path: 설정 파일 경로 (optional)
        custom_modules: 커스텀 모듈 리스트 (optional)

    Returns:
        타겟 모듈 이름 리스트
    """
    if custom_modules:
        return custom_modules

    # 설정 파일에서 로드 시도
    if config_path:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                lora_config = config.get("lora", {})

                if mode == "vision":
                    modules = lora_config.get("vision_target_modules", [])
                    if modules:
                        return modules
                elif mode == "llm":
                    modules = lora_config.get("llm_target_modules", [])
                    if modules:
                        return modules
        except Exception:
            pass

    # 기본 타겟 모듈
    if mode == "vision":
        return get_vision_target_modules()
    elif mode == "llm":
        return get_llm_target_modules()
    elif mode == "both":
        return get_vision_target_modules() + get_llm_target_modules()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'vision', 'llm', or 'both'")


def get_vision_target_modules() -> list[str]:
    """
    비전 인코더 학습을 위한 타겟 모듈을 반환합니다.

    OCR 성능 향상을 위해 비전 인코더의 어텐션과 MLP 레이어를 타겟으로 합니다.
    DeepSeek-OCR은 transformer 구조를 사용하며 qkv가 통합되어 있습니다.
    """
    return [
        # Vision Transformer Self-Attention (DeepSeek-OCR uses merged qkv_proj)
        # PEFT matches modules ending with these names
        "qkv_proj",
        "out_proj",
        # Vision Transformer MLP
        "fc1",
        "fc2",
    ]


def get_llm_target_modules() -> list[str]:
    """
    LLM 학습을 위한 타겟 모듈을 반환합니다.

    레이아웃 이해 향상을 위해 LLM의 어텐션과 MLP 레이어를 타겟으로 합니다.
    """
    return [
        # LLM Self-Attention
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # LLM MLP (Gate-Up-Down pattern for LLaMA-style models)
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def get_projector_modules() -> list[str]:
    """
    Vision-Language Projector 모듈을 반환합니다.

    비전 인코더와 LLM 사이의 프로젝터 레이어입니다.
    """
    return [
        "multi_modal_projector",
        "vision_embed_tokens",
    ]


def inspect_model_layers(model, pattern: str | None = None) -> list[str]:
    """
    모델의 모든 레이어 이름을 검사합니다.

    Args:
        model: PyTorch 모델
        pattern: 필터링할 정규식 패턴 (optional)

    Returns:
        레이어 이름 리스트
    """
    layer_names = []

    for name, _ in model.named_modules():
        if pattern:
            if re.search(pattern, name):
                layer_names.append(name)
        else:
            layer_names.append(name)

    return layer_names


def get_trainable_params_info(model) -> dict:
    """
    모델의 학습 가능한 파라미터 정보를 반환합니다.

    Returns:
        dict with:
            - total_params: 전체 파라미터 수
            - trainable_params: 학습 가능한 파라미터 수
            - trainable_percent: 학습 가능한 파라미터 비율
            - trainable_layers: 학습 가능한 레이어 이름 리스트
    """
    total_params = 0
    trainable_params = 0
    trainable_layers = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # 레이어 이름 추출 (파라미터 이름에서 .weight, .bias 제거)
            layer_name = ".".join(name.split(".")[:-1])
            if layer_name not in trainable_layers:
                trainable_layers.append(layer_name)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": (trainable_params / total_params * 100) if total_params > 0 else 0,
        "trainable_layers": trainable_layers,
    }


def print_trainable_params(model) -> None:
    """모델의 학습 가능한 파라미터 정보를 출력합니다."""
    info = get_trainable_params_info(model)

    logger.info("Trainable Parameters Info")
    logger.info(f"Total parameters: {info['total_params']:,}")
    logger.info(f"Trainable parameters: {info['trainable_params']:,}")
    logger.info(f"Trainable %: {info['trainable_percent']:.2f}%")
    logger.debug(f"Trainable layers ({len(info['trainable_layers'])})")
    for layer in info["trainable_layers"][:20]:  # 처음 20개만 출력
        logger.debug(f"  - {layer}")
    if len(info["trainable_layers"]) > 20:
        logger.debug(f"  ... and {len(info['trainable_layers']) - 20} more")
