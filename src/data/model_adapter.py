"""
Model Adapter Module

외부 모델(models/deepseek_ocr)에 대한 의존성을 추상화하여
모델 버전 변경 시 코드 수정을 최소화합니다.
"""

from typing import Any

from loguru import logger


class DeepSeekOCRAdapter:
    """DeepSeek OCR 모델 어댑터"""

    _image_transform_cls = None
    _dynamic_preprocess_fn = None
    _format_messages_fn = None
    _text_encode_fn = None

    @classmethod
    def _load_module(cls):
        """모델 모듈을 lazy load합니다."""
        if cls._image_transform_cls is not None:
            return

        try:
            from models.deepseek_ocr.modeling_deepseekocr import (
                BasicImageTransform,
                dynamic_preprocess,
                format_messages,
                text_encode,
            )

            cls._image_transform_cls = BasicImageTransform
            cls._dynamic_preprocess_fn = dynamic_preprocess
            cls._format_messages_fn = format_messages
            cls._text_encode_fn = text_encode
        except ImportError as e:
            logger.error(
                f"Failed to import DeepSeek OCR model. "
                f"Make sure 'models/deepseek_ocr' exists: {e}"
            )
            raise ImportError(
                "DeepSeek OCR model not found. "
                "Run 'python main.py train' to download the model first."
            ) from e

    @classmethod
    def get_image_transform(cls, **kwargs) -> Any:
        """BasicImageTransform 인스턴스를 반환합니다."""
        cls._load_module()
        return cls._image_transform_cls(**kwargs)

    @classmethod
    def dynamic_preprocess(
        cls,
        image,
        min_num: int = 2,
        max_num: int = 9,
        image_size: int = 640,
        use_thumbnail: bool = False,
    ) -> tuple:
        """이미지를 동적으로 전처리합니다."""
        cls._load_module()
        return cls._dynamic_preprocess_fn(
            image,
            min_num=min_num,
            max_num=max_num,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
        )

    @classmethod
    def format_messages(cls, *args, **kwargs) -> Any:
        """메시지를 포맷합니다."""
        cls._load_module()
        return cls._format_messages_fn(*args, **kwargs)

    @classmethod
    def text_encode(cls, tokenizer, text: str, bos: bool = False, eos: bool = False) -> list:
        """텍스트를 토큰화합니다."""
        cls._load_module()
        return cls._text_encode_fn(tokenizer, text, bos=bos, eos=eos)
