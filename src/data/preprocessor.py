"""
Output Preprocessor Module

추론 결과를 마크다운으로 변환하는 전처리 모듈
팩토리 패턴 + 레지스트리 기반
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path

# 전처리기 레지스트리
_PREPROCESSOR_REGISTRY: dict[str, type["BasePreprocessor"]] = {}


def register_preprocessor(name: str):
    """전처리기를 레지스트리에 등록하는 데코레이터"""
    def decorator(cls: type["BasePreprocessor"]):
        _PREPROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_preprocessor(name: str, **kwargs) -> "BasePreprocessor":
    """레지스트리에서 전처리기를 가져옴"""
    if name not in _PREPROCESSOR_REGISTRY:
        available = list(_PREPROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown preprocessor: '{name}'. Available: {available}")
    return _PREPROCESSOR_REGISTRY[name](**kwargs)


def list_preprocessors() -> list[str]:
    """사용 가능한 전처리기 목록 반환"""
    return list(_PREPROCESSOR_REGISTRY.keys())


class BasePreprocessor(ABC):
    """전처리기 기본 클래스"""

    @abstractmethod
    def process(self, text: str) -> str:
        """추론 결과를 마크다운으로 변환"""
        pass

    def save(self, text: str, output_path: Path) -> Path:
        """전처리 후 마크다운 파일로 저장"""
        processed = self.process(text)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed)

        return output_path


@register_preprocessor("default")
class DefaultPreprocessor(BasePreprocessor):
    """기본 전처리기 - 추론 결과를 그대로 저장"""

    def process(self, text: str) -> str:
        return text.strip()


@register_preprocessor("deepseek-document")
class DeepSeekDocumentPreprocessor(BasePreprocessor):
    """
    DeepSeek OCR document task 전처리기

    입력 형식:
        <|ref|>text<|/ref|><|det|>[[238, 260, 480, 275]]<|/det|>
        '실시간 대중교통 혼잡도 예측 서비스' 개발

        <|ref|>sub_title<|/ref|><|det|>[[47, 315, 152, 333]]<|/det|>
        ## 자기소개서

    출력 형식:
        '실시간 대중교통 혼잡도 예측 서비스' 개발

        ## 자기소개서
    """

    # 태그 패턴들
    REF_DET_PATTERN = re.compile(
        r"<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>\s*\n?",
        re.DOTALL
    )
    # grounding 태그
    GROUNDING_PATTERN = re.compile(r"<\|grounding\|>")
    # 기타 특수 태그
    SPECIAL_TAGS_PATTERN = re.compile(r"<\|[^|]+\|>")

    def process(self, text: str) -> str:
        # ref/det 태그 블록 제거
        result = self.REF_DET_PATTERN.sub("", text)

        # grounding 태그 제거
        result = self.GROUNDING_PATTERN.sub("", result)

        # 연속된 빈 줄 정리 (3개 이상 -> 2개로)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()


@register_preprocessor("deepseek-ocr")
class DeepSeekOCRPreprocessor(BasePreprocessor):
    """
    DeepSeek OCR (Free OCR / other image) task 전처리기

    단순 OCR 결과에서 특수 태그만 제거
    """

    SPECIAL_TAGS_PATTERN = re.compile(r"<\|[^|]+\|>")

    def process(self, text: str) -> str:
        # 특수 태그 제거
        result = self.SPECIAL_TAGS_PATTERN.sub("", text)
        return result.strip()


def get_preprocessor_for_task(task: str, model_type: str = "deepseek") -> BasePreprocessor:
    """
    task와 model_type에 따라 적절한 전처리기 반환

    Args:
        task: 태스크 이름 (document, without layouts, etc.)
        model_type: 모델 타입 (deepseek, default, etc.)

    Returns:
        적절한 전처리기 인스턴스
    """
    if model_type == "deepseek":
        if task == "document":
            return get_preprocessor("deepseek-document")
        elif task in ["without layouts", "other image"]:
            return get_preprocessor("deepseek-ocr")
        else:
            return get_preprocessor("deepseek-ocr")

    return get_preprocessor("default")
