"""
Inference Module

이미지 디렉토리와 Teacher 설정을 받아서 추론 결과를 저장합니다.
결과는 학습용 데이터셋 형식 (JSONL/JSON)으로 저장됩니다.
"""

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

# Optional imports
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Default prompts config path
DEFAULT_PROMPTS_CONFIG = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"


def load_prompts_config(prompts_config_path: str | Path | None = None) -> dict:
    """prompts.yaml 로드"""
    path = Path(prompts_config_path) if prompts_config_path else DEFAULT_PROMPTS_CONFIG
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f).get("prompts", {})


def get_student_instruction(prompt_key: str, prompts_config_path: str | Path | None = None) -> str:
    """prompts.yaml에서 student instruction 가져오기"""
    prompts = load_prompts_config(prompts_config_path)
    if prompt_key not in prompts:
        available = list(prompts.keys())
        raise ValueError(f"Unknown prompt_key: '{prompt_key}'. Available: {available}")
    return prompts[prompt_key].get("instruction", "<image>\nFree OCR. ")


class BaseInferencer(ABC):
    """추론기 기본 클래스"""

    @abstractmethod
    def infer(self, image_path: Path) -> str | None:
        """단일 이미지 추론"""
        pass

    @abstractmethod
    def infer_batch(self, image_paths: list[Path]) -> list[dict]:
        """배치 이미지 추론"""
        pass


class APIInferencer(BaseInferencer):
    """OpenAI 호환 API를 통한 추론기"""

    def __init__(
        self,
        config: dict,
        prompt_key: str,
        prompts_config_path: str | Path | None = None,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install it with: pip install openai")

        api_config = config.get("api", {})
        gen_config = config.get("generation", {})
        req_config = config.get("request", {})

        # Teacher 프롬프트 (config에서)
        prompts = config.get("prompts", {})
        if prompt_key not in prompts:
            available_keys = list(prompts.keys())
            raise ValueError(
                f"Unknown prompt_key: '{prompt_key}' in teacher config. "
                f"Available keys: {available_keys}"
            )

        teacher_prompts = prompts[prompt_key]

        self.client = AsyncOpenAI(
            base_url=api_config.get("base_url", "http://localhost:8000/v1"),
            api_key=api_config.get("api_key", "dummy"),
            timeout=req_config.get("timeout", 120),
        )
        self.model_name = api_config.get("model_name", "model")
        self.temperature = gen_config.get("temperature", 0.1)
        self.max_tokens = gen_config.get("max_tokens", 4096)
        self.top_p = gen_config.get("top_p", 1.0)
        self.max_retries = req_config.get("max_retries", 3)
        self.concurrent_requests = req_config.get("concurrent_requests", 4)

        self.system_prompt = teacher_prompts.get("system", "Extract text from the image.")
        self.user_prompt = teacher_prompts.get("user", "Extract all text from this image.")

        # Student instruction (prompts.yaml에서)
        self.student_instruction = get_student_instruction(prompt_key, prompts_config_path)

    def _encode_image(self, image_path: Path) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """이미지 MIME 타입 반환"""
        ext = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")

    async def _infer_single(
        self,
        image_path: Path,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Path, str | None]:
        """단일 이미지 비동기 추론"""
        async with semaphore:
            image_data = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                        },
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ]

            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                    )
                    return image_path, response.choices[0].message.content
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"Failed {image_path}: {e}")
                        return image_path, None

    def infer(self, image_path: Path) -> str | None:
        """단일 이미지 추론 (동기)"""
        async def _run():
            sem = asyncio.Semaphore(1)
            _, result = await self._infer_single(image_path, sem)
            return result
        return asyncio.run(_run())

    def infer_batch(self, image_paths: list[Path]) -> list[dict]:
        """배치 이미지 추론"""
        async def _run():
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            tasks = [self._infer_single(p, semaphore) for p in image_paths]

            results = []
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Inferring (API)",
            ):
                image_path, text = await coro
                if text is not None:
                    results.append({
                        "image_path": str(image_path),
                        "text": text.strip(),
                    })
            return results

        return asyncio.run(_run())


class LocalInferencer(BaseInferencer):
    """로컬 모델을 통한 추론기"""

    def __init__(
        self,
        config: dict,
        prompt_key: str,
        prompts_config_path: str | Path | None = None,
    ):
        local_config = config.get("local", {})
        image_config = config.get("image", {})

        self.model_path = local_config.get("model_path", "./deepseek_ocr")
        self.load_in_4bit = local_config.get("load_in_4bit", False)
        self.trust_remote_code = local_config.get("trust_remote_code", True)

        self.image_size = image_config.get("image_size", 640)
        self.base_size = image_config.get("base_size", 1024)
        self.crop_mode = image_config.get("crop_mode", True)

        # prompts.yaml에서 instruction 가져오기 (teacher와 student 동일)
        self.instruction = get_student_instruction(prompt_key, prompts_config_path)
        self.student_instruction = self.instruction

        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """모델 로드 (lazy loading)"""
        if self.model is not None:
            return

        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError("unsloth is required. Install it with: pip install unsloth") from e

        from transformers import AutoModel
        import os
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

        print(f"Loading model from {self.model_path}...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_path,
            load_in_4bit=self.load_in_4bit,
            auto_model=AutoModel,
            trust_remote_code=self.trust_remote_code,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(self.model)
        print("Model loaded!")

    def infer(self, image_path: Path) -> str | None:
        """단일 이미지 추론"""
        self._load_model()

        try:
            result = self.model.infer(
                self.tokenizer,
                prompt=self.instruction,
                image_file=str(image_path),
                output_path="./temp_output",
                image_size=self.image_size,
                base_size=self.base_size,
                crop_mode=self.crop_mode,
                save_results=False,
                test_compress=False,
            )
            # result가 리스트일 수 있음
            if isinstance(result, list):
                return " ".join(str(r) for r in result if r).strip()
            return str(result).strip() if result else None
        except Exception as e:
            print(f"Failed {image_path}: {e}")
            return None

    def infer_batch(self, image_paths: list[Path]) -> list[dict]:
        """배치 이미지 추론"""
        self._load_model()

        results = []
        for image_path in tqdm(image_paths, desc="Inferring (Local)"):
            text = self.infer(image_path)
            if text is not None:
                results.append({
                    "image_path": str(image_path),
                    "text": text,
                })
        return results


class DatasetInferencer:
    """이미지 디렉토리에서 데이터셋을 생성하는 추론기"""

    def __init__(
        self,
        config_path: str | Path,
        prompt_key: str,
        prompts_config_path: str | Path | None = None,
    ):
        self.config_path = Path(config_path)
        self.prompt_key = prompt_key
        self.prompts_config_path = prompts_config_path
        self.config = self._load_config()

        # 추론기 타입에 따라 인스턴스 생성
        inferencer_type = self.config.get("type", "api")
        if inferencer_type == "api":
            self.inferencer = APIInferencer(self.config, prompt_key, prompts_config_path)
        elif inferencer_type == "local":
            self.inferencer = LocalInferencer(self.config, prompt_key, prompts_config_path)
        else:
            raise ValueError(f"Unknown inferencer type: {inferencer_type}")

        # 출력 설정
        output_config = self.config.get("output", {})
        self.output_format = output_config.get("format", "jsonl")
        self.save_images = output_config.get("save_images", False)

    def _load_config(self) -> dict:
        """설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _get_image_paths(self, source: str | Path) -> list[Path]:
        """이미지 경로 리스트 가져오기"""
        source = Path(source)

        if source.is_file():
            return [source]

        if source.is_dir():
            image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
            paths = [
                p for p in sorted(source.iterdir())
                if p.suffix.lower() in image_extensions
            ]
            if not paths:
                raise ValueError(f"No image files found in {source}")
            return paths

        raise ValueError(f"Source must be an image file or directory: {source}")

    def run(
        self,
        image_source: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        추론 실행 및 데이터셋 저장

        Args:
            image_source: 이미지 파일 또는 이미지가 있는 디렉토리
            output_path: 출력 파일 경로 (.jsonl 또는 .json)

        Returns:
            저장된 데이터셋 파일 경로
        """
        image_paths = self._get_image_paths(image_source)
        print(f"Found {len(image_paths)} images")
        print(f"Using prompt key: {self.prompt_key}")

        # 추론 실행
        infer_results = self.inferencer.infer_batch(image_paths)
        print(f"Successfully inferred {len(infer_results)} images")

        # Student instruction 가져오기
        student_instruction = self.inferencer.student_instruction

        # 학습용 데이터셋 형식으로 변환
        dataset = []

        for result in infer_results:
            item = {
                "messages": [
                    {
                        "role": "<|User|>",
                        "content": student_instruction,
                        "images": [result["image_path"]],
                    },
                    {
                        "role": "<|Assistant|>",
                        "content": result["text"],
                    },
                ],
                "metadata": {
                    "source_image": result["image_path"],
                    "prompt_key": self.prompt_key,
                    "config": str(self.config_path),
                },
            }
            dataset.append(item)

        # 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "jsonl" or output_path.suffix == ".jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved to {output_path}")
        return output_path


def run_inference(
    image_source: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    prompt_key: str,
    prompts_config_path: str | Path | None = None,
) -> Path:
    """
    추론 실행 편의 함수

    Args:
        image_source: 이미지 파일 또는 디렉토리
        output_path: 출력 파일 경로
        config_path: Teacher 설정 yaml 경로
        prompt_key: 프롬프트 key (prompts.yaml 및 teacher config의 key)
        prompts_config_path: prompts.yaml 경로 (None이면 기본 경로)

    Returns:
        저장된 데이터셋 파일 경로

    Example:
        >>> from src.data.infer import run_inference
        >>> run_inference(
        ...     image_source="./images",
        ...     output_path="./dataset.jsonl",
        ...     config_path="./config/teacher_api.yaml",
        ...     prompt_key="ocr_markdown",
        ... )
    """
    inferencer = DatasetInferencer(config_path, prompt_key, prompts_config_path)
    return inferencer.run(image_source, output_path)


def list_available_prompts(config_path: str | Path | None = None) -> list[str]:
    """prompts.yaml에서 사용 가능한 prompt key 목록 반환"""
    prompts = load_prompts_config(config_path)
    return list(prompts.keys())
