# DeepSeek-OCR 파인튜닝 툴킷

<div align="center">

![OCR Finetune](https://via.placeholder.com/150x150?text=OCR+Finetune)

**문서 이미지를 마크다운으로, 더 정확하게**<br/>
DeepSeek-OCR 모델을 위한 프로페셔널 파인튜닝 및 추론 도구

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![uv](https://img.shields.io/badge/Package-uv-orange?style=flat-square)](https://github.com/astral-sh/uv)
[![Transformers](https://img.shields.io/badge/Transformers-4.56.2-red?style=flat-square)](https://huggingface.co/docs/transformers)
[![Unsloth](https://img.shields.io/badge/Unsloth-2025.12.9-green?style=flat-square)](https://github.com/unslothai/unsloth)

[English Documentation](README.eng.md)

</div>

---

## 소개

**DeepSeek-OCR 파인튜닝 툴킷**은 문서 OCR 작업을 위한 포괄적인 파인튜닝 및 추론 파이프라인입니다.

### 왜 이 프로젝트를 사용해야 하나요?

- **문제점**: 기존 OCR 도구들은 특정 도메인(의료, 법률, 학술 등)에서 정확도가 낮고, 복잡한 레이아웃을 처리하기 어렵습니다.
- **해결책**: LoRA 기반 효율적 파인튜닝으로 도메인별 데이터에 모델을 최적화하고, Vision Encoder와 LLM을 선택적으로 학습할 수 있습니다.

### 배경

이 프로젝트는 실제 문서 처리 워크플로우를 간소화하기 위해 개발되었습니다. PDF → 이미지 변환 → OCR → 후처리의 복잡한 과정을 단일 CLI로 통합했습니다.

---

## 주요 기능

### 완전한 워크플로우 지원
* **PDF 직접 입력**: 수동 이미지 변환 없이 PDF 파일에서 바로 추론 (`--pdf` 옵션)
* **스트리밍 마크다운 출력**: 문서별로 실시간 결과 저장, 추론 중에도 완료된 문서 확인 가능
* **자동 모델 다운로드**: Hugging Face에서 DeepSeek-OCR 자동 다운로드 (첫 실행 시)

### 효율적인 파인튜닝
* **LoRA 선택적 학습**: Vision Encoder, LLM, 또는 전체 모델을 선택적으로 파인튜닝
* **Unsloth 최적화**: 메모리 효율적인 학습 (Gradient Checkpointing, 4bit 양자화 지원)
* **유연한 이미지 처리**: Gundam 모드 (1024→640 crop) 등 다양한 해상도 프리셋

### API 및 로컬 모델 지원
* **OpenAI 호환 API**: vLLM, OpenAI API 등으로 Teacher 모델 추론
* **로컬 모델**: 로컬 GPU에서 직접 추론 (API 불필요)

---

## 빠른 시작

### 필수 요구사항

```bash
# Python 3.12 이상 필요
python --version

# uv 패키지 매니저 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/ocr_finetune.git
cd ocr_finetune

# 의존성 설치 (uv 권장)
uv sync

# 또는 pip 사용
pip install -e .
```

### 첫 번째 추론 실행

```bash
# 1. 설정 파일 준비
cp config/examples/teacher_api.example.yaml config/teacher_api.yaml
cp config/examples/prompts.example.yaml config/prompts.yaml

# 2. config/teacher_api.yaml 수정 (API URL, 모델명 설정)
# vim config/teacher_api.yaml

# 3. PDF에서 직접 마크다운 생성
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document

# 결과: document.md 파일이 생성됩니다
```

<details>
<summary><strong>이미지 폴더에서 추론하기</strong></summary>

```bash
# PDF를 이미지로 먼저 변환
uv run main.py pdf2img --source document.pdf --output ./images --dpi 300

# 이미지 폴더에서 추론
uv run main.py infer --images ./images --config config/teacher_api.yaml --task document
```

</details>

---

## 사용법

### 1. PDF를 이미지로 변환

```bash
uv run main.py pdf2img --source document.pdf --output ./images --dpi 200
```

| 옵션 | 설명 | 기본값 |
|:---:|:---:|:---:|
| `--source` | PDF 파일 또는 디렉토리 | (필수) |
| `--output` | 출력 디렉토리 | `./output_images` |
| `--dpi` | 이미지 해상도 | `200` |
| `--format` | 이미지 포맷 (png, jpg) | `png` |
| `--start-page` | 시작 페이지 번호 | `1` |
| `--end-page` | 종료 페이지 번호 | (전체) |

### 2. 추론 실행

```bash
# PDF에서 직접 추론 (권장)
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document

# 이미지에서 추론
uv run main.py infer --images ./images --config config/teacher_api.yaml --task ocr

# 사용 가능한 태스크 목록 보기
uv run main.py infer --config config/teacher_api.yaml --list-prompts
```

| 옵션 | 설명 | 기본값 |
|:---:|:---:|:---:|
| `--images` | 이미지 파일 또는 디렉토리 | - |
| `--pdf` | PDF 파일 또는 디렉토리 | - |
| `--dpi` | PDF 변환 시 해상도 | `200` |
| `--config` | Teacher 모델 설정 YAML | (필수) |
| `--task` | 태스크 이름 (prompts.yaml의 키) | (필수) |
| `--output` | 출력 경로 | `./output` |
| `--list-prompts` | 사용 가능한 태스크 목록 보기 | - |

<details>
<summary><strong>출력 형식 자세히 보기</strong></summary>

#### JSONL 형식 (학습용)
각 줄이 하나의 JSON 객체 (대용량 데이터셋에 적합):

```json
{"messages": [{"role": "<|User|>", "content": "<image>\n문서를 마크다운으로 변환하세요.", "images": ["page1.png"]}, {"role": "<|Assistant|>", "content": "# 제목\n\n내용..."}]}
```

#### Markdown 형식 (문서 검토용)
- `{name}_p{page}.png` 패턴의 이미지는 자동으로 `{name}.md`로 병합
- 각 문서 추론 완료 시 즉시 파일 저장 (스트리밍)
- 추론 중에도 완료된 문서 결과 확인 가능

**예시**:
```
images/
  report_p1.png  →  output/report.md
  report_p2.png  ↗
  invoice_p1.png →  output/invoice.md
```

</details>

### 3. 모델 파인튜닝

```bash
# Vision Encoder만 파인튜닝 (이미지 특징 추출 개선)
uv run main.py train --dataset data.jsonl --mode vision

# LLM만 파인튜닝 (텍스트 생성 개선)
uv run main.py train --dataset data.jsonl --mode llm

# 전체 모델 파인튜닝
uv run main.py train --dataset data.jsonl --mode both
```

| 옵션 | 설명 | 기본값 |
|:---:|:---:|:---:|
| `--dataset` | 학습 데이터셋 (JSONL) | (필수) |
| `--mode` | 학습 레이어: `vision`, `llm`, `both` | `vision` |
| `--config` | 학습 설정 YAML | `config/train_config.yaml` |
| `--output` | 출력 디렉토리 | `./models/finetuned` |
| `--resume` | 체크포인트에서 재개 | - |

<details>
<summary><strong>학습 모드 선택 가이드</strong></summary>

| 모드 | 적용 대상 | 사용 사례 |
|:---:|:---:|:---:|
| **vision** | Vision Encoder (qkv_proj, fc1, fc2 등) | 새로운 문서 레이아웃, 손글씨 인식 개선 |
| **llm** | Language Model (q_proj, gate_proj 등) | 도메인별 용어, 출력 형식 개선 |
| **both** | 전체 모델 | 완전히 새로운 도메인, 최대 성능 필요 시 |

**권장**: 대부분의 경우 `vision` 모드로 시작하세요. LLM은 이미 범용적인 언어 능력을 가지고 있습니다.

</details>

### 4. 모델 평가

```bash
uv run main.py evaluate --dataset eval.jsonl --task document --output results.json
```

---

## 설정 파일

설정 예제는 `config/examples/` 디렉토리에서 확인할 수 있습니다.

### 학습 설정 (`train_config.yaml`)

```yaml
# 모델 설정
model:
  base_model_path: "models/deepseek_ocr"
  load_in_4bit: false
  use_gradient_checkpointing: "unsloth"

# LoRA 설정
lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0

  # --mode vision 시 적용
  vision_target_modules:
    - "qkv_proj"
    - "out_proj"
    - "fc1"
    - "fc2"

  # --mode llm 시 적용
  llm_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# 학습 하이퍼파라미터
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_steps: 100
  logging_steps: 1
  save_steps: 50

# 이미지 처리 (Gundam 모드 권장)
image:
  image_size: 640
  base_size: 1024
  crop_mode: true
```

<details>
<summary><strong>이미지 크기 프리셋 보기</strong></summary>

| 프리셋 | base_size | image_size | crop_mode | 용도 |
|:---:|:---:|:---:|:---:|:---:|
| Tiny | 512 | 512 | false | 빠른 실험, 저메모리 |
| Small | 640 | 640 | false | 일반 문서 |
| Base | 1024 | 1024 | false | 고해상도 문서 |
| **Gundam** | **1024** | **640** | **true** | **권장 (성능/속도 균형)** |
| Large | 1280 | 1280 | false | 최고 품질 |

</details>

### Teacher 모델 설정 (`teacher_api.yaml`)

```yaml
type: "api"  # api 또는 local
model_type: "default"

# API 설정
api:
  base_url: "http://localhost:8000/v1"
  model_name: "deepseek-ocr"
  api_key: "your-api-key"

# 생성 파라미터
generation:
  temperature: 0.1
  max_tokens: 4096

# 요청 설정
request:
  timeout: 120
  max_retries: 3
  concurrent_requests: 4

# 프롬프트 정의
prompts:
  document:
    system: |
      You are a document parsing assistant.
      Convert the document image to structured Markdown.
    user: |
      Convert this document to Markdown format.

# 출력 설정
output:
  format: "md"  # jsonl, json, md
  save_images: false
```

---

## 데이터셋 형식

학습 데이터셋은 JSONL 형식을 사용합니다:

```json
{
  "messages": [
    {
      "role": "<|User|>",
      "content": "<image>\nConvert this document to Markdown.",
      "images": ["path/to/image.png"]
    },
    {
      "role": "<|Assistant|>",
      "content": "# Document Title\n\n## Section 1\n\nContent here..."
    }
  ]
}
```

**주의사항**:
- `images` 필드는 이미지 파일 경로 배열입니다
- `content`에 `<image>` 토큰이 반드시 포함되어야 합니다
- 각 줄은 독립적인 JSON 객체입니다 (JSON 배열 아님)

---

## 프로젝트 구조

```
ocr_finetune/
├── main.py                 # CLI 진입점
├── config/
│   ├── examples/           # 설정 파일 예제
│   │   ├── train_config.example.yaml
│   │   ├── teacher_api.example.yaml
│   │   ├── teacher_local.example.yaml
│   │   └── prompts.example.yaml
│   ├── train_config.yaml   # 학습 설정
│   ├── teacher_api.yaml    # API 추론 설정
│   └── prompts.yaml        # 프롬프트 정의
├── src/
│   ├── data/
│   │   ├── infer.py        # 추론 모듈
│   │   ├── pdf2img.py      # PDF → 이미지 변환
│   │   ├── collator.py     # 데이터 콜레이터
│   │   └── preprocessor.py # 출력 전처리
│   ├── train/
│   │   ├── trainer.py      # 학습 로직
│   │   └── layers.py       # 레이어 선택
│   └── eval/
│       └── metrics.py      # 평가 메트릭
├── models/
│   └── deepseek_ocr/       # 기본 모델 (자동 다운로드)
└── data/                   # 데이터셋 디렉토리
```

---

## 자동 모델 다운로드

`train`, `evaluate`, `inspect`, `infer` 명령 실행 시 `models/deepseek_ocr` 폴더가 없으면 Hugging Face에서 자동으로 다운로드됩니다.

```
Downloading unsloth/DeepSeek-OCR from Hugging Face...
✓ Model downloaded to models/deepseek_ocr
```

**수동 다운로드** (선택사항):
```bash
huggingface-cli download unsloth/DeepSeek-OCR --local-dir models/deepseek_ocr
```

---

## 비교

| 기능 | OCR Finetune | 일반 OCR 도구 | Tesseract |
|:---:|:---:|:---:|:---:|
| **도메인 적응** | ✅ LoRA 파인튜닝 | ❌ 불가능 | ⚠️ 제한적 |
| **복잡한 레이아웃** | ✅ Vision+LLM | ⚠️ 제한적 | ❌ 약함 |
| **마크다운 출력** | ✅ 구조화된 출력 | ❌ 텍스트만 | ❌ 텍스트만 |
| **PDF 직접 입력** | ✅ 지원 | ⚠️ 변환 필요 | ⚠️ 변환 필요 |
| **비용** | **무료** | API 비용 발생 | **무료** |

---

## 트러블슈팅

<details>
<summary><strong>CUDA Out of Memory 오류</strong></summary>

```yaml
# train_config.yaml에서 배치 크기 줄이기
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

# 또는 4bit 양자화 사용
model:
  load_in_4bit: true
```

</details>

<details>
<summary><strong>모델 다운로드 실패</strong></summary>

```bash
# Hugging Face 토큰 설정 (private 모델의 경우)
export HF_TOKEN="your_huggingface_token"

# 또는 수동으로 다운로드
huggingface-cli login
huggingface-cli download unsloth/DeepSeek-OCR --local-dir models/deepseek_ocr
```

</details>

<details>
<summary><strong>API 연결 오류</strong></summary>

```bash
# 1. vLLM 서버가 실행 중인지 확인
curl http://localhost:8000/v1/models

# 2. config/teacher_api.yaml의 base_url 확인
# 3. 방화벽 설정 확인
```

</details>

---

## 기술 스택

| 카테고리 | 기술 |
|:---:|:---:|
| **언어** | Python 3.12+ |
| **패키지 관리** | uv |
| **기본 모델** | DeepSeek-OCR (Unsloth) |
| **파인튜닝** | LoRA, PEFT |
| **최적화** | Unsloth, 4bit Quantization |
| **추론** | OpenAI API, Transformers |
| **문서 처리** | PyMuPDF |

---

## 라이선스

MIT License

---

<div align="center">

**더 나은 OCR을 위해**

[이슈 제기](https://github.com/your-username/ocr_finetune/issues) • [기여하기](https://github.com/your-username/ocr_finetune/pulls) • [문서](https://github.com/your-username/ocr_finetune/wiki)

Made with ❤️ by OCR Finetune Contributors

</div>
