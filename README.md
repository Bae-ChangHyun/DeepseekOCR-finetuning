# OCR Finetune

[English](README.eng.md)

DeepSeek-OCR 모델을 위한 파인튜닝 및 추론 도구입니다.

## 주요 기능

- **PDF/이미지 → 텍스트 추론**: API 또는 로컬 모델을 사용한 OCR
- **LoRA 파인튜닝**: Vision Encoder 또는 LLM 레이어 선택적 학습
- **스트리밍 마크다운 출력**: 문서별 실시간 결과 저장
- **PDF 직접 입력**: PDF → 이미지 변환 없이 바로 추론

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/ocr_finetune.git
cd ocr_finetune

# 의존성 설치 (uv 권장)
uv sync

# 또는 pip 사용
pip install -e .
```

## 빠른 시작

### 1. 설정 파일 준비

```bash
# 예제 파일 복사
cp config/examples/teacher_api.example.yaml config/teacher_api.yaml
cp config/examples/train_config.example.yaml config/train_config.yaml
cp config/examples/prompts.example.yaml config/prompts.yaml

# 필요에 따라 수정
```

### 2. 추론 실행

```bash
# PDF에서 직접 추론 (마크다운 출력)
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document

# 이미지 폴더에서 추론
uv run main.py infer --images ./images --config config/teacher_api.yaml --task ocr

# 사용 가능한 태스크 목록 확인
uv run main.py infer --config config/teacher_api.yaml --list-prompts
```

### 3. 학습 실행

```bash
# Vision Encoder 파인튜닝
uv run main.py train --dataset data.jsonl --mode vision

# LLM 파인튜닝
uv run main.py train --dataset data.jsonl --mode llm

# 모든 레이어 파인튜닝
uv run main.py train --dataset data.jsonl --mode both
```

## 명령어

### `pdf2img` - PDF를 이미지로 변환

```bash
uv run main.py pdf2img --source document.pdf --output ./images --dpi 200
```

### `infer` - 추론 실행

```bash
# 이미지에서 추론
uv run main.py infer --images ./images --config config/teacher_api.yaml --task document

# PDF에서 직접 추론
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document --dpi 300
```

| 옵션 | 설명 |
|------|------|
| `--images` | 이미지 파일 또는 디렉토리 |
| `--pdf` | PDF 파일 또는 디렉토리 (내부적으로 이미지 변환) |
| `--dpi` | PDF 변환 시 해상도 (기본: 200) |
| `--config` | Teacher 모델 설정 YAML |
| `--task` | 태스크 이름 (prompts.yaml의 키) |
| `--output` | 출력 경로 |

### `train` - 모델 학습

```bash
uv run main.py train --dataset data.jsonl --mode vision --output ./models/finetuned
```

| 옵션 | 설명 |
|------|------|
| `--dataset` | 학습 데이터셋 (JSONL) |
| `--mode` | 레이어 선택: `vision`, `llm`, `both` |
| `--config` | 학습 설정 YAML |
| `--output` | 출력 디렉토리 |
| `--resume` | 체크포인트에서 재개 |

### `evaluate` - 모델 평가

```bash
uv run main.py evaluate --dataset eval.jsonl --task document --output results.json
```

## 설정 파일

설정 예제는 `config/examples/` 디렉토리에서 확인할 수 있습니다.

### 학습 설정 (`train_config.yaml`)

```yaml
model:
  base_model_path: "models/deepseek_ocr"
  load_in_4bit: false

lora:
  r: 16
  lora_alpha: 16
  vision_target_modules:  # --mode vision
    - "qkv_proj"
    - "out_proj"
    - "fc1"
    - "fc2"
  llm_target_modules:     # --mode llm
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"

training:
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4
  max_steps: 100
```

### Teacher 설정 (`teacher_api.yaml`)

```yaml
type: "api"  # api 또는 local

api:
  base_url: "http://localhost:8000/v1"
  model_name: "your-model"

prompts:
  document:
    system: "You are a document parsing assistant..."
    user: "Convert this document to Markdown."

output:
  format: "md"  # jsonl, json, md
```

## 출력 형식

| 형식 | 설명 |
|------|------|
| `jsonl` | 학습용 데이터셋 (한 줄에 하나의 JSON) |
| `json` | JSON 배열 |
| `md` | 마크다운 파일 (문서별 자동 병합) |

### 마크다운 출력 시 동작

- `{name}_p{page}.png` 패턴의 이미지는 `{name}.md`로 자동 병합
- 각 문서 추론 완료 시 즉시 파일 저장 (스트리밍)
- 추론 중에도 완료된 문서 결과 확인 가능

## 데이터셋 형식

학습 데이터셋은 JSONL 형식을 사용합니다:

```json
{
  "messages": [
    {
      "role": "<|User|>",
      "content": "<image>\nConvert to Markdown. ",
      "images": ["path/to/image.png"]
    },
    {
      "role": "<|Assistant|>",
      "content": "# Document Title\n\nContent here..."
    }
  ]
}
```

## 프로젝트 구조

```
ocr_finetune/
├── main.py                 # CLI 진입점
├── config/
│   ├── examples/           # 설정 예제
│   │   ├── train_config.example.yaml
│   │   ├── teacher_api.example.yaml
│   │   ├── teacher_local.example.yaml
│   │   └── prompts.example.yaml
│   ├── train_config.yaml   # 학습 설정
│   ├── teacher_api.yaml    # API 추론 설정
│   └── prompts.yaml        # 프롬프트 설정
├── src/
│   ├── data/
│   │   ├── infer.py        # 추론 모듈
│   │   ├── pdf2img.py      # PDF 변환
│   │   ├── collator.py     # 데이터 콜레이터
│   │   └── preprocessor.py # 출력 전처리
│   ├── train/
│   │   ├── trainer.py      # 학습 로직
│   │   └── layers.py       # 레이어 선택
│   └── eval/
│       └── metrics.py      # 평가 메트릭
└── models/
    └── deepseek_ocr/       # 기본 모델 (자동 다운로드)
```

## 자동 모델 다운로드

`train`, `evaluate`, `inspect`, `infer` 명령 실행 시 `models/deepseek_ocr` 폴더가 없으면 Hugging Face에서 자동으로 다운로드됩니다.

## 라이선스

MIT License
