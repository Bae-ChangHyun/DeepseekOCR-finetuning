# OCR Finetune

[한국어](README.md)

A finetuning and inference tool for the DeepSeek-OCR model.

## Features

- **PDF/Image → Text Inference**: OCR using API or local model
- **LoRA Finetuning**: Selective training of Vision Encoder or LLM layers
- **Streaming Markdown Output**: Real-time document-by-document saving
- **Direct PDF Input**: Inference directly from PDF without manual image conversion

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/ocr_finetune.git
cd ocr_finetune

# Install dependencies (uv recommended)
uv sync

# Or use pip
pip install -e .
```

## Quick Start

### 1. Prepare Configuration Files

```bash
# Copy example files
cp config/examples/teacher_api.example.yaml config/teacher_api.yaml
cp config/examples/train_config.example.yaml config/train_config.yaml
cp config/examples/prompts.example.yaml config/prompts.yaml

# Modify as needed
```

### 2. Run Inference

```bash
# Inference directly from PDF (markdown output)
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document

# Inference from image folder
uv run main.py infer --images ./images --config config/teacher_api.yaml --task ocr

# List available tasks
uv run main.py infer --config config/teacher_api.yaml --list-prompts
```

### 3. Run Training

```bash
# Vision Encoder finetuning
uv run main.py train --dataset data.jsonl --mode vision

# LLM finetuning
uv run main.py train --dataset data.jsonl --mode llm

# Full model finetuning
uv run main.py train --dataset data.jsonl --mode both
```

## Commands

### `pdf2img` - Convert PDF to Images

```bash
uv run main.py pdf2img --source document.pdf --output ./images --dpi 200
```

### `infer` - Run Inference

```bash
# Inference from images
uv run main.py infer --images ./images --config config/teacher_api.yaml --task document

# Inference directly from PDF
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document --dpi 300
```

| Option | Description |
|--------|-------------|
| `--images` | Image file or directory |
| `--pdf` | PDF file or directory (converts internally) |
| `--dpi` | Resolution for PDF conversion (default: 200) |
| `--config` | Teacher model config YAML |
| `--task` | Task name (key in prompts.yaml) |
| `--output` | Output path |

### `train` - Train Model

```bash
uv run main.py train --dataset data.jsonl --mode vision --output ./models/finetuned
```

| Option | Description |
|--------|-------------|
| `--dataset` | Training dataset (JSONL) |
| `--mode` | Layer selection: `vision`, `llm`, `both` |
| `--config` | Training config YAML |
| `--output` | Output directory |
| `--resume` | Resume from checkpoint |

### `evaluate` - Evaluate Model

```bash
uv run main.py evaluate --dataset eval.jsonl --task document --output results.json
```

## Configuration Files

Configuration examples are available in the `config/examples/` directory.

### Training Config (`train_config.yaml`)

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

### Teacher Config (`teacher_api.yaml`)

```yaml
type: "api"  # api or local

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

## Output Formats

| Format | Description |
|--------|-------------|
| `jsonl` | Training dataset (one JSON per line) |
| `json` | JSON array |
| `md` | Markdown files (auto-merged by document) |

### Markdown Output Behavior

- Images with `{name}_p{page}.png` pattern are auto-merged into `{name}.md`
- Each document is saved immediately upon completion (streaming)
- View completed documents while inference is still running

## Dataset Format

Training datasets use JSONL format:

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

## Project Structure

```
ocr_finetune/
├── main.py                 # CLI entry point
├── config/
│   ├── examples/           # Config examples
│   │   ├── train_config.example.yaml
│   │   ├── teacher_api.example.yaml
│   │   ├── teacher_local.example.yaml
│   │   └── prompts.example.yaml
│   ├── train_config.yaml   # Training config
│   ├── teacher_api.yaml    # API inference config
│   └── prompts.yaml        # Prompt definitions
├── src/
│   ├── data/
│   │   ├── infer.py        # Inference module
│   │   ├── pdf2img.py      # PDF conversion
│   │   ├── collator.py     # Data collator
│   │   └── preprocessor.py # Output preprocessing
│   ├── train/
│   │   ├── trainer.py      # Training logic
│   │   └── layers.py       # Layer selection
│   └── eval/
│       └── metrics.py      # Evaluation metrics
└── models/
    └── deepseek_ocr/       # Base model (auto-downloaded)
```

## Automatic Model Download

When running `train`, `evaluate`, `inspect`, or `infer` commands, if the `models/deepseek_ocr` folder doesn't exist, it will be automatically downloaded from Hugging Face.

## License

MIT License
