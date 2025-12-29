# DeepSeek-OCR Finetuning Toolkit

<div align="center">

![OCR Finetune](https://via.placeholder.com/150x150?text=OCR+Finetune)

**From Document Images to Markdown, More Accurately**<br/>
A professional finetuning and inference toolkit for the DeepSeek-OCR model

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![uv](https://img.shields.io/badge/Package-uv-orange?style=flat-square)](https://github.com/astral-sh/uv)
[![Transformers](https://img.shields.io/badge/Transformers-4.56.2-red?style=flat-square)](https://huggingface.co/docs/transformers)
[![Unsloth](https://img.shields.io/badge/Unsloth-2025.12.9-green?style=flat-square)](https://github.com/unslothai/unsloth)

[한국어 문서](README.md)

</div>

---

## Introduction

**DeepSeek-OCR Finetuning Toolkit** is a comprehensive finetuning and inference pipeline for document OCR tasks.

### Why Use This Project?

- **Problem**: Traditional OCR tools have low accuracy in specific domains (medical, legal, academic, etc.) and struggle with complex layouts.
- **Solution**: Optimize models for domain-specific data with LoRA-based efficient finetuning, with selective training of Vision Encoder and LLM layers.

### Background

This project was developed to streamline real-world document processing workflows. It consolidates the complex process of PDF → image conversion → OCR → post-processing into a single CLI tool.

---

## Key Features

### Complete Workflow Support
* **Direct PDF Input**: Inference directly from PDF files without manual image conversion (`--pdf` option)
* **Streaming Markdown Output**: Real-time document-by-document saving, view completed documents during inference
* **Automatic Model Download**: Auto-download DeepSeek-OCR from Hugging Face on first run

### Efficient Finetuning
* **LoRA Selective Training**: Selectively finetune Vision Encoder, LLM, or the entire model
* **Unsloth Optimization**: Memory-efficient training (Gradient Checkpointing, 4-bit quantization support)
* **Flexible Image Processing**: Various resolution presets including Gundam mode (1024→640 crop)

### API and Local Model Support
* **OpenAI-Compatible API**: Teacher model inference via vLLM, OpenAI API, etc.
* **Local Models**: Direct inference on local GPUs (no API required)

---

## Quick Start

### Prerequisites

```bash
# Python 3.12+ required
python --version

# Install uv package manager (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/ocr_finetune.git
cd ocr_finetune

# Install dependencies (uv recommended)
uv sync

# Or use pip
pip install -e .
```

### First Inference Run

```bash
# 1. Prepare configuration files
cp config/examples/teacher_api.example.yaml config/teacher_api.yaml
cp config/examples/prompts.example.yaml config/prompts.yaml

# 2. Edit config/teacher_api.yaml (set API URL, model name)
# vim config/teacher_api.yaml

# 3. Generate markdown directly from PDF
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document

# Result: document.md file is created
```

<details>
<summary><strong>Inference from Image Folder</strong></summary>

```bash
# First convert PDF to images
uv run main.py pdf2img --source document.pdf --output ./images --dpi 300

# Inference from image folder
uv run main.py infer --images ./images --config config/teacher_api.yaml --task document
```

</details>

---

## Usage

### 1. Convert PDF to Images

```bash
uv run main.py pdf2img --source document.pdf --output ./images --dpi 200
```

| Option | Description | Default |
|:---:|:---:|:---:|
| `--source` | PDF file or directory | (required) |
| `--output` | Output directory | `./output_images` |
| `--dpi` | Image resolution | `200` |
| `--format` | Image format (png, jpg) | `png` |
| `--start-page` | Starting page number | `1` |
| `--end-page` | Ending page number | (all) |

### 2. Run Inference

```bash
# Inference directly from PDF (recommended)
uv run main.py infer --pdf document.pdf --config config/teacher_api.yaml --task document

# Inference from images
uv run main.py infer --images ./images --config config/teacher_api.yaml --task ocr

# List available tasks
uv run main.py infer --config config/teacher_api.yaml --list-prompts
```

| Option | Description | Default |
|:---:|:---:|:---:|
| `--images` | Image file or directory | - |
| `--pdf` | PDF file or directory | - |
| `--dpi` | Resolution for PDF conversion | `200` |
| `--config` | Teacher model config YAML | (required) |
| `--task` | Task name (key in prompts.yaml) | (required) |
| `--output` | Output path | `./output` |
| `--list-prompts` | List available tasks | - |

<details>
<summary><strong>Output Format Details</strong></summary>

#### JSONL Format (For Training)
Each line is a single JSON object (suitable for large datasets):

```json
{"messages": [{"role": "<|User|>", "content": "<image>\nConvert this document to Markdown.", "images": ["page1.png"]}, {"role": "<|Assistant|>", "content": "# Title\n\nContent..."}]}
```

#### Markdown Format (For Document Review)
- Images with `{name}_p{page}.png` pattern are auto-merged into `{name}.md`
- Each document is saved immediately upon completion (streaming)
- View completed documents while inference is still running

**Example**:
```
images/
  report_p1.png  →  output/report.md
  report_p2.png  ↗
  invoice_p1.png →  output/invoice.md
```

</details>

### 3. Model Finetuning

```bash
# Finetune Vision Encoder only (improve image feature extraction)
uv run main.py train --dataset data.jsonl --mode vision

# Finetune LLM only (improve text generation)
uv run main.py train --dataset data.jsonl --mode llm

# Finetune entire model
uv run main.py train --dataset data.jsonl --mode both
```

| Option | Description | Default |
|:---:|:---:|:---:|
| `--dataset` | Training dataset (JSONL) | (required) |
| `--mode` | Training layers: `vision`, `llm`, `both` | `vision` |
| `--config` | Training config YAML | `config/train_config.yaml` |
| `--output` | Output directory | `./models/finetuned` |
| `--resume` | Resume from checkpoint | - |

<details>
<summary><strong>Training Mode Selection Guide</strong></summary>

| Mode | Target | Use Case |
|:---:|:---:|:---:|
| **vision** | Vision Encoder (qkv_proj, fc1, fc2, etc.) | New document layouts, handwriting recognition |
| **llm** | Language Model (q_proj, gate_proj, etc.) | Domain-specific terminology, output format |
| **both** | Entire model | Completely new domain, maximum performance needed |

**Recommendation**: Start with `vision` mode in most cases. The LLM already has general language capabilities.

</details>

### 4. Model Evaluation

```bash
uv run main.py evaluate --dataset eval.jsonl --task document --output results.json
```

---

## Configuration Guide

This section explains the key options in inference configuration files.

### `type`: Inference Mode

| Value | Description | When to Use |
|:---:|:---|:---|
| `api` | Use OpenAI-compatible API server | When running vLLM, TGI, or external servers |
| `local` | Direct inference on local GPU | Single machine, no API server needed |

```yaml
# API mode - Call external server
type: "api"
api:
  base_url: "http://localhost:8000/v1"
  model_name: "deepseek-ocr"
  api_key: "your-api-key"

# Local mode - Use local GPU
type: "local"
local:
  model_path: "models/deepseek_ocr"
  load_in_4bit: false
```

### `model_type`: Output Postprocessing

Selects the preprocessor that removes special tags from model output.

| Value | Description | Tags Removed |
|:---:|:---|:---|
| `default` | No postprocessing | None (raw output) |
| `deepseek-ocr` | For OCR tasks | `<\|...\|>` special tags |
| `deepseek-document` | For document parsing | `<\|ref\|>`, `<\|det\|>`, coordinates, etc. |

```yaml
# For document parsing (removes coordinate tags)
model_type: "deepseek-document"

# For simple OCR
model_type: "deepseek-ocr"

# No postprocessing, raw output
model_type: "default"
```

<details>
<summary><strong>DeepSeek Output Example (Before/After Preprocessing)</strong></summary>

**Before preprocessing** (`deepseek-document` raw output):
```
<|ref|>text<|/ref|><|det|>[[238, 260, 480, 275]]<|/det|>
'Real-time Public Transit Congestion Prediction Service' Development

<|ref|>sub_title<|/ref|><|det|>[[47, 315, 152, 333]]<|/det|>
## Cover Letter
```

**After preprocessing** (`model_type: "deepseek-document"`):
```
'Real-time Public Transit Congestion Prediction Service' Development

## Cover Letter
```

</details>

### `--task`: Prompt Selection

Selects a key defined in the `prompts` section. Each task has `system` and `user` prompts.

```yaml
# config/teacher_api.yaml
prompts:
  ocr:                    # --task ocr
    system: "You are an OCR assistant..."
    user: "Extract all text from this image."

  document:               # --task document
    system: "You are a document parsing assistant..."
    user: "Convert this document to Markdown format."

  invoice:                # --task invoice (custom task)
    system: "You are an invoice parser..."
    user: "Extract invoice details as JSON."
```

```bash
# Usage example
uv run main.py infer --pdf invoice.pdf --config config/teacher_api.yaml --task invoice
```

### `output.format`: Output Format

| Value | Description | File Format |
|:---:|:---|:---|
| `jsonl` | Training dataset | One JSON per line |
| `json` | JSON array | All results in a single JSON array |
| `md` | Markdown files | Separate `.md` file per document |

```yaml
output:
  format: "md"        # Save as markdown files
  save_images: false  # Whether to save images alongside
```

**Auto-merge for Markdown output**: Images with `{name}_p{page}.png` pattern are automatically merged into `{name}.md`.

---

## Configuration Files

Configuration examples are available in the `config/examples/` directory.

### Training Config (`train_config.yaml`)

```yaml
# Model settings
model:
  base_model_path: "models/deepseek_ocr"
  load_in_4bit: false
  use_gradient_checkpointing: "unsloth"

# LoRA configuration
lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0

  # Applied with --mode vision
  vision_target_modules:
    - "qkv_proj"
    - "out_proj"
    - "fc1"
    - "fc2"

  # Applied with --mode llm
  llm_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# Training hyperparameters
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_steps: 100
  logging_steps: 1
  save_steps: 50

# Image processing (Gundam mode recommended)
image:
  image_size: 640
  base_size: 1024
  crop_mode: true
```

<details>
<summary><strong>Image Size Presets</strong></summary>

| Preset | base_size | image_size | crop_mode | Use Case |
|:---:|:---:|:---:|:---:|:---:|
| Tiny | 512 | 512 | false | Quick experiments, low memory |
| Small | 640 | 640 | false | General documents |
| Base | 1024 | 1024 | false | High-resolution documents |
| **Gundam** | **1024** | **640** | **true** | **Recommended (performance/speed balance)** |
| Large | 1280 | 1280 | false | Maximum quality |

</details>

### Teacher Model Config (`teacher_api.yaml`)

```yaml
type: "api"  # api or local
model_type: "default"

# API settings
api:
  base_url: "http://localhost:8000/v1"
  model_name: "deepseek-ocr"
  api_key: "your-api-key"

# Generation parameters
generation:
  temperature: 0.1
  max_tokens: 4096

# Request settings
request:
  timeout: 120
  max_retries: 3
  concurrent_requests: 4

# Prompt definitions
prompts:
  document:
    system: |
      You are a document parsing assistant.
      Convert the document image to structured Markdown.
    user: |
      Convert this document to Markdown format.

# Output settings
output:
  format: "md"  # jsonl, json, md
  save_images: false
```

---

## Dataset Format

Training datasets use JSONL format:

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

**Notes**:
- `images` field is an array of image file paths
- `content` must include the `<image>` token
- Each line is an independent JSON object (not a JSON array)

---

## Project Structure

```
ocr_finetune/
├── main.py                 # CLI entry point
├── config/
│   ├── examples/           # Configuration examples
│   │   ├── train_config.example.yaml
│   │   ├── teacher_api.example.yaml
│   │   ├── teacher_local.example.yaml
│   │   └── prompts.example.yaml
│   ├── train_config.yaml   # Training configuration
│   ├── teacher_api.yaml    # API inference configuration
│   └── prompts.yaml        # Prompt definitions
├── src/
│   ├── data/
│   │   ├── infer.py        # Inference module
│   │   ├── pdf2img.py      # PDF → image conversion
│   │   ├── collator.py     # Data collator
│   │   └── preprocessor.py # Output preprocessing
│   ├── train/
│   │   ├── trainer.py      # Training logic
│   │   └── layers.py       # Layer selection
│   └── eval/
│       └── metrics.py      # Evaluation metrics
├── models/
│   └── deepseek_ocr/       # Base model (auto-downloaded)
└── data/                   # Dataset directory
```

---

## Automatic Model Download

When running `train`, `evaluate`, `inspect`, or `infer` commands, if the `models/deepseek_ocr` folder doesn't exist, it will be automatically downloaded from Hugging Face.

```
Downloading unsloth/DeepSeek-OCR from Hugging Face...
✓ Model downloaded to models/deepseek_ocr
```

**Manual Download** (optional):
```bash
huggingface-cli download unsloth/DeepSeek-OCR --local-dir models/deepseek_ocr
```

---

## Comparison

| Feature | OCR Finetune | Generic OCR Tools | Tesseract |
|:---:|:---:|:---:|:---:|
| **Domain Adaptation** | ✅ LoRA finetuning | ❌ Not possible | ⚠️ Limited |
| **Complex Layouts** | ✅ Vision+LLM | ⚠️ Limited | ❌ Weak |
| **Markdown Output** | ✅ Structured output | ❌ Text only | ❌ Text only |
| **Direct PDF Input** | ✅ Supported | ⚠️ Conversion needed | ⚠️ Conversion needed |
| **Cost** | **Free** | API costs | **Free** |

---

## Troubleshooting

<details>
<summary><strong>CUDA Out of Memory Error</strong></summary>

```yaml
# Reduce batch size in train_config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

# Or use 4-bit quantization
model:
  load_in_4bit: true
```

</details>

<details>
<summary><strong>Model Download Failure</strong></summary>

```bash
# Set Hugging Face token (for private models)
export HF_TOKEN="your_huggingface_token"

# Or manually download
huggingface-cli login
huggingface-cli download unsloth/DeepSeek-OCR --local-dir models/deepseek_ocr
```

</details>

<details>
<summary><strong>API Connection Error</strong></summary>

```bash
# 1. Check if vLLM server is running
curl http://localhost:8000/v1/models

# 2. Verify base_url in config/teacher_api.yaml
# 3. Check firewall settings
```

</details>

---

## Tech Stack

| Category | Technology |
|:---:|:---:|
| **Language** | Python 3.12+ |
| **Package Manager** | uv |
| **Base Model** | DeepSeek-OCR (Unsloth) |
| **Finetuning** | LoRA, PEFT |
| **Optimization** | Unsloth, 4-bit Quantization |
| **Inference** | OpenAI API, Transformers |
| **Document Processing** | PyMuPDF |

---

## License

MIT License

---

<div align="center">

**For Better OCR**

[Report Issues](https://github.com/your-username/ocr_finetune/issues) • [Contribute](https://github.com/your-username/ocr_finetune/pulls) • [Documentation](https://github.com/your-username/ocr_finetune/wiki)

Made with ❤️ by OCR Finetune Contributors

</div>
