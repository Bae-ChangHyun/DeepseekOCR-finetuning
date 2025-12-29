# VLM(Vision Language Model) 파인튜닝 가이드

## 1. VLM 아키텍처 개요

VLM은 일반적으로 세 가지 핵심 컴포넌트로 구성됩니다:

```
┌─────────────────────┐
│   Vision Encoder    │  ← CLIP, SigLIP 등 사전학습된 비전 모델
│   (비전 인코더)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Projector/Adapter  │  ← Linear, MLP, Q-Former, Cross-Attention 등
│   (프로젝터/어댑터)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Language Model    │  ← LLaMA, Qwen, Vicuna 등 LLM
│   (언어 모델)        │
└─────────────────────┘
```

## 2. 비전 인코더 vs LLM 분리 학습의 원리

### 2.1 왜 분리해서 학습하는가?

VLM 학습에서 비전 인코더와 LLM을 분리해서 학습하는 것은 **표준적인 관행**입니다:

| 이유 | 설명 |
|------|------|
| **사전학습 지식 보존** | CLIP 같은 비전 인코더는 이미 수억 개의 이미지-텍스트 쌍으로 학습됨 |
| **계산 효율성** | 전체 모델 학습 대비 훨씬 적은 자원 필요 |
| **안정성** | 비전 인코더 동결 시 학습이 더 안정적 |
| **모듈화** | 각 컴포넌트를 독립적으로 교체/업그레이드 가능 |

### 2.2 분리 학습이 가능한 이유

비전 인코더(특히 CLIP 계열)는 이미 텍스트와 **cross-modal alignment**가 되어있어, 프로젝터만 학습해도 LLM이 이미지를 "이해"할 수 있습니다.

## 3. VLM 학습 단계별 전략

### 3.1 Stage 1: 프로젝터 사전학습 (Projector Pretraining)

```
Vision Encoder: ❄️ Frozen (동결)
Projector:      🔥 Training (학습)
LLM:            ❄️ Frozen (동결)
```

**목적**: 비전 인코더의 출력을 LLM의 입력 공간에 정렬(align)

**데이터**: 이미지-캡션 쌍 (예: CC3M, LAION)

**특징**:
- 가장 저렴한 학습 방법
- 프로젝터만 학습하므로 파라미터 수가 적음
- 기본적인 멀티모달 능력 부여

### 3.2 Stage 2: Visual Instruction Tuning

```
Vision Encoder: ❄️ Frozen (동결)
Projector:      🔥 Training (학습)
LLM:            🔥 Training (학습) 또는 LoRA
```

**목적**: 모델이 사용자 지시를 따르고 복잡한 시각적 추론을 수행하도록 학습

**데이터**: Visual instruction 데이터셋 (예: LLaVA-Instruct, ShareGPT4V)

**특징**:
- LLM의 능력을 활용하면서 멀티모달 능력 강화
- LoRA 사용 시 메모리 효율적

### 3.3 Stage 3 (선택적): End-to-End 학습

```
Vision Encoder: 🔥 Training (학습)
Projector:      🔥 Training (학습)
LLM:            🔥 Training (학습)
```

**목적**: 특수한 도메인(의료, 문서, 위성 이미지 등)에 대한 최적화

**주의사항**:
- 매우 높은 계산 비용
- Catastrophic forgetting 위험
- 낮은 learning rate 필수

## 4. 모델별 학습 전략 비교

| 모델 | Stage 1 | Stage 2 | 비전 인코더 학습 여부 |
|------|---------|---------|-------------------|
| **LLaVA-1.5** | Projector만 | Projector + LLM | ❌ 동결 |
| **LLaVA-NeXT** | Projector만 | Projector + LLM | ❌ 동결 |
| **BLIP-2** | Q-Former만 | Q-Former + LLM | ❌ 동결 |
| **Qwen2-VL** | 전체 | 전체 | ✅ 학습 |
| **InternVL** | Projector만 | Projector + LLM | ✅ 선택적 학습 |
| **DeepSeek-VL** | Projector만 | Projector + LLM | ❌ 동결 |
| **Fuyu-8B** | 전체 E2E | 전체 E2E | ✅ 학습 (인코더 없음) |
| **KOSMOS-2** | 전체 E2E | 전체 E2E | ✅ 학습 |

## 5. 어떤 상황에서 무엇을 학습시키는가?

### 5.1 프로젝터만 학습 (가장 효율적)

**적합한 상황**:
- 제한된 GPU 자원 (24GB 이하)
- 일반적인 멀티모달 태스크
- 빠른 프로토타이핑

**예상 결과**:
- 기본적인 이미지 이해 능력
- 복잡한 추론은 제한적

### 5.2 프로젝터 + LLM(LoRA) 학습 (권장)

**적합한 상황**:
- 중간 수준의 GPU 자원 (40-80GB)
- 도메인 특화 학습 (OCR, 의료 등)
- 성능과 효율성의 균형 필요

**예상 결과**:
- 우수한 instruction following
- 도메인 특화 성능 향상

### 5.3 프로젝터 + LLM(Full) 학습

**적합한 상황**:
- 충분한 GPU 자원
- 대규모 고품질 데이터셋 보유
- 최고 성능 필요

**예상 결과**:
- 최적의 멀티모달 성능
- 학습 시간 증가

### 5.4 비전 인코더까지 학습

**적합한 상황**:
- 특수한 도메인 이미지 (의료, 위성, 문서)
- 기존 CLIP 인코더가 커버하지 못하는 영역
- 매우 세밀한 시각적 이해 필요 (OCR, 미세 텍스트)

**주의사항**:
- 학습 불안정 위험 높음
- 매우 낮은 learning rate 사용 (예: 1e-6)
- 마지막 단계에서만 짧게 학습 권장
- 충분한 데이터 필요

## 6. 모든 모델에서 분리 학습이 가능한가?

### 6.1 분리 학습 지원 여부

**대부분의 VLM에서 지원됨**:

| 프레임워크 | 분리 학습 지원 | 설정 방법 |
|-----------|--------------|----------|
| **LLaMA-Factory** | ✅ | `freeze_vision_tower`, `freeze_llm` |
| **lmms-finetune** | ✅ | Vision encoder LoRA 별도 설정 |
| **Qwen-VL-Finetune** | ✅ | `--freeze_vision_tower`, `--freeze_llm` |
| **TRL (HuggingFace)** | ✅ | `model.vision_tower.requires_grad_(False)` |

### 6.2 예외 케이스

일부 모델은 **아키텍처 특성상** 분리 학습이 제한됩니다:

- **Fuyu-8B**: 별도의 비전 인코더 없음 (이미지 패치를 직접 LLM에 입력)
- **EVEv2**: LLM 내부에 비전 인식 기능 통합
- **Native Multimodal Models**: 처음부터 멀티모달로 학습된 모델 (GPT-4V, Gemini)

### 6.3 프레임워크별 설정 예시

**LLaMA-Factory**:
```yaml
# freeze 설정
freeze_vision_tower: true
freeze_llm: false
lora_target: all  # LLM에 LoRA 적용
```

**lmms-finetune**:
```bash
# 비전 인코더 동결, LLM에 LoRA 적용
python train.py \
    --freeze_vision_tower \
    --lora_enable \
    --lora_r 64
```

**PyTorch 직접 설정**:
```python
# 비전 인코더 동결
for param in model.vision_tower.parameters():
    param.requires_grad = False

# LLM에 LoRA 적용
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)
```

## 7. 실전 권장 사항

### 7.1 OCR/문서 이해 파인튜닝 시

```
추천 전략: Stage 2 (Projector + LLM LoRA)
비전 인코더: 동결 유지 (안정성)
추가 고려: 고해상도 입력 지원 확인
```

### 7.2 비전 인코더 학습 시 주의사항

1. **Learning rate를 매우 낮게** (LLM의 1/10 ~ 1/100)
2. **학습 후반부에만** unfreeze
3. **충분한 데이터** 확보 (최소 수만 샘플)
4. **Gradient checkpointing** 필수

### 7.3 Catastrophic Forgetting 방지

- **Replay 전략**: 기존 데이터 일부를 학습에 포함
- **LoRA 사용**: 원본 가중치 보존
- **낮은 epoch 수**: 1-3 epoch 권장
- **Early stopping**: validation loss 모니터링

## 8. VILA 연구 결과 요약

NVIDIA의 VILA 논문에서 밝힌 주요 발견:

| 발견 | 의미 |
|------|------|
| 프로젝터만 SFT → 성능 저하 | LLM 학습이 필수적 |
| Pretraining 시 LLM 동결 → 0-shot은 유지, in-context learning 저하 | 일반화 능력에 영향 |
| 단순한 Linear projector > 복잡한 Transformer projector | 단순함이 때로는 더 효과적 |

## 9. 현재 프로젝트 구현 (DeepSeek-OCR)

### 9.1 아키텍처

```
Vision Encoder (SigLIP-SO400M-384)
        │
        ▼
   MLP Projector (2-layer)  ← multi_modal_projector
        │
        ▼
   LLM (DeepSeekMoE)
```

### 9.2 학습 모드

| 모드 | 학습 대상 |
|------|----------|
| `vision` | Projector + Vision Encoder |
| `llm` | Projector + LLM |
| `both` | Projector + Vision Encoder + LLM |

**모든 모드에서 Projector는 기본 포함됩니다.**

### 9.3 설정 예시 (`config/train_config.yaml`)

```yaml
lora:
  # Projector: 모든 모드에서 자동 포함
  projector_target_modules:
    - "multi_modal_projector"
    - "vision_embed_tokens"

  # Vision Encoder (mode: vision)
  vision_target_modules:
    - "qkv_proj"
    - "out_proj"
    - "fc1"
    - "fc2"

  # LLM (mode: llm)
  llm_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### 9.4 사용법

```bash
# LLM + Projector 학습 (권장)
TRAINING_MODE=llm uv run main.py train --dataset data.jsonl

# Vision Encoder + Projector 학습
TRAINING_MODE=vision uv run main.py train --dataset data.jsonl

# 전체 학습
TRAINING_MODE=both uv run main.py train --dataset data.jsonl
```

---

## 10. 결론

1. **비전 인코더와 LLM을 분리하여 학습하는 것은 표준 관행**입니다.
2. **대부분의 경우 비전 인코더는 동결**하고 프로젝터 + LLM을 학습합니다.
3. **도메인 특화 시각 이해가 필요한 경우**에만 비전 인코더 학습을 고려합니다.
4. **모든 주요 VLM 파인튜닝 프레임워크**에서 이러한 분리 학습을 지원합니다.

---

## 참고 자료

- [Vision Language Models Explained - Hugging Face](https://huggingface.co/blog/vlms)
- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design)
- [VILA: On Pre-training for Visual Language Models (CVPR 2024)](https://arxiv.org/html/2312.07533v3)
- [Fine-Tuning VLM with TRL - Hugging Face Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen-VL-Series-Finetune GitHub](https://github.com/2U1/Qwen-VL-Series-Finetune)
- [lmms-finetune GitHub](https://github.com/zjysteven/lmms-finetune)
- [VLM Training Process - Medium](https://medium.com/@hexiangnan/how-vision-language-models-are-trained-a-deep-dive-into-the-vlm-training-process-1ba1d8704bb0)
