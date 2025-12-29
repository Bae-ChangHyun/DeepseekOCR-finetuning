# TODO

## 현재 이슈
- (없음)

## 완료된 작업

### Projector 학습 기본 포함 (2024-12-29)
- [x] 모든 모드(vision/llm/both)에 프로젝터 자동 포함
- [x] `src/train/layers.py`: `get_target_modules()`에서 프로젝터 모듈 기본 추가
- [x] `config/train_config.yaml`: `projector_target_modules` 섹션 추가
- [x] `docs/vlm_파인튜닝.md`: 현재 프로젝트 구현 섹션 추가

### CLI 인자 정리 (2024-12-29)

#### 변경 사항
- [x] 모든 인자에 단축형 추가 (`-d`, `-c`, `-o`, `-t`, `-r` 등)
- [x] `--images` → `--img` (infer 커맨드)
- [x] `--checkpoint` → `--resume` (infer 커맨드)
- [x] README.md, README.eng.md 업데이트

### 체크포인트 메타데이터 및 검증 기능 (2024-12-29)

#### 학습 (train)
- [x] 학습 메타데이터 저장 (`training_meta.json`)
  - dataset, config_path, mode, output_dir 등 저장
  - 학습 시작 시 자동 저장
- [x] `--resume` 시 메타데이터 검증/자동로드
  - 인자 생략 시 메타데이터에서 자동 로드
  - 인자 제공 시 메타데이터와 비교 검증
  - `--dataset` 선택적으로 변경 (resume 시 자동 로드 가능)

#### 추론 (infer)
- [x] 추론 체크포인트에 메타데이터 추가
  - image_source, config_path, task, output_path 등 저장
- [x] `--checkpoint` CLI 옵션 추가
  - `--checkpoint`: 체크포인트 모드 활성화 (자동 경로)
  - `--checkpoint path.json`: 특정 체크포인트에서 재개
- [x] 재개 시 메타데이터 검증/자동로드
  - 인자 생략 시 메타데이터에서 자동 로드
  - 인자 제공 시 메타데이터와 비교 검증
  - `--config` 선택적으로 변경 (checkpoint 재개 시 자동 로드 가능)

### 코드 리뷰 개선 사항 (2024-12-29)

#### Critical (P0)
- [x] [C2] 임시 파일 정리 로직 추가 (`main.py`)
- [x] [C3] GPU 메모리 통계 에러 처리 (`trainer.py`)
- [x] [C1] 모델 의존성 어댑터 패턴 적용 (`model_adapter.py`)

#### High Priority (P1)
- [x] [H1] `src/data/__init__.py` export 완성
- [x] [H2] 하드코딩된 설정 경로 제거 (`trainer.py`)
- [x] [H7] Graceful Shutdown 구현 (`main.py`)

#### Medium Priority (P2)
- [x] [M1] asyncio 중첩 호출 문제 해결 (`infer.py`)
- [x] [M4] 추론 체크포인트 기능 추가 (`infer.py`)

### 이전 완료 작업

#### 1. vision_target_modules 수정
- DeepSeek-OCR 모델의 실제 레이어 구조에 맞게 수정

#### 2. infer 병렬 추론 + 실시간 마크다운 저장
- 문서 그룹별 병렬/순차 추론 + 콜백 방식

## 메모

### 사용 예시

#### 학습 재개
```bash
# 처음 학습
uv run main.py train --dataset data.jsonl --config config/train.yaml --output ./output

# 재개 (인자 자동 로드)
uv run main.py train --resume ./output/checkpoints/checkpoint-100

# 재개 (인자 제공 시 검증)
uv run main.py train --resume ./output/checkpoints/checkpoint-100 --dataset data.jsonl
```

#### 추론 체크포인트
```bash
# 체크포인트 모드로 추론
uv run main.py infer --images ./images --config config/teacher.yaml --checkpoint

# 체크포인트에서 재개 (인자 자동 로드)
uv run main.py infer --checkpoint ./data/result.checkpoint.json

# 체크포인트에서 재개 (인자 제공 시 검증)
uv run main.py infer --checkpoint ./data/result.checkpoint.json --task document
```
