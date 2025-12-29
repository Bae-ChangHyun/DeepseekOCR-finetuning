# TODO

## 현재 이슈
- (없음)

## 완료된 작업

### 코드 리뷰 개선 사항 (2024-12-29)

#### Critical (P0)
- [x] [C2] 임시 파일 정리 로직 추가 (`main.py`)
  - `tempfile.TemporaryDirectory` 사용으로 자동 정리
- [x] [C3] GPU 메모리 통계 에러 처리 (`trainer.py`)
  - CUDA 사용 불가 시 안전하게 처리
- [x] [C1] 모델 의존성 어댑터 패턴 적용 (`collator.py`)
  - `src/data/model_adapter.py` 생성하여 외부 모델 의존성 추상화

#### High Priority (P1)
- [x] [H1] `src/data/__init__.py` export 완성
  - `preprocessor`, `model_adapter`, `run_inference_markdown`, `list_available_prompts` 추가
- [x] [H2] 하드코딩된 설정 경로 제거 (`trainer.py`)
  - `_config_path` 인스턴스 변수 사용
- [x] [H7] Graceful Shutdown 구현 (`main.py`)
  - `GracefulShutdown` 클래스로 SIGINT/SIGTERM 핸들링
  - 학습 중단 시 체크포인트 자동 저장

#### Medium Priority (P2)
- [x] [M1] asyncio 중첩 호출 문제 해결 (`infer.py`)
  - `run_async()` 함수로 Jupyter Notebook 호환성 확보
  - `nest_asyncio` 또는 스레드 풀 폴백 지원
- [x] [M4] 추론 체크포인트 기능 추가 (`infer.py`)
  - `DatasetInferencer.run_with_checkpoint()` 메서드 추가
  - 중단 시 자동 저장, 재개 시 처리된 이미지 건너뛰기

### 이전 완료 작업

#### 1. vision_target_modules 수정
- DeepSeek-OCR 모델의 실제 레이어 구조에 맞게 수정
- `encoder.layers` → `transformer.layers`
- `q_proj, k_proj, v_proj` → `qkv_proj` (통합)
- `o_proj` → `out_proj`
- PEFT는 `.{name}`으로 끝나는 모듈을 매칭하므로 짧은 이름 사용

#### 2. infer 병렬 추론 + 실시간 마크다운 저장
- `group_images_by_document()`: 이미지를 `{name}_p{page}` 패턴으로 문서별 그룹화
- `APIInferencer.infer_by_document_streaming()`: 문서 그룹별 병렬 추론 + 콜백
- `LocalInferencer.infer_by_document_streaming()`: 문서 그룹별 순차 추론 + 콜백
- `DatasetInferencer.run_markdown()`: 스트리밍 방식으로 변경
  - 각 문서 추론 완료 시 즉시 마크다운 파일 저장
  - 추론 진행 중에도 완료된 문서 결과 확인 가능

## 메모

### 추론 모드별 동작
| 출력 형식 | API 모드 | Local 모드 |
|----------|---------|-----------|
| md | 문서 그룹별 병렬 추론, 완료 시 즉시 저장 | 문서 그룹별 순차 추론, 완료 시 즉시 저장 |
| json/jsonl | 전체 병렬 추론 후 일괄 저장 | 전체 순차 추론 후 일괄 저장 |

### 새로 추가된 기능
- `run_with_checkpoint()`: 대량 추론 시 중단/재개 지원
- `GracefulShutdown`: 학습 중 Ctrl+C 시 체크포인트 자동 저장
- `DeepSeekOCRAdapter`: 외부 모델 의존성 추상화
