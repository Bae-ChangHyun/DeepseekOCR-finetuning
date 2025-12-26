# TODO

## 현재 이슈
- (없음)

## 완료된 작업

### 1. vision_target_modules 수정
- DeepSeek-OCR 모델의 실제 레이어 구조에 맞게 수정
- `encoder.layers` → `transformer.layers`
- `q_proj, k_proj, v_proj` → `qkv_proj` (통합)
- `o_proj` → `out_proj`
- PEFT는 `.{name}`으로 끝나는 모듈을 매칭하므로 짧은 이름 사용

### 2. infer 병렬 추론 + 실시간 마크다운 저장
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
