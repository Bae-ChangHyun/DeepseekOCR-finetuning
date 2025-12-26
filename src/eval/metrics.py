"""
Evaluation Metrics Module

OCR 및 레이아웃 분석 성능 평가를 위한 메트릭
"""

from pathlib import Path

from loguru import logger

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER)을 계산합니다.

    CER = (Substitutions + Insertions + Deletions) / Total Characters in Reference

    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트

    Returns:
        CER 값 (0.0 ~ 1.0+, 낮을수록 좋음)
    """
    if not reference:
        return 1.0 if hypothesis else 0.0

    if JIWER_AVAILABLE:
        # jiwer는 word 단위이므로 character 단위로 변환
        ref_chars = " ".join(list(reference))
        hyp_chars = " ".join(list(hypothesis))
        return jiwer.wer(ref_chars, hyp_chars)
    else:
        # Levenshtein distance 기반 수동 계산
        return _levenshtein_cer(reference, hypothesis)


def _levenshtein_cer(reference: str, hypothesis: str) -> float:
    """Levenshtein distance 기반 CER 계산"""
    m, n = len(reference), len(hypothesis)

    if m == 0:
        return 1.0 if n > 0 else 0.0

    # DP 테이블
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[m][n] / m


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate (WER)을 계산합니다.

    WER = (Substitutions + Insertions + Deletions) / Total Words in Reference

    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트

    Returns:
        WER 값 (0.0 ~ 1.0+, 낮을수록 좋음)
    """
    if JIWER_AVAILABLE:
        return jiwer.wer(reference, hypothesis)
    else:
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if not ref_words:
            return 1.0 if hyp_words else 0.0

        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    )

        return dp[m][n] / m


def compute_accuracy(reference: str, hypothesis: str) -> float:
    """
    정확도를 계산합니다 (1 - CER).

    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트

    Returns:
        정확도 (0.0 ~ 1.0, 높을수록 좋음)
    """
    cer = compute_cer(reference, hypothesis)
    return max(0.0, 1.0 - cer)


def evaluate_batch(
    references: list[str],
    hypotheses: list[str],
) -> dict:
    """
    배치 단위로 평가를 수행합니다.

    Args:
        references: 정답 텍스트 리스트
        hypotheses: 예측 텍스트 리스트

    Returns:
        평가 결과 딕셔너리
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    cer_scores = []
    wer_scores = []
    accuracy_scores = []

    for ref, hyp in zip(references, hypotheses):
        cer_scores.append(compute_cer(ref, hyp))
        wer_scores.append(compute_wer(ref, hyp))
        accuracy_scores.append(compute_accuracy(ref, hyp))

    return {
        "cer": {
            "mean": sum(cer_scores) / len(cer_scores),
            "min": min(cer_scores),
            "max": max(cer_scores),
            "scores": cer_scores,
        },
        "wer": {
            "mean": sum(wer_scores) / len(wer_scores),
            "min": min(wer_scores),
            "max": max(wer_scores),
            "scores": wer_scores,
        },
        "accuracy": {
            "mean": sum(accuracy_scores) / len(accuracy_scores),
            "min": min(accuracy_scores),
            "max": max(accuracy_scores),
            "scores": accuracy_scores,
        },
        "num_samples": len(references),
    }


def evaluate_model(
    model,
    tokenizer,
    eval_data: list[dict],
    image_size: int = 640,
    base_size: int = 1024,
    crop_mode: bool = True,
    prompt: str = "<image>\nFree OCR. ",
    verbose: bool = True,
) -> dict:
    """
    모델을 평가합니다.

    Args:
        model: 평가할 모델
        tokenizer: 토크나이저
        eval_data: 평가 데이터 (messages 형식)
        image_size: 이미지 크기
        base_size: 베이스 크기
        crop_mode: 크롭 모드
        prompt: 추론 프롬프트
        verbose: 상세 출력 여부

    Returns:
        평가 결과 딕셔너리
    """
    try:
        from unsloth import FastVisionModel
    except ImportError as e:
        raise ImportError(
            "unsloth is required. Install it with: pip install unsloth"
        ) from e

    from tqdm import tqdm

    FastVisionModel.for_inference(model)

    references = []
    hypotheses = []
    results = []

    for sample in tqdm(eval_data, desc="Evaluating", disable=not verbose):
        messages = sample.get("messages", [])

        # 이미지 경로 추출
        image_path = None
        reference_text = None

        for msg in messages:
            if msg.get("role") == "<|User|>" and "images" in msg:
                images = msg["images"]
                if images:
                    image_path = images[0] if isinstance(images[0], str) else None
            elif msg.get("role") == "<|Assistant|>":
                reference_text = msg.get("content", "")

        if image_path is None or reference_text is None:
            continue

        # 추론 (eval_mode=True로 결과 반환받음)
        try:
            hypothesis = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path="./eval_output",
                image_size=image_size,
                base_size=base_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False,
                eval_mode=True,  # 결과를 반환받기 위해 필수!
            )
            hypothesis = hypothesis.strip() if hypothesis else ""

        except Exception as e:
            if verbose:
                logger.error(f"Error processing {image_path}: {e}")
            hypothesis = ""

        references.append(reference_text)
        hypotheses.append(hypothesis)

        cer = compute_cer(reference_text, hypothesis)
        wer = compute_wer(reference_text, hypothesis)
        results.append({
            "image_path": image_path,
            "reference": reference_text,
            "hypothesis": hypothesis,
            "cer": cer,
            "wer": wer,
        })

    # 전체 평가
    eval_results = evaluate_batch(references, hypotheses)
    eval_results["detailed_results"] = results

    if verbose:
        logger.info("Evaluation Results")
        logger.info(f"Samples: {eval_results['num_samples']}")
        logger.info(f"Mean CER: {eval_results['cer']['mean']:.4f}")
        logger.info(f"Mean WER: {eval_results['wer']['mean']:.4f}")
        logger.info(f"Mean Accuracy: {eval_results['accuracy']['mean']:.4f}")

    return eval_results


def compare_models(
    baseline_results: dict,
    finetuned_results: dict,
) -> dict:
    """
    베이스라인과 파인튜닝 모델의 결과를 비교합니다.

    Args:
        baseline_results: 베이스라인 평가 결과
        finetuned_results: 파인튜닝 평가 결과

    Returns:
        비교 결과 딕셔너리
    """
    baseline_cer = baseline_results["cer"]["mean"]
    finetuned_cer = finetuned_results["cer"]["mean"]

    cer_improvement = baseline_cer - finetuned_cer
    cer_improvement_pct = (cer_improvement / baseline_cer * 100) if baseline_cer > 0 else 0

    baseline_wer = baseline_results["wer"]["mean"]
    finetuned_wer = finetuned_results["wer"]["mean"]

    wer_improvement = baseline_wer - finetuned_wer
    wer_improvement_pct = (wer_improvement / baseline_wer * 100) if baseline_wer > 0 else 0

    comparison = {
        "cer": {
            "baseline": baseline_cer,
            "finetuned": finetuned_cer,
            "improvement": cer_improvement,
            "improvement_pct": cer_improvement_pct,
        },
        "wer": {
            "baseline": baseline_wer,
            "finetuned": finetuned_wer,
            "improvement": wer_improvement,
            "improvement_pct": wer_improvement_pct,
        },
    }

    logger.info("Model Comparison")
    logger.info(f"CER: {baseline_cer:.4f} -> {finetuned_cer:.4f} ({cer_improvement_pct:+.1f}%)")
    logger.info(f"WER: {baseline_wer:.4f} -> {finetuned_wer:.4f} ({wer_improvement_pct:+.1f}%)")

    return comparison
