# =============================================================================
# scripts/run_v3_benchmark.py
# -----------------------------------------------------------------------------
# v3 학습 종료 후 5-way 비교 벤치마크 자동화.
# v3 만 신규 추론하고 기존 base/v1/v2-prejosa/v2-corrected 결과는 재사용.
#
# 사용:
#   python scripts/run_v3_benchmark.py
#
# 사전 조건:
#   - vlm/train/output/qwen3vl-lora-v3/adapter_model.safetensors 존재
#   - vlm/bench/eval_set.jsonl + 4개 기존 결과 jsonl 존재
#
# 자동 단계:
#   1. v3 어댑터 + 학습 메트릭(train/eval loss) 검증
#   2. v3 50건 추론 → results_lora_v3.jsonl
#   3. 5-way scorer → score_report.md (덮어쓰기)
#   4. 결과 콘솔 출력 + over-fit 경고 (v2-corrected 와 비교)
#
# 시간: ~25-50분 (v2-corrected 21.7s/req 기준 50건, v3 는 rank 2배라 더 길 수 있음)
# =============================================================================

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

BENCH_DIR  = ROOT / "vlm" / "bench"
ADAPTER_V3 = ROOT / "vlm" / "train" / "output" / "qwen3vl-lora-v3"
RESULTS_V3 = BENCH_DIR / "results_lora_v3.jsonl"

# v2-corrected 메트릭 (over-fit 비교 기준)
V2_CORRECTED_TRAIN_LOSS = 0.166
V2_CORRECTED_EVAL_LOSS  = 0.079


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def step_1_validate_v3():
    """v3 학습 산출물 + 메트릭 검증."""
    safe_print(f"\n{'=' * 70}\n[1/4] v3 학습 산출물 검증\n{'=' * 70}")

    adapter_file = ADAPTER_V3 / "adapter_model.safetensors"
    if not adapter_file.exists():
        safe_print(f"[FAIL] {adapter_file} 없음 — 학습 종료 안 됨")
        return None

    train_results = ADAPTER_V3 / "train_results.json"
    eval_results  = ADAPTER_V3 / "eval_results.json"
    metrics: dict = {}
    if train_results.exists():
        with open(train_results, encoding="utf-8") as f:
            tr = json.load(f)
        metrics["train_loss"]    = tr.get("train_loss")
        metrics["train_runtime"] = tr.get("train_runtime")
        safe_print(f"  train_loss:    {metrics['train_loss']}")
        safe_print(f"  train_runtime: {metrics['train_runtime']}s ({metrics['train_runtime']/3600:.2f}h)")
    if eval_results.exists():
        with open(eval_results, encoding="utf-8") as f:
            er = json.load(f)
        metrics["eval_loss"] = er.get("eval_loss")
        safe_print(f"  eval_loss:     {metrics['eval_loss']}")

    # over-fit 자동 판정
    if metrics.get("train_loss") and metrics.get("eval_loss"):
        gap = metrics["eval_loss"] - metrics["train_loss"]
        if metrics["eval_loss"] < metrics["train_loss"]:
            safe_print(f"  [OK] eval < train — 과적합 없음")
        elif gap > 0.05:
            safe_print(f"  [WARN] eval-train gap {gap:.4f} > 0.05 — 과적합 우려")
        else:
            safe_print(f"  [OK] eval-train gap {gap:.4f} 정상 범위")

        if metrics["eval_loss"] < V2_CORRECTED_EVAL_LOSS:
            improvement = (V2_CORRECTED_EVAL_LOSS - metrics["eval_loss"]) / V2_CORRECTED_EVAL_LOSS * 100
            safe_print(f"  v2-corrected eval_loss 대비 개선: -{improvement:.1f}%")
        else:
            regression = (metrics["eval_loss"] - V2_CORRECTED_EVAL_LOSS) / V2_CORRECTED_EVAL_LOSS * 100
            safe_print(f"  [WARN] v2-corrected eval_loss 대비 회귀: +{regression:.1f}%")

    safe_print(f"  [OK] 어댑터: {adapter_file} ({adapter_file.stat().st_size / 1024 / 1024:.0f} MB)")
    return metrics


def step_2_inference_v3():
    """v3 어댑터로 50건 추론."""
    safe_print(f"\n{'=' * 70}\n[2/4] v3 50건 추론\n{'=' * 70}")
    cmd = [
        PYTHON, str(BENCH_DIR / "runner.py"),
        "--mode", "lora",
        "--tag", "lora_v3",
        "--adapter-path", str(ADAPTER_V3),
        "--n", "50",
        "--output", str(RESULTS_V3),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    safe_print(f"  종료. 코드={proc.returncode}, 경과={elapsed/60:.1f}분")
    return proc.returncode == 0


def step_3_scorer():
    """5-way scorer 실행."""
    safe_print(f"\n{'=' * 70}\n[3/4] 5-way scorer (base / v1 / v2-prejosa / v2-corrected / v3)\n{'=' * 70}")
    cmd = [
        PYTHON, str(BENCH_DIR / "scorer.py"),
        "--inputs",
        f"base={BENCH_DIR / 'results_base.jsonl'}",
        f"lora_v1={BENCH_DIR / 'results_lora_v1.jsonl'}",
        f"lora_v2_prejosa={BENCH_DIR / 'results_lora_v2_prejosa.jsonl'}",
        f"lora_v2_corrected={BENCH_DIR / 'results_lora_v2_corrected.jsonl'}",
        f"lora_v3={RESULTS_V3}",
        "--baseline", "base",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode == 0


def step_4_compare_summary(train_metrics: dict | None):
    """v2-corrected vs v3 간단 종합 요약."""
    safe_print(f"\n{'=' * 70}\n[4/4] v2-corrected vs v3 종합 요약\n{'=' * 70}")
    safe_print(f"\n  학습 메트릭:")
    safe_print(f"    v2-corrected: train={V2_CORRECTED_TRAIN_LOSS}, eval={V2_CORRECTED_EVAL_LOSS}")
    if train_metrics:
        safe_print(f"    v3:           train={train_metrics.get('train_loss')}, eval={train_metrics.get('eval_loss')}")
    safe_print(f"\n  벤치 점수: vlm/bench/score_report.md 참조")
    safe_print(f"  운영 권장: 다양성/속도/환각/메트릭 종합 판단")


def main() -> int:
    overall_t0 = time.time()

    metrics = step_1_validate_v3()
    if metrics is None:
        return 1

    if not step_2_inference_v3():
        safe_print("[FAIL] step 2 추론 실패")
        return 1

    if not step_3_scorer():
        safe_print("[FAIL] step 3 scorer 실패")
        return 1

    step_4_compare_summary(metrics)

    total = time.time() - overall_t0
    safe_print(f"\n{'=' * 70}\n[ALL DONE] 총 소요: {total/60:.1f}분\n리포트: {BENCH_DIR / 'score_report.md'}\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
