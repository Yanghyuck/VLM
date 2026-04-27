# =============================================================================
# scripts/run_3way_benchmark.py
# -----------------------------------------------------------------------------
# Base / v1 LoRA / v2 LoRA 3-way 벤치마크 일괄 실행.
# 50건 held-out 샘플에 대해 세 모델 모두 추론 후 scorer 까지 실행.
#
# 동작 방법:
#   python scripts/run_3way_benchmark.py
#
# 출력:
#   vlm/bench/results_base.jsonl
#   vlm/bench/results_lora_v1.jsonl
#   vlm/bench/results_lora_v2.jsonl
#   vlm/bench/score_report.md
# =============================================================================

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

BENCH_DIR = ROOT / "vlm" / "bench"
ADAPTER_V1 = ROOT / "vlm" / "train" / "output" / "qwen3vl-lora-v1-textonly"
ADAPTER_V2 = ROOT / "vlm" / "train" / "output" / "qwen3vl-lora"

RUNS = [
    {
        "label":   "base",
        "args":    ["--mode", "base", "--tag", "base"],
        "output":  BENCH_DIR / "results_base.jsonl",
    },
    {
        "label":   "lora_v1",
        "args":    ["--mode", "lora", "--tag", "lora_v1",
                    "--adapter-path", str(ADAPTER_V1)],
        "output":  BENCH_DIR / "results_lora_v1.jsonl",
    },
    {
        "label":   "lora_v2",
        "args":    ["--mode", "lora", "--tag", "lora_v2",
                    "--adapter-path", str(ADAPTER_V2)],
        "output":  BENCH_DIR / "results_lora_v2.jsonl",
    },
]


def run_step(label: str, args: list[str], output: Path) -> bool:
    print(f"\n{'='*70}")
    print(f"[{label}] 시작")
    print(f"{'='*70}")
    cmd = [PYTHON, str(ROOT / "vlm" / "bench" / "runner.py"),
           "--n", "50", "--output", str(output)] + args
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"[{label}] 종료. 코드={proc.returncode}, 경과={elapsed/60:.1f}분")
    return proc.returncode == 0


def run_scorer():
    print(f"\n{'='*70}")
    print(f"[scorer] 3-way 비교 실행")
    print(f"{'='*70}")
    cmd = [PYTHON, str(ROOT / "vlm" / "bench" / "scorer.py"),
           "--inputs",
           f"base={BENCH_DIR / 'results_base.jsonl'}",
           f"lora_v1={BENCH_DIR / 'results_lora_v1.jsonl'}",
           f"lora_v2={BENCH_DIR / 'results_lora_v2.jsonl'}",
           "--baseline", "base"]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode == 0


def main():
    overall_t0 = time.time()
    for run in RUNS:
        ok = run_step(run["label"], run["args"], run["output"])
        if not ok:
            print(f"[FAIL] {run['label']} 실패. 중단.")
            return 1
    if not run_scorer():
        print("[FAIL] scorer 실패")
        return 1
    total = time.time() - overall_t0
    print(f"\n{'='*70}")
    print(f"[ALL DONE] 총 소요: {total/60:.1f}분")
    print(f"리포트: {BENCH_DIR / 'score_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
