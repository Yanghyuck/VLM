# =============================================================================
# scripts/run_v2_compare_benchmark.py
# -----------------------------------------------------------------------------
# v2-prejosa (어색 조사 학습본) vs v2-corrected (정제 데이터 재학습) 비교.
# base, v1 결과는 이전 산출물(`results_base.jsonl`, `results_lora_v1.jsonl`)을
# 그대로 재사용하여 4-way 스코어 리포트를 생성한다.
#
# 사용:
#   python scripts/run_v2_compare_benchmark.py
#
# 입력 (이미 존재):
#   vlm/bench/eval_set.jsonl
#   vlm/bench/results_base.jsonl
#   vlm/bench/results_lora_v1.jsonl
#
# 출력:
#   vlm/bench/results_lora_v2_prejosa.jsonl    — 신규 추론
#   vlm/bench/results_lora_v2_corrected.jsonl  — 신규 추론
#   vlm/bench/score_report.md                  — 4-way 비교 리포트 (덮어쓰기)
#
# 시간: 50건 × 2 어댑터 ≈ 50분 (RTX 4090 기준)
# =============================================================================

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

BENCH_DIR = ROOT / "vlm" / "bench"
ADAPTER_V2_PREJOSA = ROOT / "vlm" / "train" / "output" / "qwen3vl-lora-v2-prejosa"
ADAPTER_V2_CORRECTED = ROOT / "vlm" / "train" / "output" / "qwen3vl-lora"

RUNS = [
    {
        "label": "lora_v2_prejosa",
        "args": [
            "--mode", "lora",
            "--tag", "lora_v2_prejosa",
            "--adapter-path", str(ADAPTER_V2_PREJOSA),
        ],
        "output": BENCH_DIR / "results_lora_v2_prejosa.jsonl",
    },
    {
        "label": "lora_v2_corrected",
        "args": [
            "--mode", "lora",
            "--tag", "lora_v2_corrected",
            "--adapter-path", str(ADAPTER_V2_CORRECTED),
        ],
        "output": BENCH_DIR / "results_lora_v2_corrected.jsonl",
    },
]


def run_step(label: str, args: list[str], output: Path) -> bool:
    print(f"\n{'=' * 70}")
    print(f"[{label}] 시작")
    print(f"{'=' * 70}")
    cmd = [
        PYTHON, str(BENCH_DIR / "runner.py"),
        "--n", "50",
        "--output", str(output),
    ] + args
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"[{label}] 종료. 코드={proc.returncode}, 경과={elapsed/60:.1f}분")
    return proc.returncode == 0


def run_scorer() -> bool:
    print(f"\n{'=' * 70}")
    print(f"[scorer] 4-way 비교 (base / v1 / v2-prejosa / v2-corrected)")
    print(f"{'=' * 70}")
    cmd = [
        PYTHON, str(BENCH_DIR / "scorer.py"),
        "--inputs",
        f"base={BENCH_DIR / 'results_base.jsonl'}",
        f"lora_v1={BENCH_DIR / 'results_lora_v1.jsonl'}",
        f"lora_v2_prejosa={BENCH_DIR / 'results_lora_v2_prejosa.jsonl'}",
        f"lora_v2_corrected={BENCH_DIR / 'results_lora_v2_corrected.jsonl'}",
        "--baseline", "base",
    ]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0


def main() -> int:
    overall_t0 = time.time()
    for run in RUNS:
        if not run_step(run["label"], run["args"], run["output"]):
            print(f"[FAIL] {run['label']} 실패. 중단.")
            return 1
    if not run_scorer():
        print("[FAIL] scorer 실패")
        return 1
    total = time.time() - overall_t0
    print(f"\n{'=' * 70}")
    print(f"[ALL DONE] 총 소요: {total/60:.1f}분")
    print(f"리포트: {BENCH_DIR / 'score_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
