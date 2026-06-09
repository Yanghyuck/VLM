# =============================================================================
# scripts/exp_temp_sweep.py
# -----------------------------------------------------------------------------
# 요약 다양성 vs 사실성 trade-off 매핑.
#   운영 v8 을 메인 eval_set(50)에 greedy + 여러 temperature 로 추론해
#   3문장_요약 의 distinct_2(다양성) 와 사실 보존(등급/수치/성별)을 함께 측정.
#   B 실험(greedy 0.270 / temp0.3 0.279)의 후속 — 고온에서 다양성이 오르는 대신
#   사실이 언제 깨지는지 sweet spot 탐색.
#
# 사용: python scripts/exp_temp_sweep.py --temps 0.5,0.7,0.9
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG
from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report
from vlm.bench.scorer import compute_distinct_n

ADAPTER = str(ROOT / CFG.paths.lora_adapter)
GENDER_MAP = {1: "암컷", 2: "수컷", 3: "거세"}


def to_output(meta):
    img = meta.get("result_image_path")
    if img and not Path(img).exists():
        img = None
    return ThemaPAOutput(**{**meta, "result_image_path": img})


def fact_ok(summary: str, meta: dict):
    """등급/수치3종/성별 보존 여부 (bool 3종)."""
    grade = str(meta["grade"])
    gender = GENDER_MAP.get(meta["gender"], "미상")
    nums = [str(meta["backfat_average"]), str(meta["multifidus_thk"]), str(meta["body_weight"])]
    def has(n):
        return n in summary or n.rstrip("0").rstrip(".") in summary
    return (grade in summary), all(has(n) for n in nums), (gender in summary)


def run_setting(rows, label, **gen_kwargs):
    summaries, g_ok, n_ok, x_ok = [], 0, 0, 0
    for i, r in enumerate(rows, 1):
        meta = r["metadata"]
        rep = generate_report(to_output(meta), adapter_path=ADAPTER, **gen_kwargs)
        s = (rep or {}).get("3문장_요약", "")
        summaries.append(s)
        a, b, c = fact_ok(s, meta)
        g_ok += a; n_ok += b; x_ok += c
        print(f"  [{label}] {i}/{len(rows)}", flush=True)
    n = len(rows)
    return {
        "label": label,
        "distinct_2": round(compute_distinct_n(summaries, 2), 4),
        "uniq_rate": round(len(set(summaries)) / n, 3),
        "grade_keep": f"{g_ok}/{n}",
        "num_keep": f"{n_ok}/{n}",
        "gender_keep": f"{x_ok}/{n}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=str, default="0.5,0.7,0.9")
    ap.add_argument("--eval-set", type=str, default="vlm/bench/eval_set.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / args.eval_set, encoding="utf-8")]
    print(f"[sweep] 어댑터 {ADAPTER} | eval {len(rows)}건")

    results = [run_setting(rows, "greedy", sampling=False)]
    for t in [float(x) for x in args.temps.split(",")]:
        results.append(run_setting(rows, f"temp{t}", sampling=True, temperature=t))

    print("\n=== 요약 다양성 vs 사실성 trade-off ===")
    print(f"{'설정':<10} {'distinct_2':>10} {'uniq율':>7} {'등급보존':>9} {'수치보존':>9} {'성별보존':>9}")
    for r in results:
        print(f"{r['label']:<10} {r['distinct_2']:>10} {r['uniq_rate']:>7} {r['grade_keep']:>9} {r['num_keep']:>9} {r['gender_keep']:>9}")
    (ROOT / "vlm/bench/temp_sweep.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장: vlm/bench/temp_sweep.json")


if __name__ == "__main__":
    main()
