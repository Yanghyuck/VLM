# =============================================================================
# scripts/eval_error_code_faithfulness.py
# -----------------------------------------------------------------------------
# B — error_code 충실도(faithfulness) 평가.
#   harness run 결과(runs/<...>/results.jsonl)의 abnormal 케이스에서
#   `비정상_근거` 가 언급한 error_code 가 입력 error_code 와 일치하는지 측정.
#
#   - extra_rate  : 입력에 없는 코드를 끼워넣은 환각 비율 (v8 backfat 스모크의 결함)
#   - missing_rate: 입력 코드를 누락한 비율
#   - exact_match_rate: 정확히 일치(extra·missing 둘 다 없음)
#
# 사용:
#   python scripts/eval_error_code_faithfulness.py \
#       vlm/bench/runs/20260604T113154Z__eff1008__lora_v8/results.jsonl
#   # 여러 개 동시 비교:
#   python scripts/eval_error_code_faithfulness.py runs/*/results.jsonl
#
# 주의: 현재 eval_set 은 abnormal 2건뿐이라 표본이 작다. abnormal 층화 평가셋
#       (vlm/bench/dataset.py build_eval_set_from_db) 으로 보강하면 견고해진다.
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.postprocess import extract_error_codes, faithfulness
from vlm.bench.scorer import error_code_faithfulness


def _is_abnormal(ec: dict) -> bool:
    return any((v or 0) for v in (ec or {}).values())


def eval_run(results_path: Path) -> dict:
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8")]
    pairs: list[tuple[set, set]] = []
    details = []
    for r in rows:
        ec = (r.get("metadata") or {}).get("error_code") or {}
        if not _is_abnormal(ec):
            continue
        pred = r.get("prediction") or {}
        reason = pred.get("비정상_근거") if isinstance(pred, dict) else None
        detected = extract_error_codes(reason or "")
        expected = {c for c, v in ec.items() if v}
        pairs.append((detected, expected))
        extra, missing = faithfulness(detected, expected)
        details.append({"id": r.get("id"), "expected": sorted(expected),
                        "detected": sorted(detected),
                        "extra": sorted(extra), "missing": sorted(missing)})
    metric = error_code_faithfulness(pairs)
    return {"metric": metric, "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="harness run 의 results.jsonl 경로(들)")
    args = ap.parse_args()

    for p in args.results:
        path = Path(p)
        label = path.parent.name or path.name
        res = eval_run(path)
        m = res["metric"]
        print(f"\n=== error_code 충실도: {label} ===")
        if m.get("n", 0) == 0:
            print("abnormal 케이스 없음 (eval_set 에 비정상 0건)")
            continue
        print(f"abnormal n={m['n']}")
        print(f"exact_match_rate : {m['exact_match_rate']*100:.1f}%")
        print(f"extra_rate(환각) : {m['extra_rate']*100:.1f}%")
        print(f"missing_rate     : {m['missing_rate']*100:.1f}%")
        for d in res["details"]:
            flag = ""
            if d["extra"]:   flag += f" EXTRA={d['extra']}"
            if d["missing"]: flag += f" MISSING={d['missing']}"
            print(f"  id={d['id']} expected={d['expected']} detected={d['detected']}{flag}")


if __name__ == "__main__":
    main()
