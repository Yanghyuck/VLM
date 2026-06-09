# =============================================================================
# scripts/exp_sampling_abnormal.py
# -----------------------------------------------------------------------------
# A — abnormal 층화 평가셋(eval_set_abnormal.jsonl, 30 abn)에서 운영 v8 의
#     error_code 충실도를 F(후처리) 유무로 비교. abnormal 2건뿐이던 한계 해소.
# B — 메인 eval_set(50)에서 요약 distinct 를 greedy vs temp 0.3 샘플링으로 비교.
#     v9 음성결과("데이터 다양화는 greedy 다양성으로 전이 안 됨")의 후속 —
#     출력 다양성의 레버가 추론 샘플링인지 + 사실성(등급·수치) 훼손 여부 확인.
#
# 사용: python scripts/exp_sampling_abnormal.py
# =============================================================================

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG
from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report
from vlm.postprocess import extract_error_codes, enforce_error_code_grounding
from vlm.bench.scorer import compute_distinct_n, error_code_faithfulness

ADAPTER = str(ROOT / CFG.paths.lora_adapter)   # 운영(v8)


def rows(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8")]


def to_output(meta):
    img = meta.get("result_image_path")
    if img and not Path(img).exists():
        img = None
    return ThemaPAOutput(**{**meta, "result_image_path": img})


def _is_abnormal(ec):
    return any((v or 0) for v in (ec or {}).values())


def main():
    print(f"[exp] 어댑터: {ADAPTER}")

    # ---------- A: abnormal 충실도 (F 유무) ----------
    abn = [r for r in rows("vlm/bench/eval_set_abnormal.jsonl")
           if _is_abnormal(r["metadata"].get("error_code"))]
    print(f"\n[A] abnormal {len(abn)}건 추론 (greedy, F 유무 비교)")
    pairs_noF, pairs_F = [], []
    for i, r in enumerate(abn, 1):
        ec = r["metadata"]["error_code"]
        exp = {c for c, v in ec.items() if v}
        rep = generate_report(to_output(r["metadata"]), adapter_path=ADAPTER, postprocess=False)
        det_noF = extract_error_codes((rep or {}).get("비정상_근거") or "")
        pairs_noF.append((det_noF, exp))
        repF, _ = enforce_error_code_grounding(rep, ec)
        det_F = extract_error_codes((repF or {}).get("비정상_근거") or "")
        pairs_F.append((det_F, exp))
        print(f"  [{i}/{len(abn)}] id={r['id']}", flush=True)
    print("[A] 충실도 (F 미적용):", error_code_faithfulness(pairs_noF))
    print("[A] 충실도 (F 적용)  :", error_code_faithfulness(pairs_F))

    # ---------- B: 요약 distinct greedy vs temp0.3 ----------
    main_rows = rows("vlm/bench/eval_set.jsonl")
    print(f"\n[B] 메인 {len(main_rows)}건 추론 (greedy + temp0.3)")
    g_sum, s_sum = [], []
    g_grade = s_grade = g_num = s_num = 0
    for i, r in enumerate(main_rows, 1):
        meta = r["metadata"]
        out = to_output(meta)
        g = generate_report(out, adapter_path=ADAPTER)
        s = generate_report(out, adapter_path=ADAPTER, sampling=True, temperature=0.3)
        gs = (g or {}).get("3문장_요약", "")
        ss = (s or {}).get("3문장_요약", "")
        g_sum.append(gs); s_sum.append(ss)
        grade = str(meta["grade"])
        nums = [str(meta["backfat_average"]), str(meta["multifidus_thk"]), str(meta["body_weight"])]
        def has_all(t):
            return all(n in t or n.rstrip("0").rstrip(".") in t for n in nums)
        g_grade += grade in gs; s_grade += grade in ss
        g_num += has_all(gs);   s_num += has_all(ss)
        print(f"  [{i}/{len(main_rows)}] id={r['id']}", flush=True)
    n = len(main_rows)
    print(f"\n[B] 요약 distinct_2 — greedy {compute_distinct_n(g_sum,2):.4f} | temp0.3 {compute_distinct_n(s_sum,2):.4f}  (참고 v8 기존 0.270)")
    print(f"[B] 사실성 greedy : 등급포함 {g_grade}/{n}, 수치보존 {g_num}/{n}")
    print(f"[B] 사실성 temp0.3: 등급포함 {s_grade}/{n}, 수치보존 {s_num}/{n}")


if __name__ == "__main__":
    main()
