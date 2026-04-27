# =============================================================================
# scripts/analyze_failures.py
# -----------------------------------------------------------------------------
# 3-way 벤치마크 결과를 분석하여 LoRA v2 가 가장 약한 5건을 추출하고
# 정성적 비교 리포트를 마크다운으로 생성합니다.
#
# 동작 방법:
#   python scripts/analyze_failures.py
#
# 출력:
#   vlm/bench/failure_analysis.md
# =============================================================================

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "vlm" / "bench" / "failure_analysis.md"
BASE = ROOT / "vlm" / "bench" / "results_base.jsonl"
V1   = ROOT / "vlm" / "bench" / "results_lora_v1.jsonl"
V2   = ROOT / "vlm" / "bench" / "results_lora_v2.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_rouge(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return scorer.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        return -1.0


def grade_match(pred: dict | None, expected_grade: str) -> bool:
    if not pred:
        return False
    return expected_grade in (pred.get("3문장_요약", "") or "")


def number_cited(pred: dict | None, meta: dict) -> int:
    if not pred:
        return 0
    text = " ".join([
        pred.get("3문장_요약", "") or "",
        pred.get("비정상_근거", "") or "",
        pred.get("권고", "") or "",
        " ".join(pred.get("주의사항", []) or []),
    ])
    targets = [str(meta.get(k, "")) for k in ("backfat_average", "multifidus_thk", "body_weight")]
    return sum(1 for t in targets if t and t in text)


def index_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records if r.get("prediction")}


def main():
    base = index_by_id(load_jsonl(BASE))
    v1   = index_by_id(load_jsonl(V1))
    v2   = index_by_id(load_jsonl(V2))

    common_ids = set(base) & set(v1) & set(v2)
    print(f"[load] base={len(base)} v1={len(v1)} v2={len(v2)} common={len(common_ids)}")

    rows = []
    for cid in common_ids:
        v2_rec = v2[cid]
        ref = (v2_rec.get("tasks", {}).get("summary") or {}).get("reference", "")
        pred_v2 = v2_rec["prediction"].get("3문장_요약", "") if v2_rec.get("prediction") else ""
        rouge_v2 = compute_rouge(pred_v2, ref)

        v1_rec = v1[cid]
        pred_v1 = v1_rec["prediction"].get("3문장_요약", "") if v1_rec.get("prediction") else ""
        rouge_v1 = compute_rouge(pred_v1, ref)

        b_rec = base[cid]
        pred_b = b_rec["prediction"].get("3문장_요약", "") if b_rec.get("prediction") else ""
        rouge_b = compute_rouge(pred_b, ref)

        meta = v2_rec["metadata"]
        rows.append({
            "id":         cid,
            "grade":      meta["grade"],
            "backfat":    meta.get("backfat_average"),
            "weight":     meta.get("body_weight"),
            "rouge_base": rouge_b,
            "rouge_v1":   rouge_v1,
            "rouge_v2":   rouge_v2,
            "ref":        ref,
            "pred_base":  pred_b,
            "pred_v1":    pred_v1,
            "pred_v2":    pred_v2,
        })

    rows.sort(key=lambda r: r["rouge_v2"])
    bottom = rows[:5]
    top    = rows[-3:][::-1]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# 3-way 벤치마크 — 실패 케이스 정성 분석\n\n")
        f.write(f"총 {len(rows)}건 분석. v2 LoRA의 ROUGE-L 낮은 순으로 5건 추출.\n\n")
        f.write("---\n\n")

        f.write("## v2 ROUGE-L 최하위 5건\n\n")
        for i, r in enumerate(bottom, 1):
            f.write(f"### {i}. 도체번호 {r['id']} (등급 {r['grade']}, 등지방 {r['backfat']}mm, 도체중 {r['weight']}kg)\n\n")
            f.write(f"| 모델 | ROUGE-L |\n|---|---|\n")
            f.write(f"| Base | {r['rouge_base']:.4f} |\n")
            f.write(f"| v1 LoRA | {r['rouge_v1']:.4f} |\n")
            f.write(f"| v2 LoRA | **{r['rouge_v2']:.4f}** |\n\n")
            f.write("**정답 reference**:\n```\n")
            f.write(r["ref"].strip())
            f.write("\n```\n\n")
            f.write("**v2 predicted**:\n```\n")
            f.write(r["pred_v2"].strip())
            f.write("\n```\n\n")
            f.write("---\n\n")

        f.write("## (참고) v2 ROUGE-L 최상위 3건\n\n")
        for i, r in enumerate(top, 1):
            f.write(f"### {i}. 도체번호 {r['id']} (등급 {r['grade']}) — ROUGE-L {r['rouge_v2']:.4f}\n\n")
            f.write(f"- Base: {r['rouge_base']:.4f}, v1: {r['rouge_v1']:.4f}, **v2: {r['rouge_v2']:.4f}**\n\n")
            f.write("---\n\n")

        # 통계 요약
        f.write("## 분포 통계\n\n")
        for label, key in [("Base", "rouge_base"), ("v1 LoRA", "rouge_v1"), ("v2 LoRA", "rouge_v2")]:
            vals = sorted(r[key] for r in rows)
            n = len(vals)
            mn, mx = vals[0], vals[-1]
            med = vals[n // 2]
            avg = sum(vals) / n
            below_05 = sum(1 for v in vals if v < 0.5)
            f.write(f"- **{label}**: avg={avg:.4f}, median={med:.4f}, min={mn:.4f}, max={mx:.4f}, <0.5: {below_05}건\n")

    print(f"[done] {OUT}")
    print(f"[summary] v2 ROUGE-L: avg={sum(r['rouge_v2'] for r in rows)/len(rows):.4f}, "
          f"min={min(r['rouge_v2'] for r in rows):.4f}, "
          f"max={max(r['rouge_v2'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
