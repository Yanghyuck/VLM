# =============================================================================
# scripts/eval_visual_desc.py
# -----------------------------------------------------------------------------
# 기능:
#   시각 서술(visual_desc) 태스크 전용 평가 지표.
#   기존 harness(ROUGE/BERT/distinct)는 텍스트 유사도만 보므로, 시각서술에
#   필요한 (1) 항목 커버리지 (2) 수치 낭독 여부 (3) DB 오류 정합성 을 추가로 측정.
#
# 두 가지 모드:
#   1) refs-stats — 증류 reference 자체 품질 점검 (예측 없이, 지금 실행 가능)
#        python scripts/eval_visual_desc.py refs-stats \
#            --refs vlm/data/visual_desc_refs.jsonl
#
#   2) score — 모델 예측 vs reference 채점 (학습/추론 후)
#        python scripts/eval_visual_desc.py score \
#            --pred  vlm/bench/results_visual_desc_lora.jsonl \
#            --refs  vlm/data/visual_desc_refs.jsonl
#
# 입력 포맷:
#   refs  : distill_visual_desc.py 산출 — {id, error_code, abnormal,
#           visual_desc:{도체_전체_형태, 등지방층_외형}}
#   pred  : {id, visual_desc:{...}}  또는  {id, response:"도체 전체 형태: ...\n등지방층 외형: ..."}
#
# 지표:
#   field_coverage     : 두 필드 모두 비어있지 않은 비율
#   number_leak_rate   : 수치(mm/kg 등)를 낭독한 비율 (낮을수록 좋음)
#   distinct_2__{field}: 필드별 표현 다양성 (암기/천편일률 진단)
#   abnormal_consistency: 비정상 케이스가 이상 신호를 담은 비율 (정상으로 단정 X)
#   rouge_l / bert (score 모드): 예측 vs reference 텍스트 유사도
#
# 의존성: rouge-score(선택), bert-score(선택) — 없으면 해당 지표만 skip.
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

try:                                  # Windows 콘솔(cp949)에서 한글/em-dash 출력 보장
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vlm.bench.scorer import (
    compute_rouge_l, compute_distinct_n, compute_bert_score,
)

# 수치 낭독 탐지 — "22mm", "88 kg", "17 밀리" 등
NUMBER_RE = re.compile(r"\d+\s*(?:mm|cm|kg|밀리|미리|킬로)", re.IGNORECASE)
# 이상 신호 어휘 (비정상 케이스가 담아야 함)
ISSUE_WORDS = [
    "이상", "불균형", "완전하지", "완전히 정렬되지", "비대칭", "왜곡", "편차",
    "불규칙", "신뢰", "검출", "않아", "않으며", "않은", "휘어", "어긋", "불완전", "흐릿",
]


def _fields(obj) -> tuple[str, str]:
    """dict(visual_desc) 또는 라벨 텍스트(response) → (전체형태, 등지방층)."""
    if isinstance(obj, dict):
        return (obj.get("도체_전체_형태") or "").strip(), (obj.get("등지방층_외형") or "").strip()
    text = str(obj or "")
    form = backfat = ""
    m1 = re.search(r"도체\s*전체\s*형태\s*[:：]?\s*(.+?)(?=등지방층\s*외형|$)", text, re.S)
    m2 = re.search(r"등지방층\s*외형\s*[:：]?\s*(.+)$", text, re.S)
    if m1: form = m1.group(1).strip()
    if m2: backfat = m2.group(1).strip()
    if not form and not backfat:        # 라벨 없으면 전체를 두 필드 합본으로 취급
        form = text.strip()
    return form, backfat


def _pred_fields(rec: dict) -> tuple[str, str]:
    if rec.get("visual_desc"):
        return _fields(rec["visual_desc"])
    return _fields(rec.get("response") or rec.get("prediction") or "")


def _has_issue(text: str) -> bool:
    return any(w in text for w in ISSUE_WORDS)


def refs_stats(refs_path: Path) -> None:
    recs = [json.loads(l) for l in open(refs_path, encoding="utf-8")]
    valid = [r for r in recs if r.get("visual_desc")]
    n = len(recs)
    print(f"=== refs-stats: {refs_path.name} ===")
    print(f"총 {n}건 | visual_desc 파싱 성공 {len(valid)} ({len(valid)/n*100:.1f}%)")

    forms, backfats, both_ok, num_leak = [], [], 0, 0
    abn_total = abn_consistent = 0
    for r in valid:
        form, backfat = _fields(r["visual_desc"])
        forms.append(form); backfats.append(backfat)
        if len(form) >= 10 and len(backfat) >= 10:
            both_ok += 1
        if NUMBER_RE.search(form) or NUMBER_RE.search(backfat):
            num_leak += 1
        if r.get("abnormal"):
            abn_total += 1
            if _has_issue(form + " " + backfat):
                abn_consistent += 1

    m = len(valid) or 1
    print(f"field_coverage      : {both_ok/m*100:.1f}%  (두 필드 모두 ≥10자)")
    print(f"number_leak_rate    : {num_leak/m*100:.1f}%  (수치 낭독 — 낮을수록 좋음)")
    print(f"distinct_2 전체형태 : {compute_distinct_n(forms, 2):.4f}")
    print(f"distinct_2 등지방층 : {compute_distinct_n(backfats, 2):.4f}")
    if abn_total:
        print(f"abnormal_consistency: {abn_consistent}/{abn_total} "
              f"({abn_consistent/abn_total*100:.1f}%)  (이상 신호 포함)")
    else:
        print("abnormal_consistency: (비정상 케이스 없음)")


def score(pred_path: Path, refs_path: Path) -> None:
    refs = {str(json.loads(l)["id"]): json.loads(l)
            for l in open(refs_path, encoding="utf-8")}
    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8")]

    rouge_form, rouge_bf = [], []
    pred_forms, pred_bfs = [], []
    both_ok = num_leak = 0
    abn_total = abn_consistent = 0
    matched = 0
    bert_pred, bert_ref = [], []

    for p in preds:
        rid = str(p.get("id"))
        ref = refs.get(rid)
        if not ref or not ref.get("visual_desc"):
            continue
        matched += 1
        pf, pb = _pred_fields(p)
        rf, rb = _fields(ref["visual_desc"])
        pred_forms.append(pf); pred_bfs.append(pb)
        if rf and pf: rouge_form.append(compute_rouge_l(pf, rf))
        if rb and pb: rouge_bf.append(compute_rouge_l(pb, rb))
        bert_pred += [pf, pb]; bert_ref += [rf, rb]
        if len(pf) >= 10 and len(pb) >= 10: both_ok += 1
        if NUMBER_RE.search(pf) or NUMBER_RE.search(pb): num_leak += 1
        if ref.get("abnormal"):
            abn_total += 1
            if _has_issue(pf + " " + pb): abn_consistent += 1

    if not matched:
        print("[WARN] 매칭된 예측 없음 (id 불일치?)"); return
    m = matched
    print(f"=== score: {pred_path.name} vs {refs_path.name} ===")
    print(f"매칭 {matched}건")
    print(f"rouge_l 전체형태 : {mean(rouge_form):.4f}" if rouge_form else "rouge_l 전체형태 : n/a")
    print(f"rouge_l 등지방층 : {mean(rouge_bf):.4f}"   if rouge_bf   else "rouge_l 등지방층 : n/a")
    print(f"distinct_2 전체형태: {compute_distinct_n(pred_forms, 2):.4f}")
    print(f"distinct_2 등지방층: {compute_distinct_n(pred_bfs, 2):.4f}")
    print(f"field_coverage     : {both_ok/m*100:.1f}%")
    print(f"number_leak_rate   : {num_leak/m*100:.1f}%")
    if abn_total:
        print(f"abnormal_consistency: {abn_consistent}/{abn_total} ({abn_consistent/abn_total*100:.1f}%)")
    bert = compute_bert_score(bert_pred, bert_ref) if bert_pred else -1.0
    if bert >= 0:
        print(f"bert_score_f1      : {bert:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s1 = sub.add_parser("refs-stats")
    s1.add_argument("--refs", required=True)
    s2 = sub.add_parser("score")
    s2.add_argument("--pred", required=True)
    s2.add_argument("--refs", required=True)
    args = ap.parse_args()

    if args.mode == "refs-stats":
        refs_stats(Path(args.refs))
    else:
        score(Path(args.pred), Path(args.refs))


if __name__ == "__main__":
    main()
