# =============================================================================
# vlm/bench/scorer.py
# -----------------------------------------------------------------------------
# 기능:
#   베이스 / LoRA 결과 JSONL 두 개를 받아 다음 지표를 계산합니다:
#     1. JSON 파싱 성공률 (4 필드 모두 존재 + summary 비어있지 않음)
#     2. 등급 일치율 (summary 안에 정답 grade 문자열 포함 여부)
#     3. 수치 인용 정확도 (등지방/뭇갈래근/도체중 숫자가 응답에 포함된 비율)
#     4. ROUGE-L (요약 vs 정답 reference)
#     5. BERTScore (한국어 모델, optional)
#     6. 평균 추론 시간
#
# 동작 방법:
#   python vlm/bench/scorer.py \
#       --base vlm/bench/results_base.jsonl \
#       --lora vlm/bench/results_lora.jsonl \
#       --output vlm/bench/score_report.md
#
# 의존성:
#   rouge-score (필수), bert-score (선택)
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def has_4_fields(pred: dict | None) -> bool:
    if not pred:
        return False
    required = ["3문장_요약", "비정상_근거", "주의사항", "권고"]
    if not all(k in pred for k in required):
        return False
    return bool(pred.get("3문장_요약", "").strip())


def grade_in_summary(pred: dict | None, expected_grade: str) -> bool:
    if not pred:
        return False
    summary = pred.get("3문장_요약", "")
    return expected_grade in summary


def numbers_cited(pred: dict | None, meta: dict) -> float:
    """meta 의 등지방/뭇갈래근/도체중 숫자가 응답 텍스트에 몇 개 포함됐는지(0~1)."""
    if not pred:
        return 0.0
    full_text = " ".join([
        pred.get("3문장_요약", "") or "",
        pred.get("비정상_근거", "") or "",
        pred.get("권고", "") or "",
        " ".join(pred.get("주의사항", []) or []),
    ])
    targets = [
        str(meta.get("backfat_average", "")),
        str(meta.get("multifidus_thk", "")),
        str(meta.get("body_weight", "")),
    ]
    targets = [t for t in targets if t and t != "0.0" and t != "0"]
    if not targets:
        return 1.0
    hits = sum(1 for t in targets if t in full_text)
    return hits / len(targets)


def compute_rouge_l(pred_text: str, ref_text: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        score = scorer.score(ref_text, pred_text)
        return score["rougeL"].fmeasure
    except ImportError:
        return -1.0


def compute_bert_score(pred_texts: list[str], ref_texts: list[str]) -> float:
    try:
        from bert_score import score
        P, R, F = score(pred_texts, ref_texts, lang="ko", verbose=False)
        return float(F.mean())
    except ImportError:
        return -1.0
    except Exception as e:
        print(f"[WARN] bert-score skipped: {e}")
        return -1.0


def evaluate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    json_ok       = sum(1 for r in records if has_4_fields(r.get("prediction")))
    grade_ok      = sum(1 for r in records if grade_in_summary(r.get("prediction"), r["metadata"]["grade"]))
    num_acc       = mean(numbers_cited(r.get("prediction"), r["metadata"]) for r in records)
    elapsed_avg   = mean(r.get("elapsed_sec", 0) for r in records)

    # ROUGE-L (summary 태스크의 reference 와 비교)
    rouge_scores = []
    pred_texts, ref_texts = [], []
    for r in records:
        if not r.get("prediction"):
            continue
        ref = (r.get("tasks", {}).get("summary") or {}).get("reference", "")
        pred = r["prediction"].get("3문장_요약", "")
        if ref and pred:
            rl = compute_rouge_l(pred, ref)
            if rl >= 0:
                rouge_scores.append(rl)
            pred_texts.append(pred)
            ref_texts.append(ref)

    rouge_l_avg = mean(rouge_scores) if rouge_scores else -1.0

    # BERTScore (선택)
    bert_f1 = compute_bert_score(pred_texts, ref_texts) if pred_texts else -1.0

    return {
        "n":                  n,
        "json_parse_rate":    round(json_ok / n, 4),
        "grade_match_rate":   round(grade_ok / n, 4),
        "number_citation":    round(num_acc, 4),
        "rouge_l":            round(rouge_l_avg, 4),
        "bert_score_f1":      round(bert_f1, 4),
        "elapsed_avg_sec":    round(elapsed_avg, 2),
    }


def write_report(base_metrics: dict, lora_metrics: dict, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)

    def cell(v):
        if v == -1.0 or v == -1:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def diff(b, l):
        if b == -1.0 or l == -1.0 or b == 0:
            return "N/A"
        delta = (l - b) / b * 100 if b != 0 else 0
        return f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

    with open(output, "w", encoding="utf-8") as f:
        f.write("# 벤치마크 결과 — Qwen3-VL-8B Base vs LoRA\n\n")
        f.write(f"**평가셋 크기**: {base_metrics['n']} 건\n\n")
        f.write("## 점수 비교\n\n")
        f.write("| 지표 | Base | LoRA | 개선 |\n")
        f.write("|---|---|---|---|\n")
        keys = [
            ("json_parse_rate",   "JSON 파싱 성공률"),
            ("grade_match_rate",  "등급 일치율"),
            ("number_citation",   "수치 인용 정확도"),
            ("rouge_l",           "ROUGE-L (summary)"),
            ("bert_score_f1",     "BERTScore F1 (ko)"),
            ("elapsed_avg_sec",   "평균 추론 시간 (초)"),
        ]
        for key, label in keys:
            f.write(f"| {label} | {cell(base_metrics[key])} | {cell(lora_metrics[key])} | {diff(base_metrics[key], lora_metrics[key])} |\n")
        f.write("\n## 해석\n\n")
        f.write("- **JSON 파싱 성공률**: 4 필드(`3문장_요약`, `비정상_근거`, `주의사항`, `권고`) 모두 존재 + summary 비어있지 않은 비율\n")
        f.write("- **등급 일치율**: 모델이 출력한 summary 안에 정답 grade 문자열(`1+`, `1`, `2`, `등외`)이 포함된 비율\n")
        f.write("- **수치 인용 정확도**: 등지방/뭇갈래근/도체중 숫자가 응답에 정확히 포함된 비율 (0~1)\n")
        f.write("- **ROUGE-L**: 정답 요약과의 단어 시퀀스 일치도 (0~1)\n")
        f.write("- **BERTScore F1 (ko)**: 한국어 BERT 임베딩 기반 의미 유사도 (0~1)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--lora", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(ROOT / "vlm" / "bench" / "score_report.md"))
    args = parser.parse_args()

    base_records = load_jsonl(Path(args.base))
    lora_records = load_jsonl(Path(args.lora))

    print(f"베이스 결과: {len(base_records)}건")
    print(f"LoRA 결과:   {len(lora_records)}건")

    base_metrics = evaluate(base_records)
    lora_metrics = evaluate(lora_records)

    print("\n=== Base ===")
    for k, v in base_metrics.items():
        print(f"  {k:25s} = {v}")
    print("\n=== LoRA ===")
    for k, v in lora_metrics.items():
        print(f"  {k:25s} = {v}")

    write_report(base_metrics, lora_metrics, Path(args.output))
    print(f"\n리포트 저장: {args.output}")
