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


class _KoTokenizer:
    """한글 보존 토크나이저.

    rouge_score 기본 토크나이저는 `[^a-z0-9]+` 로 비ASCII 문자를 모두 제거해,
    숫자/영문이 없는 **순수 한글** 텍스트의 ROUGE 가 0이 된다(동일 문장끼리도 0).
    어절(공백) 단위 분할 + 소문자화만 수행해 한글을 보존한다.
    """

    def tokenize(self, text: str) -> list[str]:
        return (text or "").lower().split()


_KO_TOKENIZER = _KoTokenizer()


def compute_rouge_l(pred_text: str, ref_text: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False, tokenizer=_KO_TOKENIZER,
        )
        score = scorer.score(ref_text, pred_text)
        return score["rougeL"].fmeasure
    except ImportError:
        return -1.0


def compute_rouge_l_max(pred_text: str, ref_texts: list[str]) -> float:
    """A3 — paraphrase 여러 개 중 max ROUGE-L."""
    if not ref_texts:
        return -1.0
    scores = [compute_rouge_l(pred_text, r) for r in ref_texts]
    scores = [s for s in scores if s >= 0]
    return max(scores) if scores else -1.0


def compute_distinct_n(texts: list[str], n: int) -> float:
    """A4 — distinct-N: 응답 모음의 unique n-gram / total n-gram (0~1)."""
    all_ngrams: list[str] = []
    for t in texts:
        toks = (t or "").split()
        if len(toks) < n:
            continue
        all_ngrams.extend(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


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


def _is_normal_case(rec: dict) -> bool:
    ec = (rec.get("metadata") or {}).get("error_code") or {}
    return all(v == 0 for v in ec.values()) if ec else True


def _collect_field_texts(records: list[dict]) -> dict[str, list[str]]:
    """prediction 4필드 각각의 텍스트 모음 (None/list 안전 처리)."""
    out = {"3문장_요약": [], "비정상_근거": [], "주의사항": [], "권고": []}
    for r in records:
        p = r.get("prediction") or {}
        for k in out:
            v = p.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                out[k].append(" ".join(str(x) for x in v))
            else:
                out[k].append(str(v))
    return out


def evaluate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    json_ok       = sum(1 for r in records if has_4_fields(r.get("prediction")))
    grade_ok      = sum(1 for r in records if grade_in_summary(r.get("prediction"), r["metadata"]["grade"]))
    num_acc       = mean(numbers_cited(r.get("prediction"), r["metadata"]) for r in records)
    elapsed_avg   = mean(r.get("elapsed_sec", 0) for r in records)

    # ROUGE-L: 단일 reference + A3 paraphrase max
    rouge_scores: list[float] = []
    rouge_max_scores: list[float] = []
    pred_texts, ref_texts = [], []
    for r in records:
        if not r.get("prediction"):
            continue
        summary_task = r.get("tasks", {}).get("summary") or {}
        ref_single = summary_task.get("reference", "")
        ref_list   = summary_task.get("references") or ([ref_single] if ref_single else [])
        pred = r["prediction"].get("3문장_요약", "")
        if not (pred and (ref_single or ref_list)):
            continue
        if ref_single:
            rl = compute_rouge_l(pred, ref_single)
            if rl >= 0:
                rouge_scores.append(rl)
            pred_texts.append(pred)
            ref_texts.append(ref_single)
        if ref_list:
            rl_max = compute_rouge_l_max(pred, ref_list)
            if rl_max >= 0:
                rouge_max_scores.append(rl_max)

    rouge_l_avg     = mean(rouge_scores)     if rouge_scores     else -1.0
    rouge_l_max_avg = mean(rouge_max_scores) if rouge_max_scores else -1.0

    # A4 — 응답 다양성 (summary 필드, 전체)
    distinct_1 = compute_distinct_n(pred_texts, 1)
    distinct_2 = compute_distinct_n(pred_texts, 2)

    # BERTScore (선택)
    bert_f1 = compute_bert_score(pred_texts, ref_texts) if pred_texts else -1.0

    # ── 필드별 distinct-2 (v3/v4 암기 진단용) ───────────────────────────────
    field_texts = _collect_field_texts(records)
    field_distinct: dict[str, float] = {}
    for fname, texts in field_texts.items():
        field_distinct[f"distinct_2__{fname}"] = round(compute_distinct_n(texts, 2), 4) if texts else -1.0

    # ── 케이스별(normal/abnormal) 분리 메트릭 ───────────────────────────────
    split: dict[str, dict] = {}
    for case in ("normal", "abnormal"):
        wanted = [r for r in records if (_is_normal_case(r) == (case == "normal"))]
        if not wanted:
            split[case] = {"n": 0}
            continue
        s_pred, s_ref, s_rouge = [], [], []
        for r in wanted:
            if not r.get("prediction"):
                continue
            ref = (r.get("tasks", {}).get("summary") or {}).get("reference", "")
            pred = r["prediction"].get("3문장_요약", "")
            if pred and ref:
                rl = compute_rouge_l(pred, ref)
                if rl >= 0:
                    s_rouge.append(rl)
                s_pred.append(pred)
                s_ref.append(ref)
        split[case] = {
            "n":          len(wanted),
            "rouge_l":    round(mean(s_rouge), 4)               if s_rouge else -1.0,
            "distinct_2": round(compute_distinct_n(s_pred, 2), 4) if s_pred  else -1.0,
        }

    return {
        "n":                  n,
        "json_parse_rate":    round(json_ok / n, 4),
        "grade_match_rate":   round(grade_ok / n, 4),
        "number_citation":    round(num_acc, 4),
        "rouge_l":            round(rouge_l_avg, 4),
        "rouge_l_max":        round(rouge_l_max_avg, 4),
        "bert_score_f1":      round(bert_f1, 4),
        "distinct_1":         round(distinct_1, 4),
        "distinct_2":         round(distinct_2, 4),
        "elapsed_avg_sec":    round(elapsed_avg, 2),
        **field_distinct,
        "split":              split,
    }


def write_report(metrics_dict: dict[str, dict], output: Path, baseline_key: str = "base"):
    """N-way 비교 리포트 생성.

    Args:
        metrics_dict: {"base": metrics, "lora_v1": metrics, "lora_v2": metrics, ...}
        output: 출력 마크다운 경로
        baseline_key: 개선% 계산 기준 (기본 "base")
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(metrics_dict.keys())
    baseline = metrics_dict.get(baseline_key, metrics_dict[labels[0]])

    def cell(v):
        if v == -1.0 or v == -1:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def diff(b, x):
        if b == -1.0 or x == -1.0 or b == 0:
            return "—"
        delta = (x - b) / b * 100 if b != 0 else 0
        return f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

    keys = [
        ("json_parse_rate",   "JSON 파싱 성공률"),
        ("grade_match_rate",  "등급 일치율"),
        ("number_citation",   "수치 인용 정확도"),
        ("rouge_l",           "ROUGE-L (single ref)"),
        ("rouge_l_max",       "ROUGE-L max (A3 paraphrase)"),
        ("bert_score_f1",     "BERTScore F1 (ko)"),
        ("distinct_1",        "Distinct-1 (다양성, A4)"),
        ("distinct_2",        "Distinct-2 (다양성, A4)"),
        ("elapsed_avg_sec",   "평균 추론 시간 (초)"),
    ]
    field_keys = [
        ("distinct_2__3문장_요약", "Distinct-2 · 3문장_요약"),
        ("distinct_2__비정상_근거", "Distinct-2 · 비정상_근거"),
        ("distinct_2__주의사항",   "Distinct-2 · 주의사항"),
        ("distinct_2__권고",       "Distinct-2 · 권고"),
    ]

    def write_table(f, keys_):
        f.write("| 지표 |")
        for lbl in labels:
            f.write(f" {lbl} |")
            if lbl != baseline_key:
                f.write(f" Δ |")
        f.write("\n")
        f.write("|---|" + "|".join(["---"] * (len(labels) + sum(1 for l in labels if l != baseline_key))) + "|\n")
        for key, label in keys_:
            f.write(f"| {label} |")
            for lbl in labels:
                m = metrics_dict[lbl]
                v = m.get(key, -1.0)
                f.write(f" {cell(v)} |")
                if lbl != baseline_key:
                    f.write(f" {diff(baseline.get(key, -1.0), v)} |")
            f.write("\n")

    with open(output, "w", encoding="utf-8") as f:
        f.write(f"# 벤치마크 결과 — Qwen3-VL-8B {' vs '.join(labels)}\n\n")
        f.write(f"**평가셋 크기**: {baseline['n']} 건  ·  **baseline**: `{baseline_key}`\n\n")

        f.write("## 점수 비교 (전체)\n\n")
        write_table(f, keys)

        f.write("\n## 필드별 다양성 (Distinct-2)\n\n")
        f.write("v3/v4 의 암기 패턴이 어느 필드에서 발생하는지 진단.\n\n")
        write_table(f, field_keys)

        # 케이스별 split — normal vs abnormal
        f.write("\n## 케이스별 분리 (normal vs abnormal)\n\n")
        f.write("학습 분포에서 비정상(error_code 비0) 케이스의 ROUGE/distinct 가 normal 케이스와 동일하면, abnormal task 도 reference 를 암기한 것.\n\n")
        for case in ("normal", "abnormal"):
            f.write(f"### {case}\n\n")
            f.write("| 지표 |")
            for lbl in labels:
                f.write(f" {lbl} |")
            f.write("\n")
            f.write("|---|" + "|".join(["---"] * len(labels)) + "|\n")
            for sub_key in ("n", "rouge_l", "distinct_2"):
                f.write(f"| {sub_key} |")
                for lbl in labels:
                    s = (metrics_dict[lbl].get("split") or {}).get(case) or {}
                    v = s.get(sub_key, -1.0 if sub_key != "n" else 0)
                    f.write(f" {cell(v)} |")
                f.write("\n")
            f.write("\n")

        f.write("## 해석\n\n")
        f.write("- **JSON 파싱 성공률**: 4 필드(`3문장_요약`, `비정상_근거`, `주의사항`, `권고`) 모두 존재 + summary 비어있지 않은 비율\n")
        f.write("- **등급 일치율**: 모델이 출력한 summary 안에 정답 grade 문자열(`1+`, `1`, `2`, `등외`)이 포함된 비율\n")
        f.write("- **수치 인용 정확도**: 등지방/뭇갈래근/도체중 숫자가 응답에 정확히 포함된 비율 (0~1)\n")
        f.write("- **ROUGE-L (single ref)**: 단일 정답과의 단어 시퀀스 일치도 (0~1)\n")
        f.write("- **ROUGE-L max (A3)**: paraphrase 정답들 중 max — 표현 다양성 보상\n")
        f.write("- **BERTScore F1 (ko)**: 한국어 BERT 임베딩 기반 의미 유사도 (0~1)\n")
        f.write("- **Distinct-1/2 (A4)**: 응답 모음의 unique unigram/bigram 비율. 높으면 다양성 ↑\n")
        f.write("- **필드별 Distinct-2**: 4 필드 각각의 다양성. 특정 필드만 떨어지면 그 필드에 암기 집중.\n")
        f.write("- **케이스별 분리**: normal/abnormal 의 ROUGE/distinct. abnormal 도 1.0 이면 합성 reference 까지 암기.\n")
        f.write(f"- **개선 % 계산 기준**: `{baseline_key}` 대비\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="비교할 결과 파일들. 형식: 'label=path' (예: 'base=results_base.jsonl' 'lora_v1=results_lora_v1.jsonl')")
    parser.add_argument("--baseline", type=str, default="base",
                        help="개선% 계산 기준 라벨 (기본 'base')")
    parser.add_argument("--output", type=str, default=str(ROOT / "vlm" / "bench" / "score_report.md"))
    args = parser.parse_args()

    metrics_dict: dict[str, dict] = {}
    for spec in args.inputs:
        if "=" not in spec:
            raise ValueError(f"입력 형식 오류 (label=path 필요): {spec}")
        label, path_str = spec.split("=", 1)
        records = load_jsonl(Path(path_str))
        print(f"[{label}] {path_str}: {len(records)}건")
        metrics_dict[label.strip()] = evaluate(records)

    print()
    for label, metrics in metrics_dict.items():
        print(f"=== {label} ===")
        for k, v in metrics.items():
            print(f"  {k:25s} = {v}")
        print()

    write_report(metrics_dict, Path(args.output), baseline_key=args.baseline)
    print(f"리포트 저장: {args.output}")
    sys.exit(0)


# ── 하위 호환 — old CLI 형식 (--base, --lora) ────────────────────────────
def _legacy_main():
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
