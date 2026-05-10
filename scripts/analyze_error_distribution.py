# =============================================================================
# scripts/analyze_error_distribution.py
# -----------------------------------------------------------------------------
# 학습/평가 데이터의 검출 실패(error_code) 분포 분석.
# B2 작업의 입력 자료 — v4 학습 시 어떤 조합 케이스가 부족한지 식별.
#
# 사용:
#   python scripts/analyze_error_distribution.py
#
# 출력:
#   vlm/bench/error_distribution.md
# =============================================================================

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATASET_JSONL = ROOT / "vlm" / "data" / "dataset.jsonl"
EVAL_SET      = ROOT / "vlm" / "bench" / "eval_set.jsonl"
OUTPUT        = ROOT / "vlm" / "bench" / "error_distribution.md"

ERROR_LABELS = [
    "pig_RightEntry",
    "AI_Backbone_error",
    "AI_BackFat_error",
    "AI_HalfBone_error",
    "AI_multifidus_error",
    "AI_Outline_error",
]
ERROR_KO = {
    "pig_RightEntry":      "비정상 진입",
    "AI_Backbone_error":   "척추",
    "AI_BackFat_error":    "등지방",
    "AI_HalfBone_error":   "반골",
    "AI_multifidus_error": "뭇갈래근",
    "AI_Outline_error":    "윤곽선",
}


def analyze(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_active_dist: Counter[int] = Counter()
    combo_dist: Counter[frozenset] = Counter()
    grade_dist: Counter[str] = Counter()
    grade_x_n_active: Counter[tuple] = Counter()
    total = len(rows)
    for r in rows:
        meta = r["metadata"]
        ec = meta["error_code"]
        active = [k for k in ERROR_LABELS if ec.get(k, 0) == 1]
        n_active_dist[len(active)] += 1
        if active:
            combo_dist[frozenset(active)] += 1
        grade = meta.get("grade", "?")
        grade_dist[grade] += 1
        grade_x_n_active[(grade, len(active))] += 1
    return total, n_active_dist, combo_dist, grade_dist, grade_x_n_active


def fmt_combo(combo: frozenset) -> str:
    return ", ".join(sorted(ERROR_KO.get(k, k) for k in combo))


def main():
    out: list[str] = []
    out.append("# 검출 실패(error_code) 분포 분석\n")
    out.append(
        "B2 작업 입력 자료. 학습 데이터의 검출 실패 케이스 분포를 보고 "
        "v4 학습 시 어떤 조합이 부족한지 식별."
    )

    for label, path in [("학습 데이터 (dataset.jsonl)", DATASET_JSONL),
                        ("평가셋 (eval_set.jsonl)",     EVAL_SET)]:
        if not path.exists():
            out.append(f"\n## {label}\n\n(파일 없음: {path})\n")
            continue
        total, n_active, combos, grades, grade_x_n = analyze(path)

        out.append(f"\n## {label} — 총 {total}건\n")

        out.append("### 동시 실패 항목 수 분포\n")
        out.append("| 동시 실패 수 | 건수 | 비율 |")
        out.append("|---|---|---|")
        for n in sorted(n_active):
            pct = n_active[n] / total * 100
            out.append(f"| {n} | {n_active[n]} | {pct:.1f}% |")

        out.append("\n### 등급 분포\n")
        out.append("| 등급 | 건수 | 비율 |")
        out.append("|---|---|---|")
        for g in ["1+", "1", "2", "등외"]:
            c = grades.get(g, 0)
            pct = c / total * 100 if total else 0
            out.append(f"| {g} | {c} | {pct:.1f}% |")

        out.append("\n### 등급 × 동시 실패 수 교차표\n")
        max_n = max((n for _, n in grade_x_n), default=0) if grade_x_n else 0
        out.append("| 등급 | " + " | ".join(f"실패 {n}" for n in range(max_n + 1)) + " |")
        out.append("|" + "---|" * (max_n + 2))
        for g in ["1+", "1", "2", "등외"]:
            row = [g]
            for n in range(max_n + 1):
                row.append(str(grade_x_n.get((g, n), 0)))
            out.append("| " + " | ".join(row) + " |")

        if combos:
            out.append("\n### 검출 실패 조합 — Top 15\n")
            out.append("| 조합 | 건수 | 비율 (실패 케이스 중) |")
            out.append("|---|---|---|")
            n_failing = sum(combos.values())
            for combo, cnt in combos.most_common(15):
                pct = cnt / n_failing * 100
                out.append(f"| {fmt_combo(combo)} | {cnt} | {pct:.1f}% |")

    out.append(
        "\n---\n\n"
        "## 해석 가이드\n\n"
        "- **동시 실패 수** 분포가 한쪽으로 치우치면 (예: 0개 또는 1개만 다수) "
        "v4 학습 시 다중 실패 augmentation 권장\n"
        "- **등급 × 동시 실패 수** 의 빈 셀은 학습되지 않은 패턴 — 환각 위험\n"
        "- **Top 조합** 외의 드문 조합은 학습 데이터가 매우 적어 일반화 한계\n"
    )

    OUTPUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
