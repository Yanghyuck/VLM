# =============================================================================
# notebooks/dataset_analysis.py
# -----------------------------------------------------------------------------
# 기능:
#   vlm/data/dataset.jsonl 의 3,355개 도체 레코드를 분석하여
#   docs/figures/ 에 7장의 시각화 결과를 생성합니다.
#
# 분석 항목:
#   1. 등급 분포 (1+, 1, 2, 등외)
#   2. 등지방 두께 히스토그램 (등급별)
#   3. 뭇갈래근 두께 히스토그램 (등급별)
#   4. 도체중 히스토그램 (등급별)
#   5. 성별 분포 (등급별)
#   6. 등지방 vs 도체중 산점도
#   7. 도축일 분포
#
# 동작 방법:
#   conda activate vlm
#   python notebooks/dataset_analysis.py
#
# 출력:
#   docs/figures/01_grade_distribution.png
#   docs/figures/02_backfat_by_grade.png
#   docs/figures/03_multifidus_by_grade.png
#   docs/figures/04_weight_by_grade.png
#   docs/figures/05_gender_by_grade.png
#   docs/figures/06_backfat_vs_weight.png
#   docs/figures/07_slaughter_date.png
# =============================================================================

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG

# 한글 폰트 설정 (Windows)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")

DATASET_JSONL = ROOT / CFG.paths.dataset_jsonl
FIG_DIR       = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ORDER = ["1+", "1", "2", "등외"]
GRADE_COLOR = {"1+": "#2ecc71", "1": "#3498db", "2": "#f39c12", "등외": "#e74c3c"}
GENDER_LABEL = {1: "암퇘지", 2: "수퇘지", 3: "거세"}


def load_records() -> list[dict]:
    if not DATASET_JSONL.exists():
        raise FileNotFoundError(f"{DATASET_JSONL} 없음. scripts/build_dataset.py 먼저 실행.")
    with open(DATASET_JSONL, encoding="utf-8") as f:
        return [json.loads(line)["metadata"] for line in f]


def fig_grade_distribution(records: list[dict]):
    counts = Counter(r["grade"] for r in records)
    grades = [g for g in GRADE_ORDER if counts.get(g, 0) > 0]
    values = [counts[g] for g in grades]
    colors = [GRADE_COLOR[g] for g in grades]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grades, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v}\n({v/sum(values)*100:.1f}%)",
                ha="center", va="bottom", fontsize=11)
    ax.set_title("도체 등급 분포 (총 {}건)".format(sum(values)), fontsize=14)
    ax.set_xlabel("등급")
    ax.set_ylabel("건수")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_grade_distribution.png", dpi=120)
    plt.close(fig)


def _hist_by_grade(records, key, title, xlabel, fname, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    grades = sorted(set(r["grade"] for r in records), key=lambda g: GRADE_ORDER.index(g) if g in GRADE_ORDER else 999)
    for g in grades:
        values = [r[key] for r in records if r["grade"] == g and r[key] > 0]
        if values:
            ax.hist(values, bins=30, alpha=0.55, label=f"{g} (n={len(values)})",
                    color=GRADE_COLOR.get(g, "gray"))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("빈도")
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(title="등급", loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120)
    plt.close(fig)


def fig_gender_by_grade(records: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 5))
    grades = [g for g in GRADE_ORDER if any(r["grade"] == g for r in records)]
    genders = [1, 2, 3]
    bottom = [0] * len(grades)
    for gen in genders:
        counts = [sum(1 for r in records if r["grade"] == g and r["gender"] == gen) for g in grades]
        ax.bar(grades, counts, bottom=bottom, label=GENDER_LABEL[gen])
        bottom = [b + c for b, c in zip(bottom, counts)]
    ax.set_title("등급별 성별 분포", fontsize=14)
    ax.set_xlabel("등급")
    ax.set_ylabel("건수")
    ax.legend(title="성별")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_gender_by_grade.png", dpi=120)
    plt.close(fig)


def fig_backfat_vs_weight(records: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for g in GRADE_ORDER:
        xs = [r["body_weight"]      for r in records if r["grade"] == g]
        ys = [r["backfat_average"]  for r in records if r["grade"] == g]
        if xs:
            ax.scatter(xs, ys, alpha=0.4, label=f"{g} (n={len(xs)})",
                       color=GRADE_COLOR.get(g, "gray"), s=15)
    # 1+ 등급 기준 영역 표시
    ax.axhspan(17, 25, alpha=0.1, color="green", label="1+ 등지방 기준 17~25mm")
    ax.axvspan(83, 93, alpha=0.1, color="green")
    ax.set_title("등지방 두께 vs 도체중 산점도", fontsize=14)
    ax.set_xlabel("도체중 (kg)")
    ax.set_ylabel("등지방 두께 (mm)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_backfat_vs_weight.png", dpi=120)
    plt.close(fig)


def fig_slaughter_date(records: list[dict]):
    counts = Counter(r["slaughter_ymd"] for r in records)
    dates = sorted(counts.keys())
    values = [counts[d] for d in dates]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(dates)), values, color="#3498db", edgecolor="black")
    ax.set_title(f"도축일별 레코드 수 (총 {sum(values)}건, {len(dates)}일)", fontsize=14)
    ax.set_xlabel("도축일")
    ax.set_ylabel("건수")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_slaughter_date.png", dpi=120)
    plt.close(fig)


def print_stats(records: list[dict]):
    print(f"\n{'='*60}")
    print(f"데이터셋 통계 (총 {len(records)}건)")
    print(f"{'='*60}")
    grade_counts = Counter(r["grade"] for r in records)
    for g in GRADE_ORDER:
        if grade_counts.get(g, 0):
            print(f"  {g:>4} 등급: {grade_counts[g]:>5}건 ({grade_counts[g]/len(records)*100:.1f}%)")

    gender_counts = Counter(r["gender"] for r in records)
    print(f"\n성별 분포:")
    for k, v in GENDER_LABEL.items():
        if gender_counts.get(k, 0):
            print(f"  {v}: {gender_counts[k]}건")

    bfs = [r["backfat_average"] for r in records if r["backfat_average"] > 0]
    mfs = [r["multifidus_thk"]  for r in records if r["multifidus_thk"]  > 0]
    wts = [r["body_weight"]     for r in records if r["body_weight"]     > 0]
    print(f"\n측정값 통계:")
    if bfs: print(f"  등지방 두께   : 평균 {sum(bfs)/len(bfs):.1f}mm, 범위 {min(bfs):.0f}~{max(bfs):.0f}mm")
    if mfs: print(f"  뭇갈래근 두께 : 평균 {sum(mfs)/len(mfs):.1f}mm, 범위 {min(mfs):.0f}~{max(mfs):.0f}mm")
    if wts: print(f"  도체중        : 평균 {sum(wts)/len(wts):.1f}kg, 범위 {min(wts):.0f}~{max(wts):.0f}kg")

    dates = Counter(r["slaughter_ymd"] for r in records)
    print(f"\n도축 기간: {min(dates.keys())} ~ {max(dates.keys())} ({len(dates)}일)")


def main():
    records = load_records()
    print(f"[load] {DATASET_JSONL.name}: {len(records)}건")

    print(f"[plot] 1/7 grade distribution")
    fig_grade_distribution(records)

    print(f"[plot] 2/7 backfat by grade")
    _hist_by_grade(records, "backfat_average",
                   "등급별 등지방 두께 분포", "등지방 두께 (mm)",
                   "02_backfat_by_grade.png", xlim=(0, 50))

    print(f"[plot] 3/7 multifidus by grade")
    _hist_by_grade(records, "multifidus_thk",
                   "등급별 뭇갈래근 두께 분포", "뭇갈래근 두께 (mm)",
                   "03_multifidus_by_grade.png", xlim=(0, 80))

    print(f"[plot] 4/7 weight by grade")
    _hist_by_grade(records, "body_weight",
                   "등급별 도체중 분포", "도체중 (kg)",
                   "04_weight_by_grade.png", xlim=(50, 130))

    print(f"[plot] 5/7 gender by grade")
    fig_gender_by_grade(records)

    print(f"[plot] 6/7 backfat vs weight scatter")
    fig_backfat_vs_weight(records)

    print(f"[plot] 7/7 slaughter date")
    fig_slaughter_date(records)

    print_stats(records)
    print(f"\n[done] saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
