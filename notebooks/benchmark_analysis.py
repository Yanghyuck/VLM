# =============================================================================
# notebooks/benchmark_analysis.py
# -----------------------------------------------------------------------------
# 3-way 벤치마크 결과를 시각화 (ROUGE-L 분포, 산점도).
# 결과: docs/figures/benchmark_*.png
# =============================================================================

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = ROOT / "docs" / "figures"
BASE = ROOT / "vlm" / "bench" / "results_base.jsonl"
V1   = ROOT / "vlm" / "bench" / "results_lora_v1.jsonl"
V2   = ROOT / "vlm" / "bench" / "results_lora_v2.jsonl"

COLORS = {"Base": "#7f8c8d", "v1 (text-only)": "#3498db", "v2 (Vision+AI)": "#e74c3c"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_rouge(pred, ref):
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return s.score(ref, pred)["rougeL"].fmeasure


def collect_rouges(records):
    out = []
    for r in records:
        if not r.get("prediction"):
            continue
        ref = (r.get("tasks", {}).get("summary") or {}).get("reference", "")
        pred = r["prediction"].get("3문장_요약", "")
        if ref and pred:
            out.append(compute_rouge(pred, ref))
    return out


def main():
    base_rouges = collect_rouges(load(BASE))
    v1_rouges   = collect_rouges(load(V1))
    v2_rouges   = collect_rouges(load(V2))

    # 1) ROUGE-L 분포 (boxplot + scatter)
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [base_rouges, v1_rouges, v2_rouges]
    labels = ["Base", "v1 (text-only)", "v2 (Vision+AI)"]
    bp = ax.boxplot(data, labels=labels, widths=0.6, patch_artist=True,
                     showmeans=True, meanprops={"marker": "D", "markerfacecolor": "yellow"})
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS[lbl])
        patch.set_alpha(0.6)
    # 개별 점
    for i, (d, lbl) in enumerate(zip(data, labels)):
        x = [i + 1 + (j - len(d)/2) * 0.005 for j in range(len(d))]
        ax.scatter(x, d, alpha=0.4, s=10, color=COLORS[lbl])
    ax.set_title("ROUGE-L 분포 — 베이스 vs v1 vs v2 (held-out 50건)", fontsize=14)
    ax.set_ylabel("ROUGE-L F1")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="0.5 임계선")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_benchmark_rouge_distribution.png", dpi=120)
    plt.close(fig)
    print("[plot] 08_benchmark_rouge_distribution.png")

    # 2) 누적 분포 (CDF)
    fig, ax = plt.subplots(figsize=(10, 6))
    for d, lbl in zip(data, labels):
        sorted_d = sorted(d)
        n = len(sorted_d)
        cumul = [(i + 1) / n for i in range(n)]
        ax.plot(sorted_d, cumul, label=f"{lbl} (avg={sum(d)/n:.3f})",
                color=COLORS[lbl], linewidth=2)
    ax.set_title("ROUGE-L CDF — 모델별 점수 분포", fontsize=14)
    ax.set_xlabel("ROUGE-L F1")
    ax.set_ylabel("누적 비율 (≤ x)")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_benchmark_rouge_cdf.png", dpi=120)
    plt.close(fig)
    print("[plot] 09_benchmark_rouge_cdf.png")

    # 3) 페어와이즈 비교 (v2 vs Base 산점도)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(base_rouges, v2_rouges, alpha=0.6, s=40, color="#e74c3c")
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.3, label="동일선 (y=x)")
    # v2 가 base 보다 좋은 영역 표시
    ax.fill_between([0.4, 1.0], [0.4, 1.0], 1.05, alpha=0.1, color="green",
                    label="v2 > Base 영역")
    ax.set_title("샘플별 v2 vs Base ROUGE-L (50건)", fontsize=14)
    ax.set_xlabel("Base ROUGE-L")
    ax.set_ylabel("v2 LoRA ROUGE-L")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_benchmark_v2_vs_base_scatter.png", dpi=120)
    plt.close(fig)
    print("[plot] 10_benchmark_v2_vs_base_scatter.png")

    # 통계
    print(f"\n[stats] base: avg={sum(base_rouges)/len(base_rouges):.4f}, n={len(base_rouges)}")
    print(f"[stats] v1:   avg={sum(v1_rouges)/len(v1_rouges):.4f}, n={len(v1_rouges)}")
    print(f"[stats] v2:   avg={sum(v2_rouges)/len(v2_rouges):.4f}, n={len(v2_rouges)}")
    win_v2 = sum(1 for b, v in zip(base_rouges, v2_rouges) if v > b)
    print(f"[stats] v2 > Base: {win_v2}/{len(base_rouges)} ({win_v2/len(base_rouges)*100:.0f}%)")


if __name__ == "__main__":
    main()
