# =============================================================================
# scripts/augment_error_cases.py
# -----------------------------------------------------------------------------
# 학습 데이터 100% 정상 (검출 실패 0건, 등외 0건) 의 한계를 보완하기 위해,
# 정상 레코드에서 합성 검출 실패 케이스 + 등외 케이스를 생성.
#
# B2 분석(`vlm/bench/error_distribution.md`) 의 결정적 발견 대응:
#   - 학습 데이터 3,355건 모두 error_code=0
#   - 등외 등급 0건
#   - 모델이 검출 실패 입력에서 환각 발생하는 근본 원인
#
# 합성 정책:
#   1) 정상 레코드를 base 로 사용 (실제 측정 분포 유지)
#   2) 임의 error_code 1~3개 1로 설정 (가중치: 1개 70%, 2개 25%, 3개 5%)
#   3) 해당 측정값 0 으로 변경 (검출 실패의 직접 결과)
#   4) 등급을 "등외" (60%) 또는 "2" (40%) 로 강등
#   5) tasks 에 abnormal 추가
#
# 한계:
#   - 합성 측정값 분포가 실제 검출 실패 패턴과 다를 수 있음
#   - 도메인 신뢰성 보장 어려움 (실제 데이터 보강이 이상적)
#   - 그러나 학습 데이터 0건 vs 합성 N건 trade-off 에서 N건이 환각 감소에 도움
#
# 사용:
#   python scripts/augment_error_cases.py
#   python scripts/augment_error_cases.py --n 500 --seed 42
#   python scripts/augment_error_cases.py --output vlm/data/dataset_v4.jsonl
#
# 출력:
#   기본: vlm/data/dataset_v4.jsonl (원본 + 합성)
# =============================================================================

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG
from vlm.train.convert_dataset import ERROR_LABEL

DATASET   = ROOT / CFG.paths.dataset_jsonl
OUT_DEFAULT = ROOT / "vlm" / "data" / "dataset_v4.jsonl"

ERROR_KEYS = list(ERROR_LABEL.keys())

# 검출 실패 항목별 측정값 무효화 매핑
# (실제 thema_pa 운영 패턴과 정합)
INVALIDATION_MAP = {
    "AI_BackFat_error":    "backfat_average",
    "AI_multifidus_error": "multifidus_thk",
    "AI_Backbone_error":   "body_length",
    "AI_Outline_error":    "body_width",
    # pig_RightEntry: 진입 자체 비정상 → 다중 측정값 무효화 별도 처리
    # AI_HalfBone_error: 반골 → 직접 측정값과 1:1 대응 어려움 (등급에만 영향)
}

TASK_PROMPTS = {
    "summary":  "이 돼지 도체 이미지의 판정 결과를 현장 작업자에게 3문장으로 요약해주세요.",
    "grade":    "측정 수치를 근거로 이 도체의 등급 판정 이유를 단계별로 설명해주세요.",
    "abnormal": "이 도체에서 비정상으로 감지된 항목과 그 원인을 설명해주세요.",
}


def synthesize_one(base: dict, rng: random.Random) -> dict:
    """정상 레코드 → 합성 검출 실패 레코드. 깊은 복사 + 변형."""
    rec = copy.deepcopy(base)
    meta = rec["metadata"]

    # 1) 동시 실패 수 결정 (1개 70%, 2개 25%, 3개 5%)
    n_active = rng.choices([1, 2, 3], weights=[70, 25, 5])[0]
    active = rng.sample(ERROR_KEYS, n_active)

    # 2) error_code 부여
    for k in ERROR_KEYS:
        meta["error_code"][k] = 1 if k in active else 0

    # 3) 측정값 무효화 (해당 항목만)
    for k in active:
        field = INVALIDATION_MAP.get(k)
        if field:
            meta[field] = 0.0

    # pig_RightEntry: 비정상 진입은 전체 측정 신뢰도 영향 → length/width 도 0
    if "pig_RightEntry" in active:
        meta["body_length"] = 0.0
        meta["body_width"]  = 0.0

    # 4) 등급 강등 (등외 60% / 2 등급 40%)
    meta["grade"] = "등외" if rng.random() < 0.6 else "2"

    # 5) tasks 에 abnormal 추가 (summary/grade 는 이미 있음)
    rec["tasks"] = {**rec.get("tasks", {}),
                    "summary":  TASK_PROMPTS["summary"],
                    "grade":    TASK_PROMPTS["grade"],
                    "abnormal": TASK_PROMPTS["abnormal"]}

    # id 충돌 방지 — 합성 표시
    rec["id"] = f"{rec['id']}_aug{rng.randint(1000, 9999)}"
    rec["augmented"] = True
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DATASET),
                        help="입력 정상 데이터셋 jsonl")
    parser.add_argument("--output", type=str, default=str(OUT_DEFAULT),
                        help="출력 augmented jsonl (원본 + 합성)")
    parser.add_argument("--n", type=int, default=500,
                        help="합성 검출 실패 케이스 수 (기본 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"[ERROR] 입력 없음: {in_path}")
        return 1

    rng = random.Random(args.seed)

    # 원본 로드
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"원본 로드: {len(records)}건")

    # 정상 케이스만 base 로 사용
    normal_pool = [r for r in records
                   if all(v == 0 for v in r["metadata"]["error_code"].values())]
    print(f"정상 케이스 풀: {len(normal_pool)}건")

    # 합성
    synthesized = []
    bases = rng.sample(normal_pool, k=min(args.n, len(normal_pool)))
    for base in bases:
        synthesized.append(synthesize_one(base, rng))
    print(f"합성 생성: {len(synthesized)}건")

    # 통계
    n_active_dist  = {1: 0, 2: 0, 3: 0}
    grade_dist     = {"등외": 0, "2": 0}
    error_use_dist = {k: 0 for k in ERROR_KEYS}
    for r in synthesized:
        ec = r["metadata"]["error_code"]
        n_active = sum(1 for v in ec.values() if v == 1)
        n_active_dist[n_active] = n_active_dist.get(n_active, 0) + 1
        grade_dist[r["metadata"]["grade"]] += 1
        for k, v in ec.items():
            if v == 1:
                error_use_dist[k] += 1

    print(f"\n동시 실패 수 분포: {n_active_dist}")
    print(f"등급 분포:         {grade_dist}")
    print(f"error 항목별 사용:")
    for k, c in error_use_dist.items():
        print(f"  {k:25s} {c}")

    # 저장 (원본 + 합성 결합)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in synthesized:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n저장: {out_path} ({len(records) + len(synthesized)}건 = 원본 {len(records)} + 합성 {len(synthesized)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
