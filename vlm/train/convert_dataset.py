# =============================================================================
# vlm/train/convert_dataset.py
# -----------------------------------------------------------------------------
# 기능:
#   scripts/build_dataset.py 가 생성한 dataset.jsonl 을
#   LLaMA-Factory가 요구하는 ShareGPT 대화 형식 JSON으로 변환합니다.
#   각 원본 레코드에서 최대 3개의 학습 샘플을 생성합니다.
#
# 생성되는 학습 태스크:
#   - summary  : 이미지 → 3문장 도체 판정 요약 (모든 레코드)
#   - grade    : 이미지 + 수치 → 등급 판정 근거 단계별 설명 (모든 레코드)
#   - abnormal : 이미지 → 검출 오류 원인 분석 (error_code 비정상 레코드만)
#
# 출력 포맷 (ShareGPT):
#   [
#     {
#       "conversations": [
#         {"from": "human", "value": "<image>\n질문"},
#         {"from": "gpt",   "value": "답변"}
#       ],
#       "images": ["절대경로/이미지.jpg"]
#     }, ...
#   ]
#
# 동작 방법:
#   # 전체 변환
#   python vlm/train/convert_dataset.py
#
#   # 일부만 변환 (테스트용)
#   python vlm/train/convert_dataset.py --limit 100
#
#   # 출력 경로 직접 지정
#   python vlm/train/convert_dataset.py --output vlm/data/livestock_small.json
#
# 설정 (config.json):
#   paths.dataset_jsonl          : 입력 JSONL 경로
#   paths.train_json             : 출력 JSON 경로
#   grade.backfat_range          : 등지방 두께 등급별 정상 범위 (mm)
#   grade.weight_range           : 도체중 등급별 정상 범위 (kg)
#
# 전제 조건:
#   scripts/build_dataset.py 를 먼저 실행하여 dataset.jsonl 생성 필요
#   이미지 파일이 dataset.jsonl 에 기록된 경로에 실제로 존재해야 함
#   프로젝트 루트에 config.json 존재
#
# 의존성:
#   Python 표준 라이브러리만 사용 (외부 패키지 불필요)
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG

INPUT_PATH  = ROOT / CFG.paths.dataset_jsonl
OUTPUT_PATH = ROOT / CFG.paths.train_json

def _raw_config() -> dict:
    """SimpleNamespace 내부 구조에 의존하지 않도록 config.json 을 직접 다시 읽기."""
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


_raw_grade = _raw_config()["grade"]
GRADE_BACKFAT = {k: tuple(v) for k, v in _raw_grade["backfat_range"].items()}
GRADE_WEIGHT  = {k: tuple(v) for k, v in _raw_grade["weight_range"].items()}

GENDER_MAP = {1: "암퇘지", 2: "수퇘지", 3: "거세"}

ERROR_LABEL = {
    "pig_RightEntry":      ("비정상 진입",       "도체가 라인에 바르게 진입하지 않아 전체 측정값 신뢰도가 저하됩니다."),
    "AI_Backbone_error":   ("척추 검출 실패",     "척추 위치를 검출하지 못해 체장·체폭 측정이 불가능합니다."),
    "AI_BackFat_error":    ("등지방 검출 실패",   "등지방 두께를 정확히 측정할 수 없어 등급 판정에 오차가 발생할 수 있습니다."),
    "AI_HalfBone_error":   ("반골 검출 실패",     "이등분 품질 평가가 불가능합니다."),
    "AI_multifidus_error": ("뭇갈래근 검출 실패", "뭇갈래근 두께 측정값을 신뢰할 수 없습니다."),
    "AI_Outline_error":    ("윤곽선 검출 실패",   "도체 전체 형태 측정이 불가능합니다."),
}


def _summary_response(meta: dict) -> str:
    gender = GENDER_MAP.get(meta["gender"], "미상")
    grade  = meta["grade"]
    ec     = meta["error_code"]

    errors = [label for key, (label, _) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    s1 = (f"도체번호 {meta['carcass_no']}은(는) {gender}으로 "
          f"{meta['slaughter_ymd'][:4]}년 {meta['slaughter_ymd'][4:6]}월 {meta['slaughter_ymd'][6:]}일 도축되었습니다.")
    s2 = (f"등지방 두께 {meta['backfat_average']}mm, 뭇갈래근 두께 {meta['multifidus_thk']}mm, "
          f"도체중 {meta['body_weight']}kg으로 측정되었습니다.")
    if errors:
        s3 = f"AI 검출 오류({', '.join(errors)})가 발생하였으며 {grade} 등급으로 판정되었습니다."
    else:
        s3 = f"모든 AI 검출이 정상 완료되어 {grade} 등급으로 판정되었습니다."

    return f"{s1} {s2} {s3}"


def _grade_response(meta: dict) -> str:
    grade   = meta["grade"]
    backfat = meta["backfat_average"]
    weight  = meta["body_weight"]
    mf      = meta["multifidus_thk"]

    lines = [f"## 등급 판정 근거: {grade}\n"]

    bf_range = GRADE_BACKFAT.get(grade)
    if bf_range:
        in_range = bf_range[0] <= backfat <= bf_range[1]
        status   = "범위 내 (정상)" if in_range else "범위 외 (하락 요인)"
        lines.append(f"1. 등지방 두께: {backfat}mm (1+ 기준 17~25mm) -> {status}")
    else:
        lines.append(f"1. 등지방 두께: {backfat}mm -> 기준 범위 외로 등급 하락 요인")

    wt_range = GRADE_WEIGHT.get(grade)
    if wt_range:
        in_range = wt_range[0] <= weight <= wt_range[1]
        status   = "범위 내 (정상)" if in_range else "범위 외 (하락 요인)"
        lines.append(f"2. 도체중: {weight}kg (1+ 기준 83~93kg) -> {status}")
    else:
        lines.append(f"2. 도체중: {weight}kg -> 기준 범위 외로 등급 하락 요인")

    lines.append(f"3. 뭇갈래근 두께: {mf}mm (육질 지표 - 클수록 우수)")
    lines.append(f"\n최종 판정: **{grade} 등급**")
    return "\n".join(lines)


def _abnormal_response(meta: dict) -> str:
    ec     = meta["error_code"]
    failed = [(label, desc) for key, (label, desc) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    lines = ["다음 항목에서 비정상이 감지되었습니다:\n"]
    for i, (label, desc) in enumerate(failed, 1):
        lines.append(f"{i}. **{label}**: {desc}")
    lines.append("\n재촬영 또는 수동 측정을 통해 정확한 등급 판정을 권고합니다.")
    return "\n".join(lines)


def _is_normal(ec: dict) -> bool:
    return all(v == 0 for v in ec.values())


def convert(
    limit: int | None = None,
    output_path: Path = OUTPUT_PATH,
    exclude_ids: set[str] | None = None,
) -> None:
    """dataset.jsonl 을 ShareGPT 학습 JSON 으로 변환.

    Args:
        limit: 처리할 최대 원본 레코드 수
        output_path: 출력 경로
        exclude_ids: 학습 제외할 도체번호 set (벤치마크 held-out 용)
    """
    if not INPUT_PATH.exists():
        print(f"[ERROR] {INPUT_PATH} 없음. 먼저 scripts/build_dataset.py 실행 필요.")
        sys.exit(1)

    exclude_ids = exclude_ids or set()
    records = []
    skipped = 0
    excluded_count = 0

    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            if limit and len(records) >= limit * 3:
                break
            row = json.loads(line)
            meta       = row["metadata"]
            image_path = row["image_path"]
            tasks      = row.get("tasks", {})

            if str(row["id"]) in exclude_ids:
                excluded_count += 1
                continue

            if not os.path.exists(image_path):
                skipped += 1
                continue

            # ── summary task ──────────────────────────────────────────
            if "summary" in tasks:
                records.append({
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{tasks['summary']}"},
                        {"from": "gpt",   "value": _summary_response(meta)},
                    ],
                    "images": [image_path],
                })

            # ── grade task ────────────────────────────────────────────
            if "grade" in tasks:
                records.append({
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{tasks['grade']}"},
                        {"from": "gpt",   "value": _grade_response(meta)},
                    ],
                    "images": [image_path],
                })

            # ── abnormal task (오류 케이스만) ─────────────────────────
            if "abnormal" in tasks and not _is_normal(meta["error_code"]):
                records.append({
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{tasks['abnormal']}"},
                        {"from": "gpt",   "value": _abnormal_response(meta)},
                    ],
                    "images": [image_path],
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"변환 완료: {len(records)}건 → {output_path}")
    if skipped:
        print(f"이미지 없음 스킵: {skipped}건")
    if excluded_count:
        print(f"평가셋 제외: {excluded_count}건 (held-out)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, help="변환할 최대 원본 레코드 수")
    parser.add_argument("--output", type=str, help="출력 경로 (기본: vlm/data/livestock_train.json)")
    parser.add_argument("--exclude-eval-set", type=str,
                        help="평가셋 JSONL 경로 — 해당 도체번호들을 학습에서 제외")
    args = parser.parse_args()

    exclude_ids: set[str] = set()
    if args.exclude_eval_set:
        with open(args.exclude_eval_set, encoding="utf-8") as f:
            for line in f:
                exclude_ids.add(str(json.loads(line)["id"]))
        print(f"제외할 평가셋 ID 로드: {len(exclude_ids)}건")

    out = Path(args.output) if args.output else OUTPUT_PATH
    convert(limit=args.limit, output_path=out, exclude_ids=exclude_ids)
