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

GENDER_MAP = {1: "암컷", 2: "수컷", 3: "거세"}


def _eul_ro(word: str) -> str:
    """단어 끝 글자의 종성 유무에 따라 '으로'/'로' 선택.

    받침 있음 → '으로', 받침 없음 → '로'.
    한글 음절 외(숫자/영문)에는 보수적으로 '으로' 반환 (학습 데이터에서
    이런 케이스는 등장하지 않으나 안전 경로 유지).
    """
    if not word:
        return "으로"
    last = word[-1]
    if "가" <= last <= "힣":
        # (음절 코드 - 'ㅏ' 시작) % 28 == 0 이면 종성 없음
        return "로" if (ord(last) - 0xAC00) % 28 == 0 else "으로"
    return "으로"

ERROR_LABEL = {
    "pig_RightEntry":      ("비정상 진입",       "도체가 라인에 바르게 진입하지 않아 전체 측정값 신뢰도가 저하됩니다."),
    "AI_Backbone_error":   ("척추 검출 실패",     "척추 위치를 검출하지 못해 체장·체폭 측정이 불가능합니다."),
    "AI_BackFat_error":    ("등지방 검출 실패",   "등지방 두께를 정확히 측정할 수 없어 등급 판정에 오차가 발생할 수 있습니다."),
    "AI_HalfBone_error":   ("반골 검출 실패",     "이등분 품질 평가가 불가능합니다."),
    "AI_multifidus_error": ("뭇갈래근 검출 실패", "뭇갈래근 두께 측정값을 신뢰할 수 없습니다."),
    "AI_Outline_error":    ("윤곽선 검출 실패",   "도체 전체 형태 측정이 불가능합니다."),
}


# visual_desc — 이미지 기반 시각 서술 태스크 (도체 전체 형태 + 등지방층 외형).
# reference 는 scripts/distill_visual_desc.py 가 생성한 별도 파일에서 주입된다.
VISUAL_DESC_PROMPT = "이 도체 이미지를 보고 도체 전체 형태와 등지방층 외형을 시각적으로 서술해주세요."


def _visual_desc_response(vd: dict) -> str:
    """증류된 2필드 dict → 학습 타깃 텍스트 (라벨 2섹션)."""
    form    = (vd.get("도체_전체_형태") or "").strip()
    backfat = (vd.get("등지방층_외형") or "").strip()
    return f"도체 전체 형태: {form}\n등지방층 외형: {backfat}"


def _summary_response(meta: dict) -> str:
    gender = GENDER_MAP.get(meta["gender"], "미상")
    grade  = meta["grade"]
    ec     = meta["error_code"]

    errors = [label for key, (label, _) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    s1 = (f"도체번호 {meta['carcass_no']}은(는) {gender}{_eul_ro(gender)} "
          f"{meta['slaughter_ymd'][:4]}년 {meta['slaughter_ymd'][4:6]}월 {meta['slaughter_ymd'][6:]}일 도축되었습니다.")
    s2 = (f"등지방 두께 {meta['backfat_average']}mm, 뭇갈래근 두께 {meta['multifidus_thk']}mm, "
          f"도체중 {meta['body_weight']}kg으로 측정되었습니다.")
    if errors:
        s3 = f"AI 검출 오류({', '.join(errors)})가 발생하였으며 {grade} 등급으로 판정되었습니다."
    else:
        s3 = f"모든 AI 검출이 정상 완료되어 {grade} 등급으로 판정되었습니다."

    return f"{s1} {s2} {s3}"


def _summary_response_alt(meta: dict) -> str:
    """A3 — reference paraphrase 변형 (등급/측정값 우선 순서)."""
    gender = GENDER_MAP.get(meta["gender"], "미상")
    grade  = meta["grade"]
    ec     = meta["error_code"]

    errors = [label for key, (label, _) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    s1 = f"도체번호 {meta['carcass_no']} 의 최종 판정 등급은 {grade} 입니다."
    s2 = (f"{gender}{_eul_ro(gender)} "
          f"{meta['slaughter_ymd'][:4]}년 {meta['slaughter_ymd'][4:6]}월 {meta['slaughter_ymd'][6:]}일 도축되었으며, "
          f"도체중 {meta['body_weight']}kg, 등지방 {meta['backfat_average']}mm, "
          f"뭇갈래근 {meta['multifidus_thk']}mm 가 측정되었습니다.")
    if errors:
        s3 = f"AI 검출 오류({', '.join(errors)})로 인해 신뢰도 저하 가능성이 있습니다."
    else:
        s3 = "모든 AI 검출이 정상 완료되었습니다."

    return f"{s1} {s2} {s3}"


def _summary_response_bullet(meta: dict) -> str:
    """B1 — 헤더 + 불릿 강조형. 구조 자체가 다른 응답."""
    gender = GENDER_MAP.get(meta["gender"], "미상")
    grade  = meta["grade"]
    ec     = meta["error_code"]
    errors = [label for key, (label, _) in ERROR_LABEL.items() if ec.get(key, 0) == 1]
    ymd    = meta["slaughter_ymd"]
    date_str = f"{ymd[:4]}년 {ymd[4:6]}월 {ymd[6:]}일"

    header = f"### 판정 요약 — {grade} 등급"
    bullets = [
        f"- 도체번호: {meta['carcass_no']}",
        f"- 성별/도축일: {gender} / {date_str}",
        f"- 측정값: 등지방 {meta['backfat_average']}mm · 뭇갈래근 {meta['multifidus_thk']}mm · 도체중 {meta['body_weight']}kg",
    ]
    if errors:
        bullets.append(f"- 검출 오류: {', '.join(errors)}")
    else:
        bullets.append("- 검출 상태: 모든 AI 항목 정상")
    return header + "\n" + "\n".join(bullets)


def _summary_response_table(meta: dict) -> str:
    """B1 — 측정값 마크다운 표 + 한 줄 결론. 가장 구조적으로 다른 형식."""
    gender = GENDER_MAP.get(meta["gender"], "미상")
    grade  = meta["grade"]
    ec     = meta["error_code"]
    errors = [label for key, (label, _) in ERROR_LABEL.items() if ec.get(key, 0) == 1]
    ymd    = meta["slaughter_ymd"]
    date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"

    rows = [
        "| 항목 | 값 |",
        "|---|---|",
        f"| 도체번호 | {meta['carcass_no']} |",
        f"| 성별 | {gender} |",
        f"| 도축일 | {date_str} |",
        f"| 등지방 두께 | {meta['backfat_average']} mm |",
        f"| 뭇갈래근 두께 | {meta['multifidus_thk']} mm |",
        f"| 도체중 | {meta['body_weight']} kg |",
        f"| 판정 등급 | **{grade}** |",
    ]
    status = ", ".join(errors) if errors else "정상"
    conclusion = f"\n검출 상태: {status}."
    return "\n".join(rows) + conclusion


def _summary_response_all(meta: dict) -> list[str]:
    """v6 — summary 4 paraphrase 모음 (round-robin 학습용).

    순서:
      0: _summary_response          (3문장 정형)
      1: _summary_response_alt      (등급/측정값 우선)
      2: _summary_response_bullet   (헤더 + 불릿)
      3: _summary_response_table    (마크다운 표)
    """
    return [
        _summary_response(meta),
        _summary_response_alt(meta),
        _summary_response_bullet(meta),
        _summary_response_table(meta),
    ]


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
    """원본 형태 — 번호 매긴 항목 리스트 + 권고 마무리."""
    ec     = meta["error_code"]
    failed = [(label, desc) for key, (label, desc) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    lines = ["다음 항목에서 비정상이 감지되었습니다:\n"]
    for i, (label, desc) in enumerate(failed, 1):
        lines.append(f"{i}. **{label}**: {desc}")
    lines.append("\n재촬영 또는 수동 측정을 통해 정확한 등급 판정을 권고합니다.")
    return "\n".join(lines)


def _abnormal_response_narrative(meta: dict) -> str:
    """B1 변형 — 자연어 서술형 (번호 없이 문장 결합)."""
    ec     = meta["error_code"]
    failed = [(label, desc) for key, (label, desc) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    if not failed:
        return "검출 결과 비정상 항목은 발견되지 않았습니다."

    if len(failed) == 1:
        label, desc = failed[0]
        return f"본 도체에서는 {label}가 발생했습니다. {desc} 정확한 판정을 위해 재촬영 또는 수동 측정이 필요합니다."

    labels_str = ", ".join(label for label, _ in failed)
    descs = " ".join(desc for _, desc in failed)
    return (f"본 도체에서는 {labels_str} 등 {len(failed)}건의 비정상이 동시에 발생했습니다. "
            f"{descs} 측정값 신뢰도가 크게 저하되어 재촬영 또는 수동 측정이 필요합니다.")


def _abnormal_response_recommend_first(meta: dict) -> str:
    """B1 변형 — 권고 우선형 (재촬영 권고가 먼저, 사유 뒤)."""
    ec     = meta["error_code"]
    failed = [(label, desc) for key, (label, desc) in ERROR_LABEL.items() if ec.get(key, 0) == 1]

    if not failed:
        return "별도 조치 없이 정상 출하 가능합니다."

    labels_str = ", ".join(label for label, _ in failed)
    lines = [f"재촬영 또는 수동 측정을 권고드립니다. 사유는 다음과 같습니다:\n"]
    for label, desc in failed:
        lines.append(f"- {label}: {desc}")
    lines.append(f"\n총 {len(failed)}건의 검출 오류({labels_str})로 인해 등급 판정의 정확성을 보장할 수 없습니다.")
    return "\n".join(lines)


def _abnormal_response_all(meta: dict) -> list[str]:
    """B1 — 학습 데이터/평가 reference 용 abnormal 응답 paraphrase 모음.

    첫 번째가 default(이전 호환 reference). 나머지는 paraphrase.
    """
    return [
        _abnormal_response(meta),
        _abnormal_response_narrative(meta),
        _abnormal_response_recommend_first(meta),
    ]


def _is_normal(ec: dict) -> bool:
    return all(v == 0 for v in ec.values())


def convert(
    limit: int | None = None,
    output_path: Path = OUTPUT_PATH,
    exclude_ids: set[str] | None = None,
    input_path: Path | None = None,
    paraphrase_mode: str = "single",
    visual_desc_refs: dict[str, dict] | None = None,
    summary_refs: dict[str, list[str]] | None = None,
) -> None:
    """dataset.jsonl 을 ShareGPT 학습 JSON 으로 변환.

    Args:
        limit: 처리할 최대 원본 레코드 수
        output_path: 출력 경로
        exclude_ids: 학습 제외할 도체번호 set (벤치마크 held-out 용)
        input_path: 입력 jsonl (기본 None → INPUT_PATH). v4 augmented 데이터처럼
                    별도 입력 사용 시 명시.
        paraphrase_mode:
          - "single"      : v4/v5 동작 — summary/abnormal 각 1개 paraphrase 만
          - "round_robin" : v6 — 도체 ID 결정적 round-robin
                            summary: 4 paraphrase 중 (id % 4)
                            abnormal: 3 paraphrase 중 (id % 3)
                            학습 샘플 수는 single 과 동일, 노출 paraphrase 만 다양화.
    """
    src = input_path if input_path else INPUT_PATH
    if not src.exists():
        print(f"[ERROR] {src} 없음. 먼저 scripts/build_dataset.py 실행 필요.")
        sys.exit(1)

    assert paraphrase_mode in ("single", "round_robin"), \
        f"paraphrase_mode invalid: {paraphrase_mode}"

    exclude_ids = exclude_ids or set()
    records = []
    skipped = 0
    excluded_count = 0
    visual_desc_count = 0
    summary_distilled = 0      # C — 증류 다양화 요약 사용 건수
    para_counts = {"summary": [0, 0, 0, 0], "abnormal": [0, 0, 0]}

    with open(src, encoding="utf-8") as f:
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

            try:
                id_int = int(str(row["id"]))
            except (ValueError, TypeError):
                id_int = abs(hash(str(row["id"])))

            # ── summary task ──────────────────────────────────────────
            if "summary" in tasks:
                sref = summary_refs.get(str(row["id"])) if summary_refs else None
                if sref:
                    # C — 증류된 다양한 요약(도체당 K변형) 중 id 결정적 선택.
                    # 템플릿 암기 천장(distinct ~0.27) 돌파용. 없으면 기존 경로로 fallback.
                    summary_value = sref[id_int % len(sref)]
                    summary_distilled += 1
                elif paraphrase_mode == "round_robin":
                    paras = _summary_response_all(meta)
                    idx = id_int % len(paras)
                    summary_value = paras[idx]
                    para_counts["summary"][idx] += 1
                else:
                    summary_value = _summary_response(meta)
                records.append({
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{tasks['summary']}"},
                        {"from": "gpt",   "value": summary_value},
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
                if paraphrase_mode == "round_robin":
                    paras = _abnormal_response_all(meta)
                    idx = id_int % len(paras)
                    abnormal_value = paras[idx]
                    para_counts["abnormal"][idx] += 1
                else:
                    abnormal_value = _abnormal_response(meta)
                records.append({
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{tasks['abnormal']}"},
                        {"from": "gpt",   "value": abnormal_value},
                    ],
                    "images": [image_path],
                })

            # ── visual_desc task (증류 reference 가 있는 레코드만) ──────
            if visual_desc_refs:
                vd = visual_desc_refs.get(str(row["id"]))
                if vd:
                    records.append({
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{VISUAL_DESC_PROMPT}"},
                            {"from": "gpt",   "value": _visual_desc_response(vd)},
                        ],
                        "images": [image_path],
                    })
                    visual_desc_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"변환 완료: {len(records)}건 → {output_path}")
    if skipped:
        print(f"이미지 없음 스킵: {skipped}건")
    if excluded_count:
        print(f"평가셋 제외: {excluded_count}건 (held-out)")
    if visual_desc_count:
        print(f"visual_desc 샘플: {visual_desc_count}건 (증류 reference 매칭)")
    if summary_refs is not None:
        print(f"summary 증류 다양화: {summary_distilled}건 사용 (ref 없는 도체는 기존 경로 fallback)")
    if paraphrase_mode == "round_robin":
        print(f"summary  paraphrase 분포 (id%4): {para_counts['summary']}")
        print(f"abnormal paraphrase 분포 (id%3): {para_counts['abnormal']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, help="입력 jsonl (기본: config.json paths.dataset_jsonl)")
    parser.add_argument("--limit",  type=int, help="변환할 최대 원본 레코드 수")
    parser.add_argument("--output", type=str, help="출력 경로 (기본: vlm/data/livestock_train.json)")
    parser.add_argument("--exclude-eval-set", type=str,
                        help="평가셋 JSONL 경로 — 해당 도체번호들을 학습에서 제외")
    parser.add_argument("--paraphrase-mode", type=str, default="single",
                        choices=["single", "round_robin"],
                        help="single (v4/v5 호환) | round_robin (v6 — id 결정적 paraphrase 다양화)")
    parser.add_argument("--visual-desc-refs", type=str,
                        help="시각서술 증류 jsonl (distill_visual_desc.py 산출) — visual_desc 태스크 추가")
    parser.add_argument("--summary-refs", type=str,
                        help="요약 다양화 증류 jsonl (distill_summary.py 산출) — summary 타깃을 증류 변형으로 대체")
    args = parser.parse_args()

    exclude_ids: set[str] = set()
    if args.exclude_eval_set:
        with open(args.exclude_eval_set, encoding="utf-8") as f:
            for line in f:
                exclude_ids.add(str(json.loads(line)["id"]))
        print(f"제외할 평가셋 ID 로드: {len(exclude_ids)}건")

    vd_refs: dict[str, dict] | None = None
    if args.visual_desc_refs:
        vd_refs = {}
        with open(args.visual_desc_refs, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("visual_desc"):   # 파싱 성공한 것만
                    vd_refs[str(rec["id"])] = rec["visual_desc"]
        print(f"visual_desc reference 로드: {len(vd_refs)}건")

    sum_refs: dict[str, list[str]] | None = None
    if args.summary_refs:
        sum_refs = {}
        with open(args.summary_refs, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                variants = rec.get("summaries") or []
                if variants:                 # 검증 통과 변형이 있는 것만
                    sum_refs[str(rec["id"])] = variants
        print(f"summary 증류 reference 로드: {len(sum_refs)}건")

    inp = Path(args.input) if args.input else None
    out = Path(args.output) if args.output else OUTPUT_PATH
    convert(limit=args.limit, output_path=out, exclude_ids=exclude_ids,
            input_path=inp, paraphrase_mode=args.paraphrase_mode,
            visual_desc_refs=vd_refs, summary_refs=sum_refs)
