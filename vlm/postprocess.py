# =============================================================================
# vlm/postprocess.py
# -----------------------------------------------------------------------------
# LoRA 추론 결과 dict 에 적용하는 한국어 후처리.
#
# A3: 한국어 조사 정규화 — "거세으로" → "거세로", "1+으로" → "1+로", "등외으로" → "등외로"
# A4: 등급 정합성 보강 — 응답 내 등급 단언이 입력 grade 와 다르면 입력 grade 로 강제
#
# 설계 원칙:
#   - 순수 파이썬 (re 만 사용). torch/transformers/pydantic 비의존.
#   - 학습 데이터 패턴이 좁아 도메인 어휘 한정 보수적 매칭. 비교/참조 문맥은 보존.
#   - 변경 흔적은 응답 dict 의 '_postprocess' 메타필드에 기록 (디버깅·로그용).
#
# 사용:
#   from vlm.postprocess import apply_postprocess
#   raw   = generate_report(output)        # 모델 원본 응답 (이미 적용된 상태)
#   fixed = apply_postprocess(raw, expected_grade=output.grade)
# =============================================================================

from __future__ import annotations

import re
from typing import Any, Callable

VALID_GRADES: tuple[str, ...] = ("1+", "1", "2", "등외")
VALID_GENDERS: tuple[str, ...] = ("암컷", "수컷", "거세")


# A3 — 한국어 조사 정규화
# 받침 없는 글자/숫자/+ 뒤에 어색하게 붙은 '으로' 교정.
# 일반 한국어 받침 검사를 무차별 적용하면 정상 텍스트도 깨질 수 있어
# 학습 데이터에서 실제 관찰된 도메인 패턴만 좁혀서 처리.
_JOSA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"거세으로"), "거세로"),
    (re.compile(r"(\d)\+으로"), r"\1+로"),                 # 1+으로, 2+으로 → 1+로
    (re.compile(r"(?<![가-힣\d])([12])으로"), r"\1로"),    # 단독 등급 숫자 (앞에 한글/숫자 없음)
    (re.compile(r"등외으로"), "등외로"),
]


def normalize_josa(text: str) -> tuple[str, bool]:
    """A3 한국어 조사 정규화. (변경된 텍스트, 변경 여부) 반환."""
    changed = False
    for pat, repl in _JOSA_PATTERNS:
        new_text = pat.sub(repl, text)
        if new_text != text:
            changed = True
            text = new_text
    return text, changed


# A4 — 등급 정합성
# 명확한 단언 표현만 교체:
#   (1) "<X> 등급으로 판정" — '판정' 동사가 뒤따를 때
#   (2) "최종 <X> 등급" — '최종' 이 앞에 올 때
#   (3) "이의 신청 가능: <X> 등급" — 등급 안내성 단언
# 비교 문맥 ("1+ 등급 도체 대비") 은 위 패턴에 안 걸려 보존됨.
def _grade_pattern(expected: str) -> re.Pattern[str]:
    others = [g for g in VALID_GRADES if g != expected]
    if not others:
        return re.compile(r"(?!)")  # 매칭 불가능 패턴
    alt = "|".join(re.escape(g) for g in others)
    return re.compile(
        rf"(?P<g1>{alt})(?P<gap1>\s*등급)(?=\s*으로\s*판정)"
        rf"|최종(?P<sp>\s+)(?P<g2>{alt})(?P<gap2>\s*등급)"
        rf"|(?P<lead>이의\s*신청\s*가능\s*[:：]\s*)(?P<g3>{alt})(?P<gap3>\s*등급)"
    )


def enforce_grade(text: str, expected_grade: str) -> tuple[str, bool]:
    """A4 등급 정합성. 명확한 단언 패턴만 입력 grade 로 교체."""
    if expected_grade not in VALID_GRADES:
        return text, False

    pat = _grade_pattern(expected_grade)

    def repl(m: re.Match[str]) -> str:
        if m.group("g1"):
            return f"{expected_grade}{m.group('gap1')}"
        if m.group("g2"):
            return f"최종{m.group('sp')}{expected_grade}{m.group('gap2')}"
        return f"{m.group('lead')}{expected_grade}{m.group('gap3')}"

    new_text, n = pat.subn(repl, text)
    return new_text, n > 0


# A5 — 성별 정합성
# 명확한 단언 패턴만 교체:
#   (1) "<other>으로/로 판정" — 입력과 다른 성별을 판정 동사로 단언
#   (2) "<other>으로/로 판정되어야" — 정정 표현 (모델이 잘못된 정정을 출력)
# 비교/참조 ("X 기준", "X 대비")는 보존.
def _gender_pattern(expected: str) -> re.Pattern[str]:
    others = [g for g in VALID_GENDERS if g != expected]
    if not others:
        return re.compile(r"(?!)")
    alt = "|".join(re.escape(g) for g in others)
    # "거세" 받침 없음 → "거세로", 나머지("암컷"/"수컷") 받침 ㅅ → "으로"
    return re.compile(
        rf"(?P<g1>{alt})(?P<josa1>으로|로)(?=\s*판정)"
    )


def enforce_gender(text: str, expected_gender: str) -> tuple[str, bool]:
    """성별 정합성. 입력과 다른 성별을 단언하는 패턴만 교체."""
    if expected_gender not in VALID_GENDERS:
        return text, False

    expected_josa = "로" if expected_gender == "거세" else "으로"
    pat = _gender_pattern(expected_gender)

    def repl(m: re.Match[str]) -> str:
        return f"{expected_gender}{expected_josa}"

    new_text, n = pat.subn(repl, text)
    return new_text, n > 0


def detect_gender_conflict(text: str, expected_gender: str) -> bool:
    """입력 성별과 다른 성별이 응답에 등장하는지 단순 검출 (정정 X, 메타데이터용)."""
    if expected_gender not in VALID_GENDERS:
        return False
    others = [g for g in VALID_GENDERS if g != expected_gender]
    return any(g in text for g in others)


def _walk(obj: Any, fn: Callable[[str], tuple[str, bool]]) -> tuple[Any, bool]:
    """dict/list/str 트리의 모든 문자열에 fn 적용. 변경 누적."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        out_list: list[Any] = []
        any_changed = False
        for x in obj:
            new_x, ch = _walk(x, fn)
            out_list.append(new_x)
            any_changed = any_changed or ch
        return out_list, any_changed
    if isinstance(obj, dict):
        out_dict: dict[Any, Any] = {}
        any_changed = False
        for k, v in obj.items():
            new_v, ch = _walk(v, fn)
            out_dict[k] = new_v
            any_changed = any_changed or ch
        return out_dict, any_changed
    return obj, False


def apply_postprocess(
    report: dict,
    expected_grade: str | None = None,
    expected_gender: str | None = None,
) -> dict:
    """LoRA 응답 dict 에 A3 + A4 + A5 후처리를 일괄 적용.

    Args:
        report: generate_report() 출력 (예: 4 필드 dict)
        expected_grade: 입력 ThemaPAOutput.grade — A4 적용 대상. None 이면 A3 만.
        expected_gender: 입력 성별 라벨("암컷"/"수컷"/"거세") — A5 적용 대상.

    Returns:
        후처리된 dict. 변경/검출 발생 시 '_postprocess' 메타필드 기록.
    """
    new_report, josa_changed = _walk(report, normalize_josa)

    grade_changed = False
    if expected_grade and expected_grade in VALID_GRADES:
        def grade_fn(text: str) -> tuple[str, bool]:
            return enforce_grade(text, expected_grade)
        new_report, grade_changed = _walk(new_report, grade_fn)

    gender_changed = False
    gender_conflict = False
    if expected_gender and expected_gender in VALID_GENDERS:
        def gender_fn(text: str) -> tuple[str, bool]:
            return enforce_gender(text, expected_gender)
        new_report, gender_changed = _walk(new_report, gender_fn)

        # 정정 후에도 다른 성별이 남아 있으면 잠재 환각 — 메타데이터로만 노출
        def conflict_walk(obj):
            if isinstance(obj, str):
                return detect_gender_conflict(obj, expected_gender)
            if isinstance(obj, list):
                return any(conflict_walk(x) for x in obj)
            if isinstance(obj, dict):
                return any(conflict_walk(v) for v in obj.values())
            return False
        gender_conflict = conflict_walk(new_report)

    if josa_changed or grade_changed or gender_changed or gender_conflict:
        meta = dict(new_report.get("_postprocess", {}))
        if josa_changed:
            meta["josa_normalized"] = True
        if grade_changed:
            meta["grade_enforced"] = True
        if gender_changed:
            meta["gender_enforced"] = True
        if gender_conflict:
            meta["gender_conflict_detected"] = True
        new_report["_postprocess"] = meta

    return new_report
