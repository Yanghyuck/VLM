"""vlm/postprocess.py 단위 테스트.

A3 한국어 조사 정규화 + A4 등급 정합성 후처리 검증.
"""
from __future__ import annotations

import pytest

from vlm.postprocess import (
    apply_postprocess,
    detect_gender_conflict,
    enforce_gender,
    enforce_grade,
    normalize_josa,
)


# ---------------------------------------------------------------------------
# A3 — 한국어 조사 정규화
# ---------------------------------------------------------------------------

class TestNormalizeJosa:
    def test_거세으로(self):
        text, changed = normalize_josa("성별이 거세으로 판정됩니다.")
        assert text == "성별이 거세로 판정됩니다."
        assert changed is True

    def test_1plus_으로(self):
        text, changed = normalize_josa("최종 1+으로 출하 가능.")
        assert text == "최종 1+로 출하 가능."
        assert changed is True

    def test_등외으로(self):
        text, changed = normalize_josa("등외으로 처리됩니다.")
        assert text == "등외로 처리됩니다."
        assert changed is True

    def test_단독_숫자_등급(self):
        text, changed = normalize_josa("2으로 분류")
        assert text == "2로 분류"
        assert changed is True

    def test_정상_텍스트_미변경(self):
        original = "거세로 판정되며 1+로 출하 가능합니다."
        text, changed = normalize_josa(original)
        assert text == original
        assert changed is False

    def test_받침있는_단어_미변경(self):
        # "수컷으로" 의 "컷" 은 받침 있어 '으로' 가 정상
        original = "수컷으로 판정"
        text, changed = normalize_josa(original)
        assert text == original
        assert changed is False

    def test_복합_케이스(self):
        text, changed = normalize_josa("거세으로 판정, 1+으로 출하, 등외으로 분류")
        assert text == "거세로 판정, 1+로 출하, 등외로 분류"
        assert changed is True


# ---------------------------------------------------------------------------
# A4 — 등급 정합성
# ---------------------------------------------------------------------------

class TestEnforceGrade:
    def test_등외_입력_2등급_단언_교체(self):
        text, changed = enforce_grade("최종 2 등급으로 판정됩니다.", "등외")
        assert "2 등급" not in text
        assert "등외 등급" in text
        assert changed is True

    def test_등외_입력_최종_2등급(self):
        text, changed = enforce_grade("최종 2 등급", "등외")
        assert text == "최종 등외 등급"
        assert changed is True

    def test_입력과_같은_등급_미변경(self):
        text, changed = enforce_grade("최종 1+ 등급으로 판정", "1+")
        assert text == "최종 1+ 등급으로 판정"
        assert changed is False

    def test_비교_문맥_미변경(self):
        # "X 등급 도체 평균" 같은 비교 표현은 '으로 판정' 도 '최종' 도 없으므로 보존
        original = "1+ 등급 도체 평균보다 낮습니다."
        text, changed = enforce_grade(original, "2")
        assert text == original
        assert changed is False

    def test_invalid_expected_grade(self):
        text, changed = enforce_grade("최종 2 등급으로 판정", "invalid")
        assert text == "최종 2 등급으로 판정"
        assert changed is False

    def test_이의신청가능_패턴_교체(self):
        # 스모크 회귀: 입력 등외, 응답에 "이의 신청 가능: 1+ 등급" 환각
        text, changed = enforce_grade(
            "등급 이의 신청 가능: 1+ 등급", "등외"
        )
        assert "1+ 등급" not in text
        assert "이의 신청 가능: 등외 등급" in text
        assert changed is True

    def test_이의신청가능_콜론_변형_허용(self):
        # 전각 콜론 / 공백 변형
        text, changed = enforce_grade(
            "이의 신청 가능 ： 2 등급", "등외"
        )
        assert "등외 등급" in text
        assert changed is True

    def test_이의신청가능_입력과_같으면_미변경(self):
        text, changed = enforce_grade(
            "이의 신청 가능: 1+ 등급", "1+"
        )
        assert text == "이의 신청 가능: 1+ 등급"
        assert changed is False


# ---------------------------------------------------------------------------
# A5 — 성별 정합성
# ---------------------------------------------------------------------------

class TestEnforceGender:
    def test_거세_입력_암컷_단언_교체(self):
        # 입력 거세, 모델이 "암컷으로 판정" 환각 → "거세로 판정"
        text, changed = enforce_gender("암컷으로 판정됨", "거세")
        assert "암컷으로" not in text
        assert "거세로 판정" in text
        assert changed is True

    def test_암컷_입력_거세_단언_교체(self):
        # 입력 암컷, 모델이 "거세로 판정" → "암컷으로 판정"
        text, changed = enforce_gender("거세로 판정되어야 합니다", "암컷")
        assert "거세로" not in text
        assert "암컷으로 판정" in text
        assert changed is True

    def test_입력과_같은_성별_미변경(self):
        text, changed = enforce_gender("거세로 판정됨", "거세")
        assert text == "거세로 판정됨"
        assert changed is False

    def test_비교_문맥_미변경(self):
        # "X 기준" 같은 비교 표현은 보존 ('판정' 동사 부재)
        original = "암컷 기준 17~25mm 적용"
        text, changed = enforce_gender(original, "거세")
        assert text == original
        assert changed is False

    def test_invalid_expected_gender(self):
        text, changed = enforce_gender("암컷으로 판정", "invalid")
        assert text == "암컷으로 판정"
        assert changed is False


class TestDetectGenderConflict:
    def test_다른_성별_언급_검출(self):
        # 입력 거세, 응답에 "암컷" 등장 → conflict
        assert detect_gender_conflict("거세 암컷 판정 기준", "거세") is True

    def test_동일_성별만_등장_미검출(self):
        assert detect_gender_conflict("거세로 판정", "거세") is False

    def test_성별_언급_없음_미검출(self):
        assert detect_gender_conflict("등급은 1+ 입니다", "암컷") is False

    def test_invalid_expected(self):
        assert detect_gender_conflict("암컷", "invalid") is False


class TestApplyPostprocessWithGender:
    def test_성별_단언_교체_및_충돌_검출(self):
        report = {
            "비정상_근거": "비정상 진입으로 거세 암컷 판정 오류 발생",
        }
        result = apply_postprocess(report, expected_gender="거세")
        # 단언 표현은 정정 (현재 메시지엔 명확한 단언 없음 → 변화 없음)
        # 그러나 "암컷" 단어 잔존 → conflict 검출
        assert result["_postprocess"]["gender_conflict_detected"] is True

    def test_명확한_단언_정정(self):
        report = {"비정상_근거": "거세로 판정되어야 합니다"}
        result = apply_postprocess(report, expected_gender="암컷")
        assert "거세로" not in result["비정상_근거"]
        assert result["_postprocess"]["gender_enforced"] is True

    def test_None_이면_A5_미적용(self):
        report = {"비정상_근거": "거세 암컷 판정 오류"}
        result = apply_postprocess(report, expected_gender=None)
        assert "_postprocess" not in result  # 변경/검출 없음


# ---------------------------------------------------------------------------
# apply_postprocess — 통합
# ---------------------------------------------------------------------------

class TestApplyPostprocess:
    def test_4필드_dict_조사정규화(self):
        report = {
            "3문장_요약": "거세으로 판정되었으며 1+으로 출하 가능합니다.",
            "비정상_근거": None,
            "주의사항": ["등외으로 처리될 가능성"],
            "권고": "정상 출하",
        }
        result = apply_postprocess(report, expected_grade="1+")
        assert result["3문장_요약"] == "거세로 판정되었으며 1+로 출하 가능합니다."
        assert result["주의사항"] == ["등외로 처리될 가능성"]
        assert result["_postprocess"]["josa_normalized"] is True

    def test_등급_단언_강제_주입(self):
        report = {
            "3문장_요약": "최종 2 등급으로 판정",
            "비정상_근거": "검출 실패",
            "주의사항": [],
            "권고": "재촬영 권장",
        }
        result = apply_postprocess(report, expected_grade="등외")
        assert "2 등급" not in result["3문장_요약"]
        assert "등외 등급" in result["3문장_요약"]
        assert result["_postprocess"]["grade_enforced"] is True

    def test_변경_없으면_메타필드_미추가(self):
        report = {
            "3문장_요약": "정상 도체이며 1+ 등급으로 판정.",
            "비정상_근거": None,
            "주의사항": [],
            "권고": "정상 출하",
        }
        result = apply_postprocess(report, expected_grade="1+")
        assert "_postprocess" not in result
        # 원본 dict 보존 (depth 1)
        assert result["3문장_요약"] == "정상 도체이며 1+ 등급으로 판정."

    def test_expected_grade_None_이면_A3_만(self):
        report = {"권고": "거세으로 처리, 최종 2 등급으로 판정"}
        result = apply_postprocess(report, expected_grade=None)
        # A3 적용
        assert "거세로 처리" in result["권고"]
        # A4 미적용 — '2 등급' 그대로 유지
        assert "2 등급" in result["권고"]
        assert result["_postprocess"]["josa_normalized"] is True
        assert "grade_enforced" not in result["_postprocess"]

    def test_중첩_리스트_재귀_적용(self):
        report = {
            "주의사항": ["거세으로 표기", "1+으로 출하"],
        }
        result = apply_postprocess(report)
        assert result["주의사항"] == ["거세로 표기", "1+로 출하"]

    def test_None_과_타입_보존(self):
        report = {
            "3문장_요약": "거세으로 처리",
            "비정상_근거": None,
            "주의사항": [],
            "권고": "확인 필요",
            "model_used": "lora",
        }
        result = apply_postprocess(report, expected_grade="1+")
        assert result["비정상_근거"] is None
        assert result["주의사항"] == []
        assert result["model_used"] == "lora"
