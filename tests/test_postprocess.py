"""vlm/postprocess.py 단위 테스트.

A3 한국어 조사 정규화 + A4 등급 정합성 후처리 검증.
"""
from __future__ import annotations

import pytest

from vlm.postprocess import (
    apply_postprocess,
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
