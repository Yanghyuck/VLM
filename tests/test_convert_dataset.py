"""vlm/train/convert_dataset.py 의 한국어 조사 처리 + abnormal paraphrase 검증.

학습 데이터 생성 단계에서 '거세으로'/'암퇘지으로' 같은 어색한 조사가
박히는 것을 방지하기 위한 회귀 테스트. (b2aeac1 직전 시점의 livestock_train.json
에 거세으로 1,666건 + 암퇘지/수퇘지으로 1,639건이 박혀 있던 사고에 대한 가드.
이후 성별 표기를 암컷/수컷/거세 로 통일.)

B1 abnormal 응답 paraphrase (3종) 도 함께 검증.
"""
from __future__ import annotations

import pytest

from vlm.train.convert_dataset import (
    _abnormal_response,
    _abnormal_response_all,
    _abnormal_response_narrative,
    _abnormal_response_recommend_first,
    _eul_ro,
    _summary_response,
    _summary_response_alt,
)


# ---------------------------------------------------------------------------
# _eul_ro — 종성 검사 기반 '으로'/'로' 선택
# ---------------------------------------------------------------------------

class TestEulRo:
    @pytest.mark.parametrize("word", ["거세", "라디오", "노래"])
    def test_받침_없으면_로(self, word):
        assert _eul_ro(word) == "로"

    @pytest.mark.parametrize("word", ["암컷", "수컷", "코끝", "물건", "사람", "감"])
    def test_받침_있으면_으로(self, word):
        assert _eul_ro(word) == "으로"

    def test_빈_문자열(self):
        assert _eul_ro("") == "으로"

    def test_비한글_보수적(self):
        # 숫자/특수기호: 학습 데이터에 등장하지 않으므로 보수 경로로 '으로'
        assert _eul_ro("1+") == "으로"
        assert _eul_ro("ABC") == "으로"


# ---------------------------------------------------------------------------
# _summary_response — 조사 회귀 (학습 데이터 버그 방지)
# ---------------------------------------------------------------------------

def _meta(gender: int, grade: str = "1+", error_code: dict | None = None) -> dict:
    """테스트용 메타 dict (필수 필드만)."""
    return {
        "carcass_no": 3473,
        "slaughter_ymd": "20260422",
        "gender": gender,
        "grade": grade,
        "backfat_average": 20.0,
        "multifidus_thk": 30.0,
        "body_weight": 88.0,
        "error_code": error_code or {
            "pig_RightEntry": 0,
            "AI_Backbone_error": 0,
            "AI_BackFat_error": 0,
            "AI_HalfBone_error": 0,
            "AI_multifidus_error": 0,
            "AI_Outline_error": 0,
        },
    }


class TestSummaryResponseJosa:
    def test_거세_조사_정확(self):
        # "거세" 받침 없음 → "거세로"
        text = _summary_response(_meta(gender=3))
        assert "거세로" in text
        assert "거세으로" not in text  # 회귀 가드

    def test_암컷_조사_정확(self):
        # "암컷" 의 "컷" 받침 ㅅ → "암컷으로"
        text = _summary_response(_meta(gender=1))
        assert "암컷으로" in text
        assert "암컷로" not in text  # 잘못된 조사 회귀 가드

    def test_수컷_조사_정확(self):
        text = _summary_response(_meta(gender=2))
        assert "수컷으로" in text
        assert "수컷로" not in text

    def test_정상_케이스_등급_언급(self):
        text = _summary_response(_meta(gender=3, grade="1+"))
        assert "1+ 등급으로 판정" in text

    def test_오류_케이스_검출_언급(self):
        ec = {
            "pig_RightEntry": 0,
            "AI_Backbone_error": 0,
            "AI_BackFat_error": 1,
            "AI_HalfBone_error": 0,
            "AI_multifidus_error": 0,
            "AI_Outline_error": 0,
        }
        text = _summary_response(_meta(gender=3, grade="등외", error_code=ec))
        assert "AI 검출 오류" in text
        assert "등지방 검출 실패" in text
        assert "등외 등급으로 판정" in text
        assert "거세로" in text  # 조사 회귀


# ---------------------------------------------------------------------------
# A3 paraphrase — _summary_response_alt
# ---------------------------------------------------------------------------

class TestSummaryResponseAlt:
    def test_정상_케이스_등급_먼저(self):
        text = _summary_response_alt(_meta(gender=3, grade="1+"))
        # 등급/측정값 우선 순서
        assert text.startswith("도체번호 3473 의 최종 판정 등급은 1+ 입니다.")
        assert "거세로" in text  # 조사 정확

    def test_원본과_다른_표현(self):
        meta = _meta(gender=1, grade="2")
        original = _summary_response(meta)
        alt      = _summary_response_alt(meta)
        assert original != alt  # paraphrase 가 동일 X
        assert "암컷으로" in alt


# ---------------------------------------------------------------------------
# B1 abnormal response paraphrase
# ---------------------------------------------------------------------------

def _meta_with_errors(*error_keys: str) -> dict:
    """abnormal 테스트용 메타 — 지정된 error_code 만 1, 나머지 0."""
    ec = {
        "pig_RightEntry": 0,
        "AI_Backbone_error": 0,
        "AI_BackFat_error": 0,
        "AI_HalfBone_error": 0,
        "AI_multifidus_error": 0,
        "AI_Outline_error": 0,
    }
    for k in error_keys:
        ec[k] = 1
    return _meta(gender=3, grade="등외", error_code=ec)


class TestAbnormalResponseVariants:
    def test_원본_번호_매김_단일_실패(self):
        text = _abnormal_response(_meta_with_errors("AI_BackFat_error"))
        assert "다음 항목에서 비정상이 감지되었습니다" in text
        assert "1." in text
        assert "등지방 검출 실패" in text
        assert "재촬영" in text

    def test_원본_다중_실패_번호_매김(self):
        text = _abnormal_response(_meta_with_errors("AI_BackFat_error", "AI_Backbone_error"))
        assert "1." in text and "2." in text

    def test_narrative_단일_실패(self):
        text = _abnormal_response_narrative(_meta_with_errors("AI_BackFat_error"))
        # 번호 매김이 아닌 자연어 서술
        assert "1." not in text
        assert "등지방 검출 실패" in text
        assert "재촬영" in text or "수동 측정" in text

    def test_narrative_다중_실패_갯수_언급(self):
        text = _abnormal_response_narrative(
            _meta_with_errors("AI_BackFat_error", "AI_Backbone_error", "AI_multifidus_error")
        )
        assert "3건" in text

    def test_recommend_first_권고_먼저(self):
        text = _abnormal_response_recommend_first(_meta_with_errors("AI_BackFat_error"))
        assert text.startswith("재촬영 또는 수동 측정을 권고")
        assert "등지방 검출 실패" in text

    def test_all_세_가지_변형_반환(self):
        variants = _abnormal_response_all(_meta_with_errors("AI_BackFat_error"))
        assert len(variants) == 3
        assert len(set(variants)) == 3  # 모두 다른 텍스트
