# =============================================================================
# tests/test_schema.py
# -----------------------------------------------------------------------------
# 기능:
#   vlm/schema/thema_pa_output.py 의 Pydantic 모델을 검증합니다.
#   실제 샘플 JSON(vlm/schema/samples/)을 로드하여 파싱 정확성과
#   비즈니스 로직(is_normal, failed_parts, summary, 등급 유효성)을 테스트합니다.
#
# 테스트 케이스:
#   test_normal_case         — 정상 케이스: is_normal=True, grade="1+"
#   test_backfat_error_case  — 등지방·척추 오류: failed_parts에 해당 부위 포함
#   test_entry_error_case    — 비정상 진입: failed_parts에 "비정상 진입" 포함
#   test_summary_contains_grade — summary() 출력에 등급 포함 여부
#   test_invalid_grade_raises   — 허용 외 grade 값 입력 시 예외 발생 확인
#
# 동작 방법:
#   # VLM/ 루트에서 실행
#   pytest tests/test_schema.py -v
#
# 전제 조건:
#   vlm/schema/samples/ 에 normal_case.json, backfat_error_case.json,
#   entry_error_case.json 이 존재해야 함
#
# 의존성:
#   pytest, pydantic>=2.0
# =============================================================================

import json
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "vlm", "schema", "samples")

import sys
sys.path.insert(0, ROOT)
from vlm.schema.thema_pa_output import ThemaPAOutput


def load(filename):
    with open(os.path.join(SAMPLES_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def test_normal_case():
    data = load("normal_case.json")
    r = ThemaPAOutput(**data)
    assert r.error_code.is_normal()
    assert r.error_code.failed_parts() == []
    assert r.grade == "1+"


def test_backfat_error_case():
    data = load("backfat_error_case.json")
    r = ThemaPAOutput(**data)
    assert not r.error_code.is_normal()
    assert "등지방" in r.error_code.failed_parts()
    assert "척추" in r.error_code.failed_parts()


def test_entry_error_case():
    data = load("entry_error_case.json")
    r = ThemaPAOutput(**data)
    assert not r.error_code.is_normal()
    assert "비정상 진입" in r.error_code.failed_parts()


def test_summary_contains_grade():
    data = load("normal_case.json")
    r = ThemaPAOutput(**data)
    assert r.grade in r.summary()


def test_invalid_grade_raises():
    data = load("normal_case.json")
    data["grade"] = "3"
    with pytest.raises(Exception):
        ThemaPAOutput(**data)
