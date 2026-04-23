"""
report/generator.py 단위 테스트.

- test_template_selection_*: API 호출 없이 템플릿 분기 로직만 검증
- test_generate_report_*: 실제 Claude API 호출 (ANTHROPIC_API_KEY 필요)
  pytest -m integration 으로 실행
"""

import json
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.report.generator import _select_template, generate_report

SAMPLES_DIR = os.path.join(ROOT, "vlm", "schema", "samples")


def load(filename):
    with open(os.path.join(SAMPLES_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


# ─── 템플릿 분기 단위 테스트 (API 호출 없음) ─────────────────────────────


def test_template_normal_case():
    data = load("normal_case.json")
    output = ThemaPAOutput(**data)
    template = _select_template(output)
    assert "정상 판정" in template


def test_template_backfat_error_uses_failure_analysis():
    data = load("backfat_error_case.json")
    output = ThemaPAOutput(**data)
    template = _select_template(output)
    assert "등지방 측정 실패" in template or "비정상 진입" in template


def test_template_entry_error_uses_failure_analysis():
    data = load("entry_error_case.json")
    output = ThemaPAOutput(**data)
    template = _select_template(output)
    assert "등지방 측정 실패" in template or "비정상 진입" in template


def test_template_generic_error_uses_error_case():
    data = load("normal_case.json")
    data["error_code"]["AI_HalfBone_error"] = 1
    output = ThemaPAOutput(**data)
    template = _select_template(output)
    assert "AI 검출 오류" in template


# ─── 통합 테스트 (실제 Claude API 호출) ──────────────────────────────────


@pytest.mark.integration
def test_generate_report_normal():
    data = load("normal_case.json")
    output = ThemaPAOutput(**data)
    report = generate_report(output)

    assert isinstance(report, dict)
    assert "3문장_요약" in report
    assert "비정상_근거" in report
    assert "주의사항" in report
    assert "권고" in report

    assert report["비정상_근거"] is None
    assert isinstance(report["주의사항"], list)
    assert output.grade in report["3문장_요약"]


@pytest.mark.integration
def test_generate_report_backfat_error():
    data = load("backfat_error_case.json")
    output = ThemaPAOutput(**data)
    report = generate_report(output)

    assert isinstance(report, dict)
    assert report.get("비정상_근거") is not None
    assert len(report.get("주의사항", [])) > 0


@pytest.mark.integration
def test_generate_report_entry_error():
    data = load("entry_error_case.json")
    output = ThemaPAOutput(**data)
    report = generate_report(output)

    assert isinstance(report, dict)
    assert report.get("비정상_근거") is not None
