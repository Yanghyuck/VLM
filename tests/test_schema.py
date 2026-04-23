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
