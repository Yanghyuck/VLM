# =============================================================================
# tests/test_eval_harness.py
# -----------------------------------------------------------------------------
# Eval harness 정적 검증 — CI/CD 없는 로컬 환경에서 회귀를 빠르게 차단.
#
# 검증 범위:
#   1. registry.yaml 가 yaml.safe_load 로 파싱되고 필수 필드 존재
#   2. 등록 모델 라벨 유일성
#   3. baseline 라벨이 등록 모델 중 하나
#   4. 회귀 임계치 키가 실제 scorer.evaluate() 가 반환하는 메트릭에 존재
#   5. 가장 최근 regression.json 가 있으면, baseline 자기 자신은 위반 0
#      (등록 모델 중 baseline 이 아닌 후보가 위반해도 fail 하지 않음 —
#       그 판단은 harness check / score --check 의 책임)
#
# 의도:
#   - pytest 한 번으로 "harness 가 import 되고 registry 가 valid" 한지 보장
#   - 운영 채택 모델(baseline)이 회귀 검사 자기 비교에서 0 violation 임을 보장
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
REGISTRY = ROOT / "vlm" / "bench" / "registry.yaml"
REGRESSION = ROOT / "vlm" / "bench" / "regression.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    assert REGISTRY.exists(), f"registry missing: {REGISTRY}"
    with open(REGISTRY, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_registry_required_fields(registry):
    for k in ("eval_set", "baseline", "regression", "models"):
        assert k in registry, f"registry missing field: {k}"
    es = registry["eval_set"]
    for k in ("source", "n", "seed", "path"):
        assert k in es, f"eval_set missing: {k}"
    assert isinstance(registry["models"], list) and registry["models"]


def test_model_labels_unique(registry):
    labels = [m["label"] for m in registry["models"]]
    assert len(labels) == len(set(labels)), f"duplicate labels: {labels}"


def test_model_required_fields(registry):
    for m in registry["models"]:
        assert "label" in m and "mode" in m
        assert m["mode"] in {"base", "lora"}
        if m["mode"] == "lora":
            assert m.get("adapter_path"), f"lora model needs adapter_path: {m['label']}"


def test_baseline_is_registered(registry):
    labels = {m["label"] for m in registry["models"]}
    assert registry["baseline"] in labels, \
        f"baseline '{registry['baseline']}' not in registered labels {labels}"


def test_regression_keys_known(registry):
    """회귀 임계치 키가 scorer.evaluate() 반환 metric 에 존재해야 한다."""
    import sys
    sys.path.insert(0, str(ROOT))
    from vlm.bench.scorer import evaluate

    # 빈 records 호출은 {"n": 0} 반환 — 키 목록 확인용으로 부적합.
    # 대신 known metric whitelist 와 cross-check.
    known = {
        "json_parse_rate", "grade_match_rate", "number_citation",
        "rouge_l", "rouge_l_max", "bert_score_f1",
        "distinct_1", "distinct_2", "elapsed_avg_sec",
    }
    for metric in registry.get("regression", {}):
        assert metric in known, \
            f"regression metric '{metric}' 가 scorer.evaluate() known set 에 없음. 오타이거나 메트릭 추가 필요."


def test_harness_importable():
    """harness 모듈 import + load_registry 호출 가능 여부."""
    import sys
    sys.path.insert(0, str(ROOT))
    from vlm.bench import harness
    reg = harness.load_registry()
    assert "models" in reg


@pytest.mark.skipif(not REGRESSION.exists(), reason="regression.json 없음 (harness score --check 먼저 실행)")
def test_baseline_self_zero_violation():
    """regression.json 의 위반 목록 중 baseline 자신이 candidate 인 항목은 없어야 한다.
    (baseline 은 자기 자신과만 비교하지 않으므로 항상 비교 대상에서 제외됨)"""
    with open(REGRESSION, encoding="utf-8") as f:
        payload = json.load(f)
    baseline = payload.get("baseline")
    for reg in payload.get("regressions", []):
        assert reg["candidate"] != baseline, \
            f"baseline '{baseline}' 가 candidate 로 비교 — 회귀 로직 버그"
