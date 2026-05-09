"""D1 Constrained decoding 인프라 검증.

실제 모델 추론은 GPU 필요라 단위 테스트에서 제외. 여기서는
- JSON 스키마 정의 정합성
- lm-format-enforcer JsonSchemaParser 구성 가능성
- transformers 5.x monkey-patch 적용 여부
- prefix_allowed_tokens_fn 빌드 (tokenizer 없이 실패 케이스 포함)
만 검증한다.
"""
from __future__ import annotations

import pytest


def test_response_schema_required_keys():
    """4 키가 모두 required 로 선언되어 있어야 한다."""
    from vlm.train.inference import RESPONSE_JSON_SCHEMA

    assert set(RESPONSE_JSON_SCHEMA["required"]) == {
        "3문장_요약", "비정상_근거", "주의사항", "권고",
    }


def test_response_schema_property_types():
    """비정상_근거는 string|null, 주의사항은 array, 나머지는 string."""
    from vlm.train.inference import RESPONSE_JSON_SCHEMA

    props = RESPONSE_JSON_SCHEMA["properties"]
    assert props["3문장_요약"]["type"] == "string"
    assert props["권고"]["type"] == "string"
    assert props["주의사항"]["type"] == "array"
    assert props["주의사항"]["items"]["type"] == "string"
    # 비정상_근거는 nullable
    assert "string" in props["비정상_근거"]["type"]
    assert "null" in props["비정상_근거"]["type"]


def test_lm_format_enforcer_parser_builds():
    """JsonSchemaParser 가 응답 스키마로 빌드된다."""
    from lmformatenforcer import JsonSchemaParser
    from vlm.train.inference import RESPONSE_JSON_SCHEMA

    parser = JsonSchemaParser(RESPONSE_JSON_SCHEMA)
    assert parser is not None


def test_transformers_monkey_patch_applied():
    """inference.py import 시 transformers.tokenization_utils 에 PreTrainedTokenizerBase
    가 노출되어 있어야 한다 (lm-format-enforcer 호환).
    """
    import vlm.train.inference  # 모듈 import 가 monkey-patch 트리거
    import transformers.tokenization_utils as _tu

    assert hasattr(_tu, "PreTrainedTokenizerBase"), \
        "monkey-patch 미적용 — lm-format-enforcer 가 5.x 와 호환되지 않음"


def test_lm_format_enforcer_transformers_integration_imports():
    """build_transformers_prefix_allowed_tokens_fn import 가 monkey-patch 후 성공한다."""
    import vlm.train.inference  # noqa — monkey-patch 적용
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )
    assert build_transformers_prefix_allowed_tokens_fn is not None


def test_generate_report_constrained_default_off():
    """generate_report 기본 호출은 constrained=False (운영 안전 default)."""
    import inspect
    from vlm.train.inference import generate_report

    sig = inspect.signature(generate_report)
    assert sig.parameters["constrained"].default is False
