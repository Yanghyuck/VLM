# tests/test_json_extraction.py — _find_balanced_json / _extract_json 단위 테스트

from vlm.train.json_utils import _find_balanced_json, _extract_json


def test_find_nested_json():
    text = 'prefix {"a": {"b": 1}} suffix'
    assert _find_balanced_json(text) == '{"a": {"b": 1}}'


def test_find_json_with_code_fence():
    text = '```json\n{"key": "value"}\n```'
    assert _find_balanced_json(text) == '{"key": "value"}'


def test_find_json_with_string_containing_brace():
    text = '{"message": "hello {world}"}'
    assert _find_balanced_json(text) == '{"message": "hello {world}"}'


def test_find_json_with_escaped_quotes():
    text = r'{"key": "she said \"hi\""}'
    assert _find_balanced_json(text) == r'{"key": "she said \"hi\""}'


def test_find_no_json():
    assert _find_balanced_json("no json here") is None


def test_extract_valid_nested_json():
    text = '{"3문장_요약": "text", "주의사항": ["a", "b"]}'
    result = _extract_json(text)
    assert result["3문장_요약"] == "text"
    assert result["주의사항"] == ["a", "b"]


def test_extract_json_fallback_on_invalid():
    text = "this is not json at all"
    result = _extract_json(text)
    assert result["3문장_요약"] == "this is not json at all"
    assert result["비정상_근거"] is None
    assert result["주의사항"] == []


def test_extract_json_from_codeblock_with_nested():
    text = '```json\n{"outer": {"inner": {"deep": 1}}}\n```'
    result = _extract_json(text)
    assert result == {"outer": {"inner": {"deep": 1}}}
