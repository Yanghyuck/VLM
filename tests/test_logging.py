# tests/test_logging.py — vlm/logging_config.py JSON 포맷 테스트

import io
import json
import logging

from vlm.logging_config import JsonFormatter, configure_logging, get_logger


def test_json_formatter_basic():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="vlm.test", level=logging.INFO, pathname="x.py", lineno=10,
        msg="테스트 메시지", args=(), exc_info=None,
    )
    output = fmt.format(record)
    data = json.loads(output)
    assert data["message"] == "테스트 메시지"
    assert data["level"] == "INFO"
    assert data["logger"] == "vlm.test"
    assert "timestamp" in data


def test_json_formatter_with_extra():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="vlm.test", level=logging.INFO, pathname="x.py", lineno=10,
        msg="msg", args=(), exc_info=None,
    )
    record.carcass_no = 3010
    record.elapsed_ms = 22.9
    output = fmt.format(record)
    data = json.loads(output)
    assert data["carcass_no"] == 3010
    assert data["elapsed_ms"] == 22.9


def test_json_formatter_handles_non_serializable():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="vlm.test", level=logging.INFO, pathname="x.py", lineno=10,
        msg="msg", args=(), exc_info=None,
    )
    record.weird_obj = object()  # not JSON-serializable
    output = fmt.format(record)
    data = json.loads(output)
    # str() fallback applied
    assert "weird_obj" in data
    assert isinstance(data["weird_obj"], str)


def test_configure_logging_text_mode(capsys):
    configure_logging(level="DEBUG", fmt="text")
    log = get_logger("test.text")
    log.warning("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "WARNING" in captured.out
