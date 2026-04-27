# tests/test_env_override.py — 환경변수 config override 단위 테스트

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlm.config import _to_namespace, _apply_env_overrides


@pytest.fixture
def base_cfg() -> SimpleNamespace:
    return _to_namespace({
        "db": {"host": "127.0.0.1", "port": 3306, "user": "root", "password": "old", "name": "db"},
        "paths": {"image_dir": "/old", "lora_adapter": "/adapter"},
        "api": {"host": "0.0.0.0", "port": 8000, "api_keys": []},
        "logging": {"level": "INFO", "format": "json"},
    })


def test_db_password_override(monkeypatch, base_cfg):
    monkeypatch.setenv("VLM_DB_PASSWORD", "new-secret-123")
    result = _apply_env_overrides(base_cfg)
    assert result.db.password == "new-secret-123"
    assert result.db.host == "127.0.0.1"  # 다른 값 유지


def test_api_port_int_cast(monkeypatch, base_cfg):
    monkeypatch.setenv("VLM_API_PORT", "9000")
    result = _apply_env_overrides(base_cfg)
    assert result.api.port == 9000
    assert isinstance(result.api.port, int)


def test_api_keys_csv_split(monkeypatch, base_cfg):
    monkeypatch.setenv("VLM_API_KEYS", "key-A, key-B ,key-C")
    result = _apply_env_overrides(base_cfg)
    assert result.api.api_keys == ["key-A", "key-B", "key-C"]


def test_no_env_no_change(base_cfg):
    original_pwd = base_cfg.db.password
    result = _apply_env_overrides(base_cfg)
    assert result.db.password == original_pwd


def test_invalid_int_skip(monkeypatch, base_cfg, capsys):
    monkeypatch.setenv("VLM_API_PORT", "not-a-number")
    result = _apply_env_overrides(base_cfg)
    # 기존 값 유지
    assert result.api.port == 8000
    # 경고 출력
    captured = capsys.readouterr()
    assert "캐스팅 실패" in captured.out


def test_image_dir_string_override(monkeypatch, base_cfg):
    monkeypatch.setenv("VLM_IMAGE_DIR", "/data/new/images")
    result = _apply_env_overrides(base_cfg)
    assert result.paths.image_dir == "/data/new/images"


def test_log_level_override(monkeypatch, base_cfg):
    monkeypatch.setenv("VLM_LOG_LEVEL", "DEBUG")
    result = _apply_env_overrides(base_cfg)
    assert result.logging.level == "DEBUG"
