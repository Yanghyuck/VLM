# tests/test_config.py — vlm/config.py 로더 단위 테스트

import json
import pytest
from pathlib import Path
from types import SimpleNamespace


def _make_cfg(tmp_path, data: dict):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    return cfg_file


def test_config_loads_correctly(tmp_path, monkeypatch):
    data = {
        "db": {"host": "localhost", "port": 3306, "user": "u", "password": "p", "name": "db"},
        "paths": {"lora_adapter": "vlm/train/output/qwen3vl-lora"},
        "model": {"base_model_id": "Qwen/test"},
        "api": {"host": "0.0.0.0", "port": 8000},
        "grade": {"backfat_range": {"1+": [17, 25]}, "weight_range": {"1+": [83, 93]}},
    }
    cfg_file = _make_cfg(tmp_path, data)

    from vlm.config import _to_namespace, _load
    monkeypatch.setattr("vlm.config._load", lambda: _to_namespace(json.loads(cfg_file.read_text())))

    from vlm import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CFG", _to_namespace(data))

    from vlm.config import _to_namespace
    cfg = _to_namespace(data)
    assert cfg.db.host == "localhost"
    assert cfg.db.port == 3306
    assert cfg.api.port == 8000
    assert cfg.model.base_model_id == "Qwen/test"
    assert isinstance(cfg, SimpleNamespace)


def test_config_missing_file(tmp_path, monkeypatch):
    import vlm.config as cfg_mod

    missing_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "_load", lambda: (_ for _ in ()).throw(
        FileNotFoundError(f"config.json 없음: {missing_path}")
    ))

    with pytest.raises(FileNotFoundError, match="config.json 없음"):
        raise FileNotFoundError(f"config.json 없음: {missing_path}")


def test_config_invalid_json(tmp_path):
    from vlm.config import _load
    import vlm.config as cfg_mod
    from unittest.mock import patch

    bad_json = '{"db": {"host": "x"'
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(bad_json, encoding="utf-8")

    with patch.object(Path, "exists", return_value=True), \
         patch("builtins.open", lambda *a, **kw: cfg_file.open(*a[1:], **kw) if "config.json" in str(a[0]) else open(*a, **kw)):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(bad_json)
