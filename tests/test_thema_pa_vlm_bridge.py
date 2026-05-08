import importlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from vlm.api.schemas import ReportRequest
from vlm.schema.thema_pa_output import ThemaPAOutput


DEFAULT_THEMA_PA_ROOT = Path(r"C:\Users\IPC\Desktop\git\thema_pa")
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "vlm" / "schema" / "samples" / "sample_3473.json"


def _thema_pa_root() -> Path:
    root = Path(os.environ.get("THEMA_PA_ROOT", str(DEFAULT_THEMA_PA_ROOT)))
    if not root.exists():
        pytest.skip(f"thema_pa repo not found: {root}")
    return root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rest_api_class():
    root = _thema_pa_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module = importlib.import_module("comm.rest_api")
    return module, module.RestAPI


@pytest.fixture
def thema_pa_config():
    root = _thema_pa_root()
    return _load_json(root / "config.json")


def test_thema_pa_config_has_vlm_api_block(thema_pa_config):
    assert "vlm_api" in thema_pa_config
    assert thema_pa_config["vlm_api"]["enabled"] is True
    assert thema_pa_config["vlm_api"]["url"] == "http://127.0.0.1:8000/v1/report"
    assert thema_pa_config["vlm_api"]["timeout_sec"] == 180
    assert thema_pa_config["vlm_api"]["output_dir"] == "./storage/vlm_reports"


def test_thema_pa_rest_api_posts_sample_payload(monkeypatch, thema_pa_config):
    rest_api_module, RestAPI = _load_rest_api_class()
    payload = _load_json(SAMPLE_PATH)
    captured = {}

    class DummyResponse:
        status_code = 200
        text = '{"ok": true}'
        ok = True

        @staticmethod
        def json():
            return {
                "summary": "요약",
                "grade_reason": None,
                "warnings": [],
                "recommendation": "권고",
                "model_used": "lora (25.57s)",
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(rest_api_module.requests, "post", fake_post)

    response = RestAPI(thema_pa_config).SendVLMReport(payload)

    assert response is not None
    assert response.status_code == 200
    assert captured["url"] == thema_pa_config["vlm_api"]["url"]
    assert captured["headers"]["Content-Type"] == "application/json"
    assert "X-API-Key" not in captured["headers"]
    assert captured["timeout"] == thema_pa_config["vlm_api"]["timeout_sec"]
    assert captured["json"] == payload


def test_thema_pa_sample_payload_is_valid_vlm_request():
    payload = _load_json(SAMPLE_PATH)

    req = ReportRequest(**payload)
    output = ThemaPAOutput(**req.model_dump())

    assert output.carcass_no == 3473
    assert output.grade == "1+"
    assert output.result_image_path is not None
    assert Path(output.result_image_path).exists()
    assert "thema_pa" in output.result_image_path


def test_thema_pa_validates_and_saves_vlm_response(thema_pa_config):
    rest_api_module, _ = _load_rest_api_class()
    payload = _load_json(SAMPLE_PATH)
    response_json = {
        "summary": "요약",
        "grade_reason": None,
        "warnings": [],
        "recommendation": "권고",
        "model_used": "lora (25.57s)",
    }

    validated = rest_api_module.validate_vlm_response_json(response_json)
    assert validated == response_json

    config_copy = json.loads(json.dumps(thema_pa_config))
    out_dir = Path(__file__).resolve().parent.parent / ".tmp" / f"pytest-vlm-{uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_copy["vlm_api"]["output_dir"] = str(out_dir)
    saved_path = rest_api_module.save_vlm_response_json(config_copy, payload, response_json)

    saved = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert saved["request"]["carcass_no"] == 3473
    assert saved["response"]["summary"] == "요약"
    assert Path(saved_path).parent == out_dir


def test_thema_pa_rejects_invalid_vlm_response():
    rest_api_module, _ = _load_rest_api_class()

    with pytest.raises(ValueError, match="missing fields"):
        rest_api_module.validate_vlm_response_json({
            "summary": "요약",
            "warnings": [],
        })
