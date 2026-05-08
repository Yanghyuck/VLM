import importlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from vlm.api.schemas import ReportRequest
from vlm.schema.thema_pa_output import ThemaPAOutput


DEFAULT_THEMA_PA_ROOT = Path(r"C:\Users\IPC\Desktop\git\thema_pa_VLM")
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


def test_save_vlm_response_resolves_relative_path_under_project_root(
    thema_pa_config, tmp_path, monkeypatch
):
    """B2 회귀 — 상대 `output_dir` 는 cwd 와 무관하게 thema_pa_VLM 루트 기준으로 해석.

    수정 전 동작: cwd 를 바꾸면 `<cwd>/.tmp/...` 에 결과가 떨어졌음.
    수정 후 동작: PROJECT_ROOT 기준으로 항상 thema_pa_VLM 루트 아래에 저장.
    """
    rest_api_module, _ = _load_rest_api_class()
    payload = _load_json(SAMPLE_PATH)
    response_json = {
        "summary": "요약",
        "grade_reason": None,
        "warnings": [],
        "recommendation": "권고",
        "model_used": "lora (test)",
    }

    config_copy = json.loads(json.dumps(thema_pa_config))
    rel_dir = Path(".tmp") / f"pytest-b2-{uuid4().hex}"
    config_copy["vlm_api"]["output_dir"] = str(rel_dir)

    # cwd 를 임시 디렉터리로 변경 — B2 미적용이라면 여기에 잘못 떨어졌을 것
    monkeypatch.chdir(tmp_path)
    saved_path = Path(rest_api_module.save_vlm_response_json(config_copy, payload, response_json))

    project_root = rest_api_module.PROJECT_ROOT
    expected_dir = (project_root / rel_dir).resolve()

    try:
        assert saved_path.parent.resolve() == expected_dir, (
            f"저장 위치가 thema_pa_VLM 루트 기준이 아님. "
            f"saved={saved_path}, expected_parent={expected_dir}"
        )
        # cwd 아래에 잘못 생성된 디렉터리가 없어야 함 (회귀 가드)
        assert not (tmp_path / rel_dir).exists(), (
            f"cwd 의존 회귀 — {tmp_path}/{rel_dir} 가 잘못 생성됨"
        )
    finally:
        if saved_path.exists():
            saved_path.unlink()
        if expected_dir.exists() and not any(expected_dir.iterdir()):
            expected_dir.rmdir()
            # .tmp/ 부모도 비어 있으면 정리
            parent = expected_dir.parent
            if parent.name == ".tmp" and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()


def test_save_vlm_response_absolute_path_unchanged(thema_pa_config, tmp_path):
    """B2 회귀 — 절대경로 `output_dir` 는 그대로 사용 (외부 마운트 등 명시 경로 존중)."""
    rest_api_module, _ = _load_rest_api_class()
    payload = _load_json(SAMPLE_PATH)
    response_json = {
        "summary": "요약",
        "grade_reason": None,
        "warnings": [],
        "recommendation": "권고",
        "model_used": "lora (test)",
    }

    config_copy = json.loads(json.dumps(thema_pa_config))
    abs_dir = tmp_path / "explicit_abs"
    config_copy["vlm_api"]["output_dir"] = str(abs_dir)

    saved_path = Path(rest_api_module.save_vlm_response_json(config_copy, payload, response_json))
    assert saved_path.parent.resolve() == abs_dir.resolve()
