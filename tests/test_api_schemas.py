# tests/test_api_schemas.py — vlm/api/schemas.py 단위 테스트

import pytest
from pydantic import ValidationError
from vlm.api.schemas import ReportRequest, ReportResponse, ErrorCodeInput


def _base_request(**overrides) -> dict:
    data = {
        "carcass_no": 1001,
        "slaughter_ymd": "20240101",
        "backfat_average": 20.5,
        "multifidus_thk": 15.0,
        "body_weight": 90.0,
        "gender": 1,
        "grade": "1+",
    }
    data.update(overrides)
    return data


def test_request_default_error_code():
    req = ReportRequest(**_base_request())
    assert req.error_code.pig_RightEntry == 0
    assert req.error_code.AI_BackFat_error == 0


def test_request_with_error_code():
    req = ReportRequest(**_base_request(
        error_code={"pig_RightEntry": 1, "AI_BackFat_error": 1}
    ))
    assert req.error_code.pig_RightEntry == 1
    assert req.error_code.AI_BackFat_error == 1


def test_request_invalid_error_code_value():
    with pytest.raises(ValidationError):
        ReportRequest(**_base_request(
            error_code={"pig_RightEntry": 2}
        ))


def test_request_optional_image_path():
    req = ReportRequest(**_base_request())
    assert req.result_image_path is None

    req2 = ReportRequest(**_base_request(result_image_path="/path/to/img.jpg"))
    assert req2.result_image_path == "/path/to/img.jpg"


def test_response_defaults():
    resp = ReportResponse(
        summary="요약 텍스트",
        recommendation="재측정 권고",
        model_used="lora (3.2s)",
    )
    assert resp.grade_reason is None
    assert resp.warnings == []


def test_response_with_warnings():
    resp = ReportResponse(
        summary="요약",
        warnings=["등지방 범위 초과", "척추 검출 실패"],
        recommendation="수동 측정 권고",
        model_used="base (5.1s)",
    )
    assert len(resp.warnings) == 2


def test_error_code_input_all_zero():
    ec = ErrorCodeInput()
    assert all(v == 0 for v in ec.model_dump().values())
