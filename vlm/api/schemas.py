# =============================================================================
# vlm/api/schemas.py
# -----------------------------------------------------------------------------
# 기능:
#   FastAPI 서버의 요청/응답 Pydantic 모델 정의.
#   POST /v1/report 엔드포인트의 입력과 출력 형식을 규정합니다.
#
# 동작 방법:
#   from vlm.api.schemas import ReportRequest, ReportResponse
#
# 의존성:
#   pydantic>=2.0
# =============================================================================

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ErrorCodeInput(BaseModel):
    pig_RightEntry: int = Field(0, ge=0, le=1)
    AI_Backbone_error: int = Field(0, ge=0, le=1)
    AI_BackFat_error: int = Field(0, ge=0, le=1)
    AI_HalfBone_error: int = Field(0, ge=0, le=1)
    AI_multifidus_error: int = Field(0, ge=0, le=1)
    AI_Outline_error: int = Field(0, ge=0, le=1)


class BackboneSlopeInput(BaseModel):
    has_large_slope: bool = False
    threshold: Optional[float] = None


class ReportRequest(BaseModel):
    carcass_no: int = Field(description="도체 번호")
    slaughter_ymd: str = Field(description="도축일 (YYYYMMDD)")
    backfat_average: float = Field(description="평균 등지방 두께 (mm)")
    multifidus_thk: float = Field(description="뭇갈래근 두께 (mm)")
    body_length: float = Field(0.0, description="체장 (cm)")
    body_width: float = Field(0.0, description="체폭 (cm)")
    body_weight: float = Field(description="도체중 (kg)")
    gender: int = Field(description="성별 (1=암, 2=수, 3=거세)")
    grade: str = Field(description="등급 (1+/1/2/등외)")
    error_code: ErrorCodeInput = Field(default_factory=ErrorCodeInput)
    backbone_slope: BackboneSlopeInput = Field(default_factory=BackboneSlopeInput)
    result_image_path: Optional[str] = Field(None, description="도체 이미지 절대경로")


class ReportResponse(BaseModel):
    summary: str = Field(description="도체 판정 3문장 요약")
    grade_reason: Optional[str] = Field(None, description="등급 판정 근거 (비정상 시 포함)")
    warnings: list[str] = Field(default_factory=list, description="주의사항 목록")
    recommendation: str = Field(description="현장 조치 권고")
    model_used: str = Field(description="추론에 사용된 모델 (lora / base)")
