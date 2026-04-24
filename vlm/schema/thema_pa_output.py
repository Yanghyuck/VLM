# =============================================================================
# vlm/schema/thema_pa_output.py
# -----------------------------------------------------------------------------
# 기능:
#   thema_pa AI 시스템이 출력하는 돼지 도체 판정 결과 JSON을
#   Pydantic v2 모델로 정의합니다. 세 프로젝트(QA Copilot / Benchmark /
#   Multimodal API)가 공통으로 import하는 핵심 스키마 파일입니다.
#
# 주요 클래스:
#   - Gender       : 성별 열거형 (1=암, 2=수, 3=거세)
#   - ErrorCode    : AI 검출 오류 6종 플래그 (0=정상, 1=오류)
#                    is_normal() → 모든 플래그 0이면 True
#                    failed_parts() → 오류 발생 부위 한국어 이름 리스트
#   - BackboneSlope: 척추 기울기 이상 여부
#   - ThemaPAOutput: 전체 판정 결과 (측정값 + 등급 + 오류 + 이미지 경로)
#                    summary() → 프롬프트 주입용 한국어 한 줄 요약 반환
#
# 동작 방법:
#   # DB 조회 결과 또는 JSON 파일로 인스턴스 생성
#   import json
#   with open("vlm/schema/samples/normal_case.json", encoding="utf-8") as f:
#       data = json.load(f)
#   output = ThemaPAOutput(**data)
#
#   print(output.grade)             # "1+"
#   print(output.error_code.is_normal())   # True
#   print(output.summary())         # 한 줄 요약 문자열
#
# 의존성:
#   pydantic>=2.0
# =============================================================================

from __future__ import annotations

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Gender(IntEnum):
    SOW = 1       # 암퇘지
    BOAR = 2      # 수퇘지
    BARROW = 3    # 거세

    def label(self) -> str:
        return {1: "암퇘지", 2: "수퇘지", 3: "거세"}[self.value]


class Grade(str):
    """허용값: '1+', '1', '2', '등외'"""
    VALID = {"1+", "1", "2", "등외"}

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        if v not in cls.VALID:
            raise ValueError(f"grade must be one of {cls.VALID}, got '{v}'")
        return v


class ErrorCode(BaseModel):
    pig_RightEntry: int = Field(0, ge=0, le=1, description="비정상 진입 감지 (1=감지)")
    AI_Backbone_error: int = Field(0, ge=0, le=1, description="척추 검출 실패")
    AI_BackFat_error: int = Field(0, ge=0, le=1, description="등지방 검출 실패")
    AI_HalfBone_error: int = Field(0, ge=0, le=1, description="반골 검출 실패")
    AI_multifidus_error: int = Field(0, ge=0, le=1, description="뭇갈래근 검출 실패")
    AI_Outline_error: int = Field(0, ge=0, le=1, description="윤곽선 검출 실패")

    def is_normal(self) -> bool:
        return all(v == 0 for v in self.model_dump().values())

    def failed_parts(self) -> list[str]:
        label_map = {
            "pig_RightEntry": "비정상 진입",
            "AI_Backbone_error": "척추",
            "AI_BackFat_error": "등지방",
            "AI_HalfBone_error": "반골",
            "AI_multifidus_error": "뭇갈래근",
            "AI_Outline_error": "윤곽선",
        }
        return [label_map[k] for k, v in self.model_dump().items() if v == 1]


class BackboneSlope(BaseModel):
    has_large_slope: bool = Field(description="기준 임계값 초과 척추 기울기 존재 여부")
    threshold: Optional[float] = Field(None, description="판정에 사용된 마지막 임계값")


class ThemaPAOutput(BaseModel):
    # 식별
    carcass_no: int = Field(description="도체 번호")
    slaughter_ymd: str = Field(description="도축일 (YYYYMMDD)")

    # 측정값
    backfat_average: float = Field(description="평균 등지방 두께 (mm)")
    multifidus_thk: float = Field(description="뭇갈래근 두께 (mm)")
    body_length: float = Field(description="체장 (cm)")
    body_width: float = Field(description="체폭 (cm)")
    body_weight: float = Field(description="도체중 (kg)")

    # 판정
    gender: Gender = Field(description="성별 (1=암, 2=수, 3=거세)")
    grade: str = Field(description="등급 (1+/1/2/등외)")

    # 에러 및 슬로프
    error_code: ErrorCode
    backbone_slope: BackboneSlope

    # 이미지
    result_image_path: Optional[str] = Field(None, description="AI 결과 이미지 경로")

    @model_validator(mode="after")
    def _validate_grade(self) -> ThemaPAOutput:
        if self.grade not in Grade.VALID:
            raise ValueError(f"grade must be one of {Grade.VALID}, got '{self.grade}'")
        return self

    def summary(self) -> str:
        """한국어 한 줄 요약 (프롬프트 빌드용 헬퍼)"""
        parts = [
            f"도체번호 {self.carcass_no}",
            f"도축일 {self.slaughter_ymd}",
            f"{self.gender.label()}",
            f"도체중 {self.body_weight}kg",
            f"등지방 {self.backfat_average}mm",
            f"뭇갈래근 {self.multifidus_thk}mm",
            f"체장 {self.body_length}cm / 체폭 {self.body_width}cm",
            f"판정 등급: {self.grade}",
        ]
        if not self.error_code.is_normal():
            parts.append(f"검출 실패 부위: {', '.join(self.error_code.failed_parts())}")
        if self.backbone_slope.has_large_slope:
            parts.append("척추 기울기 이상 감지")
        return " | ".join(parts)
