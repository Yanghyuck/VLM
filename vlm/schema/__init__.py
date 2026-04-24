# =============================================================================
# vlm/schema/__init__.py
# -----------------------------------------------------------------------------
# 기능:
#   thema_pa 판정 결과 스키마 패키지.
#   ThemaPAOutput 및 관련 모델을 외부에 노출합니다.
#
# 주요 export:
#   ThemaPAOutput  — 전체 판정 결과 모델 (측정값 + 등급 + 오류 + 이미지 경로)
#   ErrorCode      — AI 검출 오류 6종 플래그
#   Gender         — 성별 열거형
#   BackboneSlope  — 척추 기울기 이상 여부
#
# 사용 예시:
#   from vlm.schema import ThemaPAOutput
# =============================================================================

from vlm.schema.thema_pa_output import ThemaPAOutput, ErrorCode, Gender, BackboneSlope

__all__ = ["ThemaPAOutput", "ErrorCode", "Gender", "BackboneSlope"]
