# =============================================================================
# vlm/report/generator.py
# -----------------------------------------------------------------------------
# Qwen3-VL LoRA 기반 판정 리포트 생성기.
# 실제 추론 로직은 vlm/train/inference.py 에 위치하며,
# 이 모듈은 하위 호환성을 위한 진입점 역할을 합니다.
#
# 사용 예시:
#   from vlm.report.generator import generate_report
#   from vlm.schema.thema_pa_output import ThemaPAOutput
#   import json
#
#   with open("vlm/schema/samples/normal_case.json", encoding="utf-8") as f:
#       data = json.load(f)
#   report = generate_report(ThemaPAOutput(**data))
#
# 전제 조건:
#   vlm/train/output/qwen3vl-lora/ 에 LoRA 어댑터 존재 (없으면 베이스 모델 사용)
#   GPU 필수 (RTX 4090 권장, 최소 16GB VRAM)
# =============================================================================

from vlm.train.inference import generate_report

__all__ = ["generate_report"]
