# =============================================================================
# vlm/train/__init__.py
# -----------------------------------------------------------------------------
# 기능:
#   Qwen3-VL LoRA 학습 및 추론 패키지.
#   학습 파이프라인과 추론 진입점을 외부에 노출합니다.
#
# 주요 모듈:
#   inference.py       — Qwen3-VL LoRA 추론 (generate_report)
#   convert_dataset.py — dataset.jsonl → LLaMA-Factory ShareGPT 포맷 변환
#
# 학습 설정:
#   qwen3vl_lora.yaml  — LLaMA-Factory SFT 설정 (rank 64, 3 epoch, RTX 4090)
#   dataset_info.json  — LLaMA-Factory 데이터셋 등록 정보
#
# 사용 예시:
#   from vlm.train.inference import generate_report
# =============================================================================

from vlm.train.inference import generate_report

__all__ = ["generate_report"]
