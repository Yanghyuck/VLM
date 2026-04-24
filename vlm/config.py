# =============================================================================
# vlm/config.py
# -----------------------------------------------------------------------------
# 기능:
#   프로젝트 루트의 config.json을 읽어 전역 설정 객체(CFG)를 제공합니다.
#   모든 모듈은 이 파일을 통해 설정값에 접근합니다.
#   외부 라이브러리 없이 표준 라이브러리만 사용합니다.
#
# 제공 설정 키:
#   CFG.db.host / .user / .password / .name  : MySQL 접속 정보
#   CFG.paths.image_dir                      : 도체 이미지 디렉터리
#   CFG.paths.dataset_jsonl                  : 학습 JSONL 경로
#   CFG.paths.train_json                     : ShareGPT JSON 경로
#   CFG.paths.lora_adapter                   : LoRA 어댑터 디렉터리
#   CFG.paths.samples_dir                    : 샘플 JSON 디렉터리
#   CFG.model.base_model_id                  : HuggingFace 베이스 모델 ID
#   CFG.api.host / .port                     : FastAPI 서버 바인딩
#   CFG.grade.backfat_range / .weight_range  : 등급별 정상 수치 범위
#
# 사용 예시:
#   from vlm.config import CFG
#   print(CFG.db.host)           # "127.0.0.1"
#   print(CFG.paths.image_dir)   # "C:/Users/..."
#   print(CFG.api.port)          # 8000
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _load() -> SimpleNamespace:
    cfg_path = Path(__file__).parent.parent / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        return _to_namespace(json.load(f))


CFG = _load()
