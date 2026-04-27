# =============================================================================
# vlm/config.py
# -----------------------------------------------------------------------------
# 기능:
#   프로젝트 루트의 config.json을 읽어 전역 설정 객체(CFG)를 제공합니다.
#   환경변수가 설정돼 있으면 config.json 값을 override 합니다 (운영 보안).
#   모든 모듈은 이 파일을 통해 설정값에 접근합니다.
#   외부 라이브러리 없이 표준 라이브러리만 사용합니다.
#
# 환경변수 override (있으면 config.json보다 우선):
#   VLM_DB_HOST       → CFG.db.host
#   VLM_DB_PORT       → CFG.db.port (int)
#   VLM_DB_USER       → CFG.db.user
#   VLM_DB_PASSWORD   → CFG.db.password   ⭐ 운영에서 필수 (config 평문 금지)
#   VLM_DB_NAME       → CFG.db.name
#   VLM_API_HOST      → CFG.api.host
#   VLM_API_PORT      → CFG.api.port (int)
#   VLM_API_KEYS      → CFG.api.api_keys (콤마 구분, 예: "key1,key2,key3")
#   VLM_LORA_ADAPTER  → CFG.paths.lora_adapter
#   VLM_IMAGE_DIR     → CFG.paths.image_dir
#   VLM_LOG_LEVEL     → CFG.logging.level
#   VLM_LOG_FORMAT    → CFG.logging.format
#
# 제공 설정 키:
#   CFG.db.host / .user / .password / .name  : MySQL 접속 정보
#   CFG.paths.image_dir                      : 도체 이미지 디렉터리
#   CFG.paths.dataset_jsonl                  : 학습 JSONL 경로
#   CFG.paths.train_json                     : ShareGPT JSON 경로
#   CFG.paths.lora_adapter                   : LoRA 어댑터 디렉터리
#   CFG.paths.samples_dir                    : 샘플 JSON 디렉터리
#   CFG.model.base_model_id                  : HuggingFace 베이스 모델 ID
#   CFG.model.quantize / .quantize_mode      : INT4/INT8 양자화 옵션
#   CFG.api.host / .port                     : FastAPI 서버 바인딩
#   CFG.api.api_keys                         : X-API-Key 인증 (빈 배열=비활성화)
#   CFG.api.rate_limit_per_minute            : Rate limit
#   CFG.api.inference_timeout_sec            : 추론 타임아웃
#   CFG.grade.backfat_range / .weight_range  : 등급별 정상 수치 범위
#   CFG.logging.level / .format              : 로깅 설정
#
# 사용 예시:
#   from vlm.config import CFG
#   print(CFG.db.host)           # "127.0.0.1"
#   print(CFG.paths.image_dir)   # "C:/Users/..."
#   print(CFG.api.port)          # 8000
#
#   # 운영 배포 (DB 비밀번호 환경변수)
#   $ export VLM_DB_PASSWORD="real_secret"
#   $ python vlm/api/server.py
# =============================================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


# (env_key, dotted_path, type_caster)
_ENV_OVERRIDES = [
    ("VLM_DB_HOST",      "db.host",            str),
    ("VLM_DB_PORT",      "db.port",            int),
    ("VLM_DB_USER",      "db.user",            str),
    ("VLM_DB_PASSWORD",  "db.password",        str),
    ("VLM_DB_NAME",      "db.name",            str),
    ("VLM_API_HOST",     "api.host",           str),
    ("VLM_API_PORT",     "api.port",           int),
    ("VLM_API_KEYS",     "api.api_keys",       lambda s: [k.strip() for k in s.split(",") if k.strip()]),
    ("VLM_LORA_ADAPTER", "paths.lora_adapter", str),
    ("VLM_IMAGE_DIR",    "paths.image_dir",    str),
    ("VLM_LOG_LEVEL",    "logging.level",      str),
    ("VLM_LOG_FORMAT",   "logging.format",     str),
]


def _apply_env_overrides(ns: SimpleNamespace) -> SimpleNamespace:
    """환경변수가 있으면 SimpleNamespace 의 해당 path 값을 덮어쓴다."""
    overridden: list[str] = []
    for env_key, path, caster in _ENV_OVERRIDES:
        val = os.environ.get(env_key)
        if val is None:
            continue
        try:
            casted = caster(val)
        except (ValueError, TypeError) as e:
            print(f"[config] {env_key} 캐스팅 실패: {e}, skip")
            continue

        # path 따라가며 마지막 속성 set
        parts = path.split(".")
        target = ns
        for p in parts[:-1]:
            if not hasattr(target, p):
                setattr(target, p, SimpleNamespace())
            target = getattr(target, p)
        setattr(target, parts[-1], casted)
        overridden.append(env_key)

    if overridden:
        print(f"[config] env overrides applied: {', '.join(overridden)}")
    return ns


def _load() -> SimpleNamespace:
    cfg_path = Path(__file__).parent.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.json 없음: {cfg_path}\n"
            "config.example.json 을 복사하여 config.json 을 생성하고 값을 채워주세요.\n"
            "  cp config.example.json config.json"
        )
    try:
        with open(cfg_path, encoding="utf-8") as f:
            ns = _to_namespace(json.load(f))
    except json.JSONDecodeError as e:
        raise ValueError(f"config.json 파싱 오류: {e}") from e

    return _apply_env_overrides(ns)


CFG = _load()
