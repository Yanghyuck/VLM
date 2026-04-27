# =============================================================================
# vlm/logging_config.py
# -----------------------------------------------------------------------------
# 기능:
#   Python logging 을 JSON 또는 텍스트 형식으로 설정합니다.
#   FastAPI / Streamlit / 학습 스크립트 등에서 공통 호출.
#
# 사용:
#   from vlm.logging_config import configure_logging, get_logger
#   configure_logging()
#   log = get_logger(__name__)
#   log.info("판정 완료", extra={"carcass_no": 3010, "elapsed_ms": 22.9})
#
# 설정 (config.json > logging):
#   level   : "DEBUG" | "INFO" | "WARNING" | "ERROR"
#   format  : "json"  | "text"
# =============================================================================

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """structured JSON 로그 포맷."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
        }
        # 표준 LogRecord 속성 외 extra 로 들어온 필드 모두 포함
        skip = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "taskName",
        }
        for key, val in record.__dict__.items():
            if key in skip:
                continue
            try:
                json.dumps(val)
                data[key] = val
            except TypeError:
                data[key] = str(val)

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "json"):
    """Python logging 전역 설정.

    Args:
        level: "DEBUG" / "INFO" / "WARNING" / "ERROR"
        fmt:   "json" / "text"
    """
    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper() if isinstance(level, str) else level)


def configure_from_config():
    """config.json 의 logging 섹션 기반으로 자동 설정."""
    try:
        from vlm.config import CFG
        level = getattr(CFG, "logging", None)
        if level:
            lvl = getattr(level, "level", "INFO")
            fmt = getattr(level, "format", "json")
            configure_logging(level=lvl, fmt=fmt)
            return
    except Exception:
        pass
    configure_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
