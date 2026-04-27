# =============================================================================
# vlm/api/auth.py
# -----------------------------------------------------------------------------
# 기능:
#   X-API-Key 헤더 검증. config.json > api.api_keys 에 등록된 키만 허용.
#   빈 배열일 때는 인증 비활성화 (개발 모드).
#
# 사용:
#   from fastapi import Depends
#   from vlm.api.auth import verify_api_key
#
#   @app.post("/v1/report", dependencies=[Depends(verify_api_key)])
#   async def generate_report(...): ...
# =============================================================================

from __future__ import annotations

from fastapi import Header, HTTPException

from vlm.config import CFG


_API_KEYS: set[str] = set(getattr(CFG.api, "api_keys", []) or [])


def is_auth_enabled() -> bool:
    return len(_API_KEYS) > 0


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """FastAPI Depends 용 검증자.

    - api_keys 가 빈 배열이면 인증 스킵 (개발 모드)
    - 헤더 없으면 401
    - 등록되지 않은 키면 401
    """
    if not _API_KEYS:
        return True
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key 헤더가 필요합니다.")
    if x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="유효하지 않은 API key 입니다.")
    return True
