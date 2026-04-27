# =============================================================================
# vlm/api/server.py
# -----------------------------------------------------------------------------
# 기능:
#   Qwen3-VL-8B LoRA 모델을 FastAPI로 서빙하는 Multimodal API 서버.
#   thema_pa JSON + 이미지 경로를 받아 한국어 판정 리포트를 반환합니다.
#
# 엔드포인트:
#   POST /v1/report  : 도체 판정 리포트 생성
#   GET  /v1/health  : 서버 및 모델 로드 상태 확인
#   GET  /docs       : Swagger UI (자동 생성)
#
# 동작 방법:
#   # 직접 실행 (config.json의 api.host, api.port 사용)
#   python vlm/api/server.py
#
#   # uvicorn으로 실행
#   uvicorn vlm.api.server:app --reload
#
#   # 요청 예시
#   curl -X POST http://localhost:8000/v1/report \
#        -H "Content-Type: application/json" \
#        -d @vlm/schema/samples/normal_case.json
#
# 설정 (config.json):
#   api.host                  : 바인딩 호스트 (기본: 0.0.0.0)
#   api.port                  : 바인딩 포트 (기본: 8000)
#   api.allowed_origins       : CORS 화이트리스트
#   api.api_keys              : X-API-Key 인증 (빈 배열이면 비활성화)
#   api.rate_limit_per_minute : 분당 요청 제한 (기본 60)
#   api.inference_timeout_sec : 추론 타임아웃 (기본 180초)
#   paths.lora_adapter        : LoRA 어댑터 경로 (없으면 베이스 모델로 추론)
#   logging.level             : DEBUG / INFO / WARNING
#   logging.format            : json / text
#
# 보안:
#   X-API-Key: api_keys 배열에 키 등록 시 인증 필수, 없으면 401
#   Rate limit: 분당 N회 초과 시 429 (slowapi)
#   경로 traversal: result_image_path 가 image_dir 외부면 403
#
# 전제 조건:
#   GPU 필수 (RTX 4090 권장, 최소 16GB VRAM)
#   프로젝트 루트에 config.json 존재
#
# 의존성:
#   fastapi, uvicorn, pydantic>=2.0, slowapi, python-json-logger
# =============================================================================

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 sys.path 에 추가 (직접 실행 시에도 vlm.* import 가능하게)
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from vlm.api.auth import is_auth_enabled, verify_api_key
from vlm.api.schemas import ReportRequest, ReportResponse
from vlm.config import CFG
from vlm.logging_config import configure_from_config, get_logger
from vlm.schema.thema_pa_output import ThemaPAOutput, ErrorCode, BackboneSlope

# 로깅 초기화 (config.json 기반)
configure_from_config()
log = get_logger("vlm.api")

ADAPTER_PATH = Path(__file__).parent.parent.parent / CFG.paths.lora_adapter

_inference_module: Optional[object] = None
_model_ready: bool = False
_model_used: str = "not_loaded"


def _load_model_sync():
    """모델을 실제로 GPU에 올리는 동기 함수 (lifespan에서 executor로 실행)."""
    global _inference_module, _model_ready, _model_used
    from vlm.train import inference
    inference._load_model()
    _inference_module = inference
    _model_used = "lora" if ADAPTER_PATH.exists() else "base"
    _model_ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API 부팅: 모델 로드 시작")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model_sync)
    log.info("API 준비 완료", extra={"model_used": _model_used, "adapter_exists": ADAPTER_PATH.exists()})
    yield
    log.info("API 종료")


# Rate Limiter
_RATE_LIMIT = getattr(CFG.api, "rate_limit_per_minute", 60)
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{_RATE_LIMIT}/minute"])

app = FastAPI(
    title="Livestock VLM API",
    description="Qwen3-VL-8B LoRA 기반 한국어 돼지 도체 판정 리포트 API",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    log.warning("Rate limit exceeded", extra={"client": get_remote_address(request), "path": request.url.path})
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit 초과 ({_RATE_LIMIT}/min). 잠시 후 재시도하세요."},
    )


# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000, 1)
    log.info(
        f"{request.method} {request.url.path}",
        extra={
            "request_id":  request_id,
            "method":      request.method,
            "path":        request.url.path,
            "status_code": response.status_code,
            "elapsed_ms":  elapsed_ms,
            "client":      get_remote_address(request),
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


_cors_origins = getattr(CFG.api, "allowed_origins", None) or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
)

_INFERENCE_TIMEOUT = getattr(CFG.api, "inference_timeout_sec", 180)

_IMAGE_ROOT = Path(CFG.paths.image_dir).resolve()


def _validate_image_path(path: str | None) -> str | None:
    """경로 traversal 방지 — 허용된 이미지 디렉터리 하위인지만 허용."""
    if not path:
        return None
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=422, detail=f"잘못된 이미지 경로: {path}")
    if not resolved.exists():
        raise HTTPException(status_code=422, detail=f"이미지 파일 없음: {path}")
    try:
        resolved.relative_to(_IMAGE_ROOT)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"허용되지 않은 경로. image_dir({_IMAGE_ROOT}) 하위만 허용됩니다.",
        )
    return str(resolved)


def _build_thema_output(req: ReportRequest) -> ThemaPAOutput:
    validated_path = _validate_image_path(req.result_image_path)
    return ThemaPAOutput(
        carcass_no=req.carcass_no,
        slaughter_ymd=req.slaughter_ymd,
        backfat_average=req.backfat_average,
        multifidus_thk=req.multifidus_thk,
        body_length=req.body_length,
        body_width=req.body_width,
        body_weight=req.body_weight,
        gender=req.gender,
        grade=req.grade,
        error_code=ErrorCode(**req.error_code.model_dump()),
        backbone_slope=BackboneSlope(**req.backbone_slope.model_dump()),
        result_image_path=validated_path,
    )


@app.get("/v1/health")
def health():
    return {
        "status": "ready" if _model_ready else "loading",
        "model_used": _model_used,
        "adapter_exists": ADAPTER_PATH.exists(),
        "auth_enabled": is_auth_enabled(),
        "rate_limit_per_minute": _RATE_LIMIT,
    }


@app.post("/v1/report", response_model=ReportResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{_RATE_LIMIT}/minute")
async def generate_report(request: Request, req: ReportRequest):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다. /v1/health 로 상태 확인 후 재시도하세요.")

    try:
        output = _build_thema_output(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"입력 데이터 오류: {e}")

    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _inference_module.generate_report, output),
            timeout=_INFERENCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"추론 타임아웃 ({_INFERENCE_TIMEOUT}초 초과)",
        )
    elapsed = round(time.time() - t0, 2)

    log.info(
        "report generated",
        extra={
            "carcass_no":  req.carcass_no,
            "grade":       req.grade,
            "elapsed_sec": elapsed,
            "model":       _model_used,
        },
    )

    return ReportResponse(
        summary=result.get("3문장_요약", ""),
        grade_reason=result.get("비정상_근거"),
        warnings=result.get("주의사항", []),
        recommendation=result.get("권고", ""),
        model_used=f"{_model_used} ({elapsed}s)",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vlm.api.server:app", host=CFG.api.host, port=CFG.api.port, reload=True)
