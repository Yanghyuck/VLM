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
import hashlib
import sys
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 sys.path 에 추가 (직접 실행 시에도 vlm.* import 가능하게)
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

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


def _warmup_model_sync():
    """첫 추론의 CUDA 커널 컴파일/캐싱 비용을 startup 으로 옮겨 사용자 첫 호출을 안정화.

    E2E 측정 (RTX 4090, v2 LoRA): 첫 호출 ~130s → warmed-up ~25s. 이미지 없는 dummy 입력으로
    1회 generate 만 하면 이후 사용자 요청은 안정적인 추론 시간을 가진다. 실패해도 무시.
    """
    if not _model_ready or _inference_module is None:
        return
    try:
        dummy = ThemaPAOutput(
            carcass_no=0,
            slaughter_ymd="20260101",
            backfat_average=20.0,
            multifidus_thk=15.0,
            body_length=70.0,
            body_width=30.0,
            body_weight=85.0,
            gender=3,
            grade="1+",
            error_code=ErrorCode(),
            backbone_slope=BackboneSlope(has_large_slope=False),
            result_image_path=None,
        )
        t0 = time.time()
        _inference_module.generate_report(dummy)
        log.info(f"warm-up 추론 완료 ({time.time() - t0:.1f}s)")
    except Exception as e:
        log.warning(f"warm-up 추론 실패 (무시 가능): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API 부팅: 모델 로드 시작")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model_sync)
    log.info("API 준비 완료", extra={"model_used": _model_used, "adapter_exists": ADAPTER_PATH.exists()})

    if getattr(CFG.api, "warmup_on_startup", True):
        log.info("warm-up 추론 시작 (CUDA 커널 캐싱)")
        await loop.run_in_executor(None, _warmup_model_sync)

    METRIC_MODEL_READY.set(1 if _model_ready else 0)
    yield
    METRIC_MODEL_READY.set(0)
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


# 요청 로깅 + metrics 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    elapsed_ms = round(elapsed * 1000, 1)
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
    # /metrics 자체는 측정에서 제외 (자기 호출 제외)
    if request.url.path != "/metrics":
        METRIC_REQUESTS.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
        METRIC_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
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

# ---- Prometheus metrics ----
# 직접 실행(`python vlm/api/server.py`) 시 모듈이 __main__ + vlm.api.server 두 번 import 되어
# default REGISTRY 에 중복 등록되는 문제 회피 — 전용 CollectorRegistry 사용.
METRIC_REGISTRY = CollectorRegistry()
METRIC_REQUESTS = Counter(
    "vlm_requests_total",
    "VLM API 요청 수",
    labelnames=("endpoint", "status"),
    registry=METRIC_REGISTRY,
)
METRIC_LATENCY = Histogram(
    "vlm_request_duration_seconds",
    "VLM API 요청 처리 시간 (초)",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 240),
    registry=METRIC_REGISTRY,
)
METRIC_INFERENCE = Histogram(
    "vlm_inference_duration_seconds",
    "VLM 추론 본체 시간 (캐시 hit 제외)",
    buckets=(1, 2, 5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 180, 240),
    registry=METRIC_REGISTRY,
)
METRIC_CACHE = Counter(
    "vlm_cache_total",
    "응답 캐시 hit/miss",
    labelnames=("result",),
    registry=METRIC_REGISTRY,
)
METRIC_MODEL_READY = Gauge("vlm_model_ready", "모델 로드 완료 (1/0)", registry=METRIC_REGISTRY)
METRIC_CACHE_SIZE = Gauge("vlm_cache_size", "현재 캐시된 응답 수", registry=METRIC_REGISTRY)


# ---- 응답 캐시 (in-memory LRU) ----
# 같은 payload(+이미지 mtime/size) 재호출을 즉시 반환. 운영에선 도체별 unique 라
# hit 가 적지만 디버깅/재시도/E2E 반복에 유용.
_CACHE_MAX = int(getattr(CFG.api, "response_cache_max", 256))
_response_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = asyncio.Lock()
_cache_hits = 0
_cache_misses = 0


def _cache_key(req: "ReportRequest", validated_path: Optional[str]) -> str:
    # 원본 result_image_path 는 슬래시 차이(\ vs /)로 같은 파일도 다른 key 가 되므로
    # payload 에서 빼고 정규화된 validated_path + 파일 mtime/size 만 image_meta 로 포함.
    payload = req.model_dump()
    payload.pop("result_image_path", None)
    payload["__image_meta"] = None
    if validated_path:
        try:
            st = Path(validated_path).stat()
            payload["__image_meta"] = [
                str(Path(validated_path).resolve()).replace("\\", "/"),
                st.st_mtime_ns,
                st.st_size,
            ]
        except OSError:
            pass
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


async def _cache_get(key: str) -> Optional[dict]:
    global _cache_hits, _cache_misses
    async with _cache_lock:
        if key in _response_cache:
            _response_cache.move_to_end(key)
            _cache_hits += 1
            return _response_cache[key]
        _cache_misses += 1
        return None


async def _cache_put(key: str, value: dict) -> None:
    async with _cache_lock:
        _response_cache[key] = value
        _response_cache.move_to_end(key)
        while len(_response_cache) > _CACHE_MAX:
            _response_cache.popitem(last=False)


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
    total = _cache_hits + _cache_misses
    return {
        "status": "ready" if _model_ready else "loading",
        "model_used": _model_used,
        "adapter_exists": ADAPTER_PATH.exists(),
        "auth_enabled": is_auth_enabled(),
        "rate_limit_per_minute": _RATE_LIMIT,
        "cache": {
            "size": len(_response_cache),
            "max": _CACHE_MAX,
            "hits": _cache_hits,
            "misses": _cache_misses,
            "hit_ratio": round(_cache_hits / total, 3) if total else 0.0,
        },
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

    cache_key = _cache_key(req, output.result_image_path)
    cached = await _cache_get(cache_key)
    if cached is not None:
        METRIC_CACHE.labels(result="hit").inc()
        log.info("report cache hit", extra={
            "carcass_no": req.carcass_no, "cache_key": cache_key[:8],
        })
        cached_view = dict(cached)
        cached_view["model_used"] = f"{cached_view.get('model_used', _model_used)} (cached)"
        return ReportResponse(**cached_view)
    METRIC_CACHE.labels(result="miss").inc()

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
    inference_time = time.time() - t0
    METRIC_INFERENCE.observe(inference_time)
    elapsed = round(inference_time, 2)

    log.info(
        "report generated",
        extra={
            "carcass_no":  req.carcass_no,
            "grade":       req.grade,
            "elapsed_sec": elapsed,
            "model":       _model_used,
        },
    )

    response = ReportResponse(
        summary=result.get("3문장_요약", ""),
        grade_reason=result.get("비정상_근거"),
        warnings=result.get("주의사항", []),
        recommendation=result.get("권고", ""),
        model_used=f"{_model_used} ({elapsed}s)",
    )
    await _cache_put(cache_key, response.model_dump())
    METRIC_CACHE_SIZE.set(len(_response_cache))
    return response


@app.get("/metrics")
def metrics():
    METRIC_MODEL_READY.set(1 if _model_ready else 0)
    METRIC_CACHE_SIZE.set(len(_response_cache))
    return PlainTextResponse(generate_latest(METRIC_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/report/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{_RATE_LIMIT}/minute")
async def stream_report(request: Request, req: ReportRequest):
    """NDJSON streaming. 각 줄이 독립 JSON.

    형식:
      {"event":"start","carcass_no":...}\\n
      {"event":"token","text":"..."}\\n   (반복)
      {"event":"done","result":{...},"elapsed_sec":...}\\n
    """
    if not _model_ready:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다. /v1/health 로 상태 확인 후 재시도하세요.")

    try:
        output = _build_thema_output(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"입력 데이터 오류: {e}")

    async def event_stream():
        loop = asyncio.get_event_loop()
        t0 = time.time()
        accumulated: list[str] = []

        yield json.dumps({"event": "start", "carcass_no": req.carcass_no}, ensure_ascii=False) + "\n"

        sync_gen = _inference_module.stream_response(output)

        def _next_chunk():
            try:
                return next(sync_gen)
            except StopIteration:
                return None

        try:
            while True:
                chunk = await asyncio.wait_for(
                    loop.run_in_executor(None, _next_chunk),
                    timeout=_INFERENCE_TIMEOUT,
                )
                if chunk is None:
                    break
                accumulated.append(chunk)
                yield json.dumps({"event": "token", "text": chunk}, ensure_ascii=False) + "\n"
        except asyncio.TimeoutError:
            yield json.dumps({"event": "error", "detail": f"추론 타임아웃 ({_INFERENCE_TIMEOUT}초)"}, ensure_ascii=False) + "\n"
            return

        elapsed = round(time.time() - t0, 2)
        full_text = "".join(accumulated)

        # 후처리
        from vlm.train.inference import _extract_json
        from vlm.postprocess import apply_postprocess
        try:
            parsed = _extract_json(full_text)
            parsed = apply_postprocess(
                parsed,
                expected_grade=req.grade,
                expected_gender=output.gender.label(),
            )
        except Exception as e:
            yield json.dumps({"event": "error", "detail": f"후처리 실패: {e}", "raw": full_text}, ensure_ascii=False) + "\n"
            return

        result = {
            "summary":        parsed.get("3문장_요약", ""),
            "grade_reason":   parsed.get("비정상_근거"),
            "warnings":       parsed.get("주의사항", []),
            "recommendation": parsed.get("권고", ""),
            "model_used":     f"{_model_used} ({elapsed}s)",
        }
        log.info("stream report generated", extra={
            "carcass_no": req.carcass_no, "grade": req.grade,
            "elapsed_sec": elapsed, "model": _model_used, "stream": True,
        })
        yield json.dumps({"event": "done", "result": result, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import os

    import uvicorn

    # 기본 reload=False (운영 안정 + watchfiles 가 server_run.log 변화로 무한 reload 트리거 + prometheus
    # Counter 중복 등록 방지). dev 시 VLM_API_RELOAD=1 로 강제.
    reload = os.environ.get("VLM_API_RELOAD", "0") == "1"
    uvicorn.run("vlm.api.server:app", host=CFG.api.host, port=CFG.api.port, reload=reload)
