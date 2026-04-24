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
#   api.host             : 바인딩 호스트 (기본: 0.0.0.0)
#   api.port             : 바인딩 포트 (기본: 8000)
#   paths.lora_adapter   : LoRA 어댑터 경로 (없으면 베이스 모델로 추론)
#
# 전제 조건:
#   GPU 필수 (RTX 4090 권장, 최소 16GB VRAM)
#   프로젝트 루트에 config.json 존재
#
# 의존성:
#   fastapi, uvicorn, pydantic>=2.0
# =============================================================================

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from vlm.api.schemas import ReportRequest, ReportResponse
from vlm.schema.thema_pa_output import ThemaPAOutput, ErrorCode, BackboneSlope
from vlm.config import CFG

ADAPTER_PATH = Path(__file__).parent.parent.parent / CFG.paths.lora_adapter

_inference_module: Optional[object] = None
_model_used: str = "not_loaded"


def _load_inference():
    global _inference_module, _model_used
    if _inference_module is not None:
        return
    from vlm.train import inference
    _inference_module = inference
    _model_used = "lora" if ADAPTER_PATH.exists() else "base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_inference()
    yield


app = FastAPI(
    title="Livestock VLM API",
    description="Qwen3-VL-8B LoRA 기반 한국어 돼지 도체 판정 리포트 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_thema_output(req: ReportRequest) -> ThemaPAOutput:
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
        result_image_path=req.result_image_path,
    )


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "model_used": _model_used,
        "adapter_exists": ADAPTER_PATH.exists(),
    }


@app.post("/v1/report", response_model=ReportResponse)
def generate_report(req: ReportRequest):
    if _inference_module is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다. 잠시 후 재시도하세요.")

    try:
        output = _build_thema_output(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"입력 데이터 오류: {e}")

    t0 = time.time()
    result = _inference_module.generate_report(output)
    elapsed = round(time.time() - t0, 2)

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
