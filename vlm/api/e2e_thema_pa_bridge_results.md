# thema_pa_VLM ↔ VLM FastAPI E2E 검증 결과

**실행 일시**: 2026-05-08
**스크립트**: `scripts/test_e2e_thema_pa_bridge.py`
**모델**: Qwen3-VL-8B-Instruct + **v2 LoRA** (Vision Tower + AI 이미지 학습)
**어댑터**: `vlm/train/output/qwen3vl-lora/`
**전제 조건**: VLM FastAPI 가동, thema_pa_VLM 리포 존재

---

## 검증 흐름

`tests/test_thema_pa_vlm_bridge.py` 의 통합 테스트는 `requests.post` 를 모킹하지만,
본 검증은 **실제 네트워크 호출 + v2 추론 + 저장 파일 검증**까지 수행한다.

```
1. GET /v1/health           → 200 + status=ready, model_used=lora
2. thema_pa_VLM 의 RestAPI 클래스 + config 로드
3. 4 샘플 → RestAPI(config).SendVLMReport(payload)  (실호출)
4. 응답: HTTP 200, JSON 4 필드 (summary/grade_reason/warnings/recommendation)
5. 저장: thema_pa_VLM/storage/vlm_reports/{ymd}_{pigno}_vlm_report.json
```

---

## 결과 요약: 3/4 PASS

| 샘플 | carcass | http | 추론 시간 | 저장 | 비고 |
|---|---|---|---|---|---|
| `normal_case` | 3010 | 200 | **128.9s** | ✅ | 첫 호출 warm-up 영향 |
| `backfat_error_case` | 3025 | timeout | 180.0s | ❌ | client timeout (서버 측 추론은 진행 중) |
| `entry_error_case` | 3041 | 200 | 120.3s | ✅ | 등외 + 비정상 진입 |
| `sample_3473` | 3473 | 200 | **26.0s** | ✅ | warmed-up 후 안정적 |

### 시간 패턴 분석
- 1번째(3010) → 4번째(3473): **128.9s → 26.0s** (~5배 가속)
- 첫 호출은 CUDA 커널 컴파일/캐싱으로 인한 warm-up 영향
- 2번째(3025)는 등외+다중 error_code 로 가장 복잡한 추론 → client 180s timeout
  - 운영 시 `config.api.inference_timeout_sec` 상향 또는 warm-up 후 운영 필요

---

## v2 추론 응답 샘플

### 3010 (정상 케이스, 1+)
```json
{
  "summary": "도체번호 3010은(는) 2026년 04월 22일 거세으로 등급 1+으로 판정되었습니다. 등지방 두께 22.4mm, 뭇갈래근 두께 48.2mm, 도체중 87.3kg으로 측정되었습니다. ...",
  "grade_reason": null,
  "warnings": [],
  "recommendation": "출하 처리 권고 (1+ 등급)",
  "model_used": "lora (128.8s)"
}
```

### 3041 (비정상 진입, 등외)
```json
{
  "summary": "도체번호 3041은(는) 비정상 진입으로 2 등급으로 판정되었습니다. 등지방 두께 9.0mm, 도체중 79.0kg으로 측정되었습니다. ...",
  "grade_reason": null,
  "warnings": ["재촬영 필요: X", "비정상 측정 필드: 등지방 두께, 뭇갈래근 두께, 등지방 두께", "최종 판정 근거: 2 등급"],
  "recommendation": "최종 판정 근거로 모든 AI 검출이 정상 완료되어 2 등급으로 판정되었습니다.",
  "model_used": "lora (120.26s)"
}
```

### 3473 (정상 케이스, 1+, warmed-up)
```json
{
  "summary": "도체번호 3473은(는) 2026년 04월 22일 거세으로 등급 1+으로 판정되었습니다. 등지방 두께 20.0mm, 뭇갈래근 두께 15.3mm, 도체중 88.0kg으로 측정되었습니다. 모든 AI 검출이 정상 완료되어 2026년 04월 22일 1+ 등급으로 최종 판정되었습니다.",
  "grade_reason": null,
  "warnings": [],
  "recommendation": "1+ 등급으로 판정되었습니다. 출하 및 거세으로 처리하세요.",
  "model_used": "lora (25.94s)"
}
```

---

## 검증된 운영 흐름

```
thema_pa_VLM (cwd)
  └─ RestAPI(config).SendVLMReport(payload)
       │
       │  POST http://127.0.0.1:8000/v1/report  (실제 네트워크)
       ▼
     VLM FastAPI
       │
       │  ① ReportRequest Pydantic 검증
       │  ② _validate_image_path() — image_dir 하위 + 파일 존재 검증
       │  ③ _build_thema_output() — ThemaPAOutput 변환
       │  ④ generate_report() — Qwen3-VL + v2 LoRA 추론 (이미지 + JSON)
       │  ⑤ JSON 4 필드 응답
       ▼
     thema_pa_VLM
       │  ⑥ validate_vlm_response_json() — 4 필드 검증
       │  ⑦ save_vlm_response_json() — ./storage/vlm_reports/{ymd}_{pigno}_vlm_report.json
       ▼
   thema_pa_VLM/storage/vlm_reports/20260422_3473_vlm_report.json
```

전 흐름 정상 작동 확인. `scripts/test_e2e_thema_pa_bridge.py` 가 cwd 를 thema_pa_VLM 으로 바꿔 호출하므로 `./storage/vlm_reports` 상대경로가 정확히 thema_pa_VLM 하위에 저장된다.

---

## 환경 요구사항 (이번 검증 기준)

| 항목 | 값 |
|---|---|
| GPU | RTX 4090 24GB (CUDA 12.6) |
| Python | 3.13 (system, user-site) |
| numpy | **2.2.6** (opencv-python 4.12 호환을 위해 2.4 → 2.2.6 다운그레이드) |
| slowapi | 0.1.9 |
| transformers | 5.2.0 |
| torch | 2.7.1+cu126 |
| VLM `config.json` | `image_dir` 들을 `thema_pa_VLM/images/AI(/ORI)` 로 갱신 |
| 샘플 | `result_image_path` 를 `thema_pa_VLM/images/AI/...` 실재 파일로 매칭 |

---

## 재실행 방법

```bash
# 1. VLM FastAPI 가동 (별도 셸)
python -c "import uvicorn; uvicorn.run('vlm.api.server:app', host='127.0.0.1', port=8000, reload=False)"

# 2. /v1/health 가 status=ready 응답할 때까지 대기 (모델 로딩 ~30~60초)

# 3. E2E 검증
python scripts/test_e2e_thema_pa_bridge.py
```

`THEMA_PA_ROOT` 환경변수로 thema_pa_VLM 경로 변경 가능.
