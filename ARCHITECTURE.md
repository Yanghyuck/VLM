# 한국어 축산 AI 판정 코파일럿 — 아키텍처

> 최종 업데이트: 2026-04-27
> 기반 코드: `thema_pa` (YOLOv11 기반 돼지 도체 분석 시스템)
> 메인 엔진: **Qwen3-VL-8B-Instruct + LoRA 파인튜닝** (로컬 추론)

---

## 1. 전체 그림

```
┌─────────────────────────────────────────────────────────────────┐
│                        현장 카메라 / RFID                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 이미지 + 도체번호
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         thema_pa (기존)                          │
│  YOLOv11 seg → PostProcess → GradeCalc → DB / REST / MQTT       │
│                                                                  │
│  출력 JSON 예시 (ThemaPAOutput):                                 │
│  {                                                               │
│    "carcass_no": 3010,                                           │
│    "backfat_average": 22,        # mm                            │
│    "multifidus_thk": 48,         # mm                            │
│    "body_length": 73.6, "body_width": 34.1,   # cm              │
│    "gender": 3,  "grade": "1+",                                  │
│    "error_code": { "AI_BackFat_error": 0, ... },                 │
│    "backbone_slope": {"has_large_slope": false, ...},            │
│    "result_image_path": "..."                                    │
│  }                                                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ structured JSON + result image
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│  PROJECT 1      │  │  PROJECT 3      │  │  PROJECT 2           │
│  QA Copilot     │  │  Multimodal API │  │  VLM Benchmark       │
│  Streamlit UI   │  │  FastAPI /v1    │  │  LoRA vs Base        │
└────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘
         │                    │                       │
         └────────────────────┴───────────────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │ vlm/train/inference.py │
                   │ Qwen3-VL-8B + LoRA     │
                   └────────────────────────┘
```

**중요한 설계 결정 (2026-04-24 전환)**:
초기에는 Claude API 기반으로 설계했으나, API 비용 및 네트워크 의존성 제거를 위해
**로컬 Qwen3-VL LoRA 파인튜닝**으로 전환. 세 프로젝트 모두 `vlm/train/inference.py` 를 공유합니다.

---

## 2. 공통 레이어 (세 프로젝트 공유)

### 2.1 스키마 — `vlm/schema/thema_pa_output.py`

thema_pa JSON을 Pydantic v2 모델로 정의. 세 프로젝트가 모두 import.

```python
class ThemaPAOutput(BaseModel):
    carcass_no: int
    slaughter_ymd: str
    backfat_average: float      # mm
    multifidus_thk: float       # mm
    body_length: float          # cm
    body_width: float           # cm
    body_weight: float          # kg
    gender: Gender              # 1=암, 2=수, 3=거세
    grade: str                  # '1+', '1', '2', '등외'
    error_code: ErrorCode       # 6개 AI 검출 플래그
    backbone_slope: BackboneSlope
    result_image_path: Optional[str]
```

### 2.2 중앙 설정 — `config.json` + `vlm/config.py`

모든 환경 변동값(DB 비밀번호, 경로, 모델 ID, 포트 등)을 `config.json` 한 곳에 집중.
`vlm/config.py` 가 로드해 `CFG` 객체로 제공.

```python
from vlm.config import CFG
CFG.db.host                  # MySQL 접속
CFG.paths.lora_adapter       # LoRA 어댑터 경로
CFG.model.base_model_id      # Qwen/Qwen3-VL-8B-Instruct
CFG.api.allowed_origins      # CORS 화이트리스트
CFG.grade.backfat_range      # 등급별 정상 범위
```

`config.json` 은 `.gitignore` 로 제외되며 실제 값은 `config.example.json` 을 복사해 생성.

### 2.3 로컬 추론 — `vlm/train/inference.py`

세 프로젝트가 공유하는 추론 진입점.

```python
def generate_report(output: ThemaPAOutput) -> dict:
    """ThemaPAOutput → 한국어 판정 리포트 dict."""
    # 1. 모델 lazy load (첫 호출 시)
    # 2. error_code 에 따라 프롬프트 템플릿 선택
    # 3. 이미지 있으면 멀티모달, 없으면 텍스트만
    # 4. Qwen3-VL 생성 → JSON 파싱 (brace-balanced parser)
    # 5. dict 반환
```

반환 형식:
```json
{
  "3문장_요약": "도체번호 3010은 거세 수컷으로 ...",
  "비정상_근거": null,
  "주의사항": ["체폭/체장 정상", "척추 기울기 정상"],
  "권고": "등지방 기준 범위 상단 근접. 다음 개체 주의 관찰."
}
```

### 2.4 프롬프트 템플릿 — `vlm/prompt/`

| 템플릿 | 사용 조건 |
|---|---|
| `system_prompt.txt` | 항상 system 메시지로 포함 (도메인 지식) |
| `normal_case.txt` | `error_code` 모두 0 |
| `failure_analysis.txt` | `pig_RightEntry=1` 또는 `AI_BackFat_error=1` |
| `error_case.txt` | 그 외 error_code 비정상 |

---

## 3. Project 1 — Explainable Livestock QA Copilot

### 목표
thema_pa의 구조화 JSON과 결과 이미지를 입력받아,
**현장 작업자가 이해할 수 있는 한국어 판정 요약과 원인 분석**을 자동 생성.

### 구현

```
vlm/demo/app.py — Streamlit 3패널 데모
  ┌─────────────────┬──────────────────────┬─────────────────┐
  │ 왼쪽            │ 가운데               │ 오른쪽          │
  │ - 도체 이미지   │ - 3문장 요약         │ - 등급 배지     │
  │ - JSON 업로드   │ - 비정상 근거        │ - 측정값 테이블 │
  │ - 샘플 선택     │ - 주의사항 리스트    │ - AI 검출 상태  │
  │                 │ - 권고               │                 │
  └─────────────────┴──────────────────────┴─────────────────┘
```

### 파일 구성

```
vlm/
├── schema/thema_pa_output.py    # 공통 Pydantic
├── prompt/*.txt                 # 4개 템플릿
├── train/inference.py           # Qwen3-VL LoRA 추론
├── demo/app.py                  # Streamlit 데모
└── report/generator.py          # inference.py 위임 shim (하위 호환)
```

### 사용자 흐름

1. 사이드바에서 샘플 JSON 선택 또는 업로드
2. (옵션) 이미지 업로드 → `vlm/data/tmp/{uuid}.jpg` 로 저장 (1시간 TTL)
3. "판정 실행" 버튼 → `inference.generate_report()` 호출
4. 3패널에 결과 표시

---

## 4. Project 2 — Korean Livestock VLM Benchmark (4주차 예정)

### 목표
thema_pa 수치를 ground truth 로 삼아 **LoRA 파인튜닝 모델 vs 베이스 모델**을
50~100건에 대해 비교 측정. 한국어 축산 VLM 최초 벤치마크.

### 평가 태스크 3종

```
Task A — 이미지 기반 한국어 설명 (Captioning)
  지표: ROUGE-L, BERTScore (ko)

Task B — 진단 Reasoning (Chain-of-Thought QA)
  평가: 수동 루브릭 (5점 × 정확성 · 완결성 · 현장 가독성)

Task C — 실패 케이스 해설 (Error Analysis)
  지표: error_code 기반 원인 식별 정확도
```

### 예상 구조

```
vlm/bench/
├── dataset/                   # 평가용 샘플 (train 제외)
│   ├── normal/                # error_code 모두 0 (30건)
│   ├── backfat_fail/          # AI_BackFat_error=1 (15건)
│   ├── outline_fail/          # AI_Outline_error=1 (10건)
│   └── entry_fail/            # pig_RightEntry=1 (10건)
├── annotations/               # 수동 검증 레퍼런스
├── runner.py                  # 베이스 vs LoRA 비교 실행
└── scorer.py                  # ROUGE / BERTScore / 수동 루브릭 집계
```

---

## 5. Project 3 — Multimodal Inspection API

### 목표
Qwen3-VL LoRA 추론을 **FastAPI REST 엔드포인트**로 서빙.
thema_pa 기존 REST/MQTT 흐름에 VLM 설명 레이어 추가.

### 엔드포인트

```
GET  /v1/health
  공개. 응답: {
    "status": "ready"|"loading",
    "model_used": "lora"|"base",
    "adapter_exists": bool,
    "auth_enabled": bool,
    "rate_limit_per_minute": int
  }

POST /v1/report
  보안: X-API-Key 헤더 검증 (api_keys 등록 시)
  Rate limit: config.api.rate_limit_per_minute (default 60/min)
  요청: ThemaPAOutput 호환 JSON
  응답: {
    "summary": "3문장 요약",
    "grade_reason": "비정상 근거 (optional)",
    "warnings": ["주의사항1", ...],
    "recommendation": "권고",
    "model_used": "lora (3.2s)"
  }
  응답 헤더: X-Request-ID (UUID8, 분산 추적)

GET  /docs  — Swagger UI 자동 생성
```

### 보안 & 안정성 장치 (7가지)

| 항목 | 구현 | 응답 코드 |
|---|---|---|
| **X-API-Key 인증** | `vlm/api/auth.py` 의 `verify_api_key` Depends | 401 |
| **Rate limiting** | `slowapi` Limiter, 분당 N회 | 429 |
| **추론 타임아웃** | `asyncio.wait_for(timeout=180)` | 504 |
| **경로 traversal 차단** | `result_image_path` 를 `image_dir` 하위로 제한 | 403 |
| **CORS 화이트리스트** | `config.api.allowed_origins` | — |
| **이벤트 루프 블로킹 방지** | `run_in_executor` 로 추론 위임 | — |
| **모델 로드 상태 표시** | `_model_ready` 플래그 → `/v1/health` 반영 | 503 |

### 관측성 (Observability)

| 항목 | 구현 |
|---|---|
| **JSON 구조적 로그** | `vlm/logging_config.py` 의 `JsonFormatter` |
| **요청 추적** | `X-Request-ID` (UUID8, 응답 헤더 + 로그) |
| **자동 기록 필드** | request_id, method, path, status_code, elapsed_ms, client |
| **추론 결과 로그** | carcass_no, grade, elapsed_sec, model |
| **설정** | `config.logging.level` / `config.logging.format` (`json`\|`text`) |

### 파일 구성

```
vlm/api/
├── __init__.py
├── schemas.py    # ReportRequest / ReportResponse Pydantic
├── auth.py       # X-API-Key 검증 (Depends 용)
└── server.py     # FastAPI 앱 + lifespan + 미들웨어 + 엔드포인트

vlm/
└── logging_config.py  # JSON / text 로거 설정 (서버·학습 공용)
```

### 실행

```bash
python vlm/api/server.py           # config.json 의 host/port 사용
uvicorn vlm.api.server:app --reload
```

### 호출 예시

```bash
# api_keys 미등록 시 (개발 모드)
curl -X POST http://localhost:8000/v1/report \
     -H "Content-Type: application/json" \
     -d @vlm/schema/samples/normal_case.json

# api_keys 등록 후 (운영 모드)
curl -X POST http://localhost:8000/v1/report \
     -H "Content-Type: application/json" \
     -H "X-API-Key: my-secret-key-123" \
     -d @vlm/schema/samples/normal_case.json
```

---

## 6. 학습 파이프라인

### 이미지 소스

`thema_pa/images/` 하위 두 폴더:

| 폴더 | 패턴 | 개수 | 용도 |
|---|---|---|---|
| `ORI/` | `0716_ori_{시각}_{pigno}_SP_CAM7.jpg` | 3,355 | 원본 도체 사진 |
| `AI/` | `0716_ai_{시각}_{pigno}_{pigno+offset}_SP_CAM7.jpg` | 3,473 | AI 분석 결과 오버레이 (등지방/뭇갈래근/척추 시각화) — **v2 학습 사용** |

`config.json` 의 `paths.image_dir` 가 학습 시 사용할 폴더를 결정합니다.

### 데이터 흐름

```
thema_pa MySQL DB (tb_act_result)            thema_pa/images/AI/
        │                                            │
        ├── pigno_cnt                                │ filename:
        ├── act_backfat_thk, act_centhk              │   0716_ai_{ts}_{pigno}_{pigno+N}_SP_CAM7.jpg
        └── act_grade                                │
                   │                                 │
                   └─────────────┬───────────────────┘
                                 │ pigno 매칭
                                 ▼
                   scripts/build_dataset.py
                     - DB 조회 (grade IS NOT NULL)
                     - 이미지 디렉터리 스캔 (FILENAME_RE_AI / RE_ORI)
                     - ThemaPAOutput 생성 + 태스크 프롬프트 할당
                                 │
                                 ▼
                   vlm/data/dataset.jsonl  (원본 1건 = 1 JSON 라인)
                                 │
                                 ▼
                   vlm/bench/dataset.py (벤치마크 평가셋 50건 추출)
                                 │
                                 ▼
                   vlm/bench/eval_set.jsonl  (held-out 50건)
                                 │
                                 ▼
                   vlm/train/convert_dataset.py --exclude-eval-set ...
                     - summary (모든 레코드)
                     - grade (모든 레코드)
                     - abnormal (error_code 비정상만)
                     - 평가셋 도체번호는 학습에서 제외
                                 │
                                 ▼
                   vlm/data/livestock_train.json  (ShareGPT, 6,610 샘플)
                                 │
                                 ▼
                   LLaMA-Factory 학습 (qwen3vl_lora_v2.yaml)
                                 │
                                 ▼
                   vlm/train/output/qwen3vl-lora/  (LoRA 어댑터)
```

### 학습 버전 비교

| 항목 | v1 (text-only LoRA) | **v2 (Vision LoRA + AI)** |
|---|---|---|
| 학습 대상 | LLM LoRA만 | **LLM + Vision Tower + Projector LoRA** |
| 이미지 소스 | ORI | **AI** (시각 오버레이) |
| 학습 데이터 | 6,710 (전체) | **6,610** (평가셋 50건 제외) |
| `freeze_vision_tower` | true (default) | **false** |
| `freeze_multi_modal_projector` | true (default) | **false** |
| `image_max_pixels` | 200,704 | 100,352 (메모리 보전) |
| `gradient_accumulation_steps` | 16 | 32 (effective batch 동일) |
| 학습 시간 | 12.6h | ~16~18h |
| 어댑터 위치 | `qwen3vl-lora-v1-textonly/` (백업) | `qwen3vl-lora/` |
| 평가셋 노출 여부 | ⚠️ 학습 시 봤음 | ✅ held-out (한 번도 안 봄) |

### 학습 실행

```bash
# 1. DB → JSONL (AI 이미지 매칭)
python scripts/build_dataset.py

# 2. 평가셋 50건 추출
python vlm/bench/dataset.py --source jsonl --n 50

# 3. JSONL → ShareGPT JSON (평가셋 50건 제외)
python vlm/train/convert_dataset.py --exclude-eval-set vlm/bench/eval_set.jsonl
cp vlm/data/livestock_train.json $LLAMAFACTORY_DATA_DIR/

# 4. LoRA 학습 (v2: Vision + AI)
llamafactory-cli train vlm/train/qwen3vl_lora_v2.yaml
```

### 성능 최적화 이력

| 이슈 | 조치 | 결과 |
|---|---|---|
| 초기 속도 254s/step (79시간 예상) | `image_max_pixels` 1,003,520 → 200,704 | 38s/step (12시간) |
| CP949 인코딩 에러 (`✓`, `→`, `—`) | 학습 문자열 ASCII 화 | 에러 해소 |
| Windows 멀티프로세싱 느림 | `dataloader_num_workers=0` | 안정화 |
| Vision frozen 시 시각 학습 효과 부재 | v2: `freeze_vision_tower: false` | Vision LoRA 추가 |
| 평가셋 학습 노출 (in-distribution) | v2: `--exclude-eval-set` 옵션으로 50건 held-out | 공정한 벤치마크 가능 |

---

## 7. 테스트 전략

**6개 파일, 31개 테스트, 모두 통과 (2026-04-27 기준)**

```
tests/
├── test_schema.py          # ThemaPAOutput 유효성 (5)
├── test_api_schemas.py     # ReportRequest/Response (7)
├── test_config.py          # config.py 로더 (3)
├── test_json_extraction.py # brace-balanced JSON parser (8)
├── test_auth.py            # X-API-Key 인증 (4, asyncio)
└── test_logging.py         # JSON 구조적 로깅 (4)
```

```bash
pytest tests/
```

`pytest.ini` 에 `asyncio_mode=auto` + `markers=integration` 등록.
통합 테스트는 기본 실행에서 제외 (`-m "not integration"`).

### CI/CD

- GitHub Actions (`.github/workflows/ci.yml`)
- Python 3.11 + 3.13 매트릭스
- `vlm/train/json_utils.py` 로 JSON 파서 분리하여 torch 미설치 환경에서도 모든 단위 테스트 통과
- 로컬 환경과 동일하게 31/31 통과

---

## 8. 기술 스택

| 레이어 | 선택 | 이유 |
|---|---|---|
| VLM | **Qwen3-VL-8B-Instruct** + LoRA | 한국어 성능, RTX 4090 GPU 메모리 적합, 오픈 모델 |
| 학습 프레임워크 | LLaMA-Factory | VLM 파인튜닝 지원, ShareGPT 포맷 호환 |
| PEFT | LoRA (rank=64, α=128) | 가중치 1.95%만 학습 (174M/8.9B) |
| 데모 UI | Streamlit | 빠른 3패널 레이아웃, `@st.cache_resource` |
| API 서버 | FastAPI + Uvicorn | 비동기 지원, Swagger 자동, Pydantic 통합 |
| 스키마 | Pydantic v2 | thema_pa JSON → 타입 안전 |
| DB | MySQL 8.x (`ai_grade_judg_dvlp`) | thema_pa 기존 DB 재사용 |
| 평가 | ROUGE-L + BERTScore (ko) | 한국어 NLG 표준 지표 |
| 테스트 | pytest + pytest-asyncio | testpaths, markers, asyncio_mode |
| **인증** | X-API-Key 헤더 | 운영 보안 |
| **Rate limit** | slowapi | 분당 N회 제한, 429 응답 |
| **로깅** | python-json-logger | request_id, latency 자동 |
| **CI** | GitHub Actions | Python 3.11/3.13 매트릭스 |
| **컨테이너** | Docker + docker-compose | NVIDIA GPU + 볼륨 마운트 |
| 설정 | JSON + SimpleNamespace 로더 | 외부 의존성 없음 |

---

## 9. 브랜치 전략

| 브랜치 | 역할 |
|---|---|
| `main` | 1주차 결과 (스키마 + DB/이미지 매칭) |
| `claude` | 2주차 Claude API 버전 (포기된 경로, 참고용) |
| `local-vlm-train` | **메인 개발 브랜치** — 2주차 + 3주차 + 로컬 Qwen3-VL LoRA |

---

## 10. 세 프로젝트 연결 구조

```
┌──────────────────────────────────────────────────────────────┐
│  PORTFOLIO STORY                                             │
│                                                              │
│  thema_pa (기존 산업 AI)                                     │
│      │                                                        │
│      ├──► Project 1: QA Copilot                              │
│      │    "기존 모델을 바꾸지 않고 한국어 설명 레이어 추가"  │
│      │    → 현장 작업자 교육비 절감, QA 자동화               │
│      │                                                        │
│      ├──► Project 2: VLM Benchmark                           │
│      │    "도메인 특화 한국어 VLM 평가셋 + LoRA vs Base"     │
│      │    → 파인튜닝 효과 정량화, 재현 가능한 측정           │
│      │                                                        │
│      └──► Project 3: Multimodal API                          │
│           "로컬 VLM + FastAPI end-to-end 서빙"               │
│           → API 비용 Zero, 운영 자립 가능                    │
│                                                              │
│  공유 컴포넌트:                                              │
│    vlm/schema/thema_pa_output.py  ← 세 프로젝트 import       │
│    vlm/train/inference.py         ← 세 프로젝트 추론 엔진    │
│    vlm/prompt/*.txt               ← 템플릿 공유              │
│    config.json + vlm/config.py    ← 설정 중앙화              │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. 주차별 실제 진행 상황

### 1주차 — 기반 구축 ✅
- [x] `vlm/schema/thema_pa_output.py` Pydantic 모델
- [x] thema_pa 샘플 JSON 수집 (`vlm/schema/samples/`)
- [x] DB → 샘플 export (`scripts/export_from_db.py`)
- [x] DB + 이미지 매칭 빌드 (`scripts/build_dataset.py`)

### 2주차 — LoRA 학습 파이프라인 ✅
- [x] 프롬프트 템플릿 4종
- [x] ShareGPT 변환기 (`vlm/train/convert_dataset.py`)
- [x] LoRA 학습 설정 (`vlm/train/qwen3vl_lora.yaml`)
- [x] 로컬 추론 엔진 (`vlm/train/inference.py`)
- [ ] LoRA 학습 완료 (2026-04-24 23:30 경 예정)

### 3주차 — 데모 & API & 중앙 설정 ✅
- [x] Streamlit 3패널 데모 (`vlm/demo/app.py`)
- [x] FastAPI 서버 (`vlm/api/schemas.py`, `server.py`)
- [x] `config.json` 중앙 설정 + `vlm/config.py` 로더
- [x] 보안 강화 (git 히스토리 비밀번호 제거, CORS, 경로 검증, 타임아웃)
- [x] 테스트 23개 추가
- [ ] 학습 완료 모델로 데모/API 실제 동작 확인

### 4주차 — 벤치마크 + 문서화 ⏳
- [ ] 평가 샘플 50~100건 수집 (train 제외)
- [ ] 레퍼런스 답변 작성
- [ ] `vlm/bench/scorer.py` (ROUGE / BERTScore)
- [ ] 실패 케이스 사례 문서화
- [ ] README + 포트폴리오 설명 페이지
