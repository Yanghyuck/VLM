# 🐷 Korean Livestock VLM Copilot

> **thema_pa** (YOLOv11 돼지 도체 AI) 위에 **Qwen3-VL-8B LoRA 파인튜닝** 모델을 한국어 판정 레이어로 추가하는 3종 포트폴리오

[![CI](https://github.com/Yanghyuck/VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Yanghyuck/VLM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.55+-ffb71b.svg)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-38%20passed-success.svg)](#테스트)

---

## 🎯 프로젝트 한눈에

도축장의 YOLO 기반 판정 수치만으로는 **"왜 이 등급이 나왔는지"** 를 현장 작업자가 바로 이해하기 어렵습니다.
이 프로젝트는 **Qwen3-VL-8B 를 LoRA 로 파인튜닝** 하여, 도체 이미지 + 측정 수치를 입력받아
한국어 판정 리포트(3문장 요약 + 근거 + 주의사항 + 권고)를 자동 생성합니다.

```
도체 이미지 + thema_pa JSON  ─►  Qwen3-VL LoRA  ─►  한국어 판정 리포트
```

### 핵심 결과 (held-out 50건 벤치마크)

| 지표 | Base | v1 (text-only) | **v2 (Vision+AI)** |
|---|---|---|---|
| **ROUGE-L** | 0.696 | 0.739 (+6%) | **0.876 (+26%)** ⭐ |
| **BERTScore (ko)** | 0.842 | 0.901 (+7%) | **0.957 (+14%)** ⭐ |
| **v2 > Base 비율** | — | — | **50/50 (100%)** ⭐ |
| **v2 worst case** | — | — | **0.786 ≥ Base max** |

→ Vision LoRA + AI 이미지 학습으로 **베이스 대비 ROUGE +26%, BERTScore +14%** 개선.
**모든 50건에서 v2가 Base를 초과** (단 한 건도 빠짐없음).
상세 결과: [`vlm/bench/score_report.md`](./vlm/bench/score_report.md), [`failure_analysis.md`](./vlm/bench/failure_analysis.md), [`quantization_report.md`](./vlm/train/quantization_report.md)

### 30초 어필 (포트폴리오 요약)

> **문제**: 도축장 YOLO 시스템은 등급 수치만 출력 — 현장 작업자는 "왜 이 등급인지" 즉각 이해 불가
> **접근**: Qwen3-VL-8B 를 LoRA 로 두 번 파인튜닝
>   - v1: 텍스트만 학습 (베이스라인)
>   - v2: Vision Tower + AI 분석 이미지까지 학습 (메인)
> **결과**: held-out 50건 정량 평가에서 **ROUGE-L +26%, BERTScore +14%, 100% sample-wise 우월**
> **엔지니어링**: FastAPI (auth + rate limit + JSON 로깅), Docker (GPU), GitHub Actions CI, **38 단위 테스트** 전 통과

| 영역 | 산출물 |
|---|---|
| **데이터 파이프라인** | MySQL 3,355건 + AI 이미지 매칭 → ShareGPT 6,610 학습 샘플 |
| **모델 학습** | LLaMA-Factory + LoRA (rank 64), v1 12.6h / v2 5.1h |
| **벤치마크** | 50건 held-out, 3-way (Base/v1/v2), ROUGE-L+BERTScore+JSON+Grade |
| **데모 UI** | Streamlit 3패널 (이미지/리포트/측정값) |
| **운영 API** | FastAPI + Swagger + X-API-Key + slowapi rate limit |
| **관측성** | JSON 구조적 로깅, X-Request-ID 분산 추적 |
| **재현성** | Makefile, Dockerfile, docker-compose, GitHub Actions |
| **보안** | git history 비밀번호 제거(filter-repo), 환경변수 override |
| **양자화 평가** | INT4 NF4: VRAM -70%, 품질 trade-off 정직 분석 |

---

## 📸 데모 동작 검증

```
┌─────────────────┬──────────────────────┬─────────────────┐
│ 도체 이미지     │ 3문장 요약           │ 🟢 1+ 등급      │
│ 원본 JSON       │ 비정상 근거          │ 등지방 22mm     │
│ 샘플 선택       │ 주의사항 리스트      │ 뭇갈래근 48mm   │
│                 │ 권고                 │ AI 검출: 정상   │
└─────────────────┴──────────────────────┴─────────────────┘
```

### LoRA 추론 결과 (2026-04-27)

| 샘플 | 추론 시간 | 등급 | 통과 여부 |
|---|---|---|---|
| `normal_case` | ~41s | 🟢 1+ | ✅ |
| `backfat_error_case` | ~35s | 🔴 등외 | ✅ (검출 실패 식별 + 재촬영 권고) |
| `entry_error_case` | ~33s | 🔴 등외 | ✅ (비정상 진입 인식) |
| `sample_3473` | ~39s | 🟢 1+ | ✅ |

**Streamlit 파이프라인 검증**: 4/4 통과 ([상세 결과](./vlm/train/demo_pipeline_results.md))

---

## 🚀 빠른 시작

```bash
# 1. 환경 설정
conda activate vlm
pip install -r requirements.txt
cp config.example.json config.json   # → DB 비밀번호, 경로 수정

# 2. 테스트
pytest tests/                        # ✅ 23 passed

# 3. 실행 (학습 완료 후)
streamlit run vlm/demo/app.py        # 데모 UI        : http://localhost:8501
python vlm/api/server.py             # FastAPI 서버   : http://localhost:8000/docs
```

---

## 🏗️ 세 프로젝트 구성

| # | 이름 | 한 줄 설명 | 주요 파일 |
|---|---|---|---|
| 1 | **QA Copilot** | 도체 이미지+JSON → 한국어 판정 리포트 Streamlit 데모 | `vlm/demo/app.py` |
| 2 | **VLM Benchmark** | LoRA 파인튜닝 vs 베이스 모델 50~100건 비교 평가 | `vlm/bench/` (4주차) |
| 3 | **Multimodal API** | FastAPI로 Qwen3-VL LoRA end-to-end 서빙 | `vlm/api/server.py` |

세 프로젝트 모두 동일한 `vlm/train/inference.py` 추론 엔진을 공유합니다.

---

## 🧱 아키텍처

```
thema_pa MySQL DB + 도체 이미지
        │
        ├──► scripts/build_dataset.py      # DB + 이미지 매칭
        │          │
        │          ▼
        │    vlm/data/dataset.jsonl
        │          │
        │          ▼
        │    vlm/train/convert_dataset.py  # ShareGPT 변환
        │          │
        │          ▼
        │    vlm/data/livestock_train.json (6,710 샘플)
        │          │
        │          ▼
        │    LLaMA-Factory LoRA 학습
        │          │
        │          ▼
        │    vlm/train/output/qwen3vl-lora/ (LoRA 어댑터)
        │                    │
        └──► vlm/train/inference.py ◄──────┘
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
    Streamlit   FastAPI     Benchmark
     (데모)       (API)     (평가)
```

상세 구조는 [ARCHITECTURE.md](./ARCHITECTURE.md) 를 참고하세요.

---

## 📦 기술 스택

| 레이어 | 선택 | 이유 |
|---|---|---|
| VLM | **Qwen3-VL-8B-Instruct** + LoRA (rank=64) | 한국어 성능, RTX 4090 24GB 적합 |
| 학습 | LLaMA-Factory | VLM LoRA 파인튜닝 표준 |
| 데모 UI | Streamlit | 빠른 3패널 프로토타이핑 |
| API | FastAPI + Uvicorn | 비동기, Swagger 자동 |
| 스키마 | Pydantic v2 | 타입 안전 + JSON 호환 |
| DB | MySQL 8.x | thema_pa 기존 DB 재사용 |
| 평가 | ROUGE-L + BERTScore (ko) | 한국어 NLG 표준 |
| 테스트 | pytest + pytest-asyncio | markers + testpaths + async |
| API 보안 | slowapi + X-API-Key | rate limit + 인증 |
| 로깅 | python-json-logger | request_id, latency 자동 |
| CI | GitHub Actions | Python 3.11/3.13 매트릭스 |
| 컨테이너 | Docker + docker-compose | NVIDIA GPU 지원 |

---

## 📁 프로젝트 구조

```
VLM/
├── README.md                  # ← 본 문서
├── ARCHITECTURE.md            # 전체 아키텍처 상세
├── PROGRESS.md                # 진행 현황 추적
├── config.example.json        # 설정 템플릿
├── requirements.txt
├── pytest.ini
├── Makefile                   # 공통 명령 모음
├── Dockerfile                 # GPU 컨테이너 이미지
├── docker-compose.yml         # API + Demo 서비스 정의
│
├── .github/workflows/ci.yml   # GitHub Actions CI (Python 3.11/3.13)
│
├── scripts/
│   ├── build_dataset.py       # DB + 이미지 → JSONL
│   ├── export_from_db.py      # DB → 샘플 JSON
│   ├── test_inference.py      # 학습된 LoRA 추론 검증
│   ├── test_demo_pipeline.py  # Streamlit 데모 파이프라인 검증
│   └── test_api.py            # FastAPI 엔드포인트 검증
│
├── vlm/
│   ├── config.py              # config.json 로더
│   ├── logging_config.py      # JSON 구조적 로깅
│   ├── schema/                # Pydantic 공통 모델
│   ├── prompt/                # 4종 프롬프트 템플릿
│   ├── train/
│   │   ├── inference.py       # 로컬 LoRA 추론
│   │   ├── json_utils.py      # JSON 파서 (torch 비의존)
│   │   ├── convert_dataset.py # ShareGPT 변환 (held-out 지원)
│   │   ├── qwen3vl_lora.yaml  # v1 학습 설정
│   │   └── qwen3vl_lora_v2.yaml # v2 (Vision LoRA + AI 이미지)
│   ├── api/
│   │   ├── schemas.py         # 요청/응답 모델
│   │   ├── auth.py            # X-API-Key 검증
│   │   └── server.py          # FastAPI 앱 (auth + rate limit + 로깅)
│   ├── bench/
│   │   ├── dataset.py         # 평가셋 빌드 (jsonl / db)
│   │   ├── runner.py          # base / lora 추론 실행
│   │   └── scorer.py          # ROUGE-L / BERTScore / 일치율
│   ├── demo/                  # Streamlit 데모
│   └── report/                # inference.py 위임 shim
│
├── notebooks/
│   ├── dataset_analysis.py    # 데이터셋 분석 스크립트
│   └── dataset_analysis.md    # 분석 리포트 + 시각화 narrative
│
├── docs/figures/              # 7장 시각화 (등급 분포, 측정값 등)
│
└── tests/                     # 31 테스트 (스키마/API/config/JSON/auth/로깅)
```

---

## 🔐 설정 관리

모든 환경 변동값은 `config.json` 한 곳에 집중되어 있습니다:

```json
{
  "db":    { "host": "...", "password": "...", "name": "..." },
  "paths": { "image_dir": "...", "lora_adapter": "..." },
  "model": { "base_model_id": "Qwen/Qwen3-VL-8B-Instruct", "quantize": false },
  "api":   { "port": 8000, "api_keys": [], "rate_limit_per_minute": 60 },
  "grade": { "backfat_range": {"1+": [17, 25]}, "weight_range": {...} },
  "logging": { "level": "INFO", "format": "json" }
}
```

`config.json` 은 `.gitignore` 로 제외되며, 팀 공유용 템플릿은 `config.example.json` 입니다.

### 환경변수 override (운영 보안)

`config.json` 에 평문으로 저장하지 않고 환경변수로 주입 가능 — 운영에서 권장:

```bash
export VLM_DB_PASSWORD="real_secret"
export VLM_API_KEYS="prod-key-1,prod-key-2"
export VLM_API_PORT=9000
python vlm/api/server.py
# [config] env overrides applied: VLM_DB_PASSWORD, VLM_API_KEYS, VLM_API_PORT
```

지원 환경변수: `VLM_DB_HOST/PORT/USER/PASSWORD/NAME`, `VLM_API_HOST/PORT/KEYS`,
`VLM_LORA_ADAPTER`, `VLM_IMAGE_DIR`, `VLM_LOG_LEVEL/FORMAT`

---

## 🧪 테스트

**단위 테스트: 38/38 통과** (2026-04-28 기준)

| 파일 | 대상 | 테스트 수 |
|---|---|---|
| `tests/test_schema.py` | `ThemaPAOutput` 유효성 | 5 |
| `tests/test_api_schemas.py` | `ReportRequest`/`Response` | 7 |
| `tests/test_config.py` | config 로더 | 3 |
| `tests/test_json_extraction.py` | brace-balanced JSON 파서 | 8 |
| `tests/test_auth.py` | X-API-Key 인증 | 4 |
| `tests/test_logging.py` | JSON 구조적 로깅 | 4 |
| `tests/test_env_override.py` | 환경변수 config override | 7 |

```bash
pytest tests/                # 기본 실행 (integration 제외)
pytest tests/ -m integration # 통합 테스트만 (실제 모델/DB 필요)
```

**End-to-End 검증**

| 스크립트 | 검증 내용 | 결과 |
|---|---|---|
| `scripts/test_inference.py` | 학습된 LoRA 어댑터 추론 (3샘플) | ✅ 3/3 |
| `scripts/test_demo_pipeline.py` | Streamlit 데모 동일 코드 경로 (4샘플) | ✅ 4/4 |
| `scripts/test_api.py` | FastAPI `/v1/health` + `/v1/report` (4샘플) | ✅ 4/4 |

---

## 📡 API 사용

### `/v1/health`

```bash
curl http://localhost:8000/v1/health
# { "status": "ready", "model_used": "lora", "adapter_exists": true }
```

### `/v1/report`

```bash
curl -X POST http://localhost:8000/v1/report \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_KEY" \
     -d @vlm/schema/samples/normal_case.json
```

**검증된 추론 시간**: 20~34초 (4샘플 평균 27초, lifespan 모델 사전 로드)

### 보안 / 운영 기능

| 기능 | 동작 |
|---|---|
| **X-API-Key 인증** | `config.api.api_keys` 배열 비어있으면 비활성화 (개발 모드) |
| **Rate limiting** | 분당 N회 (`config.api.rate_limit_per_minute`, 기본 60), 초과 시 429 |
| **추론 타임아웃** | `config.api.inference_timeout_sec` (기본 180초), 초과 시 504 |
| **CORS 화이트리스트** | `config.api.allowed_origins` 명시 출처만 |
| **경로 traversal 차단** | `result_image_path` 가 `image_dir` 외부면 403 |
| **JSON 구조적 로깅** | request_id, latency, status_code 등 자동 기록 |
| **요청 추적** | 응답에 `X-Request-ID` 헤더 포함 |

응답:
```json
{
  "summary": "도체번호 3010은 거세 수컷으로 ...",
  "grade_reason": null,
  "warnings": ["체폭/체장 측정 정상", "척추 기울기 정상"],
  "recommendation": "등지방 기준 범위 내이나 상단 근접. 다음 개체 주의 관찰.",
  "model_used": "lora (3.2s)"
}
```

Swagger UI: http://localhost:8000/docs

---

## 🗓️ 진행 현황

| 주차 | 목표 | 상태 |
|---|---|---|
| 1주차 | Pydantic 스키마 + 샘플 수집 | ✅ 완료 |
| 2주차 | 프롬프트 4종 + **v1 LoRA 학습** (text-only, eval_loss=0.130) | ✅ 완료 |
| 3주차 | Streamlit 데모 + FastAPI + 중앙 설정 + 추론 검증 | ✅ 완료 |
| 3.5주차 | **v2 재학습** (Vision LoRA + AI 이미지, eval_loss=0.080) | ✅ 완료 |
| 4주차 | **3-way 벤치마크** Base/v1/v2 + 결과 분석 | ✅ 완료 |

상세 진행 이력은 [PROGRESS.md](./PROGRESS.md) 를 참고하세요.

---

## 🌿 브랜치

| 브랜치 | 역할 |
|---|---|
| `main` | 1주차 결과 (스키마 + DB/이미지 매칭) |
| `claude` | 2주차 Claude API 버전 (참고용) |
| `local-vlm-train` | **메인 개발 브랜치** — 로컬 Qwen3-VL LoRA |

---

## 📖 더 읽을거리

- [ARCHITECTURE.md](./ARCHITECTURE.md) — 전체 아키텍처 설계
- [PROGRESS.md](./PROGRESS.md) — 주차별 진행 이력
- [CHANGELOG.md](./CHANGELOG.md) — 버전별 변경 이력
- [notebooks/dataset_analysis.md](./notebooks/dataset_analysis.md) — 데이터셋 분석 리포트 (등급 분포, 측정값 통계)
- [vlm/bench/score_report.md](./vlm/bench/score_report.md) — 3-way 벤치마크 점수표
- [vlm/bench/failure_analysis.md](./vlm/bench/failure_analysis.md) — 실패 케이스 정성 분석
- [vlm/train/quantization_report.md](./vlm/train/quantization_report.md) — INT4 양자화 평가
- `vlm/prompt/system_prompt.txt` — 도메인 지식 + 출력 형식 정의

## 🛠️ Makefile 명령

```bash
make help           # 전체 타겟 목록
make test           # pytest 실행
make demo           # Streamlit 데모
make api            # FastAPI 서버
make analyze        # 데이터셋 분석 그림 생성
make build-eval     # 평가셋 50건 (held-out)
make convert        # 학습 데이터 변환 (eval 제외)
make train-v2       # v2 학습 실행 (Vision LoRA + AI)
make bench-all      # 베이스 + LoRA + 점수
```

## 🐳 Docker 실행

NVIDIA Container Toolkit 설치 환경에서 GPU 추론이 가능합니다.

```bash
# 빌드
docker build -t vlm-livestock .

# FastAPI + Streamlit 동시 실행
docker compose up

# 개별 실행
docker compose up api    # http://localhost:8000
docker compose up demo   # http://localhost:8501
```

**볼륨 마운트**: 모델 가중치(8.9B)와 LoRA 어댑터(666MB)는 이미지에 포함하지 않고
호스트의 `vlm/train/output/qwen3vl-lora/` 를 마운트합니다 (이미지 크기 절감).
