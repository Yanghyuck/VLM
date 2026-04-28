# VLM 프로젝트 진행 현황

**최종 업데이트**: 2026-04-28 (v1.0.0 릴리스, main 동기화 완료)
**현재 브랜치**: `main` (default), `local-vlm-train` (개발)
**리포지토리**: https://github.com/Yanghyuck/VLM
**릴리스**: [`v1.0.0`](https://github.com/Yanghyuck/VLM/releases/tag/v1.0.0)

---

## 프로젝트 개요

`thema_pa` (YOLOv11 돼지 도체 AI 시스템) 위에 **Qwen3-VL-8B LoRA 파인튜닝 모델**을
한국어 판정 레이어로 추가하는 3종 포트폴리오.

### 세 프로젝트 구성

| # | 이름 | 역할 |
|---|---|---|
| 1 | **QA Copilot** | thema_pa JSON + 이미지 → Qwen3-VL LoRA → 한국어 판정 리포트 (Streamlit 데모) |
| 2 | **VLM Benchmark** | LivestockVLM-Ko-Bench. LoRA 파인튜닝 vs 베이스 모델 비교 (50~100건) |
| 3 | **Multimodal API** | FastAPI로 Qwen3-VL LoRA end-to-end 서빙 |

---

## 아키텍처 핵심

### 공유 컴포넌트

- `vlm/schema/thema_pa_output.py` — 세 프로젝트 공통 Pydantic v2 스키마
- `vlm/train/inference.py` — Qwen3-VL-8B + LoRA 로컬 추론 (모든 프로젝트 공유)
- `vlm/report/generator.py` — `inference.py` 위임 shim (하위 호환)
- `vlm/prompt/` — 4종 한국어 프롬프트 템플릿
- `config.json` + `vlm/config.py` — 중앙 설정 관리

### 데이터 파이프라인

```
thema_pa MySQL DB ──► scripts/build_dataset.py ──► vlm/data/dataset.jsonl
                                                        │
도체 이미지 ──────────────────────────────────────────► │
                                                        ▼
                              vlm/train/convert_dataset.py
                                         │
                                         ▼
                            vlm/data/livestock_train.json (ShareGPT)
                                         │
                                         ▼
                           LLaMA-Factory + qwen3vl_lora.yaml
                                         │
                                         ▼
                     vlm/train/output/qwen3vl-lora/ (LoRA 어댑터)
```

---

## 4주 일정

| 주차 | 목표 | 상태 | 마감 |
|---|---|---|---|
| 1주차 | Pydantic 스키마 + 샘플 10건 수집 | ✅ 완료 | 2026-04-28 |
| 2주차 | 프롬프트 템플릿 4종 + Qwen3-VL LoRA 학습 (학습 완료) | ✅ 완료 | 2026-05-05 |
| 3주차 | Streamlit 데모 + FastAPI 서빙 + 중앙 설정 + 추론 검증 | ✅ 완료 | 2026-05-12 |
| 3.5주차 | v2 재학습 (Vision LoRA + AI 이미지, held-out 50건) | ✅ 완료 | 2026-04-27 |
| 4주차 | 3-way 벤치마크 (Base / v1 / v2) + 결과 분석 | ✅ 완료 | 2026-04-27 |
| Plan C | 양자화 + 환경변수 override + CHANGELOG + main 머지 | ✅ 완료 | 2026-04-28 |
| **v1.0.0** | **`main` 동기화 + `v1.0.0` 태그 푸시** | ✅ 완료 | 2026-04-28 |

---

## 주차별 상세 진행

### 1주차 — 스키마 정의 ✅

**커밋**: `fb31174`, `d88f137`, `741d711`, `6158536`, `8fb6358`, `79ac217`

- `vlm/schema/thema_pa_output.py` — `ThemaPAOutput`, `ErrorCode`, `BackboneSlope`, `Gender`
- `vlm/schema/samples/` — 샘플 JSON (normal_case, error_case, failure_analysis)
- `scripts/export_from_db.py` — MySQL → 샘플 JSON 변환
- `scripts/build_dataset.py` — DB + 이미지 매칭 → JSONL 빌드
- `tests/test_schema.py` — 스키마 유효성 단위 테스트

### 2주차 — 프롬프트 & 학습 파이프라인 ✅

**커밋**: `9541782` (claude 브랜치), `b7c6b3f` (local-vlm-train)

#### 프롬프트 템플릿 4종 (`vlm/prompt/`)
- `system_prompt.txt` — 도메인 지식 (등급 기준, 오류 코드 해석)
- `normal_case.txt` — 정상 케이스
- `error_case.txt` — 일부 AI 검출 오류 케이스
- `failure_analysis.txt` — 심각한 검출 실패 케이스

#### 학습 파이프라인
- `vlm/train/convert_dataset.py` — ShareGPT 포맷 변환 (원본 1건 → 최대 3 학습 샘플)
- `vlm/train/qwen3vl_lora.yaml` — LoRA 설정 (rank=64, α=128, bf16)
- `vlm/train/inference.py` — 로컬 LoRA 추론 (`generate_report(ThemaPAOutput) → dict`)
- `vlm/report/generator.py` — Claude API 버전 + inference.py 위임 shim

#### 학습 데이터 생성 완료
- **입력**: DB 레코드 3,355건 + 이미지 매칭
- **출력**: `livestock_train.json` — **6,710 학습 샘플** (summary / grade / abnormal)

#### LoRA 학습 완료 ✅ (2026-04-25 00:00)
- **모델**: Qwen3-VL-8B-Instruct
- **최적화**: `image_max_pixels` 400k → 200k, `cutoff_len` 2048 → 1024
- **속도**: 38초/step (초기 254초/step → **6.7배 개선**)
- **총 학습 시간**: **12시간 37분** (45,428초)
- **에폭**: 3.0 / 3.0 (1,134 / 1,134 steps)
- **train_loss**: 0.187
- **eval_loss**: **0.130** (eval < train, 과적합 없음)
- **어댑터 크기**: 666 MB (`adapter_model.safetensors`)
- **체크포인트**: `checkpoint-800`, `checkpoint-1000`, `checkpoint-1134` 3종 보존

### 3주차 — 데모 & API & 설정 관리 ✅

**커밋**: `7e6081a`, `790f29f`, `f435d4e`, `29116f3`, `5cb7fa0`, `03b9da1`, `4bcaf70`

#### FastAPI 서버 (`vlm/api/`)
- `schemas.py` — `ReportRequest`, `ReportResponse` Pydantic 모델
- `server.py` — POST `/v1/report`, GET `/v1/health`
  - 비동기 추론 (`asyncio.run_in_executor`)
  - 추론 타임아웃 (기본 180초)
  - 경로 traversal 방어 (image_dir 하위만 허용)
  - CORS 허용 origin 제한

#### Streamlit 데모 (`vlm/demo/app.py`)
- 3패널 UI (이미지 / 리포트 / 측정값)
- 샘플 선택 / JSON 업로드 / 이미지 업로드
- 업로드 이미지 uuid 파일명 + 1시간 경과 자동 삭제

#### 중앙 설정 관리
- `config.json` — DB, 경로, 모델, API, 등급 범위
- `config.example.json` — 더미 값 템플릿 (git 추적)
- `vlm/config.py` — `CFG` 로더 (SimpleNamespace)
- 모든 모듈이 `from vlm.config import CFG` 사용

#### 문서화
- `README.md` — 프로젝트 랜딩 페이지 (배지, 빠른 시작, 아키텍처 요약, API 예제)
- `PROGRESS.md` — 전체 진행 상황 통합 뷰
- `ARCHITECTURE.md` — Qwen3-VL LoRA 기반 현재 구조 반영 (초안 업데이트)

#### 추론 검증 ✅ (2026-04-25 ~ 2026-04-27)

**추론 단위 테스트** (`scripts/test_inference.py` → `vlm/train/test_inference_results.md`)

| 샘플 | 추론 시간 | 결과 |
|---|---|---|
| `normal_case` | 45.9초 | ✅ 1+ 등급, 정상 요약 |
| `backfat_error_case` | 34.6초 | ✅ 등외, 검출 실패 식별 + 주의사항 3개 |
| `entry_error_case` | 32.6초 | ✅ 등외, 비정상 진입 인식 + 재촬영 권고 |

**Streamlit 데모 파이프라인 검증** (`scripts/test_demo_pipeline.py` → `vlm/train/demo_pipeline_results.md`)

- Streamlit 서버 가동 확인 (HTTP 200, http://localhost:8501)
- 데모와 동일한 코드 경로(`_build_output()` → `generate_report()`)로 4개 샘플 자동 검증
- **통과율: 4/4 (100%)**
- 4개 필드(`3문장_요약`, `비정상_근거`, `주의사항`, `권고`) 모두 포함 확인
- 한국어 자연스러움, 도메인 지식 활용 우수

**FastAPI 엔드포인트 검증** (`scripts/test_api.py` → `vlm/api/api_test_results.md`)

- 서버 부팅 (`python vlm/api/server.py` — config.json 의 host/port 사용)
- `GET /v1/health` → **200** `{"status":"ready","model_used":"lora","adapter_exists":true}`
- `POST /v1/report` × 4 샘플 → **4/4 200 OK**
- 추론 시간: 20.3 ~ 33.8초/요청 (lifespan에서 모델 사전 로드, executor 비동기 추론)

| 샘플 | HTTP | 추론 시간 | 등급 |
|---|---|---|---|
| `normal_case` | 200 | 22.9초 | 🟢 1+ |
| `backfat_error_case` | 200 | 33.8초 | 🔴 등외 |
| `entry_error_case` | 200 | 31.3초 | 🔴 등외 |
| `sample_3473` | 200 | 20.3초 | 🟢 1+ |

---

## 4주차 — 3-way 벤치마크 결과 ✅

**평가셋**: held-out 50건 (학습에서 제외된 도체)
**실행 시각**: 2026-04-27 19:36 ~ 20:31 (56.8분)
**상세 리포트**: [`vlm/bench/score_report.md`](vlm/bench/score_report.md)

### 점수 비교

| 지표 | Base | v1 (text-only LoRA) | v2 (Vision LoRA + AI) |
|---|---|---|---|
| JSON 파싱 성공률 | 1.000 | 1.000 | 1.000 |
| 등급 일치율 | 1.000 | 1.000 | 1.000 |
| 수치 인용 정확도 | 1.000 | 1.000 | 1.000 |
| **ROUGE-L (summary)** | 0.696 | 0.739 (+6.2%) | **0.876 (+26.0%)** ⭐ |
| **BERTScore F1 (ko)** | 0.842 | 0.901 (+7.0%) | **0.957 (+13.7%)** ⭐ |
| 평균 추론 시간 | 15.8초 | 26.7초 | 24.5초 |

### 핵심 발견

1. **상한선이 이미 높음** — system_prompt 가 강력해서 베이스 모델도 JSON 형식·등급·수치는 100% 정확
2. **LoRA의 진짜 가치는 의미적·어휘적 일치도** — reference 와의 정확한 표현 패턴 학습
3. **v2 (Vision + AI) > v1 (text-only) > Base** — 일관된 우열
4. **v2 ROUGE-L +26%, BERTScore +13.7%** — 통계적으로 유의미한 개선
5. **추론 시간**: LoRA 적용으로 ~10초 증가 (PEFT layer 추가 연산), v2 가 v1보다 약간 빠름

### 시사점

- **Vision LoRA + AI 이미지 학습이 통계적으로 입증된 효과** (+26% ROUGE-L)
- "텍스트만 학습 vs 비전까지 학습" 의 정량적 차이를 처음 확인
- AI 이미지의 시각 단서(등지방 영역, 측정 라인)가 모델 응답 품질에 기여

### 분포 통계 (정성 분석)

| 모델 | avg | median | min | max | <0.5 |
|---|---|---|---|---|---|
| Base | 0.696 | 0.696 | 0.476 | 0.783 | 1건 |
| v1 LoRA | 0.739 | 0.727 | 0.615 | 0.870 | 0건 |
| **v2 LoRA** | **0.876** | **0.833** | **0.786** | **1.000** | 0건 |

> **v2 의 worst case (0.786) ≥ Base 의 max (0.783)** — 모든 분위에서 우월
> **v2 가 Base 를 초과한 비율: 50/50 (100%)** — 모든 샘플에서 개선 ⭐

### 시각화

`docs/figures/`:
- `08_benchmark_rouge_distribution.png` — 박스플롯
- `09_benchmark_rouge_cdf.png` — 누적 분포
- `10_benchmark_v2_vs_base_scatter.png` — 페어와이즈 산점도

상세 정성 분석: [`vlm/bench/failure_analysis.md`](vlm/bench/failure_analysis.md)

### 한계 (정직한 평가)

- 단일 일자(2026-04-22) 데이터로 학습/평가 → 시간적 일반화 미검증
- 등외 케이스 부재 → error_code 처리 능력은 별도 검증 필요
- 이미지 다양성 제한 (단일 카메라/조명/각도)

---

## INT4 양자화 (NF4) 평가 ⚠️

| 항목 | bf16 (기본) | INT4 NF4 | 변화 |
|---|---|---|---|
| VRAM | ~17~22 GB | **6.75 GB** | -70% ⭐ |
| 추론 시간 | 22~50초 | 46~112초 | +50~120% ⚠️ |
| 정상 케이스 품질 | 우수 | 우수 | OK |
| 오류 케이스 품질 | 우수 | **저하** (반복, 모순) | ⚠️ |

**결론**: 운영 서비스에는 bf16 권장. INT4 는 엣지 디바이스 + 정상 케이스 위주에서만 사용.
상세 분석: [`vlm/train/quantization_report.md`](vlm/train/quantization_report.md)

향후 개선 후보: GPTQ / AWQ / GGUF / 작은 모델 (Qwen3-VL-2B 등)

---

## 보안 조치

### git 히스토리 비밀번호 제거 ✅

초기에 `config.json`이 커밋되면서 DB 비밀번호가 공개 리포에 노출됐습니다.

**조치 내역**:
1. `git-filter-repo` 로 모든 히스토리에서 `config.json` 삭제 (11 커밋 재작성)
2. 강제 푸시로 원격 히스토리 덮어쓰기
3. 로컬 reflog + GC 로 blob 물리 삭제
4. `.gitignore` 에 `config.json` 등록
5. `config.example.json` 을 통한 설정 템플릿 제공

### 추가 보안 강화

- `/v1/report` 경로 검증 (traversal 차단)
- CORS 화이트리스트 (`config.api.allowed_origins`)
- 추론 타임아웃 (`config.api.inference_timeout_sec`)

> ⚠️ 공개 리포에 이미 한 번 노출된 비밀번호는 **DB 비밀번호 변경**을 추가로 권장합니다.

---

## 코드 품질 개선 이력

| 항목 | 해결 |
|---|---|
| CP949 인코딩 에러 (`✓`, `✗`, `→`, `—`) | ASCII 대체 문자 사용 |
| LoRA 학습 속도 (254s/step) | image_max_pixels 축소 → 38s/step |
| FastAPI 이벤트 루프 블로킹 | `asyncio.wait_for` + executor |
| health 엔드포인트 부정확성 | `_model_ready` 플래그 도입 |
| demo 죽은 코드 (`if run_btn and meta` → `if not meta`) | 제거 |
| 임시 이미지 파일 충돌 | uuid 파일명 |
| 임시 파일 무한 누적 | 1시간 경과 자동 삭제 |
| 중첩 JSON 파싱 실패 (regex) | brace-counting 파서 |
| SimpleNamespace 내부 의존 | `config.json` 직접 로드 방식 |
| `requirements.txt` ML 의존성 누락 | torch, transformers, peft 등 추가 |
| `pytest.ini` 미정의 | testpaths, markers, default filter |

---

## 테스트 현황

**최종 결과: 38/38 통과** (2026-04-28 기준)

| 파일 | 테스트 수 | 대상 |
|---|---|---|
| `tests/test_schema.py` | 5 | `ThemaPAOutput` 유효성 |
| `tests/test_api_schemas.py` | 7 | `ReportRequest`, `ReportResponse` |
| `tests/test_config.py` | 3 | `config.py` 로더 |
| `tests/test_json_extraction.py` | 8 | `_find_balanced_json`, `_extract_json` |
| `tests/test_auth.py` | 4 | X-API-Key 인증 (asyncio) |
| `tests/test_logging.py` | 4 | JSON 구조적 로깅 |
| `tests/test_env_override.py` | 7 | 환경변수 config override |

**End-to-End 검증 스크립트**

| 스크립트 | 검증 내용 | 결과 |
|---|---|---|
| `scripts/test_inference.py` | LoRA 어댑터 추론 (3샘플) | ✅ 3/3 |
| `scripts/test_demo_pipeline.py` | Streamlit 데모 동일 코드 경로 (4샘플) | ✅ 4/4 |
| `scripts/test_api.py` | FastAPI `/v1/health` + `/v1/report` (4샘플) | ✅ 4/4 |

```bash
pytest tests/
```

---

## 파일 구조

```
VLM/
├── config.json                       ← gitignore (로컬 전용)
├── config.example.json               ← 더미 템플릿 (git 추적)
├── requirements.txt                  ← 모든 Python 의존성
├── pytest.ini                        ← 테스트 설정 (asyncio_mode=auto)
├── Makefile                          ← 공통 명령 17개
├── Dockerfile                        ← GPU 컨테이너
├── docker-compose.yml                ← API + Demo 서비스
├── .dockerignore
├── .gitignore
├── PROGRESS.md                       ← 본 문서
├── README.md
├── ARCHITECTURE.md
├── CHANGELOG.md                      ← 버전별 변경 이력
│
├── .github/workflows/ci.yml          ← GitHub Actions CI
│
├── scripts/
│   ├── build_dataset.py              ← DB + AI 이미지 → JSONL
│   ├── export_from_db.py             ← DB → 샘플 JSON
│   ├── test_inference.py             ← 학습된 LoRA 추론 검증
│   ├── test_demo_pipeline.py         ← Streamlit 데모 파이프라인 검증
│   └── test_api.py                   ← FastAPI 엔드포인트 검증
│
├── vlm/
│   ├── config.py                     ← config.json 로더
│   ├── logging_config.py             ← JSON 구조적 로깅
│   │
│   ├── schema/
│   │   ├── thema_pa_output.py        ← 공통 Pydantic 모델
│   │   └── samples/                  ← 샘플 JSON
│   │
│   ├── prompt/
│   │   ├── system_prompt.txt
│   │   ├── normal_case.txt
│   │   ├── error_case.txt
│   │   └── failure_analysis.txt
│   │
│   ├── report/
│   │   └── generator.py              ← inference.py 위임 shim
│   │
│   ├── train/
│   │   ├── convert_dataset.py        ← ShareGPT 변환 (held-out 지원)
│   │   ├── qwen3vl_lora.yaml         ← v1 (text-only LoRA) 설정
│   │   ├── qwen3vl_lora_v2.yaml      ← v2 (Vision LoRA + AI 이미지) 설정
│   │   ├── inference.py              ← 로컬 추론 (use_adapter 토글)
│   │   ├── json_utils.py             ← JSON 파서 (torch 비의존)
│   │   └── output/
│   │       ├── qwen3vl-lora-v1-textonly/  ← v1 백업
│   │       └── qwen3vl-lora/              ← v2 (학습 중)
│   │
│   ├── api/
│   │   ├── schemas.py                ← 요청/응답 모델
│   │   ├── auth.py                   ← X-API-Key 검증
│   │   └── server.py                 ← FastAPI (auth + rate limit + 로깅)
│   │
│   ├── bench/
│   │   ├── dataset.py                ← 평가셋 빌드 (jsonl / db)
│   │   ├── runner.py                 ← base / lora 추론 실행
│   │   └── scorer.py                 ← ROUGE / BERTScore / 일치율
│   │
│   ├── demo/
│   │   └── app.py                    ← Streamlit 3패널 UI
│   │
│   └── data/                         ← gitignore (JSONL, 업로드 파일)
│
├── notebooks/
│   ├── dataset_analysis.py           ← 분석 스크립트
│   └── dataset_analysis.md           ← 리포트 + narrative
│
├── docs/figures/                     ← 7장 시각화 .png
│
└── tests/                            ← 31 테스트 (6개 파일)
    ├── test_schema.py                # ThemaPAOutput (5)
    ├── test_api_schemas.py           # Request/Response (7)
    ├── test_config.py                # config 로더 (3)
    ├── test_json_extraction.py       # JSON 파서 (8)
    ├── test_auth.py                  # X-API-Key (4, asyncio)
    └── test_logging.py               # JSON 로깅 (4)
```

---

## 브랜치 구조

| 브랜치 | 용도 | 최신 커밋 |
|---|---|---|
| `main` | **default 브랜치 (1~4주차 + Plan C 모두 반영)** | `7015c92` |
| `local-vlm-train` | 개발 브랜치 (main 과 동일 상태) | `7015c92` |
| `claude` | 2주차 Claude API 버전 (참고용 보관) | `9541782` |

**태그**: `v1.0.0` ([릴리스 페이지](https://github.com/Yanghyuck/VLM/releases/tag/v1.0.0))

---

## 실행 방법

### 초기 설정

```bash
# 1. 환경 활성화
conda activate vlm

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 설정 파일 생성
cp config.example.json config.json
# → config.json 안의 DB 비밀번호, 경로 수정

# 4. LLaMA-Factory 설치 (LoRA 학습용)
cd ../LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../VLM
```

### 테스트 실행

```bash
pytest tests/
```

### LoRA 학습

```bash
# 1. 데이터셋 빌드
python scripts/build_dataset.py

# 2. ShareGPT 변환
python vlm/train/convert_dataset.py
cp vlm/data/livestock_train.json C:/Users/IPC/Desktop/git/LLaMA-Factory/data/

# 3. 학습 시작
llamafactory-cli train vlm/train/qwen3vl_lora.yaml
```

### 데모 실행 (학습 완료 후)

```bash
# Streamlit 데모
streamlit run vlm/demo/app.py

# FastAPI 서버
python vlm/api/server.py

# API 호출 테스트
curl -X POST http://localhost:8000/v1/report \
     -H "Content-Type: application/json" \
     -d @vlm/schema/samples/normal_case.json
```

---

## 다음 할 일

### 3주차 잔여 작업
- [x] LoRA 학습 완료 (2026-04-25 00:00, 12시간 37분)
- [x] Streamlit 데모 실제 모델로 동작 확인 (HTTP 200, 4/4 샘플 통과)
- [x] 추론 단위 테스트 (3 샘플)
- [x] 데모 파이프라인 검증 (4 샘플)
- [x] FastAPI 엔드포인트 실제 요청 테스트 (4/4 200 OK)
- [~] 데모 GIF 녹화 (생략 결정)

### v2 재학습 (Vision LoRA + AI 이미지, 진행 중)
**시작**: 2026-04-27 10:55
**예상 완료**: 2026-04-27 16:30 경 (실측 33s/step × 558 steps = **~5.1 시간**)
**참고**: 초기 예상 16~18h 였으나 image_max_pixels 절반(200K→100K)으로 vision 처리 부담 감소 덕에 5h 수준

핵심 변경:
- ✅ v1 어댑터 백업 (`vlm/train/output/qwen3vl-lora-v1-textonly/`)
- ✅ 이미지 소스: ORI → **AI** (등지방/뭇갈래근 시각 오버레이 포함)
- ✅ Vision Tower LoRA 학습 활성화 (`freeze_vision_tower: false`)
- ✅ Multi-modal Projector LoRA 학습 활성화
- ✅ 평가셋 50건 학습 데이터에서 제외 (held-out 벤치마크)
- ✅ image_max_pixels 200,704 → 100,352 (vision 학습 메모리 보전)
- ✅ gradient_accumulation_steps 16 → 32 (effective batch 16 유지)
- 🔄 학습 진행 중 (`vlm/train/qwen3vl_lora_v2.yaml`)
- ⏳ v1 vs v2 벤치마크 (학습 완료 후)

**학습 데이터 변경**:
- 이미지 소스: `thema_pa/images/AI/` (3,473장 중 3,355장 매칭)
- 학습 샘플: 6,710 → **6,610** (50건 × 2 task held-out 제외)
- 평가셋 50건: `vlm/bench/eval_set.jsonl` (등급 분포 1+:23 / 1:15 / 2:12)

### 4주차 (벤치마크) ✅
- [x] 평가 데이터셋 선정 (50건 held-out, 학습 제외)
- [x] 벤치마크 러너 (`vlm/bench/runner.py`) — base/v1/v2 추론
- [x] 스코어러 (`vlm/bench/scorer.py`) — N-way 비교, ROUGE-L / BERTScore (ko) / 4종 정확도
- [x] 결과 리포트 (`vlm/bench/score_report.md`)
- [x] 3-way 자동 실행 스크립트 (`scripts/run_3way_benchmark.py`)
- [x] **실패 케이스 5건 정성 분석** (`vlm/bench/failure_analysis.md`)
- [x] **벤치마크 시각화** (3장 figures: boxplot, CDF, scatter)
- [x] **INT4 양자화 평가** (`vlm/train/quantization_report.md`)

### 포트폴리오 완성도
- [x] GitHub Actions CI (`pytest` 자동 실행 + 배지) — `.github/workflows/ci.yml`, Python 3.11/3.13 매트릭스
- [x] Dockerfile + docker-compose — GPU 컨테이너 + 모델 가중치 볼륨 마운트
- [x] 데이터셋 분석 노트북 (등급 분포, 측정값 히스토그램) — `notebooks/dataset_analysis.{py,md}` + 7장 figures
- [x] Makefile — 공통 명령 모음
- [x] **API 인증 (X-API-Key)** — `vlm/api/auth.py`, 4 테스트 통과
- [x] **구조적 로깅 (JSON)** — `vlm/logging_config.py`, request_id/latency 자동 기록
- [x] **Rate limiting (slowapi)** — 분당 N회 제한, 429 응답
- [x] **모델 양자화 (INT4 NF4)** — VRAM 22GB → 6.75GB (-70%), 품질 trade-off 문서화

### 운영 전 필요 작업
- [ ] DB 비밀번호 변경 (이전 노출 대응 — 사용자 수동 작업)
- [x] **환경변수 기반 config 로딩 추가** — `VLM_DB_PASSWORD`, `VLM_API_KEYS` 등 12개 변수
- [x] **구조적 로깅** (이미 완료, JSON + request_id)
- [x] Rate limiting (이미 완료, slowapi)
- [ ] HTTPS 리버스 프록시 (사용자 환경 의존)
- [ ] Sentry/PagerDuty 알림 (선택)
- [x] **CHANGELOG.md** — 버전별 변경 이력 + 결정 이력
