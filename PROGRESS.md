# VLM 프로젝트 진행 현황

**최종 업데이트**: 2026-04-24
**현재 브랜치**: `local-vlm-train`
**리포지토리**: https://github.com/Yanghyuck/VLM

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
| 2주차 | 프롬프트 템플릿 4종 + Qwen3-VL LoRA 학습 파이프라인 | ✅ 완료 | 2026-05-05 |
| 3주차 | Streamlit 데모 + FastAPI 서빙 + 중앙 설정 관리 | 🔄 진행 중 | 2026-05-12 |
| 4주차 | 벤치마크 50~100건 (LoRA vs 베이스) + 사례 문서화 | ⏳ 대기 | 2026-05-19 |

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

#### LoRA 학습 진행 상황
- **모델**: Qwen3-VL-8B-Instruct
- **최적화**: `image_max_pixels` 400k → 200k, `cutoff_len` 2048 → 1024
- **속도**: 38초/step (초기 254초/step → **6.7배 개선**)
- **총 스텝**: 1,134
- **예상 완료**: 2026-04-24 23:30 경

### 3주차 — 데모 & API & 설정 관리 🔄

**커밋**: `7e6081a`, `790f29f`, `f435d4e`, `29116f3`

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

**최종 결과: 23/23 통과**

| 파일 | 테스트 수 | 대상 |
|---|---|---|
| `tests/test_schema.py` | 5 | `ThemaPAOutput` 유효성 |
| `tests/test_api_schemas.py` | 7 | `ReportRequest`, `ReportResponse` |
| `tests/test_config.py` | 3 | `config.py` 로더 |
| `tests/test_json_extraction.py` | 8 | `_find_balanced_json`, `_extract_json` |

```bash
pytest tests/
```

---

## 파일 구조

```
VLM/
├── config.json                  ← gitignore (로컬 전용)
├── config.example.json          ← 더미 템플릿 (git 추적)
├── requirements.txt             ← 모든 Python 의존성
├── pytest.ini                   ← 테스트 설정
├── .gitignore
├── PROGRESS.md                  ← 본 문서
│
├── scripts/
│   ├── build_dataset.py         ← DB + 이미지 → JSONL
│   └── export_from_db.py        ← DB → 샘플 JSON
│
├── vlm/
│   ├── config.py                ← CFG 로더
│   │
│   ├── schema/
│   │   ├── thema_pa_output.py   ← 공통 Pydantic 모델
│   │   └── samples/             ← 샘플 JSON
│   │
│   ├── prompt/
│   │   ├── system_prompt.txt
│   │   ├── normal_case.txt
│   │   ├── error_case.txt
│   │   └── failure_analysis.txt
│   │
│   ├── report/
│   │   └── generator.py         ← inference.py 위임 shim
│   │
│   ├── train/
│   │   ├── convert_dataset.py   ← ShareGPT 변환
│   │   ├── qwen3vl_lora.yaml    ← LoRA 학습 설정
│   │   ├── inference.py         ← 로컬 추론
│   │   └── output/qwen3vl-lora/ ← LoRA 어댑터 (학습 완료 후)
│   │
│   ├── api/
│   │   ├── schemas.py           ← 요청/응답 모델
│   │   └── server.py            ← FastAPI 앱
│   │
│   ├── demo/
│   │   └── app.py               ← Streamlit 3패널 UI
│   │
│   └── data/                    ← gitignore (JSONL, 업로드 파일)
│
└── tests/
    ├── test_schema.py
    ├── test_api_schemas.py
    ├── test_config.py
    └── test_json_extraction.py
```

---

## 브랜치 구조

| 브랜치 | 용도 | 최신 커밋 |
|---|---|---|
| `main` | 1주차 결과 (스키마 + DB/이미지 매칭) | `79ac217` |
| `claude` | 2주차 Claude API 버전 | `9541782` |
| `local-vlm-train` | 2주차 + 3주차 (Qwen3-VL LoRA) | `29116f3` |

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

### 단기 (3주차 완료)
- [ ] LoRA 학습 완료 대기 (2026-04-24 23:30 경)
- [ ] Streamlit 데모 실제 모델로 동작 확인
- [ ] FastAPI 엔드포인트 실제 요청 테스트
- [ ] 3주차 결과 `local-vlm-train` 브랜치 최종 커밋

### 4주차 (벤치마크)
- [ ] 평가 데이터셋 선정 (50~100건, train/val 제외)
- [ ] 벤치마크 러너 (`vlm/bench/`)
  - 베이스 모델 vs LoRA 모델 응답 비교
  - BERTScore / ROUGE / 수치 정확도 측정
- [ ] 결과 리포트 (mdbook / notion)

### 운영 전 필요 작업
- [ ] DB 비밀번호 변경 (노출됐던 이전 비밀번호)
- [ ] 환경변수 기반 config 로딩 추가 (`DB_PASSWORD` 등)
- [ ] HTTPS 리버스 프록시 + rate limiting
- [ ] 구조적 로깅 + 에러 알림
