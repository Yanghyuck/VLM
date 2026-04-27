# 🐷 Korean Livestock VLM Copilot

> **thema_pa** (YOLOv11 돼지 도체 AI) 위에 **Qwen3-VL-8B LoRA 파인튜닝** 모델을 한국어 판정 레이어로 추가하는 3종 포트폴리오

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.55+-ffb71b.svg)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-23%20passed-success.svg)](#테스트)

---

## 🎯 프로젝트 한눈에

도축장의 YOLO 기반 판정 수치만으로는 **"왜 이 등급이 나왔는지"** 를 현장 작업자가 바로 이해하기 어렵습니다.
이 프로젝트는 **Qwen3-VL-8B 를 LoRA 로 파인튜닝** 하여, 도체 이미지 + 측정 수치를 입력받아
한국어 판정 리포트(3문장 요약 + 근거 + 주의사항 + 권고)를 자동 생성합니다.

```
도체 이미지 + thema_pa JSON  ─►  Qwen3-VL LoRA  ─►  한국어 판정 리포트
```

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
| 테스트 | pytest | markers + testpaths |

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
│
├── scripts/
│   ├── build_dataset.py       # DB + 이미지 → JSONL
│   └── export_from_db.py      # DB → 샘플 JSON
│
├── vlm/
│   ├── config.py              # CFG 로더
│   ├── schema/                # Pydantic 공통 모델
│   ├── prompt/                # 4종 프롬프트 템플릿
│   ├── train/                 # LoRA 학습 + 로컬 추론
│   ├── api/                   # FastAPI 서버
│   ├── demo/                  # Streamlit 데모
│   └── report/                # inference.py 위임 shim
│
└── tests/                     # 23 테스트 (스키마, API, config, JSON 파서)
```

---

## 🔐 설정 관리

모든 환경 변동값은 `config.json` 한 곳에 집중되어 있습니다:

```json
{
  "db":    { "host": "...", "password": "...", "name": "..." },
  "paths": { "image_dir": "...", "lora_adapter": "..." },
  "model": { "base_model_id": "Qwen/Qwen3-VL-8B-Instruct" },
  "api":   { "port": 8000, "allowed_origins": [...], "inference_timeout_sec": 180 },
  "grade": { "backfat_range": {"1+": [17, 25]}, "weight_range": {...} }
}
```

`config.json` 은 `.gitignore` 로 제외되며, 팀 공유용 템플릿은 `config.example.json` 입니다.

---

## 🧪 테스트

**단위 테스트: 23/23 통과** (2026-04-27 기준)

| 파일 | 대상 | 테스트 수 |
|---|---|---|
| `tests/test_schema.py` | `ThemaPAOutput` 유효성 | 5 |
| `tests/test_api_schemas.py` | `ReportRequest`/`Response` | 7 |
| `tests/test_config.py` | config 로더 | 3 |
| `tests/test_json_extraction.py` | brace-balanced JSON 파서 | 8 |

```bash
pytest tests/                # 기본 실행 (integration 제외)
pytest tests/ -m integration # 통합 테스트만 (실제 모델/DB 필요)
```

**End-to-End 검증**

| 스크립트 | 검증 내용 | 결과 |
|---|---|---|
| `scripts/test_inference.py` | 학습된 LoRA 어댑터 추론 (3샘플) | ✅ 3/3 |
| `scripts/test_demo_pipeline.py` | Streamlit 데모 동일 코드 경로 (4샘플) | ✅ 4/4 |

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
     -d @vlm/schema/samples/normal_case.json
```

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
| 2주차 | 프롬프트 4종 + **LoRA 학습 완료** (12h 37m, eval_loss=0.130) | ✅ 완료 |
| 3주차 | Streamlit 데모 + FastAPI + 중앙 설정 + 추론 검증 | ✅ 완료 |
| 4주차 | 벤치마크 50~100건 (LoRA vs Base) + 문서화 | ⏳ 예정 |

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
- `vlm/prompt/system_prompt.txt` — 도메인 지식 + 출력 형식 정의
