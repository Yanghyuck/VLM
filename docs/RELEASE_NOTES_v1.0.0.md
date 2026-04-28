# v1.0.0 — Korean Livestock VLM Copilot

**4주차 포트폴리오 완성 릴리스**

---

## 🎯 정량 결과 (held-out 50건 벤치마크)

| 지표 | Base | v1 (text-only) | **v2 (Vision+AI)** |
|---|---|---|---|
| **ROUGE-L** | 0.696 | 0.739 (+6%) | **0.876 (+26%)** ⭐ |
| **BERTScore (ko)** | 0.842 | 0.901 (+7%) | **0.957 (+14%)** ⭐ |
| **sample-wise win rate** | — | — | **50/50 (100%)** ⭐ |

> v2 의 worst case (0.786) ≥ Base 의 max (0.783) — 모든 분위에서 우월

---

## ✨ 주요 산출물

### 모델
- **v1 LoRA** — text-only LoRA (12.6h 학습, eval_loss 0.130)
- **v2 LoRA** — Vision Tower 활성화 + AI 분석 이미지 학습 (5.1h, eval_loss 0.080)

### 데이터 파이프라인
- MySQL 3,355 건 + AI 이미지 매칭 → ShareGPT 6,610 학습 샘플
- held-out 50건 평가셋 (학습에서 제외)

### 벤치마크 인프라
- 3-way runner (`vlm/bench/runner.py`)
- N-way 스코어러 (`vlm/bench/scorer.py`) — ROUGE-L / BERTScore / JSON / Grade / Number
- 자동 실행 (`scripts/run_3way_benchmark.py`)
- 정성 분석 (`scripts/analyze_failures.py`)
- 시각화 3장 — boxplot, CDF, scatter

### 데모 + API
- Streamlit 3패널 데모 (`vlm/demo/app.py`)
- FastAPI REST 서버 (`vlm/api/server.py`)
  - X-API-Key 인증
  - slowapi rate limiting (분당 N회)
  - 추론 타임아웃 (180초)
  - 경로 traversal 차단
  - JSON 구조적 로깅 + X-Request-ID 분산 추적

### 운영 준비도
- 환경변수 config override (12개 변수)
- GitHub Actions CI (Python 3.11/3.13 매트릭스)
- Dockerfile + docker-compose (NVIDIA GPU)
- Makefile (17개 타겟)

### 양자화 평가
- INT4 NF4 trade-off 정직 분석 — VRAM -70%, 품질 일부 저하
- bf16 운영 권장, INT4 는 엣지 케이스 한정

---

## 🧪 테스트

**38/38 단위 테스트 통과**

| 영역 | 테스트 |
|---|---|
| 스키마 | 5 |
| API 모델 | 7 |
| Config 로더 | 3 |
| JSON 파서 | 8 |
| 인증 | 4 |
| 로깅 | 4 |
| 환경변수 override | 7 |

**End-to-End 검증**
- `scripts/test_inference.py` (3 샘플)
- `scripts/test_demo_pipeline.py` (4 샘플)
- `scripts/test_api.py` (4 샘플)

---

## 📚 문서

- [README.md](../README.md) — 30초 어필 + 빠른 시작
- [ARCHITECTURE.md](../ARCHITECTURE.md) — 전체 설계
- [PROGRESS.md](../PROGRESS.md) — 주차별 이력
- [CHANGELOG.md](../CHANGELOG.md) — 버전별 변경
- [notebooks/dataset_analysis.md](../notebooks/dataset_analysis.md) — 데이터셋 분석
- [vlm/bench/score_report.md](../vlm/bench/score_report.md) — 벤치마크 점수표
- [vlm/bench/failure_analysis.md](../vlm/bench/failure_analysis.md) — 실패 케이스 분석
- [vlm/train/quantization_report.md](../vlm/train/quantization_report.md) — 양자화 평가

---

## 🛠️ 기술 스택

`Qwen3-VL-8B-Instruct` · `LLaMA-Factory` · `LoRA (rank 64)` · `PyTorch 2.7` ·
`Transformers 4.55+` · `PEFT 0.18` · `bitsandbytes 0.49 (NF4)` · `Pydantic v2` ·
`FastAPI` · `Streamlit` · `slowapi` · `python-json-logger` · `pytest 8` ·
`MySQL 8` · `Docker` · `GitHub Actions`

---

## 🔍 한계 및 향후 과제

- 단일 일자(2026-04-22) 데이터로 학습/평가 → 시간적 일반화 미검증
- 등외 케이스 학습 데이터 부재 (현재 hand-crafted 샘플로만 평가)
- 단일 도축장 데이터 → 다른 시설 일반화 불확실
- INT4 NF4 품질 저하 → GPTQ/AWQ 재시도 후보

자세한 향후 계획은 [CHANGELOG.md `Unreleased` 섹션](../CHANGELOG.md) 참고.

---

🤖 [Korean Livestock VLM Copilot](https://github.com/Yanghyuck/VLM)
