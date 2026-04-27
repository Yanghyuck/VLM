# Changelog

VLM Korean Livestock Copilot 프로젝트 변경 이력.
형식은 [Keep a Changelog](https://keepachangelog.com/) 를 따릅니다.

---

## [Unreleased]

### 향후 계획
- 데이터 추가 수집 (다른 도축장, 다른 일자, 등외 케이스 포함)
- GPTQ / AWQ 양자화 재시도 (NF4 대비 품질 보존 기대)
- HTTPS 리버스 프록시 구성 가이드
- Prometheus `/metrics` 엔드포인트
- 데모 영상/GIF

---

## [v1.0.0] — 2026-04-28 (4주차 완료)

### Added — 운영 준비
- 환경변수 기반 config override (`VLM_DB_PASSWORD`, `VLM_API_KEYS` 등)
- `tests/test_env_override.py` — 7개 테스트
- `CHANGELOG.md` — 본 문서

### Added — 4주차 벤치마크
- 3-way 벤치마크 (Base / v1 / v2) 50건 held-out
- `vlm/bench/runner.py` — 모델별 추론 (`--adapter-path`, `--tag`)
- `vlm/bench/scorer.py` — N-way 비교 (ROUGE-L, BERTScore ko, JSON, Grade, Number)
- `scripts/run_3way_benchmark.py` — 일괄 실행 오케스트레이션
- `scripts/analyze_failures.py` — bottom 5 정성 분석
- `notebooks/benchmark_analysis.py` — 시각화 3장 (boxplot, CDF, scatter)

### Added — INT4 양자화
- `vlm/train/inference.py` — `BitsAndBytesConfig` 통합
- `config.model.quantize` 플래그 + `quantize_mode` (nf4/int8)
- `scripts/test_quantization.py` — VRAM + 품질 검증
- `vlm/train/quantization_report.md` — trade-off 정직 분석

### Changed
- v2 LoRA 학습 (Vision Tower 활성화 + AI 이미지 + held-out 50건)
- 학습 시간 12.6h → 5.1h (image_max_pixels 절반 효과)
- v2 ROUGE-L 평균 +26%, 50/50 sample-wise 우월

### Results
| 지표 | Base | v1 | v2 | 개선 |
|---|---|---|---|---|
| ROUGE-L | 0.696 | 0.739 | **0.876** | +26% |
| BERTScore (ko) | 0.842 | 0.901 | **0.957** | +14% |
| sample-wise win rate | — | — | **50/50 (100%)** | |

---

## [v0.9.0] — 2026-04-27 (Day 1: 인프라 강화)

### Added — 데이터 분석
- `notebooks/dataset_analysis.{py,md}` — 7장 figures (등급/측정값 분포)
- `Makefile` — 17개 공통 명령

### Added — CI/CD & 컨테이너
- `.github/workflows/ci.yml` — pytest 자동 실행 (Python 3.11/3.13 매트릭스)
- `Dockerfile` + `.dockerignore` (NVIDIA GPU + 볼륨 마운트)
- `docker-compose.yml` — api + demo 서비스
- `vlm/train/json_utils.py` — torch 의존성 분리 (CI 가벼움)

### Added — API 보안 & 관측성
- `vlm/api/auth.py` — X-API-Key 검증 (`Depends`)
- `vlm/logging_config.py` — JSON 구조적 로깅 + JsonFormatter
- `slowapi` rate limiting (분당 N회)
- 요청 미들웨어 — request_id, latency, status 자동 기록
- 응답 헤더 `X-Request-ID` (분산 추적)
- `tests/test_auth.py` (4) + `tests/test_logging.py` (4)

---

## [v0.8.0] — 2026-04-25 (3주차)

### Added — 데모 + API
- `vlm/demo/app.py` — Streamlit 3패널 UI
- `vlm/api/server.py` — FastAPI (lifespan, async, executor)
- `vlm/api/schemas.py` — ReportRequest/Response Pydantic
- `vlm/config.py` — config.json 로더 (SimpleNamespace)
- `config.example.json` — 설정 템플릿

### Added — 검증
- `scripts/test_inference.py` — 추론 단위 테스트 (3 샘플)
- `scripts/test_demo_pipeline.py` — Streamlit 코드 경로 검증 (4 샘플)
- `scripts/test_api.py` — FastAPI 엔드포인트 검증 (4 샘플)

### Security
- `git-filter-repo` 로 `config.json` 전체 git 히스토리에서 제거
- `.gitignore` 에 `config.json` 등록
- 경로 traversal 차단 (image_dir 외부 접근 403)
- CORS 화이트리스트 (`allowed_origins`)
- 추론 타임아웃 (`asyncio.wait_for`, 기본 180초)
- nested JSON 파싱 (brace-counting, regex 한계 극복)

---

## [v0.5.0] — 2026-04-25 (v1 LoRA 학습 완료)

### Added — 학습 파이프라인
- `scripts/build_dataset.py` — DB + 이미지 매칭 → JSONL
- `vlm/train/convert_dataset.py` — ShareGPT 변환
- `vlm/train/qwen3vl_lora.yaml` — LoRA 설정 (rank 64, α 128)
- `vlm/train/inference.py` — 로컬 추론

### Added — 프롬프트 4종
- `vlm/prompt/system_prompt.txt` (도메인 지식)
- `normal_case.txt`, `error_case.txt`, `failure_analysis.txt`

### Results
- 학습 시간: 12시간 37분 (RTX 4090, image_max_pixels 200K)
- train_loss 0.187, eval_loss 0.130
- 어댑터 크기 666 MB

---

## [v0.1.0] — 2026-04-21 (1주차)

### Added — 기반
- `vlm/schema/thema_pa_output.py` — Pydantic 공통 모델
- `vlm/schema/samples/` — 4종 샘플 JSON
- `scripts/export_from_db.py` — DB → 샘플 JSON
- `tests/test_schema.py` — 5 테스트

---

## 주요 결정 이력

| 시점 | 결정 | 이유 |
|---|---|---|
| 2026-04-22 | **Claude API → Qwen3-VL LoRA 전환** | API 비용 + 네트워크 의존성 제거 |
| 2026-04-25 | image_max_pixels 1M → 200K | 학습 시간 79h → 12h |
| 2026-04-27 | Vision Tower frozen → trainable (v2) | 비전 단서 학습 효과 검증 |
| 2026-04-27 | 평가셋 50건 학습 데이터에서 제외 | 공정한 held-out 벤치마크 |
| 2026-04-28 | INT4 양자화 default false 유지 | 품질 trade-off 명확 |
