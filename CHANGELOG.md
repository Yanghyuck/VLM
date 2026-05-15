# Changelog

VLM Korean Livestock Copilot 프로젝트 변경 이력.
형식은 [Keep a Changelog](https://keepachangelog.com/) 를 따릅니다.

---

## [Unreleased]

### Added — 자유 chat CLI (옵션 A)
- `scripts/chat_vlm.py` — 멀티턴 + 이미지 첨부 (`/image PATH`) + LoRA 토글
  - `--use-adapter` 로 v4 LoRA 적용, 기본은 베이스 Qwen3-VL-8B (일반 chat 권장)
  - 세션 명령: `/image PATH`, `/clear`, `/quit` (또는 `/q`)
  - `image_max_pixels=200,704` 자동 LANCZOS 리사이즈 (학습 일치)
  - Windows cp949 회피: stdout/stderr utf-8 강제
  - 스모크: 텍스트만 3.4s/33t, 이미지+텍스트 4.8s/48t
  - 주의: VLM FastAPI 서버 가동 중이면 GPU OOM (먼저 종료)

### Added — thema_pa_VLM 끝단 운영 흐름 9건 검증 (2026-05-15)
- `scripts/run_pa_then_vlm_on_new_images.py` — Phase A (PA) + Phase B (VLM) 통합 스크립트
  - **Phase A (ThematecPA)**: 새 ORI 이미지 9건 → YOLO 6 + gender + rightside + inpaint → AI 이미지 9/9 생성 (init 2.8s + 평균 0.8s/건)
  - **Phase B (SendVLMReport)**: AI 이미지 9건 → 운영 RestAPI 클래스 호출 → **9/9 PASS** (평균 18.9s/req)
  - **Phase B-ORI** (대체 모드): PA 가동 불가 환경 우회 — ORI 직호출 9/9 PASS (평균 19.1s/req)
- 산출물: `thema_pa_VLM/images/AI_run/0716_ai_*.jpg` + `storage/vlm_reports/20260212_{3..17}_vlm_report.json`
- 응답 품질: 환각 없음, 성별·등급 일관 ("암컷"/"1+"), 정상 케이스 톤 일관 ("정상 출하 처리하세요")
- 입증 흐름: ORI → ThematecPA(YOLO+gender+rightside+inpaint) → AI 결과 → SendVLMReport → VLM v4 → 한국어 리포트 → 저장

### Fixed — thema_pa_VLM `_resolve_vlm_grade` 시그니처 모순 (별도 리포)
- `business/thematec_pcw.py` — `@staticmethod` + `def _resolve_vlm_grade(self, result, ...)` 모순으로 운영 호출 시 `TypeError`
- 호출부 (`self._resolve_vlm_grade(result, ...)`) 와 정렬되도록 `@staticmethod` 제거 → instance method
- 본 회차 검증 전엔 RestAPI 직호출(E2E 우회 경로)만 검증해서 미발견. 운영 호출 흐름 점검에서 발견·수정.

### Changed — thema_pa_VLM `_send_vlm_api` 비동기화 (별도 리포)
- `business/thematec_pcw.py` — `import threading` + `_worker()` 클로저를 `daemon Thread` 로 위임
- 이유: 도체당 ~25s 동기 호출이 도축 라인 처리 속도(시간당 100-200마리, ≈20s/마리) 와 충돌
- 검증: 모킹 스모크에서 두 호출 4.9ms 즉시 반환, 워커 2개 병렬 시작 (간격 0.6ms), 2초 후 전부 종료, 워커 leak 없음
- thema_pa_VLM 측 커밋·푸시는 사용자 검토 후 (메모리 규칙 B2 정책)

### 향후 계획
- v1.2.0 GitHub Release 페이지는 웹 UI 에서 작성 (gh CLI 미설치)
- chat CLI 의 B/C 확장 — FastAPI `/v1/chat` 엔드포인트, Streamlit chat UI
- thema_pa_VLM 측 변경(_resolve_vlm_grade 수정 + 비동기화) 사용자 검토·커밋

---

## [v1.2.0] — 2026-05-15 (v4 어댑터 운영 채택 + 검출 실패 환각 근본 해결)

### Operations — v4 운영 채택 + thema_pa_VLM E2E 재검증 (2026-05-15)
- `config.json` `paths.lora_adapter`: `qwen3vl-lora` (v2-corrected) → **`qwen3vl-lora-v4`**
- `config.example.json` 동일 갱신 (운영 권장 템플릿)
- VLM FastAPI `/v1/health`: `model_used=lora`, `adapter_exists=true` 확인
- `scripts/test_e2e_thema_pa_bridge.py` **4/4 PASS** (평균 25.7s/req)
  - normal_case 20.1s / backfat_error 34.6s / entry_error 28.0s / sample_3473 20.0s
- 저장 검증: `thema_pa_VLM/storage/vlm_reports/{ymd}_{pigno}_vlm_report.json` 4건 생성
- 환각 해결 운영 흐름에서도 재현: backfat_error_case 응답이 입력 `암컷` + `검출 실패` 정확 명시

### Added — 응답 품질 후처리 (A3/A4)
- `vlm/postprocess.py` — LoRA 응답 dict 한국어 후처리 (순수 파이썬, 학습 무관 즉시 적용)
  - **A3 한국어 조사 정규화**: "거세으로" → "거세로", "1+으로" → "1+로", "등외으로" → "등외로", "2으로" → "2로"
  - **A4 등급 정합성**: 입력 `grade` 와 다른 등급 단언("최종 2 등급", "X 등급으로 판정")을 입력 grade 로 강제 교체. 비교 문맥("1+ 등급 도체 대비")은 보존.
  - 변경 발생 시 `_postprocess` 메타필드(`josa_normalized` / `grade_enforced`)에 흔적 기록
- `vlm/train/inference.py` — `generate_report(..., postprocess=True)` 인자 추가 (기본 ON, 벤치마크용 OFF 가능)
- `tests/test_postprocess.py` — 단위 테스트 18건 (정상 텍스트 미변경 + 비교 문맥 보존 회귀 포함)

### Fixed — B2 thema_pa_VLM 결과 저장 경로 cwd 의존성 제거
- `thema_pa_VLM/comm/rest_api.py` (별도 리포)
  - `PROJECT_ROOT = Path(__file__).resolve().parent.parent` 추가
  - `save_vlm_response_json` 의 `output_dir` 가 상대경로면 PROJECT_ROOT 기준으로 해석. 절대경로면 그대로.
  - 효과: cwd 가 어디서 실행되든 결과 파일이 항상 thema_pa_VLM 루트의 같은 위치에 저장. 운영 시 systemd/스케줄러 cwd 가 임의여도 산출물 추적 끊김 없음.
- `tests/test_thema_pa_vlm_bridge.py` — 회귀 테스트 2건 추가:
  - `test_save_vlm_response_resolves_relative_path_under_project_root` — cwd 변경해도 PROJECT_ROOT 기준 저장 확인 + cwd 아래 잘못 생성 가드
  - `test_save_vlm_response_absolute_path_unchanged` — 절대경로 명시 시 그대로 동작
- 전체 79/79 PASS

### Added — C5 system_prompt few-shot 예시 (큰 효과로 채택)
- `vlm/prompt/system_prompt.txt` — 정상 + 검출실패 in-context 예시 2개 추가
- 학습된 모델임에도 **환각 잔재 거의 제거**
  - backfat_error: "거세 암컷" 환각 사라짐
  - entry_error: "거세 판정 오류" 환각 사라짐
  - normal_case: "도체중 등급 하락 가능성" 도메인 깊이 ↑
- 추론 시간 4-15% 증가 (입력 토큰 ↑) — 응답 품질 대비 미미
- 후처리 메타필드(gender_conflict_detected 등) 트리거 안됨 = 응답 자체가 깨끗
- C 카테고리 가장 큰 임팩트. 운영 default ON.

### Trained — v4 LoRA 어댑터 (데이터 보강 + abnormal 학습) — **운영 권장**
- `vlm/train/qwen3vl_lora_v4.yaml` — rank 64 유지, 데이터만 v4 augmented (8,110 샘플)
- 학습 시간 **6h 19m** (v3 34h 대비 **1/5.7**)
- 메트릭: train_loss 0.160 / eval_loss 0.077 — v2-corrected 동등, over-fit 없음
- 6-way 벤치 (held-out 50건, 100% 정상): ROUGE/BERT/Distinct 모두 v3 동일 — 정상 케이스에서 동등 성능
- **검출 실패 스모크** (`vlm/train/test_inference_v4.md`) — **환각 근본 해결**
  - backfat_error: "거세 암컷" 환각 사라짐, 검출 실패 항목 명시
  - entry_error: "거세 판정 오류" 환각 사라짐, 비정상 진입 명확
- **운영 권장 변경: v2-corrected → v4**

### Trained — v3 LoRA 어댑터 (rank 128/alpha 256) — **over-fit 입증, 비채택**
- `vlm/train/qwen3vl_lora_v3.yaml` — rank 64→128, alpha 128→256 (capacity 2배)
- 학습 시간 **34h 15m** (v2-corrected 5h 2m 대비 6.8배 회귀) — GPU 메모리 한계 + Vision LoRA 곱셈 부담
- 메트릭: train_loss 0.147 (-11%), eval_loss 0.072 (-9% vs v2-corrected) — 표면적 개선
- **5-way 벤치 결정적 진단**:
  - ROUGE-L 1.0000 + BERTScore 1.0000 = 학습 reference 완전 암기
  - Distinct-1 -34.9% / Distinct-2 -30.5% (vs base) = 응답 표현 다양성 상실
  - eval_loss 가 train_loss 보다 낮은 건 val_size=0.1 (train 의 in-distribution 일부)
- **운영 비채택** — capacity 증가가 환각 해결이 아닌 표면 암기로 귀결. v2-corrected 가 다양성/속도/비용에서 우위.
- v3 가 입증한 것: 환각의 진짜 해결책은 **학습 데이터 다양화 (B 카테고리, v4)** 이지 capacity 증가가 아님

### Added — A3 reference paraphrase + A4 응답 다양성 메트릭
- `vlm/train/convert_dataset.py` — `_summary_response_alt(meta)` 헬퍼 (등급/측정값 우선 순서 paraphrase)
- `vlm/bench/dataset.py:_build_tasks` — `references` 리스트 (A3 paraphrase 1개 + 원본) 추가, `reference` 단일 필드는 호환성 유지
- `vlm/bench/scorer.py`:
  - `compute_rouge_l_max(pred, refs)` — paraphrase 들 중 max ROUGE-L
  - `compute_distinct_n(texts, n)` — A4 distinct-1/2 다양성
  - 리포트에 `rouge_l_max` / `distinct_1` / `distinct_2` 행 추가
- 4-way 결과:
  - ROUGE-L max: base 0.696→0.703 (paraphrase +1%), v2 어댑터들은 학습 패턴 강하게 매칭해 변화 없음
  - **Distinct (응답 다양성)**: base > v1 > v2-corrected > v2-prejosa
  - **v2-corrected vs v2-prejosa**: Distinct-1 +21% / Distinct-2 +21% — 데이터 정제로 표현 다양성 회복
  - 종합: v2-corrected 가 학습 데이터 표면 모방을 덜 한다는 객관적 신호. ROUGE-L 미세 하락은 다양성 trade-off

### Added — D1 Constrained decoding 인프라 (default OFF, 회귀로 비채택)
- `vlm/train/inference.py` — `RESPONSE_JSON_SCHEMA` + `generate_report(constrained=True)` 인자
- transformers 5.x ↔ lm-format-enforcer 0.11.x 호환 monkey-patch 추가 (`PreTrainedTokenizerBase` 위치)
- `tests/test_constrained_decoding.py` — 인프라 6 테스트 (스키마 + parser 빌드 + monkey-patch + default OFF)
- `requirements`: lm-format-enforcer 설치 (수동 — `requirements.txt` 미반영, 향후 정리)
- 스모크 결과: **회귀 발생** (`vlm/train/test_inference_constrained.md`)
  - 모델이 학습 패턴으로 inner JSON 시작 → outer string 값에 박혀 응답 절단
  - 운영 default 는 OFF 유지. 향후 prefix 강제 / propertyOrder 등 튜닝 후 재시도

### Added — D2 sampling + D3 beam search 인자 (default OFF, 비채택)
- `vlm/train/inference.py` — `sampling`/`temperature`/`top_p`/`num_beams` 인자
- `scripts/test_inference_modes.py` — 3샘플 × 3모드 비교
- 결과 (`vlm/train/test_inference_modes.md`):
  - D2 sampling_t03: 검출 실패 케이스에서 반복 폭주 부활 (repetition_penalty=1.05 효과 약화) — 비채택
  - D3 beam4: 학습 흔한 표현으로 회귀 (D5 가드 무시) — 케이스별 trade-off, 비채택
  - greedy(default): 가장 일관됨, 운영 default 유지

### Added — D5 system_prompt 환각 가드 + D4 A5 성별 정합성 후처리
- `vlm/prompt/system_prompt.txt` — "필수 준수사항" 4가지 추가
  1. 입력값 그대로 사용 (성별/등급/측정값 추론·변경 금지)
  2. 검출 실패 항목을 "정상 완료"로 표현 금지
  3. JSON 무결성 (코드블록·중복키 금지)
  4. 같은 단어/구절 반복 금지 (5회 이상 등장 X)
  - 효과: 스모크에서 "검출이 정상 완료" 환각 완전 해결 → "정상 완료되지 않아"로 정확 교정
- `vlm/postprocess.py` — A5 성별 정합성
  - `enforce_gender(text, expected_gender)` — "(다른성별)으로 판정" 단언만 입력 성별로 교체
  - `detect_gender_conflict(text, expected_gender)` — 다른 성별 단어 등장 검출 (정정 X, 메타데이터)
  - `apply_postprocess(..., expected_gender=)` — A5 통합 (`gender_enforced` / `gender_conflict_detected` 메타필드)
- `vlm/train/inference.py` — `apply_postprocess` 에 `output.gender.label()` 자동 전달
- `tests/test_postprocess.py` — A5 단위 테스트 12건 (전체 94/94 PASS)
- 스모크 첫 실전 트리거: backfat_error 케이스의 "거세 암컷" 환각이 `gender_conflict_detected` 로 가시화

### Fixed — 추론 안정성 + A4 패턴 보강 (v2-corrected 스모크 회귀 대응)
- `vlm/train/inference.py` — `generate(..., repetition_penalty=1.05)` 추가
  - 회귀 사례: 검출 실패 입력에서 greedy 디코딩이 같은 토큰 시퀀스 반복 폭주 (71s, 무한반복, 이중 JSON)
  - 효과: backfat_error 케이스 71.4s → 15.7s, 정상 JSON 으로 회귀 해소
- `vlm/postprocess.py` — A4 등급 정합성 패턴 확장
  - 추가: "이의 신청 가능: X 등급" / 전각 콜론 변형 허용
  - 기존 비교 문맥 보존 정책 유지
- `tests/test_postprocess.py` — A4 새 패턴 3 테스트 (전체 82/82 PASS)

### Trained — v2-corrected LoRA 어댑터 (정제 데이터 재학습)
- 데이터 정제 후 동일 YAML(`qwen3vl_lora_v2.yaml`)로 558 step / 3 epoch 재학습 (5h 2m)
- **train_loss 0.187 → 0.166** (-11%), **eval_loss 0.130 → 0.079** (-42% ⭐)
- 출력: `vlm/train/output/qwen3vl-lora/adapter_model.safetensors` (840 MB)
- 이전 어색 조사 학습본은 `qwen3vl-lora-v2-prejosa/` 로 백업 보존
- 차후 작업: v1 / v2-prejosa / v2-corrected 3-way 벤치마크 + 후처리 의존도 측정

### Changed — 학습 데이터 패턴 보강 (조사 + 성별 라벨)
- **근본 원인 수정**: `livestock_train.json` 에 박혀 있던 `거세으로` 1,666건 + `암퇘지/수퇘지으로` 1,639건 (총 3,305건) 어색 패턴 제거
- `vlm/train/convert_dataset.py`
  - `_eul_ro(word)` 헬퍼 추가 — 종성 유무 검사로 `으로`/`로` 정확 선택 (한글 음절 코드 기반)
  - `_summary_response` 가 `{gender}{_eul_ro(gender)}` 사용 — 성별 라벨에 맞는 조사 자동 선택
  - `GENDER_MAP`: 성별 라벨을 **암퇘지/수퇘지/거세** → **암컷/수컷/거세** 로 통일
- `vlm/schema/thema_pa_output.py` — `Gender.label()` 동일 변경 (입력 JSON 의 gender 정수는 그대로, 표시 텍스트만 변경)
- `vlm/demo/app.py`, `notebooks/dataset_analysis.py` — 라벨 매핑 동기화
- 재생성 결과: 어색한 패턴 0건, 정상 패턴(`거세로`/`암컷으로`/`수컷으로`) 3,305건
- `tests/test_convert_dataset.py` — 16 단위 테스트 (조사 회귀 가드 포함). 전체 77/77 PASS.

### Roadmap (v1.2.0 시점)
- 데이터 추가 수집 (다른 도축장, 다른 일자, 등외 케이스 포함)
- GPTQ / AWQ 양자화 재시도 (NF4 대비 품질 보존 기대)
- HTTPS 리버스 프록시 구성 가이드
- Prometheus `/metrics` 엔드포인트
- 데모 영상/GIF
- thema_pa_VLM `main.py` 파이프라인 자동 트리거 검증

---

## [v1.1.0] — 2026-05-08 (5주차 — thema_pa 시스템 통합)

### Added — thema_pa ↔ VLM 브릿지
- 연동 대상 폴더를 `thema_pa` → `thema_pa_VLM` 으로 전환 (원본 미수정 사본 사용)
- `scripts/export_from_db.py` — 이미지 경로 자동 매칭 (AI / ORI 패턴, `scan_images()` map → `result_image_path` 자동 채움)
- `tests/test_thema_pa_vlm_bridge.py` — 통합 테스트 5건 (`THEMA_PA_ROOT` 환경변수 기반, 미존재 시 skip)
- `storage/vlm_reports/` — 호출 결과 누적 (gitignore + `.gitkeep` 만 추적)
- `scripts/test_e2e_thema_pa_bridge.py` — 실 네트워크 + 추론 + 저장까지 검증하는 E2E 스크립트

### Added — API 안정성 (lifespan warm-up)
- `vlm/api/server.py` — `lifespan` 단계에서 더미 추론 1회 수행 (cold-start 제거)
- `config.api.warmup_on_startup` 플래그
- `config.api.inference_timeout_sec` 기본 180 → 240 (복잡 입력 여유)

### Changed
- `requirements.txt` — `numpy<2.3` 핀 (opencv-python 4.12 호환)

### Results — E2E 실호출 검증 (v2 어댑터)
| 회차 | 결과 | 평균 추론 | 비고 |
|---|---|---|---|
| 1차 (cold) | 3/4 | — | 1건 180s timeout (첫 호출 130s warm-up 영향) |
| 2차 (warm-up + timeout 240) | **4/4** ⭐ | **25.5초/req** | 첫 호출도 20.5s 안정 |

- 첫 호출 cold-start 128.9s → 20.5s (**6배 가속**)
- 통합 테스트 5/5 + E2E 4/4 모두 PASS
- 상세: [`vlm/api/e2e_thema_pa_bridge_results.md`](vlm/api/e2e_thema_pa_bridge_results.md)

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
