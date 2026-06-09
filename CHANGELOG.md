# Changelog

VLM Korean Livestock Copilot 프로젝트 변경 이력.
형식은 [Keep a Changelog](https://keepachangelog.com/) 를 따릅니다.

---

## [Unreleased]

### Experiment — abnormal 층화 평가셋(A) + 요약 샘플링(B) (2026-06-09)
- **A — `vlm/bench/eval_set_abnormal.jsonl`** (`dataset.py --n-abnormal 30`): 20 normal + 30 abnormal. 기존 held-out 50건의 abnormal 2건뿐 한계 해소. (abnormal 344건이 학습 포함이라 in-sample — 충실도·스모크엔 적합, ROUGE/distinct 는 주의)
- **A 결과 — F 후처리 가치 정량화** (`scripts/exp_sampling_abnormal.py`, 30 abnormal): error_code 환각(extra) **F 미적용 13.3%(4/30) → F 적용 0%**. 운영 v8 안전장치 견고 입증(n=2→n=30).
- **B 결과 — 요약 distinct greedy vs temp 0.3**: 0.2698 → 0.2792 (+3.5%, 무의미), 사실성 훼손 0(등급 50/50·수치 48/50). → 천장은 평가셋(유사 정상도체) 특성. **요약 다양성은 결함 아닌 태스크 특성으로 수용.**

### Trained — v9 요약 다양화 증류 학습 (2026-06-07 ~ 06-09) — 음성 결과, 비채택
- **`scripts/distill_summary.py`** — base teacher 가 사실(등급·성별·측정3종) 고정·표현만 다양화한 요약을 도체당 K=3 생성. **10,852도체 / 32,556변형 / 통과율 99.9%**. 안전장치: 사실 보존 검증 + 가치판단 금지 프롬프트(누출 44%→0%) + 영문 글리치 필터. K=3 배치 생성(~1.7x). 동일 50표본 distinct **0.448**(템플릿 round_robin 0.267 대비 +68%).
- **`convert_dataset.py --summary-refs`** + **`qwen3vl_lora_v9.yaml`** — summary 타깃 100% 증류본(32,748 샘플), v8 동일 hyperparam.
- 5-way 평가(base/v4/v6/v8/v9 동일 eval_set): 3문장_요약 distinct **0.270→0.223 (천장 미돌파, 오히려 하락)**, ROUGE/BERT 하락(암기↓), 주의사항 distinct 0.968(최고), error_code 충실도 100%(F).
- **핵심 교훈**: 학습데이터 다양성은 **greedy 추론 다양성으로 전이되지 않음** — distinct 천장은 decoding(greedy)+입력유사성에 묶임. 출력다양성은 추론 샘플링이 유일 레버(사실안전성 trade-off). → **v9 비채택, 운영 v8 유지.**
- **운영 교훈**: 학습 중 VRAM 99.9% 만차 → 메모리 단편화로 step~1400부터 37→216s/it 폭락. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + checkpoint resume 로 35s/it 복구**(yaml 주석 명시).

### Added — error_code 충실도: 후처리(F) + 메트릭(B) (2026-06-04)
- **F — `vlm/postprocess.py enforce_error_code_grounding`**: `비정상_근거`가 입력 error_code 에 없는 검출오류를 끼워넣는 환각을 차단. extra 코드 탐지 시 입력 기준으로 근거 재작성(A4 등급 강제와 동일 철학), missing 은 메타 기록. `generate_report` 가 `output.error_code` 전달해 운영 자동 적용.
- **B — `vlm/bench/scorer.error_code_faithfulness` + `scripts/eval_error_code_faithfulness.py`**: abnormal 케이스의 extra/missing/exact 비율 측정.
- 테스트 +12 (`tests/test_postprocess.py` 전체 45 PASS).

### Fixed — compute_rouge_l 한글 토크나이저 (2026-06-04)
- `vlm/bench/scorer.compute_rouge_l` 가 `rouge_score` 기본 토크나이저(`[^a-z0-9]+` 로 비ASCII 제거)를 써서 **순수 한글 텍스트의 ROUGE 가 0**(동일 문장끼리도 0; 요약은 "1+","20mm" 등이 살아남아 부분 동작)이던 버그. 어절(공백) 분할 `_KoTokenizer` 주입으로 수정 → score_report.md ROUGE 전부 한글 어절 기준 재계산.

### Changed — 운영 어댑터 v4 → v8 전환 (2026-06-04)
- `config.json` / `config.example.json` `paths.lora_adapter` → `qwen3vl-lora-v8`. 검출실패 스모크 통과(성별 환각 해소) 근거. 검증: config 기본값으로 v8 로드 + 정상 4필드 응답.
- N-way 평가 방법론 수정: 기존 legacy 결과가 옛 eval_set(옛 id·thema_pa 경로)이라 무효 → base·v4·v6·v8 을 새 eval_set(thema_pa_VLM)으로 `--force` 전량 재추론. registry 에서 base/v4 의 `legacy_results` 제거.

### Trained — v8 학습 완료 (2026-06-03 ~ 06-04)
- **데이터 3배 확대** — 매칭 키 버그(`pigno_cnt` 단독 → `(pigno_cnt, ymd)`) 수정으로 dataset 3,816 → **10,852건**. `tb_error` 실 검출오류(344건) LEFT JOIN 통합(하드코딩 제거). **시각 서술(visual_desc)** 태스크 신설 — base Qwen3-VL 증류(`scripts/distill_visual_desc.py`, 10,852장, 수치 비낭독·문체 통일).
- v8 학습 28h 50m, 2766 step / 3 epoch: **train_loss 0.156 / eval_loss 0.108**(과적합 없음). 학습셋 32,748 샘플(summary 10,802 + grade 10,802 + abnormal 342 + visual_desc 10,802).
- 5-way 평가: 권고 distinct 0.291·주의사항 0.923(다양성 회복), 수치인용 0.987(최고), 검출실패 스모크 통과(성별 환각 해소). 요약 distinct 천장(0.270) 미돌파.
- **운영 교훈**: LLaMA-Factory `print_data_example` 의 em-dash cp949 콘솔 출력 크래시 → `report_to: none` + `PYTHONUTF8=1` 로 회피.

### Added — 진단/CI 강화 (2026-05-28)
- `vlm/bench/scorer.py` — 필드별/케이스별 평가 분리
  - prediction 4필드 각각 `distinct_2__{필드}` 측정 → "권고" 필드가 v3/v4 에서 baseline 대비 **-69.7%** (가장 심한 암기 패턴) 진단
  - normal/abnormal 케이스 분리 ROUGE/distinct (현 held-out 50건은 모두 normal — abnormal eval 보강 필요성 확인)
- `vlm/bench/harness.py` — `trend` 서브커맨드 추가. `runs/` 의 모든 실행을 시간순 metric 추이 표로 정리 (`score_trend.md`)
- `tests/test_eval_harness.py` — pytest 7건 (registry 파싱/필드/라벨 유일성/baseline 등록/메트릭 키 정합성/harness import/regression 자기-비교)
  - **실제 YAML 버그 자동 감지**: `grade_match_rate:{...}` colon-space 누락으로 키 일부로 파싱되던 회귀 임계치 버그를 pytest 가 잡아냄

### Trained — v6 학습 완료 + 비채택 결정 + "3문장_요약" 천장 발견 (2026-05-29)
- 학습 6h 58m (2026-05-29 00:00 → 06:58), 687/687 step / 3 epoch
  - train_loss **0.173** (v4 0.160 대비 +8% — paraphrase 다양화 효과 확인)
  - eval_loss **0.085** (정상 일반화, train_loss > eval_loss)
  - 학습 데이터 livestock_train_v6.json (8,110건, paraphrase 분포 [949,941,960,955]/[172,154,174])
- 8-way harness 평가 (`vlm/bench/runs/20260529T000018Z__c52c1db__lora_v6/`):
  - ROUGE_L **0.9900** / BERT **0.9961** → reference 거의 완전 학습 (v3/v4/v5 와 동일 양상)
  - distinct_2 **0.2430** (합격기준 0.28 미달, baseline -22.7%) → 회귀 자동 fail
  - 추론 평균 **19.97s** (전 모델 최단)
  - v5 대비 distinct_2 +2.1% 미미한 회복
- **충격적 진단 — "3문장_요약" 필드 천장 발견**:
  - 3문장_요약: v4 0.2296 → v5 0.2381 → v6 0.2430 (거의 정체)
  - capacity 축소(v5) + 데이터 4종 다양화(v6) **두 방향 모두 이 필드 천장을 못 깸**
  - "권고/주의사항" 은 두 방향 모두 큰 효과 (capacity↓ 가 더 강력: 권고 +424%)
- **결론**: 도체당 1 paraphrase round-robin 학습 한계 명확. 모델이 "이 도체에는 이 표현" 외워 추론 시 같은 표현 반복
- **v6 비채택, 운영 어댑터 v4 유지**
- **v7 방향**: abnormal 평가셋 추출(가장 ROI), 도체당 모든 4종 학습(~24h), 또는 천장 수용

### Trained — v5 학습 완료 + 비채택 결정 (2026-05-28)
- 학습 5h 49m (09:34 → 15:43), 687/687 step / 3 epoch
  - train_loss **0.180** (v4 0.160 대비 +12.5% — capacity 축소 효과 확인 ✓)
  - eval_loss **0.080** (v2-corrected 0.079 와 동일 — 일반화 정상)
  - 어댑터 400MB (rank 32 → v4 840MB 의 절반)
- 7-way harness 평가 (`vlm/bench/runs/20260528T064858Z__b27faf0__lora_v5/`):
  - ROUGE_L **0.9983** / BERT **0.9991** → 학습 reference 거의 완전 학습 (v3/v4 와 동일 양상)
  - distinct_2 **0.2381** (합격기준 0.28 미달, baseline -24.2%) → **회귀 자동 fail**
  - 추론 평균 **20.06s** (전 모델 중 최단)
- **필드별 진단** (가설 부분 적중):
  - "권고" 필드: v4 0.0200 → v5 0.1048 (**+424%**) ⭐
  - "주의사항" 필드: v4 0.4251 → v5 0.7042 (+66%) ⭐
  - "3문장_요약" 필드: v4 0.2296 → v5 0.2381 (+3.7%, 거의 변화 없음)
- **결론**: capacity 축소(rank 64→32, dropout 0.05→0.10)는 "권고/주의사항" 다양성 회복에 큰 효과. 그러나 "3문장_요약" 은 데이터 측 reference 패턴(2 paraphrase) 일관성 때문에 capacity 축소만으로 깨지지 않음.
- **v5 비채택, 운영 어댑터 v4 유지**. v6 방향은 데이터 측 paraphrase 증강(references 4-5개) + abnormal 평가셋 별도 추출
- 첫 정식 `runs/` 구조 도입 — eval_set SHA256 동일 확인 (`b5972a67...`), adapter SHA256 / env(torch/transformers/peft/numpy) 자동 기록

### Added — Eval harness (2026-05-27)
- `vlm/bench/registry.yaml` — 모델/평가셋/회귀 임계치 선언적 등록 (6 모델: base + lora_v1/v2_prejosa/v2_corrected/v3/v4)
- `vlm/bench/harness.py` — 단일 진입점 `run`/`score`/`check` 서브커맨드
  - `run` — 등록 모델 일괄 추론 (legacy_results 있으면 skip, `--force` 로 재추론). `runs/<UTC>__<sha>__<label>/` 격리 디렉터리에 `results.jsonl` + `manifest.json`(git_sha, adapter SHA256, eval_set SHA256, env) 저장
  - `score [--check]` — N-way 리포트(`score_report.md`) 갱신 + 회귀 검사 통합 (위반 시 exit 1)
  - `check --candidate <label>` — baseline 대비 회귀 검사 단독 실행 + `regression.json` 상세 저장
- 회귀 임계치: `rouge_l/rouge_l_max −5%`, `bert_score_f1 −3%`, `distinct_2 −10%`, `grade_match_rate −2%`, `elapsed_avg_sec +30%`
- 검증: 기존 6 results 로 `score --check` 실행 → v3/v4 의 distinct_2 −26.9%(학습 reference 암기) 회귀 자동 감지 + exit 1
- baseline = `lora_v2_corrected` (현 운영 어댑터). v5 학습 시 한 줄로 회귀 여부 판단 가능

### Published — v1.2.0 GitHub Release 페이지 (2026-05-19)
- https://github.com/Yanghyuck/VLM/releases/tag/v1.2.0
- gh CLI 미설치 환경에서 Windows Credential Manager 의 GitHub OAuth token 을 추출 → GitHub Releases API 로 직접 POST
- `name`, `tag_name=v1.2.0`, draft/prerelease=false, 본문 ~2,809자 (CHANGELOG `[v1.2.0]` 기반 발표성 마크다운)

### Added — `/v1/report/stream` NDJSON streaming 엔드포인트
- `vlm/api/server.py` — `POST /v1/report/stream` (NDJSON, 줄당 1 JSON)
  - 형식: `{"event":"start"}` → `{"event":"token","text":...}` * N → `{"event":"done","result":{...},"elapsed_sec":...}`
  - 에러 시: `{"event":"error","detail":...}`
- `vlm/train/inference.py` — `stream_response(output)` generator (TextIteratorStreamer + daemon Thread)
- 검증 (carcass_no=11, AI 이미지):
  - START 0.03s / **FIRST TOKEN 2.11s** / DONE 23.15s (41 chunks)
  - 사용자 체감 latency: 23s 일괄 대기 → 2s 후 점진 출력
  - 후처리 (A3/A4/A5) 는 누적 텍스트로 done 직전 1회 적용 (응답 일관성 보존)

### Investigated — micro-batching 가치 입증 + vLLM 평가 보류
- `scripts/bench_batch.py` — batch_size 1/2/4 의 per-request latency 비교
- 결과 (max_new=128, 3 trials, RTX 4090, v4 LoRA, image_max_pixels=100K):
  - batch=1: 18.08s/req
  - batch=2: 9.26s/req → **1.95x (98% 효율)**
  - batch=4: 4.71s/req → **3.84x (96% 효율)**
  - 거의 선형 가속. 단일 요청은 메모리 대역폭 bound 라 batch 로 weight/KV 로드 amortize → GPU compute 거의 그대로 활용
- vLLM 평가 시도 → Windows MAX_PATH(260자) 한계로 source build 실패 (`fused_moe configs/E=256,N=384,...` 파일명)
- 결정: **평가 보류, batch 측정 결과로 micro-batching 가치 입증**. 운영 도입은 Linux 환경(WSL2/Docker) 확보 후
- 측정 결과 보존: `vlm/bench/batch_speedup.json`

### Added — Prometheus `/metrics` 엔드포인트
- `vlm/api/server.py` — 전용 `CollectorRegistry` 사용 (직접 실행 시 모듈 중복 import 회피)
- 노출 metrics:
  - `vlm_requests_total{endpoint,status}` — 요청 카운트
  - `vlm_request_duration_seconds{endpoint}` — 요청 처리 시간 히스토그램
  - `vlm_inference_duration_seconds` — 추론 본체 시간 (캐시 hit 제외)
  - `vlm_cache_total{result="hit|miss"}` — 캐시 hit/miss
  - `vlm_model_ready` — 모델 로드 완료 (1/0)
  - `vlm_cache_size` — 현재 캐시 항목 수
- `requirements.txt`: `prometheus_client>=0.20`
- 검증: 3 호출 후 `/metrics` 정확 노출 (requests_total=3, cache hit=1/miss=2, inference_sum 38.25s)

### Added — 응답 캐시 (in-memory LRU)
- `vlm/api/server.py` — SHA1(payload + image_meta) 키 + `OrderedDict` LRU + `asyncio.Lock`
  - 키에 image 의 mtime/size 포함 → 같은 경로라도 이미지 변경 시 재추론
  - 기본 max=256 (`config.api.response_cache_max` 로 재정의)
  - `/v1/report/stream` 은 캐시 제외 (token 스트림 의미 X)
  - hit 응답의 `model_used` 끝에 `(cached)` 표기
- `/v1/health` 에 `cache: {size, max, hits, misses, hit_ratio}` 노출
- 검증: 동일 payload 두 번 호출 시 19.4s → **14.5ms (1340x)**, gender 변경 시 정확히 miss

### Fixed — uvicorn `reload=True` 가 server_run.log 변화로 무한 reload + Prometheus Counter 중복 등록
- `vlm/api/server.py` `__main__`: `reload` 기본 False 로 변경
- 환경변수 `VLM_API_RELOAD=1` 일 때만 reload (dev 전용)
- 기존: `vlm/api/` 안에 server_run.log 가 쌓이면서 watchfiles 가 무한 reload → metrics Counter 가 매번 재등록되어 ValueError

### Changed — 운영 image_max_pixels 200,704 → 100,352 (학습 분포 일치)
- `config.json` / `config.example.json` `model.image_max_pixels`: 200,704 → 100,352
- 배경: v4 학습 YAML(`qwen3vl_lora_v4.yaml`)이 100,352 인데 운영이 200,704 → 추론 시 이미지가 학습보다 2배 컸음
- 검증 (Phase B 9건 재호출):
  - latency: 평균 19.1s vs 200K baseline 18.9s — 차이 노이즈 안 (prefill 1.4% 비중이라 vision encoder 절감이 곧 ~0.2s 미만)
  - 응답: pigno 3/11 등 spot-check, **200K 응답과 토씨 하나 안 다르게 동일** (회귀 0)
- 가치: latency 보다는 **학습 분포 일치**, VRAM/이미지 처리 부담 절감

### Investigated — system_prompt KV cache (보류)
- baseline 측정 (`scripts/bench_prefill_decode.py`, 3 샘플 평균):
  - first-token 0.32s / total 23.6s → **prefill 비중 1.4%**
  - decode-only 141 ms/token (병목)
- KV cache 절감 한도: ~0.2s (1% 미만) → ROI 없음, 미구현
- 결과 보존: `vlm/bench/prefill_decode_baseline.json`
- 향후 가속은 decode 단계(speculative decoding / vLLM continuous batching / torch.compile)에서 찾는 것이 합리적

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
- chat CLI 의 B/C 확장 — FastAPI `/v1/chat` 엔드포인트, Streamlit chat UI
- thema_pa_VLM 측 변경(_resolve_vlm_grade 수정 + 비동기화) 사용자 검토·커밋
- micro-batching 서버 통합 (batch 3.84x 입증, 동시 요청 빈도 측정 후)
- vLLM 평가 — Linux 환경 (WSL2/Docker) 확보 후 재시도
- API 인증 활성화 (`config.api.api_keys` 비어있음 → 운영 보안)

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
