# =============================================================================
# Makefile — VLM 프로젝트 공통 명령
# =============================================================================
# 사용:
#   make <target>
#
# 주요 타겟:
#   make help            도움말 출력
#   make install         pip 의존성 설치
#   make test            pytest 실행
#   make demo            Streamlit 데모 실행
#   make api             FastAPI 서버 실행
#   make analyze         데이터셋 분석 그림 생성
#   make build-data      DB → JSONL 빌드
#   make build-eval      평가셋 50건 생성 (held-out)
#   make convert         학습 데이터 변환 (평가셋 제외)
#   make train-v1        v1 학습 (text-only LoRA)
#   make train-v2        v2 학습 (Vision LoRA + AI)
#   make bench-base      베이스 모델 추론
#   make bench-lora      LoRA 모델 추론
#   make score           벤치마크 점수 계산
#   make clean           로그·임시파일 삭제
# =============================================================================

# Conda 환경 + Python 경로 (Windows)
PY := C:/Users/IPC/.conda/envs/vlm/python.exe
LF := C:/Users/IPC/.conda/envs/vlm/Scripts/llamafactory-cli.exe

# 경로 변수
EVAL_SET := vlm/bench/eval_set.jsonl
TRAIN_JSON := vlm/data/livestock_train.json
LF_DATA_DIR := C:/Users/IPC/Desktop/git/LLaMA-Factory/data

.PHONY: help install test demo api analyze \
        build-data build-eval convert train-v1 train-v2 \
        bench-base bench-lora score clean

help:
	@echo "VLM 프로젝트 — Makefile 타겟 목록"
	@echo ""
	@echo "  make install      - pip install -r requirements.txt"
	@echo "  make test         - pytest tests/"
	@echo "  make demo         - streamlit run vlm/demo/app.py"
	@echo "  make api          - python vlm/api/server.py"
	@echo "  make analyze      - 데이터셋 분석 그림 생성"
	@echo ""
	@echo "  make build-data   - DB to JSONL 빌드"
	@echo "  make build-eval   - 평가셋 50건 (held-out)"
	@echo "  make convert      - 학습 데이터 변환 (eval 제외)"
	@echo "  make train-v1     - v1 학습 (text-only LoRA)"
	@echo "  make train-v2     - v2 학습 (Vision LoRA + AI)"
	@echo ""
	@echo "  make bench-base   - 베이스 모델 추론"
	@echo "  make bench-lora   - LoRA 모델 추론"
	@echo "  make score        - 벤치마크 점수"
	@echo ""
	@echo "  make clean        - 로그·임시파일 삭제"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/

demo:
	$(PY) -m streamlit run vlm/demo/app.py --server.port 8501

api:
	$(PY) vlm/api/server.py

analyze:
	$(PY) notebooks/dataset_analysis.py

# ── 데이터 파이프라인 ────────────────────────────────────────
build-data:
	$(PY) scripts/build_dataset.py

build-eval:
	$(PY) vlm/bench/dataset.py --source jsonl --n 50

convert: build-eval
	$(PY) vlm/train/convert_dataset.py --exclude-eval-set $(EVAL_SET)
	cp $(TRAIN_JSON) $(LF_DATA_DIR)/

# ── 학습 ─────────────────────────────────────────────────────
train-v1: convert
	$(LF) train vlm/train/qwen3vl_lora.yaml

train-v2: convert
	$(LF) train vlm/train/qwen3vl_lora_v2.yaml

# ── 벤치마크 ────────────────────────────────────────────────
bench-base:
	$(PY) vlm/bench/runner.py --mode base --n 50

bench-lora:
	$(PY) vlm/bench/runner.py --mode lora --n 50

score:
	$(PY) vlm/bench/scorer.py \
	    --base vlm/bench/results_base.jsonl \
	    --lora vlm/bench/results_lora.jsonl

bench-all: bench-base bench-lora score

# 3-way 벤치마크 (Base / v1 / v2 동시)
bench-3way:
	$(PY) scripts/run_3way_benchmark.py

clean:
	rm -f vlm/train/train.log vlm/train/train_err.log
	rm -f vlm/train/test_inference.log vlm/train/streamlit.log
	rm -f vlm/train/demo_pipeline.log vlm/train/train_v2.log
	rm -f vlm/api/api_server.log vlm/api/api_test.log
	rm -f vlm/bench/runner_base.log vlm/bench/runner_lora.log
	rm -rf vlm/data/tmp/
	@echo "임시파일 정리 완료"
