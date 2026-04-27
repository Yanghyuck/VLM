# =============================================================================
# Dockerfile — VLM Korean Livestock Copilot
# -----------------------------------------------------------------------------
# 베이스: PyTorch + CUDA 12.6 (RTX 4090 호환)
# 용도: FastAPI 서버 또는 Streamlit 데모를 컨테이너로 실행
#
# 빌드:
#   docker build -t vlm-livestock .
#
# 실행:
#   # GPU 필수
#   docker run --gpus all -p 8000:8000 \
#       -v $(pwd)/config.json:/app/config.json \
#       -v $(pwd)/vlm/train/output:/app/vlm/train/output \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       vlm-livestock
#
# 주의: 모델 가중치(8.9B)와 LoRA 어댑터(666MB)는 이미지에 포함하지 않음.
#       볼륨 마운트로 호스트의 모델을 사용.
# =============================================================================

FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리
WORKDIR /app

# Python 의존성 (변경이 적은 항목 → 자주 변경되는 항목 순)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사 (대용량 파일 .dockerignore 로 제외)
COPY vlm/    /app/vlm/
COPY scripts/ /app/scripts/
COPY tests/  /app/tests/
COPY config.example.json /app/config.example.json
COPY pytest.ini /app/pytest.ini

# 기본 설정 (런타임에 마운트로 덮어쓰기 가능)
RUN cp config.example.json config.json

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# 기본은 FastAPI 서버 실행 (config.json 의 host/port 사용)
EXPOSE 8000 8501

CMD ["python", "vlm/api/server.py"]
