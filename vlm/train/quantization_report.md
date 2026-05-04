# Qwen3-VL-8B + LoRA 양자화 종합 리포트

**최종 업데이트**: 2026-04-28
**대상 모델**: Qwen3-VL-8B-Instruct + v2 LoRA 어댑터
**환경**: Windows 11 + Python 3.13 + PyTorch 2.7.1 + transformers 5.2.0

---

## 종합 비교표

| 방식 | VRAM | 추론 시간 (3샘플 평균) | 품질 (정상 케이스) | 품질 (오류 케이스) | 상태 |
|---|---|---|---|---|---|
| **bf16 (기본)** | 17~22 GB | 26 초 | ⭐ 매우 우수 | ⭐ 매우 우수 | ✅ 권장 |
| **INT8 (bitsandbytes)** | **10.15 GB (-54%)** | 96 초 (3.7×) | ✅ 양호 | ⚠️ 환각 일부 ("2+" 등급 등) | ✅ 메모리 제약 시 |
| **INT4 NF4 (bitsandbytes)** | **6.75 GB (-69%)** | 70 초 (2.7×) | ✅ 양호 | ❌ 반복·모순 명확 | ⚠️ 정상 케이스만 |
| GPTQ 4-bit (auto-gptq) | — | — | — | — | ❌ 환경 한계 |
| AWQ 4-bit | — | — | — | — | ❌ 환경 한계 |
| GGUF (llama.cpp) | — | — | — | — | ⏸ 미시도 |

---

## bitsandbytes 결과 상세

### bf16 (기본, 운영 권장)

3-way 벤치마크 50건 held-out 결과 (참고):
- ROUGE-L 0.876, BERTScore (ko) 0.957
- 모든 50건에서 Base 대비 우월 (sample-wise win rate 100%)
- 추론 시간 안정 (24.5초/샘플)

### INT8 (`bnb_8bit`)

```python
BitsAndBytesConfig(load_in_8bit=True)
```

- **VRAM**: 22GB → **10.15GB (-54%)**
- **속도**: 2~4배 느려짐 (Windows GPU bnb 커널 최적화 부족)
- **품질 검증** (3 샘플):
  - normal_case → ✅ 정확 (1+ 등급, 측정값 인용 OK)
  - backfat_error_case → ⚠️ 환각 "2+ 등급" (존재하지 않는 등급) 출력
  - entry_error_case → ⚠️ 권고와 비정상_근거 텍스트 거의 동일 (반복)
- **결론**: 정상 케이스는 OK, 오류 케이스에 약점. NF4보다는 안정적.

### INT4 NF4 (`bnb_4bit_quant_type="nf4"`)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

- **VRAM**: 22GB → **6.75GB (-69%)** ⭐ 가장 작음
- **속도**: 2배 느림
- **품질 검증** (3 샘플):
  - normal_case → ✅ OK
  - backfat_error_case → ❌ 등급 혼동, 주의사항 반복
  - entry_error_case → ❌ "등외" → "2 등급" 으로 잘못 표기
- **결론**: 메모리는 가장 작지만 8B 모델 + 오류 케이스에서 품질 손상 큼.

---

## GPTQ / AWQ 시도 — 환경 한계 정직 보고

NF4 의 약점을 보완하기 위해 GPTQ / AWQ 4-bit 양자화를 시도했으나 모두 실패. 시도 기록:

### 시도 1 — `auto-gptq` (PyPI 표준)

```bash
pip install auto-gptq
# → torch detection 실패 (build isolation 문제)

pip install auto-gptq --no-build-isolation
# → 0.3.1 설치되나 transformers 5.x 와 import 충돌 (`no_init_weights` 함수 위치 변경)

BUILD_CUDA_EXT=0 pip install auto-gptq --no-build-isolation --no-deps
# → 0.7.1 설치되나 동일한 import 충돌
```

**근본 원인**: auto-gptq 마지막 릴리스(2024-04, 0.7.1)가 transformers 5.x(2025-)와 호환 안 됨.

### 시도 2 — `gptqmodel` (auto-gptq 후속 fork)

```bash
pip install gptqmodel
# → pypcre 의존성 빌드 실패 (Windows + Python 3.13 wheel 미존재)

pip install gptqmodel --no-build-isolation
# → metadata generation failed
```

### 시도 3 — `torchao` (PyTorch 공식)

```bash
pip install torchao  # 설치 OK
```

```
import torchao
# → "Skipping import of cpp extensions due to incompatible torch version.
#    Please upgrade to torch >= 2.11.0 (found 2.7.1+cu126)"
```

torchao 0.17.0 cpp 확장은 torch 2.11+ 필요. 우리는 2.7.1 (LLaMA-Factory 호환 위해 유지).

### 시도 4 — Hugging Face 사전 양자화 모델

[`AXERA-TECH/Qwen3-VL-8B-Instruct-GPTQ-Int4`](https://huggingface.co/AXERA-TECH/Qwen3-VL-8B-Instruct-GPTQ-Int4) 다운로드 후 로드 시도:

```
OSError: It looks like the config file at '...config.json' is not a valid JSON file.
```

해당 모델은 AXERA AI 칩용 전용 포맷으로, 표준 transformers 와 호환 안 됨.

### 결론 — 4-bit 정밀 양자화 환경 호환성 매트릭스

| 라이브러리 | Linux + Py 3.10/3.11 | **Windows + Py 3.13 (현재)** |
|---|---|---|
| `auto-gptq` | ⭕ 가능 | ❌ build / import 모두 실패 |
| `gptqmodel` | ⭕ 가능 | ❌ pypcre 빌드 실패 |
| `awq` | ⭕ 가능 | ❌ (auto-gptq 와 동일 stack) |
| `torchao` (cpp) | ⭕ 가능 | ❌ torch 2.11+ 필요 |
| `bitsandbytes` (NF4/INT8) | ⭕ 가능 | ✅ **정상 작동** |

**현재 환경에서 운영 가능한 양자화는 bitsandbytes NF4/INT8 만**. GPTQ/AWQ 평가는
Linux 환경 또는 Python 3.11 다운그레이드 시 가능 (라이선스/환경 의존).

---

## 권장 사용 가이드

| 시나리오 | 권장 양자화 | 이유 |
|---|---|---|
| **운영 서비스 (품질 우선)** | **bf16** | ROUGE-L 0.876 / 100% sample win, 속도도 가장 빠름 |
| **VRAM 16GB 미만 GPU** | INT8 | 정상 케이스 품질 유지, VRAM 10GB |
| **VRAM 8GB 미만 GPU (엣지)** | NF4 | 6.75GB 까지 축소 가능, 정상 케이스 한정 |
| **에러 케이스 처리 필수** | bf16 만 | INT8/NF4 에서 환각·반복 발생 |
| **오프라인 엣지 / 임베디드** | GGUF (향후) | llama.cpp, CPU 추론도 가능 |

---

## 향후 개선 방향

| 방향 | 환경 요구 |
|---|---|
| GPTQ 4-bit 재시도 | Linux + Python 3.11 + transformers 4.x |
| AWQ 4-bit | 동일 |
| GGUF (llama.cpp) | Qwen3-VL multimodal 지원 추가 대기 (현재 미지원) |
| ONNX 변환 → INT8 quantize | onnxruntime + Qwen3-VL ONNX export 검증 |
| vLLM 서빙 + 양자화 통합 | Linux 기반 |
| 모델 다운사이징 | Qwen3-VL-2B 평가 (8B → 2B, 양자화 없이도 8GB) |

---

## 사용 방법

### bf16 (기본)

```json
// config.json
"model": { "quantize": false }
```

### INT8 활성화

```json
"model": { "quantize": true, "quantize_mode": "int8" }
```

### NF4 (4-bit) 활성화

```json
"model": { "quantize": true, "quantize_mode": "nf4" }
```

설정 변경 후 서버/데모 재시작.

---

## 첨부 결과 파일

- [`vlm/train/quantization_test_results_int8.md`](./quantization_test_results_int8.md) — INT8 3샘플 raw 결과
- [`vlm/train/quantization_test_results_nf4.md`](./quantization_test_results_nf4.md) — NF4 3샘플 raw 결과
- [`vlm/bench/score_report.md`](../bench/score_report.md) — bf16 v2 LoRA 50건 벤치마크
