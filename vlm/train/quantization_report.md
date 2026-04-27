# INT4 양자화 (NF4) 평가 리포트

**실행 시각**: 2026-04-28
**대상 모델**: Qwen3-VL-8B-Instruct + v2 LoRA 어댑터
**양자화 라이브러리**: bitsandbytes 0.49.2 (NF4 mode + double quant)

---

## 결과 요약

| 항목 | bf16 (기본) | **INT4 NF4** | 변화 |
|---|---|---|---|
| VRAM 사용 | ~17~22 GB | **6.75 GB** | **-70%** ⭐ |
| 평균 추론 시간 | 22~50초 | 46~112초 | **+50~120%** ⚠️ |
| 응답 품질 | 우수 | **저하 (반복, 모순)** | ⚠️ |

---

## 케이스별 품질 분석

### ✅ Case 1: normal_case (1+ 등급)

bf16/INT4 모두 정상적인 응답 생성. 약간 느려졌지만 품질 유지.

```json
{
  "3문장_요약": "도체번호 3010은 2026년 04월 23일 거세으로 도축되었습니다. 등지방 22.4mm, 뭇갈래근 48.2mm, 도체중 87.3kg으로 1+ 등급 판정. 모든 AI 검출이 정상 완료되어 1+ 최종 판정.",
  "권고": "1+ 등급으로 판정 완료되어 출하 처리 가능합니다."
}
```

### ⚠️ Case 2: backfat_error_case (등외)

INT4 에서 **품질 저하 명확**:
- 잘못된 등급 표기: "2+ 등급" (실제는 등외)
- 주의사항이 같은 문구 반복
- 권고와 등급 모순 ("등외 판정으로 25% 신뢰도 감소로 거세으로 등록됨 - 거세으로 등록")

### ⚠️ Case 3: entry_error_case (등외)

INT4 에서 **품질 저하 명확**:
- "등외 판정" → "2 등급" 으로 잘못 변환
- "비정상 진입" 인식은 유지되지만 후속 추론 흐트러짐

---

## 결론

| 사용 시나리오 | 양자화 권장 여부 |
|---|---|
| **운영 서비스 (품질 우선)** | ❌ 사용하지 말 것 (bf16 권장) |
| **엣지 디바이스 (메모리 제약 > 6GB)** | ⭕ 가능 (정상 케이스 한정) |
| **테스트/검증 (빠른 프로토타입)** | ⭕ 가능 |
| **에러 케이스 처리 필수 환경** | ❌ 절대 사용 금지 |

---

## 기술적 세부사항

### 양자화 설정

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

### 활성화 방법

```json
// config.json
"model": {
  "quantize": true,
  "quantize_mode": "nf4"
}
```

### 한계

1. **품질 저하**: 8B 모델은 4-bit 양자화에 민감 (10B+ 모델보다 손실 큼)
2. **속도 저하 (Windows GPU)**: bitsandbytes 4bit kernel이 Windows에서 최적화 부족
3. **LoRA 호환성**: PEFT + 4bit 조합에서 LoRA forward가 추가 dequant 비용 발생

### 대안 검토

향후 개선 방향:
- **GPTQ / AWQ**: 4-bit 양자화 중 품질 보존 더 좋음 (Activation-aware)
- **vLLM**: 양자화 모델 서빙 최적화
- **TensorRT-LLM**: NVIDIA 전용 추가 가속
- **llama.cpp + GGUF**: 엣지 디바이스 최적
- **양자화 없이 더 작은 모델**: Qwen3-VL-2B 등 동급 vision 모델 평가
