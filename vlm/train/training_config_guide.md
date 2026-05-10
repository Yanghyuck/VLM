# LLaMA-Factory 학습 YAML 파라미터 가이드

`vlm/train/qwen3vl_lora_v2.yaml` / `qwen3vl_lora_v3.yaml` 에 등장하는 주요
하이퍼파라미터의 의미·효과·운영 가이드를 정리합니다. 본 프로젝트(Qwen3-VL-8B
+ 도체 판정 도메인) 기준 권장 값과 이유도 같이 기록합니다.

---

## 1. LoRA 핵심 (rank · alpha · dropout · target)

LoRA (Low-Rank Adaptation) 의 기본 원리:

```
output = W·x  +  (alpha/rank) · B·A·x
         ↑동결         ↑학습 (LoRA 부분)
```

원본 weight `W` 는 동결, 작은 두 행렬 `A`·`B` 만 학습합니다.

### `lora_rank` — 학습할 패턴의 차원 (capacity)

- A·B 의 작은 차원 수 (LoRA 의 "low-rank" 핵심)
- **클수록**: 표현력 ↑, 학습 파라미터 ↑, 메모리 ↑, 학습 시간 ↑
- 일반 값: 8 / 16 / 32 / 64 / 128
- v2-corrected: **64** | v3: **128** (capacity 2배)

> "기본 모델에 추가 학습할 새 차원의 폭" — 폭이 넓을수록 다양/미세한 패턴
> 학습 가능. 단 너무 크면 over-fit.

### `lora_alpha` — 학습된 패턴의 강도 (scaling factor)

- 최종 출력에 LoRA 부분이 반영되는 강도 = `alpha / rank`
- 관례: `alpha = rank × 2` 비율 유지 → scaling factor = 2 일정
- v2-corrected: **128** (alpha/rank=2) | v3: **256** (alpha/rank=2 유지)

> "새로 학습한 패턴을 모델 출력에 얼마나 강하게 섞을지" — `alpha/rank` 비율이
> 핵심이라 둘을 비례해서 같이 움직이는 게 일반적.

### `lora_dropout` — LoRA 정규화

- LoRA 부분 출력에 적용되는 dropout 확률
- 일반 값: 0.0 ~ 0.1
- 본 프로젝트: **0.05** (보수적)

> over-fit 방지 안전장치. 데이터 적거나 epoch 많으면 0.1 권장.

### `lora_target` — LoRA 를 끼워 넣을 layer

Transformer 의 후보 Linear layer:

```
[Attention block]              [FFN block (MLP)]
    q_proj                         gate_proj
    k_proj                         up_proj
    v_proj                         down_proj
    o_proj
```

VLM(Qwen3-VL)은 위에 더해 **Vision Tower** 와 **Multi-modal Projector**
의 Linear 들도 후보.

| 값 | 적용 범위 | 학습 파라미터 | 용도 |
|---|---|---|---|
| `q_proj,v_proj` | Q/V 만 (LoRA 원논문) | 매우 적음 | 가벼운 말투/스타일 |
| `q_proj,k_proj,v_proj,o_proj` | Attention 전체 | 적음 | 표준 fine-tune |
| `+ gate/up/down_proj` | Attention + FFN | 중간 | 도메인 지식 학습 |
| **`all`** | **모든 Linear (Vision/LLM/Projector)** | **많음** | **VLM 전영역 fine-tune** |

본 프로젝트: **`all`** — Vision Tower 까지 학습 (도체 이미지 시각 단서 활용
목적). 단 v3 처럼 rank 크게 키우면 메모리·시간 부담이 곱셈으로 누적되므로
주의.

---

## 2. Vision freezing 옵션

VLM 은 **Vision Tower (이미지 인코더)** 와 **LLM (텍스트 디코더)** 가 결합돼
있습니다. 어느 부분을 학습할지 선택 가능.

### `freeze_vision_tower` — 비전 인코더 동결 여부

- `true`: Vision Tower 가중치 고정 → 텍스트 layer 만 학습 (text-only fine-tune)
- `false`: Vision Tower 도 학습 (멀티모달 fine-tune)

본 프로젝트: **`false`** (v2-corrected, v3) — 도체 이미지의 시각 단서(등지방
영역, 측정 라인 오버레이)까지 학습.

> 비교: v1(text-only LoRA) 는 `freeze_vision_tower: true`. v2/v3 는 `false`.
> 4-way 벤치에서 v2 의 ROUGE-L 가 v1 보다 +21% 높은 이유 중 하나.

### `freeze_multi_modal_projector` — 비전↔LLM 연결부 동결 여부

- 비전 인코더 출력을 LLM 의 임베딩 공간으로 변환하는 작은 projection layer
- `false` 면 이 변환부도 학습 → 도메인 이미지의 LLM 표현 매핑이 미세조정됨

본 프로젝트: **`false`** — Vision Tower 와 함께 학습.

### 권장 조합

| 시나리오 | freeze_vision_tower | freeze_multi_modal_projector | 비고 |
|---|---|---|---|
| 텍스트만 fine-tune | true | true | 이미지 무시, LLM 만 |
| 멀티모달 (현재) | false | false | 이미지 시각 단서 학습 |
| 빠른 멀티모달 | false | true | Vision 만 (Projector freeze) |

---

## 3. 데이터셋

### `dataset` / `dataset_dir`

- `dataset`: LLaMA-Factory 의 `dataset_info.json` 에 등록된 데이터셋 이름
- `dataset_dir`: 데이터 파일 경로
- 본 프로젝트: `dataset: livestock_ko`, `dataset_dir: C:/Users/IPC/Desktop/git/LLaMA-Factory/data`

### `template`

- 채팅 포맷 템플릿 (`<|im_start|>user`, `<|im_end|>` 등 모델 고유 토큰)
- 본 프로젝트: **`qwen3_vl_nothink`** (Qwen3-VL 의 기본 채팅 템플릿, "thinking
  mode" 비활성)

### `cutoff_len` — 최대 시퀀스 길이 (토큰)

- 길수록: 긴 입력/출력 처리 가능, 메모리·시간 ↑
- 본 프로젝트: **1024** — 도체 판정 응답이 짧아 충분 (256 토큰 미만)

> 줄이면 학습 빨라짐. 응답 길이가 짧은 도메인이면 더 줄여도 됨 (512 등).

### `max_samples` — 학습에 사용할 최대 샘플 수

- 본 프로젝트: **30000** — 실제 데이터 6,610 보다 큼 → 전체 사용

### `overwrite_cache`

- `true`: tokenization 캐시 무시하고 재처리. 데이터 변경 시 필수.
- `false`: 캐시 재사용 (빠름)
- 본 프로젝트: **`true`** (학습 데이터 정제 후 캐시 무효화)

### `preprocessing_num_workers`

- 데이터 전처리 병렬 워커 수
- 본 프로젝트: **1** (Windows 환경 안정성)

---

## 4. 출력

### `output_dir`

- 어댑터·체크포인트·로그 저장 디렉터리
- 본 프로젝트: `vlm/train/output/qwen3vl-lora` (v2-corrected) /
  `qwen3vl-lora-v3` (v3)

### `logging_steps`

- N step 마다 로그 기록
- 본 프로젝트: **10**

### `save_steps`

- N step 마다 체크포인트 저장
- 본 프로젝트: **200** (학습 558 step 기준 200/400/558 세 시점 저장)

### `save_total_limit`

- 보존할 체크포인트 최대 개수 (오래된 것은 자동 삭제)
- 본 프로젝트: **3**

### `plot_loss`

- 학습 종료 시 loss 곡선 PNG 자동 생성
- 본 프로젝트: **`true`**

### `overwrite_output_dir`

- 기존 출력 디렉터리 내용 덮어쓰기 여부
- 본 프로젝트: **`true`** (재학습 시 필수)

---

## 5. 최적화 (Optimization)

### `per_device_train_batch_size` — GPU당 batch 크기

- VRAM 한계 + 모델 크기 + 이미지 토큰 수에 의해 제한됨
- 본 프로젝트: **1** (RTX 4090 24GB + Qwen3-VL-8B + Vision LoRA 의 한계)

### `gradient_accumulation_steps` — 기울기 누적

- 실제 batch size = `per_device_train_batch_size × gradient_accumulation_steps × GPU 수`
- micro-batch 를 N 번 누적해서 큰 batch 효과
- 본 프로젝트: **32** → **effective batch = 1×32×1 = 32**

> 메모리 압박 시 `per_device` 줄이고 `gradient_accumulation_steps` 늘리면
> 같은 effective batch 로 메모리 부담 분산. v3 의 메모리 한계 시 64까지 ↑ 고려.

### `learning_rate`

- 옵티마이저 학습률
- LoRA 는 일반 fine-tune 보다 큰 lr 가능 (LoRA 자체가 작은 perturbation)
- 본 프로젝트: **1.0e-4** (LoRA 표준 권장 범위 1e-4 ~ 5e-4)

### `num_train_epochs`

- 학습 데이터를 몇 번 반복 학습할지
- 본 프로젝트: **3.0** (6,610 샘플 × 3 epoch ≈ 558 step at effective batch 32)

> epoch 많으면 over-fit 위험. eval_loss 가 train_loss 보다 높아지기 시작하면
> 줄여야 함. 본 프로젝트는 eval < train 으로 안전.

### `lr_scheduler_type` — 학습률 스케줄러

- `cosine`: cosine 곡선으로 lr 감소 (warmup 후 부드럽게 0 으로)
- `linear`, `constant`, `polynomial` 등
- 본 프로젝트: **`cosine`** — LoRA 학습 표준

### `warmup_ratio`

- 전체 step 중 warmup 비율 (lr 0 → 목표값 선형 증가)
- 본 프로젝트: **0.1** (전체의 10% step 동안 warmup)

> 학습 초반 불안정 방지. 0.03 ~ 0.1 일반.

### `bf16` — bfloat16 mixed precision

- `true`: bfloat16 (Brain Float 16) 로 학습 → 메모리 절반, 속도 ↑
- Ampere 이상 GPU(RTX 30xx, 40xx, A100, H100) 에서 권장
- 본 프로젝트: **`true`**

> `fp16` 대비 dynamic range 가 넓어 안정성 ↑. Hopper/Ada 에서 더 빠름.

### `ddp_timeout`

- 분산 학습(DDP) 타임아웃 (초)
- 본 프로젝트: **180000000** (단일 GPU 라 무의미하지만 큰 값으로 설정)

### `dataloader_num_workers`

- 데이터 로더 병렬 워커 수
- 본 프로젝트: **0** (Windows + multimodal 안정성, 메인 프로세스에서 처리)

> Linux 환경이면 4-8 권장. Windows 는 fork 모델 미지원으로 0 안전.

---

## 6. 검증 (Evaluation)

### `val_size`

- 학습 데이터 중 validation 으로 분리할 비율
- 본 프로젝트: **0.1** (10% = 약 661 샘플 검증용)

### `per_device_eval_batch_size`

- 검증 시 GPU당 batch (학습보다 크게 가능 — gradient 계산 없음)
- 본 프로젝트: **1**

### `eval_strategy`

- `steps`: N step 마다 검증
- `epoch`: epoch 마다 검증
- `no`: 검증 안 함
- 본 프로젝트: **`steps`**

### `eval_steps`

- `eval_strategy=steps` 일 때 N step 마다 검증
- 본 프로젝트: **200** (= save_steps 와 일치)

---

## 7. Vision 입력 (Qwen3-VL 전용)

### `image_max_pixels` — 이미지당 최대 픽셀 수

- 입력 이미지가 이 값보다 크면 비율 유지하며 자동 축소
- 비전 토큰 수 = 약 `image_max_pixels / patch_size² × 1.05`
- 본 프로젝트: **100,352** (~316×316 해상도)

| 값 | 해상도 (대략) | 비전 토큰 수 (대략) |
|---|---|---|
| 50,176 | 224×224 | ~256 |
| 100,352 (현재) | 316×316 | ~512 |
| 200,704 (v1) | 448×448 | ~1024 |
| 1,000,000 | 1000×1000 | ~5000 |

> 클수록 이미지 디테일 ↑, 메모리·시간 ↑↑ (제곱 비례). 시각 정보가 필수
> 도메인 (의료/측정 라인 등) 면 큰 값, 단순 카테고리 분류면 작은 값.

### `video_max_pixels`

- 비디오 입력의 프레임당 최대 픽셀 수
- 본 프로젝트는 비디오 미사용이지만 yaml 형식상 지정: **50,176**

---

## 8. 자주 묶이는 trade-off

### 메모리 ↔ 시간

| 항목 | 메모리 ↓ 효과 | 시간 영향 |
|---|---|---|
| `gradient_checkpointing: true` | 큼 (-30~50%) | +10~30% |
| `per_device_train_batch_size` ↓ | 비례 | +비례 |
| `gradient_accumulation_steps` ↑ | 작음 | 약간 ↑ |
| `image_max_pixels` ↓ | 큼 (제곱) | -비례 |
| `cutoff_len` ↓ | 비례 | -비례 |
| `bf16: true` (이미 켬) | -50% | +10~30% |

### Capacity ↔ 비용

| 항목 | 효과 ↑ | 비용 |
|---|---|---|
| `lora_rank` ↑ | 표현력 | 메모리·시간 비례 |
| `lora_target=all` | 적용 범위 ↑ | 메모리·시간 ↑ |
| `freeze_vision_tower=false` | 시각 단서 학습 | 메모리·시간 큼 |
| `num_train_epochs` ↑ | 학습 깊이 | 시간 비례 + over-fit 위험 |

---

## 9. 본 프로젝트 학습 비교

| 항목 | v1 (text-only) | v2-corrected | v3 |
|---|---|---|---|
| `lora_rank` | 64 | 64 | **128** |
| `lora_alpha` | 128 | 128 | **256** |
| `lora_target` | all | all | all |
| `freeze_vision_tower` | true | false | false |
| `freeze_multi_modal_projector` | true | false | false |
| `image_max_pixels` | 200,704 | 100,352 | 100,352 |
| 학습 시간 | 12h 37m | **5h 2m** | **~33h** ⚠️ |
| train_loss (final) | — | 0.166 | 0.08 (step 350) |
| eval_loss (final) | 0.130 | **0.079** | TBD |

> v3 의 학습 시간 회귀 원인: rank 2배 + Vision Tower 학습 + image_max_pixels
> 100K 의 결합. GPU 메모리 24GB 한계 근접 (24.1/24.0 GB) → caching allocator
> fragmentation 으로 throughput 비선형 하락.

---

## 10. 참고 — 학습 명령

```bash
# conda env vlm 에서
conda run -n vlm --no-capture-output llamafactory-cli train vlm/train/qwen3vl_lora_v3.yaml

# 진행 모니터링
tail -3 vlm/train/output/qwen3vl-lora-v3/trainer_log.jsonl
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

---

**최종 업데이트**: 2026-05-10 (v3 학습 진행 중)
