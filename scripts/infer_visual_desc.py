# =============================================================================
# scripts/infer_visual_desc.py
# -----------------------------------------------------------------------------
# 기능:
#   학습된 LoRA 어댑터(또는 base)로 평가셋 이미지에 visual_desc(시각 서술)를
#   추론하여 예측 파일을 생성합니다. 이 출력을 eval_visual_desc.py score 의
#   --pred 로 넣어 채점합니다.
#
# 입력  : eval_set.jsonl (vlm/bench/dataset.py 산출) — 각 {id, image_path}
# 출력  : {id, image_path, abnormal, error_code, response} jsonl
#         response = "도체 전체 형태: …\n등지방층 외형: …" (학습 타깃과 동일 포맷)
#
# 사용:
#   # LoRA 어댑터 (config.json paths.lora_adapter 기본)
#   python scripts/infer_visual_desc.py --eval-set vlm/bench/eval_set.jsonl \
#       --output vlm/bench/results_visual_desc_lora.jsonl
#   # 베이스 모델 baseline
#   python scripts/infer_visual_desc.py --no-adapter \
#       --output vlm/bench/results_visual_desc_base.jsonl
#
# 평가 흐름:
#   python scripts/infer_visual_desc.py ...           # 예측 생성
#   python scripts/eval_visual_desc.py score \
#       --pred vlm/bench/results_visual_desc_lora.jsonl \
#       --refs vlm/data/visual_desc_refs.jsonl
#
# 의존성: torch, transformers, peft, qwen_vl_utils, pillow
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vlm.config import CFG
from vlm.train.convert_dataset import VISUAL_DESC_PROMPT

BASE_MODEL = CFG.model.base_model_id
ADAPTER    = str(ROOT / CFG.paths.lora_adapter)
MAX_PIXELS = int(getattr(CFG.model, "image_max_pixels", 200_704))


def _is_normal(ec: dict) -> bool:
    return all((v or 0) == 0 for v in ec.values())


def load_model(use_adapter: bool, adapter_path: str):
    label = f"LoRA={adapter_path}" if use_adapter else "base(어댑터 미적용)"
    print(f"[infer] 모델 로딩: {BASE_MODEL} ({label})")
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if use_adapter and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        if use_adapter:
            print(f"[infer] 어댑터 없음 — base 로 추론 ({adapter_path})")
        model = base
    model.eval()
    return model, proc


def describe(model, proc, image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": VISUAL_DESC_PROMPT},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=400,
            do_sample=False, num_beams=1, repetition_penalty=1.05,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=str, default="vlm/bench/eval_set.jsonl")
    ap.add_argument("--adapter", type=str, default=ADAPTER, help="LoRA 어댑터 경로")
    ap.add_argument("--no-adapter", action="store_true", help="base 모델로 추론(baseline)")
    ap.add_argument("--output", type=str, default="vlm/bench/results_visual_desc_lora.jsonl")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(ROOT / args.eval_set, encoding="utf-8")]
    print(f"[infer] 평가셋 {len(samples)}건")

    model, proc = load_model(use_adapter=not args.no_adapter, adapter_path=args.adapter)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            ec = s["metadata"]["error_code"]
            try:
                resp = describe(model, proc, s["image_path"])
            except Exception as e:
                resp = f"[ERROR] {e}"
            f.write(json.dumps({
                "id":         s["id"],
                "image_path": s["image_path"],
                "abnormal":   not _is_normal(ec),
                "error_code": ec,
                "response":   resp,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(samples)}] id={s['id']}")

    print(f"\n[infer] 완료 → {out_path}")


if __name__ == "__main__":
    main()
