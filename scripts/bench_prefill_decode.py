# =============================================================================
# scripts/bench_prefill_decode.py
# -----------------------------------------------------------------------------
# generate_report 의 시간을 prefill / decode 로 분리 측정.
#
# 방법:
#   max_new_tokens=1   → prefill + 1 token decode ≈ first-token latency
#   max_new_tokens=512 → total
#   decode_per_token  ≈ (total - first_token) / (out_tokens - 1)
#
# KV cache 효과 상한:
#   system_prompt KV 를 미리 계산해두면 prefill 의 system 부분만 절감.
#   image + user_text + decode 는 그대로.
#
# 사용:
#   conda activate vlm
#   python scripts/bench_prefill_decode.py
# =============================================================================

import io
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train import inference

SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/sample_3473.json",
]


def build_inputs(output: ThemaPAOutput):
    """generate_report 와 동일한 inputs 빌드 (단, 내부 분리 측정용)."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from vlm.config import CFG

    system_text = inference._load_prompt("system_prompt.txt")
    template = inference._select_template(output)
    user_text = template.replace("{{SUMMARY}}", output.summary())

    image_path = output.result_image_path
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
        max_pixels = getattr(CFG.model, "image_max_pixels", 200_704)
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
        text_input = inference._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = inference._processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(inference._model.device)
    else:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        text_input = inference._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = inference._processor(
            text=[text_input], padding=True, return_tensors="pt",
        ).to(inference._model.device)
    return inputs, system_text


def main():
    print("Loading model (v4 LoRA)...")
    t0 = time.time()
    inference._load_model(use_adapter=True)
    print(f"  ready in {time.time()-t0:.1f}s\n")

    import torch
    pad_id = inference._processor.tokenizer.pad_token_id or inference._processor.tokenizer.eos_token_id

    rows = []
    for rel in SAMPLES:
        p = ROOT / rel
        if not p.exists():
            continue
        output = ThemaPAOutput.model_validate_json(p.read_text(encoding="utf-8"))
        inputs, system_text = build_inputs(output)
        in_len = inputs["input_ids"].shape[1]

        # warm-up 1회 (cuda 커널 캐시)
        with torch.no_grad():
            _ = inference._model.generate(
                **inputs, max_new_tokens=1, do_sample=False, num_beams=1,
                pad_token_id=pad_id,
            )

        # Run A — first-token latency
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            _ = inference._model.generate(
                **inputs, max_new_tokens=1, do_sample=False, num_beams=1,
                repetition_penalty=1.05, pad_token_id=pad_id,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ft_lat = time.time() - t0

        # Run B — full 512
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            out_full = inference._model.generate(
                **inputs, max_new_tokens=512, do_sample=False, num_beams=1,
                repetition_penalty=1.05, pad_token_id=pad_id,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total = time.time() - t0
        out_tokens = out_full.shape[1] - in_len

        decode_only = total - ft_lat
        per_token = decode_only / max(1, out_tokens - 1)

        rows.append({
            "sample": Path(rel).stem,
            "in_tokens": in_len,
            "out_tokens": out_tokens,
            "first_token_sec": round(ft_lat, 3),
            "total_sec": round(total, 3),
            "decode_per_token_ms": round(per_token * 1000, 1),
        })
        print(f"  [{Path(rel).stem:25s}] "
              f"in={in_len:4d} out={out_tokens:3d} "
              f"first={ft_lat:5.2f}s total={total:5.2f}s "
              f"per_token={per_token*1000:5.1f}ms")

    print("\n=== Summary ===")
    if rows:
        avg_ft = sum(r["first_token_sec"] for r in rows) / len(rows)
        avg_total = sum(r["total_sec"] for r in rows) / len(rows)
        print(f"avg first_token = {avg_ft:.2f}s")
        print(f"avg total       = {avg_total:.2f}s")
        print(f"prefill 비중    = {avg_ft/avg_total*100:.1f}% of total")
        print(f"\nKV cache 효과 상한 (system_prompt 부분만 절감 가정):")
        # system_prompt 토큰 수 — 추정. prefill 의 일부.
        sys_tokens = len(inference._processor.tokenizer(rows[0]['sample'] and
                          inference._load_prompt('system_prompt.txt'))['input_ids'])
        print(f"  system_prompt 토큰 수 ≈ {sys_tokens}")
        print(f"  (참고: 평균 입력 토큰 {sum(r['in_tokens'] for r in rows)/len(rows):.0f}, "
              f"system 비중 ≈ {sys_tokens/(sum(r['in_tokens'] for r in rows)/len(rows))*100:.0f}%)")

    out_path = ROOT / "vlm" / "bench" / "prefill_decode_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
