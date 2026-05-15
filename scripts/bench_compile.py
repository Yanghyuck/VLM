# =============================================================================
# scripts/bench_compile.py
# -----------------------------------------------------------------------------
# torch.compile 적용 vs baseline 의 decode 속도 비교.
#
# 측정:
#   각 모드(compiled/baseline)에서 동일 sample 3회 generate(max_new=128).
#   첫 호출은 (compiled 의 경우) 컴파일 비용 포함이므로 별도 표기.
#   2~3 회 평균을 비교.
#
# 사용 (VLM API 서버는 꺼져 있어야 함):
#   conda activate vlm
#   python scripts/bench_compile.py
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

SAMPLE = "vlm/schema/samples/sample_3473.json"
N_TRIALS = 3
MAX_NEW = 128  # 빠른 사이클


def build_inputs(output: ThemaPAOutput):
    """generate_report 와 동일한 inputs 빌드."""
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
    return inputs


def time_generate(inputs, max_new):
    import torch
    pad_id = inference._processor.tokenizer.pad_token_id or inference._processor.tokenizer.eos_token_id
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = inference._model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, num_beams=1,
            repetition_penalty=1.05, pad_token_id=pad_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    in_len = inputs["input_ids"].shape[1]
    out_tokens = out.shape[1] - in_len
    return dt, out_tokens


def trial_block(label, inputs, n_trials, max_new):
    print(f"\n--- {label} ---")
    times = []
    for i in range(n_trials):
        dt, n_tok = time_generate(inputs, max_new)
        per_token = dt / max(1, n_tok) * 1000
        times.append((dt, n_tok, per_token))
        print(f"  trial {i+1}: total={dt:6.2f}s out={n_tok:3d} per_token={per_token:6.1f}ms")
    avg_total = sum(t[0] for t in times) / len(times)
    avg_per_tok = sum(t[2] for t in times) / len(times)
    print(f"  AVG     : total={avg_total:6.2f}s per_token={avg_per_tok:6.1f}ms")
    return {"label": label, "trials": times, "avg_total_sec": avg_total, "avg_per_token_ms": avg_per_tok}


def main():
    print("Loading model (v4 LoRA)...")
    t0 = time.time()
    inference._load_model(use_adapter=True)
    print(f"  ready in {time.time()-t0:.1f}s")

    output = ThemaPAOutput.model_validate_json(
        (ROOT / SAMPLE).read_text(encoding="utf-8"),
    )
    inputs = build_inputs(output)
    print(f"  input tokens: {inputs['input_ids'].shape[1]}")

    # ---- baseline ----
    # warm-up 1회 (cuda 커널 캐시) — 결과는 버림
    print("\n[warm-up] baseline (no compile)...")
    _ = time_generate(inputs, max_new=8)
    baseline = trial_block("baseline (no compile)", inputs, N_TRIALS, MAX_NEW)

    # ---- compiled (language_model 만) ----
    import torch
    print("\n[compile] applying torch.compile to language_model (mode=reduce-overhead)...")
    try:
        # Qwen3-VL: model.language_model 이 LM 본체
        target = getattr(inference._model, "language_model", None) or inference._model
        target_name = "model.language_model" if hasattr(inference._model, "language_model") else "model"
        compiled = torch.compile(target, mode="reduce-overhead", fullgraph=False, dynamic=True)
        if hasattr(inference._model, "language_model"):
            inference._model.language_model = compiled
        else:
            print("  (warning: language_model attribute not found, compiled whole model)")
        print(f"  compiled: {target_name}")
    except Exception as e:
        print(f"[FAIL] torch.compile 실패: {type(e).__name__}: {e}")
        print(json.dumps({"baseline": baseline, "compiled_error": str(e)}, ensure_ascii=False, indent=2))
        return 1

    # warm-up 1회 — 첫 호출 컴파일 비용 측정
    print("\n[warm-up] compiled — 컴파일 비용 (수십초 가능)...")
    t0 = time.time()
    try:
        _ = time_generate(inputs, max_new=8)
    except Exception as e:
        print(f"[FAIL] compiled forward 실패: {type(e).__name__}: {e}")
        print(json.dumps({"baseline": baseline, "compiled_warmup_error": str(e)}, ensure_ascii=False, indent=2))
        return 1
    print(f"  compile warm-up: {time.time()-t0:.1f}s")

    compiled = trial_block("compiled", inputs, N_TRIALS, MAX_NEW)

    speedup = baseline["avg_per_token_ms"] / compiled["avg_per_token_ms"]
    print(f"\n=== Summary ===")
    print(f"baseline per_token: {baseline['avg_per_token_ms']:.1f} ms")
    print(f"compiled per_token: {compiled['avg_per_token_ms']:.1f} ms")
    print(f"speedup           : {speedup:.2f}x  ({(1 - 1/speedup)*100:+.1f}% faster)")

    out_path = ROOT / "vlm" / "bench" / "compile_speedup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "baseline": baseline, "compiled": compiled, "speedup_x": speedup,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
