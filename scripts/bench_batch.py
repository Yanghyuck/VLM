# =============================================================================
# scripts/bench_batch.py
# -----------------------------------------------------------------------------
# batch_size 1/2/4 에서 per-request latency 비교.
#
# 시나리오: 같은 system_prompt + 다른 user payload 를 batch 로 generate.
#   throughput 가속이 의미 있으면 micro-batching 서버 구현 가치.
#
# 측정 방식:
#   각 batch_size 당 max_new=128 으로 3회 trial (warm-up 1회 별도).
#   per_request_sec = total_sec / batch_size.
#
# 사용 (VLM API 서버는 꺼져 있어야 함):
#   conda activate vlm
#   python scripts/bench_batch.py
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

# 9건 AI 이미지 — Phase A 산출물 (학습 분포와 동일)
AI_DIR = Path(r"C:\Users\IPC\Desktop\git\thema_pa_VLM\images\AI")
PIGNOS = [3, 4, 5, 8, 11, 13, 15, 16, 17]


def build_one(pigno: int):
    """단일 ThemaPAOutput → messages dict."""
    from PIL import Image
    from vlm.config import CFG

    img_files = list(AI_DIR.glob(f"0716_*_{pigno}_SP_CAM7.jpg"))
    if not img_files:
        raise FileNotFoundError(f"no AI image for pigno {pigno}")
    image_path = str(img_files[0])
    output = ThemaPAOutput(
        carcass_no=pigno, slaughter_ymd="20260212",
        backfat_average=22.4, multifidus_thk=48.2,
        body_length=73.6, body_width=34.1, body_weight=87.3,
        gender=1, grade="1+",
        error_code={"pig_RightEntry": 0, "AI_Backbone_error": 0, "AI_BackFat_error": 0,
                    "AI_HalfBone_error": 0, "AI_multifidus_error": 0, "AI_Outline_error": 0},
        backbone_slope={"has_large_slope": False, "threshold": 0.28},
        result_image_path=image_path,
    )

    system_text = inference._load_prompt("system_prompt.txt")
    template = inference._select_template(output)
    user_text = template.replace("{{SUMMARY}}", output.summary())

    image = Image.open(image_path).convert("RGB")
    max_pixels = getattr(CFG.model, "image_max_pixels", 100_352)
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
    ]


def build_batch_inputs(batch_messages):
    """N개 messages 리스트 → batched inputs."""
    from qwen_vl_utils import process_vision_info

    text_inputs = [
        inference._processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True,
        )
        for m in batch_messages
    ]
    # process_vision_info 는 messages 한 묶음을 받음 — 각 샘플별 호출 후 concat
    image_inputs = []
    for m in batch_messages:
        ii, _ = process_vision_info(m)
        if ii:
            image_inputs.extend(ii)

    inputs = inference._processor(
        text=text_inputs,
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(inference._model.device)
    return inputs


def time_batch(batch_size: int, max_new: int = 128):
    import torch
    msgs_list = [build_one(PIGNOS[i % len(PIGNOS)]) for i in range(batch_size)]
    inputs = build_batch_inputs(msgs_list)
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
    out_tokens = (out.shape[1] - inputs["input_ids"].shape[1]) * batch_size
    return dt, out_tokens, inputs["input_ids"].shape[1]


def main():
    print("Loading model (v4 LoRA)...")
    t0 = time.time()
    inference._load_model(use_adapter=True)
    print(f"  ready in {time.time()-t0:.1f}s\n")

    rows = []
    for bs in [1, 2, 4]:
        # warm-up
        try:
            _ = time_batch(bs, max_new=8)
        except Exception as e:
            print(f"[FAIL] batch_size={bs} warm-up: {type(e).__name__}: {e}")
            rows.append({"batch_size": bs, "error": str(e)})
            continue
        # 3 trials
        trials = []
        for i in range(3):
            try:
                dt, out_tok, in_len = time_batch(bs, max_new=128)
                per_req = dt / bs
                trials.append({"total_sec": dt, "per_request_sec": per_req,
                               "in_len": in_len, "out_tokens": out_tok})
                print(f"  bs={bs} trial {i+1}: total={dt:6.2f}s "
                      f"per_req={per_req:6.2f}s in_len={in_len} out_total={out_tok}")
            except Exception as e:
                print(f"  bs={bs} trial {i+1}: FAIL {type(e).__name__}: {e}")
                trials.append({"error": str(e)})
        ok = [t for t in trials if "error" not in t]
        avg_total = sum(t["total_sec"] for t in ok) / max(1, len(ok))
        avg_per_req = sum(t["per_request_sec"] for t in ok) / max(1, len(ok))
        rows.append({
            "batch_size": bs,
            "avg_total_sec": round(avg_total, 2),
            "avg_per_request_sec": round(avg_per_req, 2),
            "trials": trials,
        })
        print(f"  AVG bs={bs}: total={avg_total:.2f}s per_req={avg_per_req:.2f}s\n")

    # 비교
    base = next((r for r in rows if r.get("batch_size") == 1), None)
    print("\n=== Summary ===")
    if base:
        for r in rows:
            if "avg_per_request_sec" in r:
                speedup = base["avg_per_request_sec"] / r["avg_per_request_sec"]
                eff = speedup / r["batch_size"] * 100
                print(f"  bs={r['batch_size']}: per_req={r['avg_per_request_sec']:6.2f}s "
                      f"speedup={speedup:.2f}x efficiency={eff:.0f}%")

    out_path = ROOT / "vlm" / "bench" / "batch_speedup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
