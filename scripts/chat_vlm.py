# =============================================================================
# scripts/chat_vlm.py
# -----------------------------------------------------------------------------
# Qwen3-VL-8B 와 자유 chat (텍스트 + 이미지). 도체 판정 도메인 외 일반 대화 가능.
#
# 베이스 모델 권장 (LoRA 적용 시 도체 판정 톤으로 빠질 수 있음).
#
# 사용:
#   conda activate vlm
#   python scripts/chat_vlm.py                # 베이스 (Qwen3-VL-8B 원본)
#   python scripts/chat_vlm.py --use-adapter  # v4 LoRA 적용
#
# 세션 명령:
#   /image PATH    다음 user 메시지에 이미지 첨부 (PATH 는 절대/상대경로)
#   /clear         대화 기록 초기화
#   /quit (또는 /q, Ctrl+C/D) 종료
#
# 주의:
#   VLM FastAPI 서버가 가동 중이면 GPU OOM. 먼저 서버를 종료하세요.
# =============================================================================

import argparse
import io
import sys
import time
from pathlib import Path

# Windows cp949 콘솔에서 한국어 + em-dash 등 출력 깨짐 방지
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL chat CLI")
    ap.add_argument("--use-adapter", action="store_true",
                    help="v4 LoRA 적용 (기본: 베이스 모델만)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    print(f"Loading model (use_adapter={args.use_adapter})...")
    t0 = time.time()
    from vlm.train import inference
    from vlm.config import CFG
    inference._load_model(use_adapter=args.use_adapter)
    model = inference._model
    processor = inference._processor
    print(f"  ready in {time.time()-t0:.1f}s\n")

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    max_pixels = getattr(CFG.model, "image_max_pixels", 200_704)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def _resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w * h <= max_pixels:
            return img
        scale = (max_pixels / (w * h)) ** 0.5
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    history: list[dict] = []
    pending_image_path: str | None = None

    print("=" * 60)
    print("  Qwen3-VL Chat — '/image PATH' to attach, '/clear', '/quit'")
    print("=" * 60)

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(bye)")
            break

        if not user:
            continue
        if user in ("/quit", "/q", "/exit"):
            print("(bye)")
            break
        if user == "/clear":
            history = []
            pending_image_path = None
            print("  (history cleared)")
            continue
        if user.startswith("/image "):
            path = user[len("/image "):].strip().strip('"').strip("'")
            if not Path(path).exists():
                print(f"  (file not found: {path})")
                continue
            pending_image_path = path
            print(f"  (will attach: {path})")
            continue
        if user.startswith("/"):
            print("  (unknown command — /image PATH | /clear | /quit)")
            continue

        # 사용자 메시지 빌드
        if pending_image_path:
            img = _resize(Image.open(pending_image_path).convert("RGB"))
            content = [{"type": "image", "image": img},
                       {"type": "text", "text": user}]
            pending_image_path = None
        else:
            content = user
        history.append({"role": "user", "content": content})

        # 토크나이즈
        text_input = processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(history)
        if image_inputs:
            inputs = processor(
                text=[text_input], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
        else:
            inputs = processor(
                text=[text_input], padding=True, return_tensors="pt",
            ).to(model.device)

        # 생성
        t0 = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=pad_id,
            )
        dt = time.time() - t0

        input_len = inputs.input_ids.shape[1]
        output_tokens = generated[0][input_len:]
        response = processor.tokenizer.decode(
            output_tokens, skip_special_tokens=True,
        ).strip()

        print(f"\nVLM> {response}")
        print(f"     ({dt:.1f}s, {output_tokens.shape[0]} tokens)")

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
