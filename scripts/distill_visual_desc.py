# =============================================================================
# scripts/distill_visual_desc.py
# -----------------------------------------------------------------------------
# 기능:
#   로컬 base Qwen3-VL(LoRA 미적용)을 teacher 로 사용하여, AI 도체 이미지의
#   "도체 전체 형태 + 등지방층 외형" 시각 서술 reference 를 생성(증류)합니다.
#   생성 결과는 향후 visual_desc 학습 태스크의 정답(reference)으로 사용합니다.
#
# 동작:
#   1) dataset.jsonl 에서 N건 샘플 (정상/비정상 혼합 — abnormal 일부 강제 포함)
#   2) 각 이미지에 vlm/prompt/visual_desc.txt 프롬프트로 base 모델 추론
#   3) JSON({"도체_전체_형태","등지방층_외형"}) 파싱 + DB 메타(grade/error_code) 병기
#   4) 결과를 jsonl 로 저장 (레코드마다 flush — 진행 모니터링 가능)
#
# 사용:
#   python scripts/distill_visual_desc.py --n 50 --seed 42 \
#       --output vlm/data/visual_desc_refs_pilot.jsonl
#
# 의존성: torch, transformers, qwen_vl_utils, pillow
# =============================================================================

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vlm.config import CFG
from vlm.train.json_utils import _extract_json

PROMPT       = (ROOT / "vlm" / "prompt" / "visual_desc.txt").read_text(encoding="utf-8")
BASE_MODEL   = CFG.model.base_model_id
DATASET_JSONL = ROOT / CFG.paths.dataset_jsonl
MAX_PIXELS   = int(getattr(CFG.model, "image_max_pixels", 200_704))

# teacher 전용 error 힌트 — 비정상 케이스에서 teacher 가 해당 부위를 정상으로
# 단정하지 않도록 프롬프트에 주입. 학습 입력(visual_desc.txt)에는 절대 안 들어감.
ERROR_HINTS = {
    "pig_RightEntry":      "도체 진입 자세/위치 검출 오류 (정상 진입이 아닐 수 있음 — 전체 형태에 반영)",
    "AI_Backbone_error":   "척추(등뼈) 검출 오류 (절단면·정렬에 이상 가능 — 전체 형태에 반영)",
    "AI_BackFat_error":    "등지방 검출 오류 (등지방층 외형/경계 신뢰도 낮음 — 등지방층 서술에 반영)",
    "AI_HalfBone_error":   "반골(척추 분할) 검출 오류 (절단면 정렬에 이상 가능 — 전체 형태에 반영)",
    "AI_multifidus_error": "뭇갈래근 검출 오류",
    "AI_Outline_error":    "도체 외곽선 검출 오류 (전체 형태 윤곽에 이상 가능)",
}


def _is_normal(ec: dict) -> bool:
    return all((v or 0) == 0 for v in ec.values())


def _build_prompt(error_code: dict) -> str:
    """비정상이면 teacher 프롬프트에 error 힌트 블록을 덧붙인다."""
    active = [ERROR_HINTS[k] for k, v in error_code.items() if v and k in ERROR_HINTS]
    if not active:
        return PROMPT
    hint = "\n".join(f"- {h}" for h in active)
    return (
        PROMPT
        + "\n[참고 — thema_pa 시스템이 이 도체에서 다음 검출 이상을 보고했습니다]\n"
        + hint
        + "\n해당 부위를 정상이라고 단정하지 말고, 위 이상 가능성을 반영하여 신중히 서술하십시오.\n"
    )


def load_model():
    print(f"[distill] base 모델 로딩: {BASE_MODEL} (LoRA 미적용)")
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, proc


def describe(model, proc, image_path: str, prompt: str, temperature: float, top_p: float) -> str:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    # 다양성 확보를 위해 nucleus sampling (greedy 의 boilerplate 천편일률 방지)
    sample = temperature > 0
    gen_kwargs = (
        dict(do_sample=True, temperature=temperature, top_p=top_p)
        if sample else dict(do_sample=False, num_beams=1)
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=400, repetition_penalty=1.05, **gen_kwargs,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0]


def sample_records(n: int, seed: int) -> list[dict]:
    records = [json.loads(l) for l in open(DATASET_JSONL, encoding="utf-8")]
    records.sort(key=lambda r: str(r["id"]))  # 결정적 순서 (resume 안정)

    # n<=0 또는 전체 이상이면 전량 (비정상 344건 모두 포함)
    if n <= 0 or n >= len(records):
        return records

    # 파일럿: 비정상도 검증해야 하므로 최대 10건 강제 포함
    abn = [r for r in records if not _is_normal(r["metadata"]["error_code"])]
    nrm = [r for r in records if _is_normal(r["metadata"]["error_code"])]
    rng = random.Random(seed)
    n_abn = min(len(abn), max(0, min(10, n // 5)))
    picked = rng.sample(abn, n_abn) + rng.sample(nrm, min(len(nrm), n - n_abn))
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 이면 greedy, >0 이면 sampling")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--output", type=str, default="vlm/data/visual_desc_refs_pilot.jsonl")
    ap.add_argument("--overwrite", action="store_true",
                    help="기존 출력 무시하고 처음부터 (기본: 이미 생성된 id 는 건너뛰고 이어쓰기)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)  # sampling 재현성

    picked = sample_records(args.n, args.seed)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume — 이미 생성된 id 는 건너뛴다 (하루짜리 작업의 중단/재개 대비)
    done_ids: set[str] = set()
    if out_path.exists() and not args.overwrite:
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(str(json.loads(line)["id"]))
            except Exception:
                continue
    todo = [r for r in picked if str(r["id"]) not in done_ids]
    n_abn = sum(1 for r in todo if not _is_normal(r["metadata"]["error_code"]))
    print(f"[distill] 대상 {len(picked)}건 중 미완료 {len(todo)}건 처리 "
          f"(이미 완료 {len(done_ids)}, 비정상 {n_abn}) | temp={args.temperature} top_p={args.top_p}")

    model, proc = load_model()

    mode = "w" if args.overwrite else "a"
    ok = fail = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            meta = r["metadata"]
            try:
                prompt = _build_prompt(meta["error_code"])
                raw = describe(model, proc, r["image_path"], prompt, args.temperature, args.top_p)
                parsed = _extract_json(raw)
            except Exception as e:
                raw, parsed = f"[ERROR] {e}", {}
            valid = isinstance(parsed, dict) and "도체_전체_형태" in parsed and "등지방층_외형" in parsed
            ok += valid; fail += (not valid)
            rec = {
                "id":          r["id"],
                "image_path":  r["image_path"],
                "grade":       meta["grade"],
                "error_code":  meta["error_code"],
                "abnormal":    not _is_normal(meta["error_code"]),
                "visual_desc": parsed if valid else None,
                "raw":         raw,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] id={r['id']} abn={rec['abnormal']} json={'OK' if valid else 'FAIL'}")

    print(f"\n[distill] 완료: {ok} OK / {fail} FAIL → {out_path}")


if __name__ == "__main__":
    main()
