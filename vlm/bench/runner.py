# =============================================================================
# vlm/bench/runner.py
# -----------------------------------------------------------------------------
# 기능:
#   평가셋의 각 샘플에 대해 베이스 또는 LoRA 모델로 추론을 수행하고
#   결과를 JSONL 파일에 저장합니다.
#
# 동작 방법:
#   # 베이스 모델만으로 추론
#   python vlm/bench/runner.py --mode base   --output vlm/bench/results_base.jsonl
#
#   # LoRA 모델로 추론
#   python vlm/bench/runner.py --mode lora   --output vlm/bench/results_lora.jsonl
#
#   # 평가셋 크기 직접 지정
#   python vlm/bench/runner.py --mode lora --n 50
#
# 입력: vlm/bench/eval_set.jsonl (없으면 자동 생성)
# 출력: 1줄당 1샘플 결과 (JSONL)
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vlm.bench.dataset import build_eval_set, DATASET_JSONL
from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report

EVAL_SET_PATH = ROOT / "vlm" / "bench" / "eval_set.jsonl"


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def load_eval_set(n: int, seed: int) -> list[dict]:
    if EVAL_SET_PATH.exists():
        with open(EVAL_SET_PATH, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        if len(samples) >= n:
            return samples[:n]
        safe_print(f"[INFO] 기존 eval_set 작음 ({len(samples)}<{n}), 재생성")

    samples = build_eval_set(n=n, seed=seed)
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return samples


def run(mode: str, n: int, seed: int, output: Path, adapter_path: str | None = None, tag: str | None = None):
    use_adapter = (mode == "lora")
    label = tag or mode
    safe_print(f"[BENCH] mode={mode} tag={label} n={n} seed={seed} adapter={adapter_path or 'config-default'}")

    samples = load_eval_set(n=n, seed=seed)
    safe_print(f"[BENCH] 평가셋 {len(samples)}건 로드")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples, 1):
            meta = sample["metadata"]

            # 이미지 경로 검증 (없으면 None)
            img = sample.get("image_path")
            if img and not Path(img).exists():
                meta_with_img = {**meta, "result_image_path": None}
            else:
                meta_with_img = {**meta, "result_image_path": img}

            try:
                output_obj = ThemaPAOutput(**meta_with_img)
            except Exception as e:
                safe_print(f"  [{i}/{len(samples)}] BUILD FAIL: {e}")
                f.write(json.dumps({"id": sample["id"], "error": f"build: {e}"}, ensure_ascii=False) + "\n")
                continue

            t0 = time.time()
            try:
                pred = generate_report(output_obj, use_adapter=use_adapter, adapter_path=adapter_path)
                elapsed = round(time.time() - t0, 2)
                err = None
            except Exception as e:
                pred = None
                elapsed = round(time.time() - t0, 2)
                err = str(e)

            result = {
                "id": sample["id"],
                "mode": mode,
                "tag": label,
                "elapsed_sec": elapsed,
                "metadata": meta,
                "tasks": sample.get("tasks", {}),
                "prediction": pred,
                "error": err,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            status = "OK" if pred and not err else "FAIL"
            safe_print(f"  [{i}/{len(samples)}] {status} carcass={sample['id']} grade={meta['grade']} elapsed={elapsed}s")

    safe_print(f"[DONE] 결과 저장: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "lora"], required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="결과 JSONL 경로 (기본: results_{tag or mode}.jsonl)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="LoRA 어댑터 디렉터리 (None=config 기본). v1/v2 비교 시 명시")
    parser.add_argument("--tag", type=str, default=None,
                        help="결과에 붙일 라벨 (예: 'lora_v1', 'lora_v2')")
    args = parser.parse_args()

    label = args.tag or args.mode
    out = Path(args.output) if args.output else (
        ROOT / "vlm" / "bench" / f"results_{label}.jsonl"
    )
    run(args.mode, args.n, args.seed, out, adapter_path=args.adapter_path, tag=args.tag)
