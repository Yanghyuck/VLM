# =============================================================================
# scripts/test_quantization.py
# -----------------------------------------------------------------------------
# 양자화(INT4 NF4) 모델 동작 검증 + VRAM 사용량 측정.
# 3개 샘플 추론 후 결과를 vlm/train/quantization_test_results.md 에 저장.
#
# 동작 방법:
#   1. config.json 의 model.quantize 를 true 로 일시 변경
#   2. python scripts/test_quantization.py
#   3. 검증 후 quantize 를 다시 false 로 되돌리기
# =============================================================================

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG
from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report, QUANTIZE, QUANTIZE_MODE

OUTPUT_FILE = ROOT / "vlm" / "train" / "quantization_test_results.md"

SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/entry_error_case.json",
]


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024**3), 2)
    return 0.0


def main():
    safe_print(f"[CONFIG] quantize={QUANTIZE}, mode={QUANTIZE_MODE}")
    safe_print(f"[GPU] before load: {gpu_mem_gb()} GB allocated")

    results = []
    for sample_path in SAMPLES:
        full = ROOT / sample_path
        if not full.exists():
            safe_print(f"[SKIP] {sample_path}")
            continue

        with open(full, encoding="utf-8") as f:
            data = json.load(f)

        img = data.get("result_image_path")
        if img and not Path(img).exists():
            data["result_image_path"] = None

        output_obj = ThemaPAOutput(**data)

        t0 = time.time()
        try:
            report = generate_report(output_obj)
            elapsed = round(time.time() - t0, 2)
            err = None
        except Exception as e:
            report = None
            elapsed = round(time.time() - t0, 2)
            err = str(e)

        mem = gpu_mem_gb()
        safe_print(f"[{Path(sample_path).stem}] elapsed={elapsed}s, GPU={mem}GB, err={'YES' if err else 'no'}")

        results.append({
            "sample":   sample_path,
            "elapsed":  elapsed,
            "gpu_gb":   mem,
            "report":   report,
            "error":    err,
        })

    safe_print(f"\n[GPU] after all inference: {gpu_mem_gb()} GB allocated")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# 양자화 테스트 결과\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**양자화**: {QUANTIZE} (mode: {QUANTIZE_MODE})\n")
        f.write(f"**최대 GPU 사용량**: {max(r['gpu_gb'] for r in results)} GB\n\n")

        for r in results:
            ok_mark = "✅" if not r["error"] else "❌"
            f.write(f"## {ok_mark} {r['sample']}\n\n")
            f.write(f"- 추론 시간: {r['elapsed']}초\n")
            f.write(f"- GPU 사용: {r['gpu_gb']} GB\n")
            if r["report"]:
                f.write("- 결과:\n\n```json\n")
                f.write(json.dumps(r["report"], ensure_ascii=False, indent=2))
                f.write("\n```\n\n")
            else:
                f.write(f"- 에러: {r['error']}\n\n")
            f.write("---\n\n")

    safe_print(f"\n[DONE] 결과: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
