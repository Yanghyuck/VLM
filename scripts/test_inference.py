# =============================================================================
# scripts/test_inference.py
# -----------------------------------------------------------------------------
# 학습된 Qwen3-VL LoRA 어댑터 로드 후 샘플 3종에 대해 추론 결과 출력.
#
# 동작 방법:
#   python scripts/test_inference.py
# =============================================================================

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report

SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/entry_error_case.json",
]

OUTPUT_FILE = ROOT / "vlm" / "train" / "test_inference_results.md"


def safe_print(msg: str):
    """CP949 인코딩 실패 시에도 안전하게 출력."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def main():
    results = []
    for sample_path in SAMPLES:
        full_path = ROOT / sample_path
        if not full_path.exists():
            safe_print(f"[SKIP] sample not found: {sample_path}")
            continue

        safe_print(f"\n{'=' * 70}")
        safe_print(f"[SAMPLE] {sample_path}")
        safe_print(f"{'=' * 70}")

        with open(full_path, encoding="utf-8") as f:
            data = json.load(f)

        img = data.get("result_image_path")
        if img and not Path(img).exists():
            safe_print(f"[WARN] image not found, text-only inference: {img}")
            data["result_image_path"] = None

        output = ThemaPAOutput(**data)

        t0 = time.time()
        report = generate_report(output)
        elapsed = round(time.time() - t0, 2)

        safe_print(f"[TIME] {elapsed}sec")
        safe_print(f"[RESULT preview - check {OUTPUT_FILE.name} for full UTF-8 result]")

        results.append({
            "sample": sample_path,
            "input_summary": output.summary(),
            "elapsed_sec": elapsed,
            "report": report,
        })

    # UTF-8 마크다운 파일로 결과 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# 추론 테스트 결과\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in results:
            f.write(f"## {r['sample']}\n\n")
            f.write(f"**추론 시간**: {r['elapsed_sec']}초\n\n")
            f.write(f"**입력**: `{r['input_summary']}`\n\n")
            f.write("**결과**:\n```json\n")
            f.write(json.dumps(r["report"], ensure_ascii=False, indent=2))
            f.write("\n```\n\n---\n\n")

    safe_print(f"\n[DONE] full results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
