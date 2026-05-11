# =============================================================================
# scripts/test_inference_v4_smoke.py
# -----------------------------------------------------------------------------
# v4 어댑터로 검출 실패 샘플(backfat_error/entry_error) 추론.
# 평가셋 50건이 100% 정상이라 6-way 메트릭으로는 v4 abnormal 학습 효과를
# 측정 못함. 검출 실패 샘플 직접 비교로 효과 확인.
#
# 사용:
#   python scripts/test_inference_v4_smoke.py
# =============================================================================

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report

ADAPTER_V4 = str(ROOT / "vlm" / "train" / "output" / "qwen3vl-lora-v4")
SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/entry_error_case.json",
]
OUT = ROOT / "vlm" / "train" / "test_inference_v4.md"


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# v4 스모크 — 검출 실패 케이스 확인\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**어댑터**: `{ADAPTER_V4}`\n\n")
        f.write("---\n\n")
        for sp in SAMPLES:
            full = ROOT / sp
            data = json.loads(full.read_text(encoding="utf-8"))
            img = data.get("result_image_path")
            if img and not Path(img).exists():
                data["result_image_path"] = None
            output = ThemaPAOutput(**data)

            safe_print(f"\n[{sp}] 추론 시작")
            t0 = time.time()
            try:
                rep = generate_report(output, adapter_path=ADAPTER_V4)
                err = None
            except Exception as e:
                rep = None
                err = str(e)
            elapsed = round(time.time() - t0, 2)
            safe_print(f"[{sp}] {elapsed}s done")

            f.write(f"## {sp}\n\n")
            f.write(f"**입력**: `{output.summary()}`\n\n")
            f.write(f"**추론 시간**: {elapsed}초\n\n")
            if err:
                f.write(f"**ERROR**: `{err}`\n\n")
            else:
                f.write("```json\n")
                f.write(json.dumps(rep, ensure_ascii=False, indent=2))
                f.write("\n```\n\n")
            f.write("---\n\n")
            f.flush()
    safe_print(f"\n[DONE] {OUT}")


if __name__ == "__main__":
    main()
