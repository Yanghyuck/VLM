# =============================================================================
# scripts/test_inference_constrained.py
# -----------------------------------------------------------------------------
# D1 Constrained decoding 효과 검증 — 3샘플에 대해 constrained=True 추론.
#
# 사용:
#   python scripts/test_inference_constrained.py
#
# 출력:
#   vlm/train/test_inference_constrained.md
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
OUT = ROOT / "vlm" / "train" / "test_inference_constrained.md"


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# D1 Constrained decoding 결과\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"`generate_report(constrained=True)` — lm-format-enforcer JsonSchemaParser 로\n")
        f.write(f"4 키 dict (`3문장_요약`/`비정상_근거`/`주의사항`/`권고`) 강제.\n\n")
        f.write("---\n\n")

        for sp in SAMPLES:
            full = ROOT / sp
            if not full.exists():
                continue
            data = json.loads(full.read_text(encoding="utf-8"))
            img = data.get("result_image_path")
            if img and not Path(img).exists():
                data["result_image_path"] = None
            output = ThemaPAOutput(**data)

            safe_print(f"\n[{sp}] 시작")
            t0 = time.time()
            try:
                rep = generate_report(output, constrained=True)
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
