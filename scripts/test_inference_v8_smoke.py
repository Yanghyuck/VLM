# =============================================================================
# scripts/test_inference_v8_smoke.py
# -----------------------------------------------------------------------------
# v8 어댑터로 검출 실패 샘플(backfat_error/entry_error) 추론.
# 평가셋 50건이 거의 정상(비정상 2)이라 N-way 메트릭으로는 abnormal 처리 능력을
# 충분히 못 본다. 검출 실패 샘플 직접 추론으로 환각(예: 입력 "암컷" → "거세 암컷")
# 잔존 여부를 확인한다. (v4 채택 시 사용한 것과 동일 절차)
#
# 사용:
#   python scripts/test_inference_v8_smoke.py
# =============================================================================

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.train.inference import generate_report

ADAPTER_V8 = str(ROOT / "vlm" / "train" / "output" / "qwen3vl-lora-v8")
SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/entry_error_case.json",
]
OUT = ROOT / "vlm" / "train" / "test_inference_v8.md"


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# v8 스모크 — 검출 실패 케이스 확인\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**어댑터**: `{ADAPTER_V8}`\n\n")
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
                rep = generate_report(output, adapter_path=ADAPTER_V8)
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
