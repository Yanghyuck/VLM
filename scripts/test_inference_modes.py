# =============================================================================
# scripts/test_inference_modes.py
# -----------------------------------------------------------------------------
# D2 (sampling) / D3 (beam search) 효과 비교.
# 동일 샘플 3종에 대해 3개 모드(greedy / sampling / beam4)로 각각 추론하여
# 결과를 마크다운으로 저장한다.
#
# 사용:
#   python scripts/test_inference_modes.py
#
# 출력:
#   vlm/train/test_inference_modes.md
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

# (라벨, generate_report kwargs)
MODES = [
    ("greedy",        {}),
    ("sampling_t03",  {"sampling": True, "temperature": 0.3, "top_p": 0.95}),
    ("beam4",         {"num_beams": 4}),
]

OUTPUT_FILE = ROOT / "vlm" / "train" / "test_inference_modes.md"


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# 추론 모드 비교 — D2 sampling vs D3 beam vs greedy(default)\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**모드**:\n")
        f.write(f"- `greedy` — 현재 default (do_sample=False, repetition_penalty=1.05)\n")
        f.write(f"- `sampling_t03` — D2: do_sample=True, temperature=0.3, top_p=0.95\n")
        f.write(f"- `beam4` — D3: do_sample=False, num_beams=4\n\n")
        f.write("---\n\n")

        for sample_path in SAMPLES:
            full_path = ROOT / sample_path
            if not full_path.exists():
                continue

            data = json.loads(full_path.read_text(encoding="utf-8"))
            img = data.get("result_image_path")
            if img and not Path(img).exists():
                data["result_image_path"] = None
            output = ThemaPAOutput(**data)

            f.write(f"## {sample_path}\n\n")
            f.write(f"**입력**: `{output.summary()}`\n\n")

            for mode_name, mode_kwargs in MODES:
                safe_print(f"\n[{sample_path} / {mode_name}] 시작")
                t0 = time.time()
                try:
                    report = generate_report(output, **mode_kwargs)
                    elapsed = round(time.time() - t0, 2)
                    err = None
                except Exception as e:
                    report = None
                    elapsed = round(time.time() - t0, 2)
                    err = str(e)
                safe_print(f"[{sample_path} / {mode_name}] {elapsed}s done")

                f.write(f"### `{mode_name}`\n\n")
                f.write(f"**추론 시간**: {elapsed}초\n\n")
                if err:
                    f.write(f"**ERROR**: `{err}`\n\n")
                else:
                    f.write("```json\n")
                    f.write(json.dumps(report, ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")

            f.write("---\n\n")
            f.flush()

    safe_print(f"\n[DONE] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
