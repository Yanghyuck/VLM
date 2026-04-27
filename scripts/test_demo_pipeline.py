# =============================================================================
# scripts/test_demo_pipeline.py
# -----------------------------------------------------------------------------
# Streamlit 데모(`vlm/demo/app.py`)가 호출하는 동일한 코드 경로를
# 4개 샘플 전체에 대해 자동 검증.
#   1. 사이드바 샘플 목록 = SAMPLE_DIR.glob("*.json")
#   2. _build_output() 으로 ThemaPAOutput 생성
#   3. inference.generate_report() 호출
#   4. 결과가 4개 필드 모두 포함하는지 확인
# =============================================================================

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG
from vlm.schema.thema_pa_output import ThemaPAOutput, ErrorCode, BackboneSlope
from vlm.train.inference import generate_report

OUTPUT_FILE = ROOT / "vlm" / "train" / "demo_pipeline_results.md"


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def _build_output(meta: dict, image_path):
    return ThemaPAOutput(
        carcass_no=meta["carcass_no"],
        slaughter_ymd=meta["slaughter_ymd"],
        backfat_average=meta["backfat_average"],
        multifidus_thk=meta["multifidus_thk"],
        body_length=meta.get("body_length", 0.0),
        body_width=meta.get("body_width", 0.0),
        body_weight=meta["body_weight"],
        gender=meta["gender"],
        grade=meta["grade"],
        error_code=ErrorCode(**meta.get("error_code", {})),
        backbone_slope=BackboneSlope(**meta.get("backbone_slope", {"has_large_slope": False, "threshold": None})),
        result_image_path=image_path,
    )


def check_streamlit_alive():
    try:
        with urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    sample_dir = ROOT / CFG.paths.samples_dir
    samples = sorted(sample_dir.glob("*.json"))
    safe_print(f"[1/3] Streamlit health: {'OK' if check_streamlit_alive() else 'DOWN'}")
    safe_print(f"[2/3] Samples found in {sample_dir.name}: {len(samples)}")
    for s in samples:
        safe_print(f"      - {s.stem}")

    safe_print(f"\n[3/3] Running demo pipeline for each sample...")

    results = []
    for sample_path in samples:
        with open(sample_path, encoding="utf-8") as f:
            data = json.load(f)

        img = data.get("result_image_path")
        if img and not Path(img).exists():
            data["result_image_path"] = None

        try:
            output = _build_output(data, data.get("result_image_path"))
        except Exception as e:
            safe_print(f"  [{sample_path.stem}] BUILD FAIL: {e}")
            results.append({"sample": sample_path.stem, "ok": False, "error": str(e)})
            continue

        t0 = time.time()
        try:
            report = generate_report(output)
        except Exception as e:
            safe_print(f"  [{sample_path.stem}] INFERENCE FAIL: {e}")
            results.append({"sample": sample_path.stem, "ok": False, "error": str(e)})
            continue
        elapsed = round(time.time() - t0, 2)

        required = ["3문장_요약", "비정상_근거", "주의사항", "권고"]
        missing = [k for k in required if k not in report]
        all_present = len(missing) == 0
        non_empty_summary = bool(report.get("3문장_요약", "").strip())

        status = "OK" if (all_present and non_empty_summary) else "FAIL"
        safe_print(f"  [{sample_path.stem}] {status} ({elapsed}s) - missing={missing}")

        results.append({
            "sample": sample_path.stem,
            "ok": all_present and non_empty_summary,
            "elapsed": elapsed,
            "input": output.summary(),
            "report": report,
        })

    passed = sum(1 for r in results if r.get("ok"))
    safe_print(f"\n[SUMMARY] {passed}/{len(results)} samples passed")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Streamlit 데모 파이프라인 검증 결과\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Streamlit 서버**: {'정상 (HTTP 200)' if check_streamlit_alive() else '다운'}\n")
        f.write(f"**통과율**: {passed}/{len(results)}\n\n")
        for r in results:
            f.write(f"## {r['sample']}\n\n")
            if r.get("ok"):
                f.write(f"- 상태: ✅ PASS\n")
                f.write(f"- 추론 시간: {r['elapsed']}초\n")
                f.write(f"- 입력: `{r['input']}`\n\n")
                f.write("### 결과\n\n```json\n")
                f.write(json.dumps(r["report"], ensure_ascii=False, indent=2))
                f.write("\n```\n\n")
            else:
                f.write(f"- 상태: ❌ FAIL\n")
                f.write(f"- 에러: {r.get('error', 'unknown')}\n\n")
            f.write("---\n\n")

    safe_print(f"\n[DONE] Detailed results: {OUTPUT_FILE}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
