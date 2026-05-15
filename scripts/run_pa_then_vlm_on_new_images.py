# =============================================================================
# scripts/run_pa_then_vlm_on_new_images.py
# -----------------------------------------------------------------------------
# thema_pa_VLM 의 새 이미지 (images/*.jpg) 로 운영 흐름 검증.
#
# 두 단계로 진행:
#   Phase A — ThematecPA.analysis_pa 로 ORI → AI 이미지 9건 생성
#             (DB/RFID/MQTT/inpaint 모두 끄고 단일 cuda:0 에 YOLO 만)
#   Phase B — 생성된 AI 이미지로 SendVLMReport 9건 호출 (운영 흐름 동일 클래스)
#
# 전제:
#   - VLM FastAPI 서버는 Phase A 동안 꺼져 있어야 함 (cuda OOM 회피)
#   - Phase B 시작 전 사용자 또는 자동화로 VLM 서버 재가동
#
# 실행:
#   python scripts/run_pa_then_vlm_on_new_images.py [--phase a|b|both]
# =============================================================================

import argparse
import contextlib
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THEMA_PA_ROOT = Path(os.environ.get("THEMA_PA_ROOT", r"C:\Users\IPC\Desktop\git\thema_pa_VLM"))
IMAGES_DIR = THEMA_PA_ROOT / "images"
AI_OUTPUT_DIR = THEMA_PA_ROOT / "images" / "AI_run"  # Phase A 결과 — 새 이미지 전용 분리
STORAGE_DIR = THEMA_PA_ROOT / "storage" / "vlm_reports"

API_BASE = "http://127.0.0.1:8000"


def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def parse_filename(path: Path) -> dict | None:
    """0716_ori_{YYYYMMDDHHMMSS}_{pigno}_SP_CAM7.jpg → {ymd, pigno}"""
    parts = path.stem.split("_")
    if len(parts) < 6:
        return None
    return {"slaughter_ymd": parts[2][:8], "carcass_no": parts[3]}


# ---------------------------------------------------------------- Phase A ----

@contextlib.contextmanager
def chdir(path: Path):
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


def phase_a():
    safe_print("=" * 70)
    safe_print("Phase A — ThematecPA 9건 분석 (AI 이미지 생성)")
    safe_print("=" * 70)

    images = sorted(IMAGES_DIR.glob("0716_ori_*.jpg"))
    safe_print(f"입력 이미지: {len(images)} 건")
    if not images:
        safe_print("[FAIL] 새 이미지 없음")
        return 2

    # VLM 서버가 살아 있으면 cuda OOM. 사전 확인.
    try:
        with urllib.request.urlopen(f"{API_BASE}/v1/health", timeout=2) as r:
            if r.status == 200:
                safe_print("[FAIL] VLM 서버가 가동 중. Phase A 전에 종료하세요.")
                return 2
    except Exception:
        pass  # 꺼져있는 게 정상

    AI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(THEMA_PA_ROOT))
    import torch as _torch_check
    safe_print(f"  torch={_torch_check.__version__} cuda={_torch_check.cuda.is_available()} "
               f"path={_torch_check.__file__}")
    if not _torch_check.cuda.is_available():
        safe_print("[FAIL] CUDA 사용 불가 — torch 빌드 또는 드라이버 점검 필요")
        return 2
    from infra.log import init as log_init
    log_init()
    from business.thematec_pa import ThematecPA

    # config 일시 override — 운영 외 기능 모두 끔
    config = json.loads((THEMA_PA_ROOT / "config.json").read_text(encoding="utf-8"))
    config["dl_db_instrt"] = 0
    config["use_warning_light"] = 0
    config["use_rfid"] = 0
    config["use_monitor"] = 0
    # use_inpaint 는 config 기본값(1) 유지 — make_result_image 가 self.inpaint 를 직접 참조함
    config["dl_ouput_folder"] = str(AI_OUTPUT_DIR)
    config["dl_ouput_display"] = 0

    # weights/ 가 config 와 정확 일치 (사용자가 9개 .torchscript + rightside .pth 보강).
    # substitute 매핑 불필요.

    # ThematecPA 가 model_path='./weights' 등 cwd 의존 경로를 쓰므로
    # init 부터 analysis_pa 까지 모두 THEMA_PA_ROOT 에서 실행한다.
    results = []
    with chdir(THEMA_PA_ROOT):
        safe_print("\nThematecPA 초기화 (YOLO 모델 로드)...")
        t0 = time.time()
        pa = ThematecPA(config, None)
        safe_print(f"  init 완료: {time.time()-t0:.1f}s")

        for i, img_path in enumerate(images):
            meta = parse_filename(img_path)
            t1 = time.time()
            try:
                pa.analysis_pa(i, str(img_path))
                elapsed = time.time() - t1

                ai_basename = img_path.stem.replace("ori", "ai") + ".jpg"
                ai_path = AI_OUTPUT_DIR / ai_basename
                ok = ai_path.exists()
                results.append({
                    "ori": img_path.name,
                    "ai": ai_basename,
                    "ai_path": str(ai_path),
                    "ymd": meta["slaughter_ymd"] if meta else None,
                    "pigno": meta["carcass_no"] if meta else None,
                    "elapsed_sec": round(elapsed, 2),
                    "ai_image_created": ok,
                    "AI_BackFat_error": int(getattr(pa, "AI_BackFat_error", 0) or 0),
                    "AI_Backbone_error": int(getattr(pa, "AI_Backbone_error", 0) or 0),
                    "AI_HalfBone_error": int(getattr(pa, "AI_HalfBone_error", 0) or 0),
                    "AI_multifidus_error": int(getattr(pa, "AI_multifidus_error", 0) or 0),
                    "AI_Outline_error": int(getattr(pa, "AI_Outline_error", 0) or 0),
                    "pig_RightEntry": int(getattr(pa, "pig_RightEntry", 0) or 0),
                })
                tag = "OK" if ok else "MISS"
                safe_print(f"  [{tag}] [{i+1}/{len(images)}] {img_path.name} -> {ai_basename} ({elapsed:.1f}s)")
            except Exception as e:
                safe_print(f"  [ERR] [{i+1}/{len(images)}] {img_path.name} :: {type(e).__name__}: {e}")
                results.append({
                    "ori": img_path.name,
                    "ai_path": None,
                    "elapsed_sec": round(time.time()-t1, 2),
                    "ai_image_created": False,
                    "error": f"{type(e).__name__}: {e}",
                })

    out_path = AI_OUTPUT_DIR / "_phase_a_summary.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    created = sum(1 for r in results if r.get("ai_image_created"))
    safe_print(f"\nPhase A 결과: {created}/{len(images)} AI 이미지 생성")
    safe_print(f"  요약 저장: {out_path}")
    return 0 if created == len(images) else 1


# ---------------------------------------------------------------- Phase B ----

def phase_b():
    safe_print("=" * 70)
    safe_print("Phase B — SendVLMReport 9건 (운영 RestAPI 클래스 사용)")
    safe_print("=" * 70)

    summary_path = AI_OUTPUT_DIR / "_phase_a_summary.json"
    if not summary_path.exists():
        safe_print(f"[FAIL] Phase A 요약이 없음: {summary_path}. Phase A 먼저 실행.")
        return 2
    pa_results = json.loads(summary_path.read_text(encoding="utf-8"))
    pa_results = [r for r in pa_results if r.get("ai_image_created")]
    safe_print(f"AI 이미지 입력: {len(pa_results)} 건")

    # VLM API path traversal 가드: image_dir(=.../images/AI) 하위만 허용 →
    # Phase A 산출(AI_run/)을 images/AI/ 로 복사하고 ai_path 갱신.
    import shutil
    ai_dir = IMAGES_DIR / "AI"
    ai_dir.mkdir(parents=True, exist_ok=True)
    for r in pa_results:
        src = Path(r["ai_path"])
        dst = ai_dir / src.name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        r["ai_path"] = str(dst)

    # health 확인
    try:
        with urllib.request.urlopen(f"{API_BASE}/v1/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
        if health.get("status") != "ready":
            safe_print(f"[FAIL] VLM 모델 미준비: {health}")
            return 2
        safe_print(f"  VLM ready: model_used={health.get('model_used')}")
    except Exception as e:
        safe_print(f"[FAIL] VLM API 연결 실패: {e}")
        return 2

    # thema_pa_VLM RestAPI 로드
    sys.path.insert(0, str(THEMA_PA_ROOT))
    config = json.loads((THEMA_PA_ROOT / "config.json").read_text(encoding="utf-8"))
    rest_module = importlib.import_module("comm.rest_api")
    rest_api = rest_module.RestAPI(config)

    e2e = []
    for i, r in enumerate(pa_results):
        # PA 가 추출한 error_code 그대로 사용. 측정값은 합리적 디폴트.
        payload = {
            "carcass_no": int(r["pigno"]),
            "slaughter_ymd": str(r["ymd"]),
            "backfat_average": 18.0,
            "multifidus_thk": 5.5,
            "body_length": 78.0,
            "body_width": 35.0,
            "body_weight": 88.0,
            "gender": 1,            # 암컷 (디폴트, PA gender 추출은 여기선 생략)
            "grade": "1+",
            "error_code": {
                "pig_RightEntry": r.get("pig_RightEntry", 0),
                "AI_Backbone_error": r.get("AI_Backbone_error", 0),
                "AI_BackFat_error": r.get("AI_BackFat_error", 0),
                "AI_HalfBone_error": r.get("AI_HalfBone_error", 0),
                "AI_multifidus_error": r.get("AI_multifidus_error", 0),
                "AI_Outline_error": r.get("AI_Outline_error", 0),
            },
            "backbone_slope": {"has_large_slope": False, "threshold": 0.28},
            "result_image_path": r["ai_path"],
        }

        # 매 호출 결과 파일 사전 삭제 (이번 호출인지 확인)
        out_path = STORAGE_DIR / f"{payload['slaughter_ymd']}_{payload['carcass_no']}_vlm_report.json"
        if out_path.exists():
            out_path.unlink()

        t0 = time.time()
        with chdir(THEMA_PA_ROOT):
            response = rest_api.SendVLMReport(payload)
        dt = time.time() - t0

        ok = response is not None and getattr(response, "status_code", 0) == 200
        body_keys = set()
        if ok:
            try:
                body_keys = set(response.json().keys())
            except Exception:
                ok = False
        required = {"summary", "grade_reason", "warnings", "recommendation", "model_used"}
        missing = required - body_keys
        saved = out_path.exists()

        status_str = "PASS" if (ok and not missing and saved) else "FAIL"
        e2e.append({
            "i": i + 1,
            "pigno": payload["carcass_no"],
            "ai": Path(r["ai_path"]).name,
            "http": getattr(response, "status_code", None),
            "elapsed_sec": round(dt, 2),
            "missing_fields": sorted(missing),
            "saved_ok": saved,
            "status": status_str,
        })
        safe_print(f"  [{status_str}] [{i+1}/{len(pa_results)}] pigno={payload['carcass_no']} "
                   f"http={getattr(response, 'status_code', None)} elapsed={dt:5.1f}s saved={saved}")

    out = AI_OUTPUT_DIR / "_phase_b_summary.json"
    out.write_text(json.dumps(e2e, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for r in e2e if r["status"] == "PASS")
    safe_print(f"\nPhase B 결과: {passed}/{len(e2e)} PASS")
    safe_print(f"  요약 저장: {out}")
    return 0 if passed == len(e2e) else 1


# ---------------------------------------------------------------- entry ----

def phase_b_ori():
    """PA 가동 불가 환경 우회 — 새 ORI 이미지로 직접 SendVLMReport 호출.
    AI 오버레이 없이 원본 도체 이미지만으로 VLM 응답·저장을 검증."""
    safe_print("=" * 70)
    safe_print("Phase B-ORI — 새 ORI 이미지 9건으로 SendVLMReport 직접 호출")
    safe_print("=" * 70)

    # VLM API path traversal 방어: image_dir(=.../images/AI) 하위만 허용.
    # 새 ORI 이미지는 images/ 루트에 있으므로 images/AI/ 에서 같은 이름 파일을 우선 사용.
    ai_subdir = IMAGES_DIR / "AI"
    images = []
    for ori in sorted(IMAGES_DIR.glob("0716_ori_*.jpg")):
        candidate = ai_subdir / ori.name
        images.append(candidate if candidate.exists() else ori)
    safe_print(f"입력 이미지: {len(images)} 건 (image_dir/AI 우선)")
    if not images:
        safe_print("[FAIL] 새 이미지 없음")
        return 2

    try:
        with urllib.request.urlopen(f"{API_BASE}/v1/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
        if health.get("status") != "ready":
            safe_print(f"[FAIL] VLM 모델 미준비: {health}")
            return 2
        safe_print(f"  VLM ready: model_used={health.get('model_used')}")
    except Exception as e:
        safe_print(f"[FAIL] VLM API 연결 실패: {e}")
        return 2

    sys.path.insert(0, str(THEMA_PA_ROOT))
    config = json.loads((THEMA_PA_ROOT / "config.json").read_text(encoding="utf-8"))
    rest_module = importlib.import_module("comm.rest_api")
    rest_api = rest_module.RestAPI(config)

    e2e = []
    for i, img_path in enumerate(images):
        meta = parse_filename(img_path)
        if not meta:
            continue
        payload = {
            "carcass_no": int(meta["carcass_no"]),
            "slaughter_ymd": str(meta["slaughter_ymd"]),
            "backfat_average": 18.0,
            "multifidus_thk": 5.5,
            "body_length": 78.0,
            "body_width": 35.0,
            "body_weight": 88.0,
            "gender": 1,
            "grade": "1+",
            "error_code": {
                "pig_RightEntry": 0, "AI_Backbone_error": 0, "AI_BackFat_error": 0,
                "AI_HalfBone_error": 0, "AI_multifidus_error": 0, "AI_Outline_error": 0,
            },
            "backbone_slope": {"has_large_slope": False, "threshold": 0.28},
            "result_image_path": str(img_path),
        }
        out_path = STORAGE_DIR / f"{payload['slaughter_ymd']}_{payload['carcass_no']}_vlm_report.json"
        if out_path.exists():
            out_path.unlink()

        t0 = time.time()
        with chdir(THEMA_PA_ROOT):
            response = rest_api.SendVLMReport(payload)
        dt = time.time() - t0

        ok = response is not None and getattr(response, "status_code", 0) == 200
        body_keys = set()
        if ok:
            try:
                body_keys = set(response.json().keys())
            except Exception:
                ok = False
        required = {"summary", "grade_reason", "warnings", "recommendation", "model_used"}
        missing = required - body_keys
        saved = out_path.exists()

        status_str = "PASS" if (ok and not missing and saved) else "FAIL"
        e2e.append({
            "i": i + 1,
            "ori": img_path.name,
            "pigno": payload["carcass_no"],
            "ymd": payload["slaughter_ymd"],
            "http": getattr(response, "status_code", None),
            "elapsed_sec": round(dt, 2),
            "missing_fields": sorted(missing),
            "saved_ok": saved,
            "status": status_str,
        })
        safe_print(f"  [{status_str}] [{i+1}/{len(images)}] pigno={payload['carcass_no']} "
                   f"http={getattr(response, 'status_code', None)} elapsed={dt:5.1f}s saved={saved}")

    AI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = AI_OUTPUT_DIR / "_phase_b_ori_summary.json"
    out.write_text(json.dumps(e2e, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for r in e2e if r["status"] == "PASS")
    safe_print(f"\nPhase B-ORI 결과: {passed}/{len(e2e)} PASS")
    safe_print(f"  요약 저장: {out}")
    return 0 if passed == len(e2e) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "b-ori", "both"], default="both")
    args = ap.parse_args()

    if args.phase in ("a", "both"):
        rc = phase_a()
        if rc != 0:
            return rc
        if args.phase == "both":
            safe_print("\n" + "!" * 70)
            safe_print("Phase A 완료. VLM FastAPI 서버를 재가동한 뒤 --phase b 로 다시 실행하세요.")
            safe_print("!" * 70)
            return 0

    if args.phase == "b":
        return phase_b()
    if args.phase == "b-ori":
        return phase_b_ori()
    return 0


if __name__ == "__main__":
    sys.exit(main())
