# =============================================================================
# scripts/test_e2e_thema_pa_bridge.py
# -----------------------------------------------------------------------------
# thema_pa_VLM (운영 통합용 사본 리포) ↔ VLM FastAPI 의 end-to-end 검증.
#
# tests/test_thema_pa_vlm_bridge.py 의 통합 테스트는 requests.post 를 모킹하지만,
# 본 스크립트는 실제 네트워크 호출 + 모델 추론까지 수행해 운영 흐름을 검증한다.
#
# 검증 흐름:
#   1. VLM FastAPI 가동 확인 (GET /v1/health → 200, status=ready)
#   2. thema_pa_VLM/comm/rest_api.py 의 RestAPI 클래스 로드
#   3. 4개 샘플 페이로드로 RestAPI(config).SendVLMReport(payload) 호출
#       - 실제 POST http://127.0.0.1:8000/v1/report
#       - VLM 측에서 Qwen3-VL LoRA 추론 수행 (~20~35초/요청)
#   4. 응답 검증: status 200, JSON 4 필드 (summary/grade_reason/warnings/recommendation)
#   5. 저장 파일 검증: thema_pa_VLM/storage/vlm_reports/{ymd}_{pigno}_vlm_report.json
#
# 전제 조건:
#   - VLM FastAPI 서버가 이미 가동 중이어야 함 (http://127.0.0.1:8000)
#         conda activate vlm
#         python vlm/api/server.py
#   - thema_pa_VLM 리포가 C:\Users\IPC\Desktop\git\thema_pa_VLM 에 존재
#         (THEMA_PA_ROOT 환경변수로 다른 경로 지정 가능)
#
# 실행:
#   python scripts/test_e2e_thema_pa_bridge.py
# =============================================================================

import contextlib
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


@contextlib.contextmanager
def chdir(path: Path):
    """thema_pa_VLM 운영 cwd 를 모방 — save_vlm_response_json 의 ./storage/... 상대경로가 정확히 thema_pa_VLM 하위에 저장되도록."""
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG

DEFAULT_THEMA_PA_ROOT = Path(r"C:\Users\IPC\Desktop\git\thema_pa_VLM")

API_HOST = "127.0.0.1"
API_PORT = CFG.api.port
BASE_URL = f"http://{API_HOST}:{API_PORT}"

SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/entry_error_case.json",
    "vlm/schema/samples/sample_3473.json",
]


def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def http_get(path: str, timeout: int = 5) -> tuple[int, str]:
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def thema_pa_root() -> Path:
    root = Path(os.environ.get("THEMA_PA_ROOT", str(DEFAULT_THEMA_PA_ROOT)))
    if not root.exists():
        safe_print(f"[FAIL] thema_pa_VLM repo not found: {root}")
        safe_print("  THEMA_PA_ROOT 환경변수로 경로 지정 가능")
        sys.exit(2)
    return root


def load_rest_api(root: Path):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module = importlib.import_module("comm.rest_api")
    return module, module.RestAPI


def main() -> int:
    safe_print("=" * 70)
    safe_print("thema_pa_VLM <-> VLM FastAPI E2E 검증")
    safe_print("=" * 70)

    # 1. health 확인
    safe_print(f"\n[1/3] GET {BASE_URL}/v1/health")
    try:
        status, body = http_get("/v1/health", timeout=5)
    except Exception as e:
        safe_print(f"[FAIL] VLM API 연결 실패: {e}")
        safe_print(f"  먼저 'python vlm/api/server.py' 로 서버를 가동하세요.")
        return 2

    if status != 200:
        safe_print(f"[FAIL] /v1/health status={status} body={body}")
        return 2
    health = json.loads(body)
    if health.get("status") != "ready":
        safe_print(f"[FAIL] 모델 미준비: {health}")
        return 2
    safe_print(f"  OK status=ready, model_used={health.get('model_used')}, "
               f"adapter_exists={health.get('adapter_exists')}")

    # 2. thema_pa_VLM 의 RestAPI 로드
    root = thema_pa_root()
    safe_print(f"\n[2/3] thema_pa_VLM 로드: {root}")
    rest_api_module, RestAPI = load_rest_api(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not config.get("vlm_api", {}).get("enabled"):
        safe_print(f"[FAIL] config.json 의 vlm_api.enabled 가 false")
        return 2
    output_dir_rel = config["vlm_api"]["output_dir"]
    output_dir = (root / output_dir_rel).resolve() if not Path(output_dir_rel).is_absolute() \
                 else Path(output_dir_rel)
    safe_print(f"  vlm_api.url={config['vlm_api']['url']}")
    safe_print(f"  output_dir={output_dir}")

    rest_api = RestAPI(config)

    # 3. 4 샘플 호출
    safe_print(f"\n[3/3] 4개 샘플 SendVLMReport() 실호출")
    results = []
    required_fields = {"summary", "grade_reason", "warnings", "recommendation"}

    for rel in SAMPLES:
        sample_path = ROOT / rel
        if not sample_path.exists():
            safe_print(f"  [SKIP] {rel} (파일 없음)")
            continue
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        carcass_no = payload.get("carcass_no", "?")
        ymd = payload.get("slaughter_ymd", "?")

        # output_dir 의 기존 파일 삭제 (이번 호출 결과인지 확인용)
        out_path = output_dir / f"{ymd}_{carcass_no}_vlm_report.json"
        if out_path.exists():
            out_path.unlink()

        t0 = time.time()
        with chdir(root):
            response = rest_api.SendVLMReport(payload)
        dt = time.time() - t0

        ok = response is not None and response.status_code == 200
        body_keys = set()
        if ok:
            try:
                body_keys = set(response.json().keys())
            except Exception:
                ok = False

        missing = required_fields - body_keys
        saved_ok = out_path.exists()

        status_str = "PASS" if (ok and not missing and saved_ok) else "FAIL"
        results.append({
            "sample": rel,
            "carcass_no": carcass_no,
            "elapsed_sec": round(dt, 2),
            "http_status": getattr(response, "status_code", None),
            "missing_fields": sorted(missing),
            "saved_ok": saved_ok,
            "saved_path": str(out_path) if saved_ok else None,
            "status": status_str,
        })
        safe_print(f"  [{status_str}] {Path(rel).stem:25s} "
                   f"http={getattr(response, 'status_code', None)} "
                   f"elapsed={dt:5.1f}s saved={saved_ok}")

    # 결과 요약
    passed = sum(1 for r in results if r["status"] == "PASS")
    safe_print("\n" + "=" * 70)
    safe_print(f"결과: {passed}/{len(results)} PASS")
    safe_print("=" * 70)
    safe_print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
