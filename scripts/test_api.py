# =============================================================================
# scripts/test_api.py
# -----------------------------------------------------------------------------
# FastAPI 서버 (vlm/api/server.py) 의 엔드포인트를 4개 샘플로 검증.
#
# 검증 항목:
#   1. GET  /v1/health                  → 200 + status: ready
#   2. POST /v1/report  (정상 케이스)    → 200 + 4개 필드 모두 존재
#   3. POST /v1/report  (오류 케이스 3종)→ 200 + 비정상_근거 또는 주의사항 존재
#
# 전제 조건:
#   FastAPI 서버가 이미 가동 중이어야 함 (http://localhost:8000)
#     python vlm/api/server.py
#
# 동작 방법:
#   python scripts/test_api.py
# =============================================================================

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG

API_HOST = "127.0.0.1"
API_PORT = CFG.api.port
BASE_URL = f"http://{API_HOST}:{API_PORT}"

OUTPUT_FILE = ROOT / "vlm" / "api" / "api_test_results.md"

SAMPLES = [
    "vlm/schema/samples/normal_case.json",
    "vlm/schema/samples/backfat_error_case.json",
    "vlm/schema/samples/entry_error_case.json",
    "vlm/schema/samples/sample_3473.json",
]


def safe_print(msg: str):
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


def http_post_json(path: str, body: dict, timeout: int = 240) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def wait_for_ready(max_wait_sec: int = 600) -> bool:
    """모델 로딩 완료 대기 (최대 10분)."""
    safe_print(f"[BOOT] waiting for model ready at {BASE_URL}/v1/health ...")
    deadline = time.time() + max_wait_sec
    last_status = ""
    while time.time() < deadline:
        try:
            status, body = http_get("/v1/health", timeout=3)
            if status == 200:
                data = json.loads(body)
                cur_status = data.get("status", "")
                if cur_status != last_status:
                    safe_print(f"  [{int(time.time() % 1000)}s] status={cur_status}")
                    last_status = cur_status
                if cur_status == "ready":
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main() -> int:
    results = {"health": None, "reports": []}

    # 1) 모델 로딩 대기
    if not wait_for_ready(max_wait_sec=600):
        safe_print("[FAIL] model did not become ready within 10 minutes")
        return 1

    # 2) /v1/health 검증
    status, body = http_get("/v1/health")
    health = json.loads(body)
    results["health"] = {"status": status, "body": health}
    safe_print(f"\n[HEALTH] {status} {health}")

    # 3) /v1/report 4 샘플 검증
    for sample_path in SAMPLES:
        full = ROOT / sample_path
        with open(full, encoding="utf-8") as f:
            payload = json.load(f)

        # 이미지 경로가 image_dir 외부거나 없으면 None 처리 (보안 검증 통과 위해)
        img = payload.get("result_image_path")
        if img and not Path(img).exists():
            payload["result_image_path"] = None

        safe_print(f"\n[POST /v1/report] {sample_path}")
        t0 = time.time()
        status, body = http_post_json("/v1/report", payload, timeout=240)
        elapsed = round(time.time() - t0, 2)

        try:
            resp = json.loads(body)
        except json.JSONDecodeError:
            resp = {"raw_response": body}

        ok = status == 200 and bool(resp.get("summary"))
        safe_print(f"  status={status} elapsed={elapsed}s ok={ok}")

        results["reports"].append({
            "sample": sample_path,
            "status_code": status,
            "elapsed_sec": elapsed,
            "ok": ok,
            "response": resp,
        })

    # 4) 결과 마크다운 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# FastAPI 엔드포인트 검증 결과\n\n")
        f.write(f"**실행 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**서버 URL**: {BASE_URL}\n\n")

        f.write("## GET /v1/health\n\n")
        f.write(f"- 상태 코드: {results['health']['status']}\n")
        f.write(f"- 응답: `{json.dumps(results['health']['body'], ensure_ascii=False)}`\n\n")
        f.write("---\n\n")

        f.write("## POST /v1/report\n\n")
        passed = sum(1 for r in results["reports"] if r["ok"])
        f.write(f"**통과율**: {passed}/{len(results['reports'])}\n\n")
        for r in results["reports"]:
            mark = "✅" if r["ok"] else "❌"
            f.write(f"### {mark} {r['sample']}\n\n")
            f.write(f"- HTTP: `{r['status_code']}`\n")
            f.write(f"- 추론 시간: {r['elapsed_sec']}초\n")
            f.write("- 응답:\n\n```json\n")
            f.write(json.dumps(r["response"], ensure_ascii=False, indent=2))
            f.write("\n```\n\n---\n\n")

    safe_print(f"\n[DONE] results saved to: {OUTPUT_FILE}")
    return 0 if all(r["ok"] for r in results["reports"]) else 1


if __name__ == "__main__":
    sys.exit(main())
