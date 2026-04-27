# =============================================================================
# vlm/bench/dataset.py
# -----------------------------------------------------------------------------
# 기능:
#   평가셋 두 가지 빌드 방법 제공:
#   1) build_eval_set(n, seed)         — dataset.jsonl 에서 추출 (in-distribution)
#   2) build_eval_set_from_db(n, seed) — DB 직접 조회 후 dataset.jsonl 외 샘플만
#                                         (held-out, 학습에 안 쓰인 도체)
#
#   각 샘플에 대해 "정답 응답"을 convert_dataset 의 함수로 생성.
#
# 출력:
#   {
#     "id": "도체번호",
#     "metadata": ThemaPAOutput JSON,
#     "image_path": "...",
#     "tasks": {
#       "summary":   {"prompt": "...", "reference": "..."},
#       "grade":     {"prompt": "...", "reference": "..."},
#       "abnormal":  {"prompt": "...", "reference": "..."}  (오류 케이스만)
#     }
#   }
#
# 동작 방법:
#   # in-distribution (학습 데이터 분포 내)
#   python vlm/bench/dataset.py --n 50
#
#   # held-out (DB 직접 조회, 학습 데이터 외)
#   python vlm/bench/dataset.py --source db --n 50
#
# 의존성:
#   mysql-connector-python (db 모드 시)
# =============================================================================

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vlm.config import CFG
from vlm.train.convert_dataset import (
    _summary_response,
    _grade_response,
    _abnormal_response,
    _is_normal,
)

DATASET_JSONL = ROOT / CFG.paths.dataset_jsonl
IMAGE_DIR     = CFG.paths.image_dir
FILENAME_RE_AI  = re.compile(r"^.+_ai_\d+_(\d+)_\d+_.+\.jpg$",  re.IGNORECASE)
FILENAME_RE_ORI = re.compile(r"^.+_ori_\d+_(\d+)_.+\.jpg$",     re.IGNORECASE)
FILENAME_RE   = FILENAME_RE_AI if "AI" in IMAGE_DIR.upper() else FILENAME_RE_ORI

TASK_PROMPTS = {
    "summary":  "이 돼지 도체 이미지의 판정 결과를 현장 작업자에게 3문장으로 요약해주세요.",
    "grade":    "측정 수치를 근거로 이 도체의 등급 판정 이유를 단계별로 설명해주세요.",
    "abnormal": "이 도체에서 비정상으로 감지된 항목과 그 원인을 설명해주세요.",
}


def _row_to_meta(row: dict) -> dict:
    """DB row → ThemaPAOutput dict (build_dataset.row_to_output 과 동일 로직)."""
    return {
        "carcass_no":      int(row["pigno_cnt"]),
        "slaughter_ymd":   str(row["ymd"]),
        "backfat_average": float(row.get("act_backfat_thk") or 0),
        "multifidus_thk":  float(row.get("act_centhk")      or 0),
        "body_length":     float(row.get("act_length")       or 0),
        "body_width":      float(row.get("act_width")        or 0),
        "body_weight":     float(row.get("act_weight")       or 0),
        "gender":          int(row.get("act_gender")         or 1),
        "grade":           str(row.get("act_grade")          or "등외"),
        "error_code": {
            "pig_RightEntry": 0, "AI_Backbone_error": 0, "AI_BackFat_error": 0,
            "AI_HalfBone_error": 0, "AI_multifidus_error": 0, "AI_Outline_error": 0,
        },
        "backbone_slope": {"has_large_slope": False, "threshold": None},
        "result_image_path": None,
    }


def _build_tasks(meta: dict) -> dict:
    """메타데이터로부터 task별 prompt + reference 생성."""
    tasks = {
        "summary": {"prompt": TASK_PROMPTS["summary"], "reference": _summary_response(meta)},
        "grade":   {"prompt": TASK_PROMPTS["grade"],   "reference": _grade_response(meta)},
    }
    if not _is_normal(meta["error_code"]):
        tasks["abnormal"] = {"prompt": TASK_PROMPTS["abnormal"], "reference": _abnormal_response(meta)}
    return tasks


def build_eval_set(n: int = 50, seed: int = 42) -> list[dict]:
    """dataset.jsonl 에서 결정적 추출 (in-distribution)."""
    if not DATASET_JSONL.exists():
        raise FileNotFoundError(f"dataset.jsonl 없음: {DATASET_JSONL}. scripts/build_dataset.py 먼저 실행.")

    records = []
    with open(DATASET_JSONL, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    rng = random.Random(seed)
    selected = rng.sample(records, min(n, len(records)))

    return [
        {
            "id":         rec["id"],
            "image_path": rec["image_path"],
            "metadata":   rec["metadata"],
            "tasks":      _build_tasks(rec["metadata"]),
        }
        for rec in selected
    ]


def _scan_images() -> dict[str, str]:
    """이미지 디렉터리 → {pigno: path} 매핑."""
    result = {}
    if not os.path.isdir(IMAGE_DIR):
        return result
    for fname in os.listdir(IMAGE_DIR):
        m = FILENAME_RE.match(fname)
        if m:
            pigno = m.group(1).lstrip("0") or "0"
            result[pigno] = os.path.join(IMAGE_DIR, fname)
    return result


def build_eval_set_from_db(n: int = 50, seed: int = 42, exclude_jsonl: bool = True) -> list[dict]:
    """DB 직접 조회 → dataset.jsonl 에 없는 도체만 평가셋 구성 (held-out).

    Args:
        n: 추출할 샘플 수
        seed: 랜덤 시드
        exclude_jsonl: True 면 dataset.jsonl 에 있는 도체번호 제외 (default)

    Returns:
        eval samples list
    """
    import mysql.connector

    excluded_ids: set[str] = set()
    if exclude_jsonl and DATASET_JSONL.exists():
        with open(DATASET_JSONL, encoding="utf-8") as f:
            for line in f:
                excluded_ids.add(json.loads(line)["id"])

    conn = mysql.connector.connect(
        host=CFG.db.host, user=CFG.db.user, password=CFG.db.password,
        database=CFG.db.name, charset="utf8mb4",
    )
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT pigno_cnt, LEFT(date, 8) as ymd,
               act_backfat_thk, act_centhk,
               act_length, act_width, act_weight,
               act_gender, act_grade
        FROM tb_act_result
        WHERE act_grade IS NOT NULL AND act_grade != ''
    """)
    rows = cur.fetchall()
    conn.close()

    image_map = _scan_images()

    candidates = []
    for row in rows:
        pigno = str(row["pigno_cnt"])
        if pigno in excluded_ids:
            continue
        if pigno not in image_map:
            continue
        candidates.append({"row": row, "image_path": image_map[pigno]})

    print(f"[DB] 전체 레코드: {len(rows)}, 이미지 매칭: {sum(1 for r in rows if str(r['pigno_cnt']) in image_map)}")
    print(f"[DB] 학습 외 후보 (held-out + 이미지 존재): {len(candidates)}건")

    if len(candidates) < n:
        print(f"[WARN] 후보 부족: 요청 {n} > 가용 {len(candidates)}")
        n = len(candidates)

    rng = random.Random(seed)
    selected = rng.sample(candidates, n)

    eval_samples = []
    for c in selected:
        meta = _row_to_meta(c["row"])
        meta["result_image_path"] = c["image_path"]
        eval_samples.append({
            "id":         str(meta["carcass_no"]),
            "image_path": c["image_path"],
            "metadata":   meta,
            "tasks":      _build_tasks(meta),
        })

    return eval_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["jsonl", "db"], default="jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(ROOT / "vlm" / "bench" / "eval_set.jsonl"))
    args = parser.parse_args()

    if args.source == "db":
        samples = build_eval_set_from_db(n=args.n, seed=args.seed)
    else:
        samples = build_eval_set(n=args.n, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_normal   = sum(1 for s in samples if _is_normal(s["metadata"]["error_code"]))
    n_abnormal = len(samples) - n_normal
    grade_dist = {}
    for s in samples:
        g = s["metadata"]["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    print(f"평가셋 생성 ({args.source}): {len(samples)}건 -> {out_path}")
    print(f"  정상/비정상: {n_normal} / {n_abnormal}")
    print(f"  등급 분포:   {grade_dist}")
