"""
thema_pa 이미지 + DB 데이터를 VLM 학습용 JSONL로 변환.

출력: vlm/data/dataset.jsonl
  각 라인: {id, image_path, metadata, summary, tasks}

사용법 (VLM/ 루트에서):
    python scripts/build_dataset.py               # 전체 빌드
    python scripts/build_dataset.py --limit 100   # 일부만
    python scripts/build_dataset.py --stats       # 통계만 출력
"""

import argparse
import json
import os
import re
import sys

import mysql.connector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vlm.schema.thema_pa_output import ThemaPAOutput

IMAGE_DIR = r"C:\Users\IPC\Desktop\git\thema_pa\images"
OUTPUT_PATH = os.path.join(ROOT, "vlm", "data", "dataset.jsonl")

DB_CONFIG = {
    "host":     "127.0.0.1",
    "user":     "root",
    "password": "tmf32277@",
    "db_name":  "ai_grade_judg_dvlp",
}

FILENAME_RE = re.compile(r"^.+_ori_\d+_(\d+)_.+\.jpg$", re.IGNORECASE)

TASK_PROMPTS = {
    "summary":  "이 돼지 도체 이미지의 판정 결과를 현장 작업자에게 3문장으로 요약해주세요.",
    "grade":    "측정 수치를 근거로 이 도체의 등급 판정 이유를 단계별로 설명해주세요.",
    "abnormal": "이 도체에서 비정상으로 감지된 항목과 그 원인을 설명해주세요.",
}


def get_connection():
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["db_name"],
        charset="utf8mb4",
    )


def fetch_all_records() -> dict[str, dict]:
    conn = get_connection()
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
    return {str(r["pigno_cnt"]): r for r in rows}


def scan_images() -> dict[str, str]:
    result = {}
    for fname in os.listdir(IMAGE_DIR):
        m = FILENAME_RE.match(fname)
        if m:
            pigno = m.group(1).lstrip("0") or "0"
            result[pigno] = os.path.join(IMAGE_DIR, fname)
    return result


def row_to_output(row: dict, image_path: str) -> ThemaPAOutput:
    return ThemaPAOutput(
        carcass_no=int(row["pigno_cnt"]),
        slaughter_ymd=str(row["ymd"]),
        backfat_average=float(row.get("act_backfat_thk") or 0),
        multifidus_thk=float(row.get("act_centhk")      or 0),
        body_length=float(row.get("act_length")          or 0),
        body_width=float(row.get("act_width")            or 0),
        body_weight=float(row.get("act_weight")          or 0),
        gender=int(row.get("act_gender")                 or 1),
        grade=str(row.get("act_grade")                   or "등외"),
        error_code={
            "pig_RightEntry": 0, "AI_Backbone_error": 0,
            "AI_BackFat_error": 0, "AI_HalfBone_error": 0,
            "AI_multifidus_error": 0, "AI_Outline_error": 0,
        },
        backbone_slope={"has_large_slope": False, "threshold": None},
        result_image_path=image_path,
    )


def build(limit: int | None = None):
    print("DB 조회 중...")
    db_records = fetch_all_records()
    print(f"  DB 레코드: {len(db_records)}건")

    print("이미지 스캔 중...")
    image_map = scan_images()
    print(f"  이미지: {len(image_map)}장")

    matched_keys = sorted(set(db_records) & set(image_map), key=lambda x: int(x))
    if limit:
        matched_keys = matched_keys[:limit]

    print(f"  매칭: {len(matched_keys)}건 → {OUTPUT_PATH} 생성")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    grade_count = {}
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for key in matched_keys:
            row = db_records[key]
            img_path = image_map[key]
            try:
                output = row_to_output(row, img_path)
            except Exception as e:
                print(f"  [SKIP] pigno={key}: {e}")
                continue

            grade_count[output.grade] = grade_count.get(output.grade, 0) + 1

            tasks = ["summary", "grade"]
            if not output.error_code.is_normal():
                tasks.append("abnormal")

            record = {
                "id":         str(output.carcass_no),
                "image_path": img_path,
                "metadata":   output.model_dump(),
                "summary":    output.summary(),
                "tasks":      {t: TASK_PROMPTS[t] for t in tasks},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n완료: {written}건 저장")
    print("등급 분포:")
    for g, cnt in sorted(grade_count.items()):
        print(f"  {g}: {cnt}건")


def print_stats():
    db_records = fetch_all_records()
    image_map = scan_images()
    matched = set(db_records) & set(image_map)
    print(f"DB 레코드:  {len(db_records)}")
    print(f"이미지:     {len(image_map)}")
    print(f"매칭:       {len(matched)}")
    print(f"DB만 있음:  {len(set(db_records) - set(image_map))}")
    print(f"이미지만:   {len(set(image_map) - set(db_records))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="생성할 최대 샘플 수")
    parser.add_argument("--stats", action="store_true", help="통계만 출력")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        build(limit=args.limit)
