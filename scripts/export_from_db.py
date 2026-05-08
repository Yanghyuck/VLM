# =============================================================================
# scripts/export_from_db.py
# -----------------------------------------------------------------------------
# 기능:
#   thema_pa MySQL DB(tb_act_result)에서 도체 판정 결과를 조회하여
#   ThemaPAOutput 형식의 JSON 파일로 저장합니다.
#   단일 도체 조회 또는 최근 N건 일괄 export를 지원합니다.
#
# 출력 위치: config.json > paths.samples_dir  (기본: vlm/schema/samples/)
#   파일명: sample_{pigno_cnt}.json
#
# 동작 방법:
#   # 최근 10건 목록만 출력 (파일 저장 없음)
#   python scripts/export_from_db.py
#
#   # 특정 도체번호 단일 export
#   python scripts/export_from_db.py --pigno 3473
#
#   # 최근 N건 일괄 export
#   python scripts/export_from_db.py --all --limit 20
#
# 설정 (config.json):
#   db.host / db.user / db.password / db.name  : MySQL 접속 정보
#   paths.samples_dir                          : 샘플 JSON 출력 디렉터리
#
# 전제 조건:
#   - MySQL 서버 실행 중
#   - 프로젝트 루트에 config.json 존재
#
# 의존성:
#   mysql-connector-python
# =============================================================================

import argparse
import json
import os
import re
import sys

import mysql.connector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vlm.config import CFG

OUTPUT_DIR = os.path.join(ROOT, CFG.paths.samples_dir)
IMAGE_DIR = CFG.paths.image_dir

# AI 이미지: 0716_ai_{datetime}_{pigno}_{pigno+offset}_SP_CAM7.jpg
# ORI 이미지: 0716_ori_{datetime}_{pigno}_SP_CAM7.jpg
FILENAME_RE_AI = re.compile(r"^.+_ai_\d+_(\d+)_\d+_.+\.jpg$", re.IGNORECASE)
FILENAME_RE_ORI = re.compile(r"^.+_ori_\d+_(\d+)_.+\.jpg$", re.IGNORECASE)
FILENAME_RE = FILENAME_RE_AI if "AI" in IMAGE_DIR.upper() else FILENAME_RE_ORI


def get_connection():
    return mysql.connector.connect(
        host=CFG.db.host,
        user=CFG.db.user,
        password=CFG.db.password,
        database=CFG.db.name,
        charset="utf8mb4",
    )


def fetch_act(cur, pigno=None, limit=10):
    if pigno:
        cur.execute(
            """SELECT pigno_cnt, LEFT(date,8) as ymd,
                      act_backfat_thk, act_centhk,
                      act_length, act_width, act_weight,
                      act_gender, act_grade
               FROM tb_act_result
               WHERE pigno_cnt = %s
               ORDER BY date DESC LIMIT 1""",
            (str(pigno),),
        )
    else:
        cur.execute(
            f"""SELECT pigno_cnt, LEFT(date,8) as ymd,
                       act_backfat_thk, act_centhk,
                       act_length, act_width, act_weight,
                       act_gender, act_grade
                FROM tb_act_result
                ORDER BY date DESC LIMIT {int(limit)}"""
        )
    return cur.fetchall()


def scan_images() -> dict[str, str]:
    result = {}
    if not os.path.isdir(IMAGE_DIR):
        return result

    for fname in os.listdir(IMAGE_DIR):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        pigno = m.group(1).lstrip("0") or "0"
        result[pigno] = os.path.join(IMAGE_DIR, fname)
    return result


def row_to_output(row: dict, image_path: str | None) -> dict:
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
            "pig_RightEntry":      0,
            "AI_Backbone_error":   0,
            "AI_BackFat_error":    0,
            "AI_HalfBone_error":   0,
            "AI_multifidus_error": 0,
            "AI_Outline_error":    0,
        },
        "backbone_slope": {"has_large_slope": False, "threshold": None},
        "result_image_path": image_path,
    }


def export_one(pigno: int):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    rows = fetch_act(cur, pigno=pigno)
    conn.close()

    if not rows:
        print(f"pigno_cnt={pigno} 데이터 없음")
        return

    image_map = scan_images()
    output = row_to_output(rows[0], image_map.get(str(pigno)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"sample_{pigno}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    img_status = "있음" if output["result_image_path"] else "없음"
    print(f"저장: {path}  (등급={output['grade']}, 등지방={output['backfat_average']}mm, 이미지={img_status})")


def export_all(limit: int):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    rows = fetch_act(cur, limit=limit)
    conn.close()

    image_map = scan_images()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for row in rows:
        output = row_to_output(row, image_map.get(str(row["pigno_cnt"])))
        path = os.path.join(OUTPUT_DIR, f"sample_{row['pigno_cnt']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        img_status = "있음" if output["result_image_path"] else "없음"
        print(f"저장: {path}  (등급={output['grade']}, 등지방={output['backfat_average']}mm, 이미지={img_status})")


def list_recent(limit: int = 10):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    rows = fetch_act(cur, limit=limit)
    conn.close()

    print(f"{'pigno_cnt':>10}  {'날짜':>10}  {'등급':>4}  {'등지방':>6}  {'뭇갈래근':>8}  {'체중':>6}  {'성별':>4}")
    print("-" * 60)
    for r in rows:
        gender_map = {1: "암", 2: "수", 3: "거세"}
        gender = gender_map.get(int(r.get("act_gender") or 0), "?")
        print(
            f"{r['pigno_cnt']:>10}  {r['ymd']:>10}  {str(r.get('act_grade') or '?'):>4}  "
            f"{str(r.get('act_backfat_thk') or 0):>6}  {str(r.get('act_centhk') or 0):>8}  "
            f"{str(r.get('act_weight') or 0):>6}  {gender:>4}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pigno", type=int, help="단일 도체번호 export")
    parser.add_argument("--all",   action="store_true", help="최근 N건 일괄 export")
    parser.add_argument("--limit", type=int, default=10, help="일괄 export 건수 (기본 10)")
    args = parser.parse_args()

    if args.pigno:
        export_one(args.pigno)
    elif args.all:
        export_all(args.limit)
    else:
        list_recent(args.limit)
