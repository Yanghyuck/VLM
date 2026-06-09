# =============================================================================
# scripts/distill_summary.py
# -----------------------------------------------------------------------------
# C — 요약(3문장_요약) 다양화 증류.
#   `3문장_요약` 다양성 천장(v4~v8 distinct ~0.23~0.27)은 학습 reference 가
#   템플릿 기반이라 패턴이 일관적이어서 생긴다(capacity·paraphrase 4종으로도 미돌파).
#   base Qwen3-VL(teacher, 고온 sampling)로 **사실은 고정**하고 표현만 다양화한
#   요약을 도체당 K개 생성 → 학습 reference 다변화의 재료로 쓴다.
#
# 핵심 안전장치 — 사실 검증:
#   teacher 출력이 등급·성별·측정 3종(등지방/뭇갈래근/도체중) 숫자를 그대로
#   포함하지 않으면 그 변형은 폐기(환각 주입 방지). 통과분만 저장.
#
# 사용:
#   # 파일럿 (품질 확인)
#   python scripts/distill_summary.py --n 20 --k 3 --output vlm/data/summary_refs_pilot.jsonl
#   # 전량
#   python scripts/distill_summary.py --n 0 --k 3 --output vlm/data/summary_refs.jsonl
#
# 의존성: torch, transformers
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vlm.config import CFG
from vlm.train.convert_dataset import GENDER_MAP, ERROR_LABEL

BASE_MODEL = CFG.model.base_model_id
DATASET_JSONL = ROOT / CFG.paths.dataset_jsonl

# 변형별 스타일 힌트 — K개 변형이 서로 다른 구조를 갖도록 유도(다양성↑).
STYLE_HINTS = [
    "도축 정보 → 측정값 → 등급 순의 서술형으로.",
    "최종 등급을 먼저 단언하고, 이어 측정값과 도축 정보를 설명하는 순서로.",
    "측정값을 중심으로 간결하게, 마지막 문장에서 등급을 결론짓는 형태로.",
    "검출 상태와 신뢰도를 강조하는 톤으로.",
]


def _gender_label(g) -> str:
    return GENDER_MAP.get(g, "미상")


def _facts(meta: dict) -> dict:
    ymd = str(meta["slaughter_ymd"])
    errors = [label for key, (label, _) in ERROR_LABEL.items()
              if (meta.get("error_code") or {}).get(key, 0) == 1]
    return {
        "carcass_no": meta["carcass_no"],
        "gender": _gender_label(meta["gender"]),
        "date": f"{ymd[:4]}년 {ymd[4:6]}월 {ymd[6:]}일",
        "backfat": f"{meta['backfat_average']}",
        "multifidus": f"{meta['multifidus_thk']}",
        "weight": f"{meta['body_weight']}",
        "grade": str(meta["grade"]),
        "errors": errors,
    }


def _build_prompt(f: dict, style: str) -> str:
    err_line = (f"검출 오류: {', '.join(f['errors'])}" if f["errors"]
                else "검출 상태: 모든 AI 항목 정상")
    return (
        "다음은 돼지 도체 한 마리의 측정 사실입니다. 이 사실만 사용하여 "
        "도축장 판정 리포트용 한국어 3문장 요약을 작성하세요.\n\n"
        "[사실]\n"
        f"- 도체번호: {f['carcass_no']}\n"
        f"- 성별: {f['gender']}\n"
        f"- 도축일: {f['date']}\n"
        f"- 등지방 두께: {f['backfat']}mm\n"
        f"- 뭇갈래근 두께: {f['multifidus']}mm\n"
        f"- 도체중: {f['weight']}kg\n"
        f"- 판정 등급: {f['grade']}\n"
        f"- {err_line}\n\n"
        "[작성 규칙]\n"
        "- 정확히 3문장, 자연스러운 한국어 '~습니다' 체.\n"
        "- 위 수치(등지방/뭇갈래근/도체중)·등급·성별은 반드시 그대로 포함하고 절대 바꾸지 마세요.\n"
        "- 위 사실에 없는 내용은 지어내지 마세요.\n"
        "- 품질·가치 평가 표현 금지: '건강한·최고·고급·우수·훌륭·뛰어난·우량·적절한 수준' 등 "
        "주관적 판단어를 쓰지 말고, 측정 사실과 검출 상태만 객관적으로 기술하세요.\n"
        f"- 표현 스타일: {style} 이전과 겹치지 않게 매번 다른 어휘·문장 구조로 작성하세요.\n\n"
        "요약:"
    )


# 영문 글리치 탐지 — base 가 "뭇갈래근"을 "limburg 근" 처럼 영어 토큰으로
# 치환하는 디코딩 사고(파일럿에서 관찰). 단위 mm/kg/cm 만 허용, 그 외 3자+ 영문 차단.
_LATIN_GLITCH = re.compile(r"\b(?!mm\b|kg\b|cm\b)[A-Za-z]{3,}\b")


def _is_faithful(text: str, f: dict) -> bool:
    """사실 검증 — 등급·성별·측정 3종 숫자 보존 + 영문 글리치 없음."""
    if f["grade"] not in text:
        return False
    if f["gender"] not in text:
        return False
    for num in (f["backfat"], f["multifidus"], f["weight"]):
        # "22.0" 또는 "22" 형태 모두 허용 (소수점 .0 누락 케이스)
        base = num.rstrip("0").rstrip(".") if "." in num else num
        if num not in text and base not in text:
            return False
    if _LATIN_GLITCH.search(text):      # limburg 같은 영어 토큰 끼임 → 폐기
        return False
    return True


def load_model():
    print(f"[distill-summary] base 모델 로딩: {BASE_MODEL}")
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # 배치 생성 시 디코더는 left-padding 이어야 출력이 어긋나지 않음.
    if getattr(proc, "tokenizer", None) is not None:
        proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, proc


def generate_batch(model, proc, prompts: list[str], temperature: float, top_p: float) -> list[str]:
    """C1 — K개 프롬프트를 한 번의 forward 로 배치 생성(~3x 가속, 메모리대역 amortize)."""
    texts = [
        proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": p}]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = proc(text=texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=1.05,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [s.strip() for s in proc.batch_decode(gen, skip_special_tokens=True)]


def _is_abnormal(meta: dict) -> bool:
    return any((v or 0) for v in (meta.get("error_code") or {}).values())


def sample_records(n: int):
    records = [json.loads(l) for l in open(DATASET_JSONL, encoding="utf-8")]
    records.sort(key=lambda r: str(r["id"]))
    if n <= 0 or n >= len(records):
        return records                                 # 전량 (resume 안정 순서)
    # 파일럿: 비정상 케이스 검증을 위해 최대 5건 강제 포함
    abn = [r for r in records if _is_abnormal(r["metadata"])]
    nrm = [r for r in records if not _is_abnormal(r["metadata"])]
    n_abn = min(len(abn), max(0, min(5, n // 3)))
    return abn[:n_abn] + nrm[: n - n_abn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=3, help="도체당 생성·검증 통과 목표 변형 수")
    ap.add_argument("--max-tries", type=int, default=5, help="도체당 최대 시도(검증 실패 재시도 포함)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="vlm/data/summary_refs_pilot.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    picked = sample_records(args.n)
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if out_path.exists() and not args.overwrite:
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(str(json.loads(line)["id"]))
            except Exception:
                continue
    todo = [r for r in picked if str(r["id"]) not in done_ids]
    print(f"[distill-summary] 대상 {len(picked)} 중 미완료 {len(todo)} "
          f"(완료 {len(done_ids)}) | k={args.k} temp={args.temperature}")

    model, proc = load_model()

    mode = "w" if args.overwrite else "a"
    total_valid = total_try = 0
    with open(out_path, mode, encoding="utf-8") as fout:
        for i, r in enumerate(todo, 1):
            f = _facts(r["metadata"])
            variants: list[str] = []
            rounds = 0
            while len(variants) < args.k and rounds < args.max_tries:
                need = args.k - len(variants)
                prompts = [
                    _build_prompt(f, STYLE_HINTS[(len(variants) + j + rounds * args.k) % len(STYLE_HINTS)])
                    for j in range(need)
                ]
                raws = generate_batch(model, proc, prompts, args.temperature, args.top_p)
                rounds += 1
                total_try += len(raws)
                for raw in raws:
                    cand = re.sub(r"\s+", " ", raw).strip()       # 줄바꿈 제거·공백 정규화
                    if cand and _is_faithful(cand, f) and cand not in variants:
                        variants.append(cand)
            total_valid += len(variants)
            rec = {
                "id": r["id"],
                "carcass_no": f["carcass_no"],
                "grade": f["grade"],
                "error_code": r["metadata"].get("error_code"),
                "summaries": variants,        # 검증 통과한 다양한 표현들
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  [{i}/{len(todo)}] id={r['id']} 통과 {len(variants)}/{args.k} (배치 {rounds}회)")

    rate = (total_valid / total_try * 100) if total_try else 0
    print(f"\n[distill-summary] 완료: 변형 {total_valid}개 / 시도 {total_try} "
          f"(검증 통과율 {rate:.1f}%) → {out_path}")


if __name__ == "__main__":
    main()
