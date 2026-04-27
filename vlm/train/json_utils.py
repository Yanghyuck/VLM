# =============================================================================
# vlm/train/json_utils.py
# -----------------------------------------------------------------------------
# 기능:
#   모델 응답 텍스트에서 JSON 객체를 추출합니다.
#   torch 의존성 없이 단독 import 가능 (CI 환경 최소 deps 호환).
#
# 주요 함수:
#   _find_balanced_json(text) — brace counting 으로 첫 완성된 {...} 블록 추출
#   _extract_json(text)        — JSON 파싱 시도, 실패 시 fallback dict 반환
# =============================================================================

from __future__ import annotations

import json
import re


def _find_balanced_json(text: str) -> str | None:
    """텍스트에서 첫 번째 완성된 JSON 객체를 brace counting으로 추출.

    중첩 객체({"a": {"b": 1}})도 정확히 매칭. 코드블록이 있으면 우선 처리.
    """
    fence = re.search(r"```(?:json)?\s*", text)
    if fence:
        text = text[fence.end():]

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = _find_balanced_json(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {
        "3문장_요약": text[:300],
        "비정상_근거": None,
        "주의사항": [],
        "권고": "",
    }
