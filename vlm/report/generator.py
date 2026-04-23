"""
thema_pa 판정 결과 → 한국어 리포트 생성기.

사용법:
    from vlm.report.generator import generate_report
    from vlm.schema.thema_pa_output import ThemaPAOutput

    output = ThemaPAOutput(**data)
    report = generate_report(output)
    # report: {"3문장_요약": "...", "비정상_근거": null, "주의사항": [...], "권고": "..."}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from vlm.schema.thema_pa_output import ThemaPAOutput

PROMPT_DIR = Path(__file__).parent.parent / "prompt"
MODEL = "claude-opus-4-7"

_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _load(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _select_template(output: ThemaPAOutput) -> str:
    """error_code 상태에 따라 프롬프트 템플릿 선택."""
    ec = output.error_code
    if ec.pig_RightEntry == 1 or ec.AI_BackFat_error == 1:
        return _load("failure_analysis.txt")
    if not ec.is_normal():
        return _load("error_case.txt")
    return _load("normal_case.txt")


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 마크다운 코드블록 안에 JSON이 있을 경우 추출
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"LLM 응답에서 JSON 파싱 실패:\n{text[:300]}")


def generate_report(output: ThemaPAOutput) -> dict:
    """ThemaPAOutput → 한국어 판정 리포트 dict를 반환.

    반환 형식:
        {
            "3문장_요약": str,
            "비정상_근거": str | None,
            "주의사항": list[str],
            "권고": str,
        }
    """
    client = _get_client()
    system_text = _load("system_prompt.txt")
    template = _select_template(output)
    user_content = template.replace("{{SUMMARY}}", output.summary())

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    for block in response.content:
        if block.type == "text":
            return _extract_json(block.text)

    raise ValueError("LLM 응답에 text 블록이 없습니다")
