# =============================================================================
# vlm/train/inference.py
# -----------------------------------------------------------------------------
# 기능:
#   LLaMA-Factory로 학습된 Qwen3-VL-8B LoRA 모델을 로컬에서 실행하여
#   한국어 판정 리포트를 생성합니다.
#   vlm/report/generator.py 와 동일한 인터페이스를 제공하므로
#   두 파일을 교체하여 사용할 수 있습니다.
#
# 동작 순서:
#   1. 첫 호출 시 config.json > model.base_model_id 의 베이스 모델을 GPU에 로드
#   2. config.json > paths.lora_adapter 경로에 어댑터가 있으면 자동 적용
#      (없으면 베이스 모델로 추론)
#   3. error_code 상태에 따라 vlm/prompt/ 템플릿 선택
#   4. 이미지 경로가 유효하면 이미지+텍스트 멀티모달 입력으로 처리
#      (이미지 없으면 텍스트만 입력)
#   5. JSON 형식으로 응답 파싱 후 반환
#
# 사용 예시:
#   import json
#   from vlm.schema.thema_pa_output import ThemaPAOutput
#   from vlm.train.inference import generate_report
#
#   with open("vlm/schema/samples/normal_case.json", encoding="utf-8") as f:
#       data = json.load(f)
#   report = generate_report(ThemaPAOutput(**data))
#   print(report)
#
# 반환 형식:
#   {
#     "3문장_요약": str,        # 도체번호·등급·측정값 3문장
#     "비정상_근거": str|None,  # 오류 있을 때만 문자열
#     "주의사항": list[str],    # 확인 필요 항목 목록
#     "권고": str               # 현장 조치 권고
#   }
#
# 설정 (config.json):
#   model.base_model_id  : 베이스 모델 HuggingFace ID
#   paths.lora_adapter   : LoRA 어댑터 디렉터리 경로
#
# 전제 조건:
#   GPU 필수 (RTX 4090 권장, 최소 16GB VRAM)
#   프로젝트 루트에 config.json 존재
#
# 의존성:
#   torch, transformers, peft, pillow, qwen_vl_utils
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from vlm.schema.thema_pa_output import ThemaPAOutput
from vlm.config import CFG

BASE_MODEL_ID = CFG.model.base_model_id
ADAPTER_PATH  = str(Path(__file__).parent.parent.parent / CFG.paths.lora_adapter)
PROMPT_DIR    = Path(__file__).parent.parent / "prompt"

_model:     Optional[object] = None
_processor: Optional[object] = None


def _load_model():
    global _model, _processor

    if _model is not None:
        return

    print(f"[inference] 모델 로딩: {BASE_MODEL_ID}")
    _processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter = Path(ADAPTER_PATH)
    if adapter.exists():
        print(f"[inference] LoRA 어댑터 로딩: {ADAPTER_PATH}")
        _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    else:
        print(f"[inference] 어댑터 없음 — 베이스 모델로 추론 ({ADAPTER_PATH})")
        _model = base

    _model.eval()


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _select_template(output: ThemaPAOutput) -> str:
    ec = output.error_code
    if ec.pig_RightEntry == 1 or ec.AI_BackFat_error == 1:
        return _load_prompt("failure_analysis.txt")
    if not ec.is_normal():
        return _load_prompt("error_case.txt")
    return _load_prompt("normal_case.txt")


def _find_balanced_json(text: str) -> str | None:
    """텍스트에서 첫 번째 완성된 JSON 객체를 brace counting으로 추출.

    중첩 객체({"a": {"b": 1}})도 정확히 매칭. 코드블록이 있으면 우선 처리.
    """
    # 코드블록 제거
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

    # JSON 파싱 실패 시 텍스트를 요약 필드에 담아 반환
    return {
        "3문장_요약": text[:300],
        "비정상_근거": None,
        "주의사항": [],
        "권고": "",
    }


def generate_report(output: ThemaPAOutput) -> dict:
    """ThemaPAOutput → 한국어 판정 리포트 dict.

    반환 형식:
        {
            "3문장_요약": str,
            "비정상_근거": str | None,
            "주의사항": list[str],
            "권고": str,
        }
    """
    _load_model()

    system_text = _load_prompt("system_prompt.txt")
    template    = _select_template(output)
    user_text   = template.replace("{{SUMMARY}}", output.summary())

    image_path = output.result_image_path
    if image_path and Path(image_path).exists():
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system",  "content": system_text},
            {"role": "user",    "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
    else:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ]

    text_input = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if image_path and Path(image_path).exists():
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_model.device)
    else:
        inputs = _processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response  = _processor.batch_decode(generated, skip_special_tokens=True)[0]

    return _extract_json(response)
