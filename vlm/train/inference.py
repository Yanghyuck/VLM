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
_loaded_mode: Optional[str] = None  # "lora:<path>" | "base"


def _load_model(use_adapter: bool = True, adapter_path: str | None = None):
    """모델 로드.

    Args:
        use_adapter: True 면 LoRA 어댑터 적용. False 면 베이스 모델만 사용.
        adapter_path: 사용할 어댑터 경로. None 이면 CFG.paths.lora_adapter 사용.
                      여러 어댑터(v1, v2)를 비교할 때 명시.
    """
    global _model, _processor, _loaded_mode

    effective_path = adapter_path if adapter_path else ADAPTER_PATH
    desired_mode = f"lora:{effective_path}" if use_adapter else "base"

    if _model is not None and _loaded_mode == desired_mode:
        return
    if _model is not None and _loaded_mode != desired_mode:
        print(f"[inference] 모드 전환: {_loaded_mode} -> {desired_mode}")
        del _model
        _model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[inference] 모델 로딩: {BASE_MODEL_ID} (mode={desired_mode})")
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter = Path(effective_path)
    if use_adapter and adapter.exists():
        print(f"[inference] LoRA 어댑터 로딩: {effective_path}")
        _model = PeftModel.from_pretrained(base, str(effective_path))
        _loaded_mode = desired_mode
    else:
        if use_adapter and not adapter.exists():
            print(f"[inference] 어댑터 없음 — 베이스 모델로 추론 ({effective_path})")
        else:
            print(f"[inference] 베이스 모델 단독 추론 (use_adapter=False)")
        _model = base
        _loaded_mode = "base"

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


from vlm.train.json_utils import _find_balanced_json, _extract_json  # noqa: F401


def generate_report(
    output: ThemaPAOutput,
    use_adapter: bool = True,
    adapter_path: str | None = None,
) -> dict:
    """ThemaPAOutput → 한국어 판정 리포트 dict.

    Args:
        output: 도체 판정 결과
        use_adapter: True 면 LoRA 어댑터 적용, False 면 베이스 모델만 사용 (벤치마크용)
        adapter_path: 사용할 어댑터 경로 (None 이면 config 기본값)

    반환 형식:
        {
            "3문장_요약": str,
            "비정상_근거": str | None,
            "주의사항": list[str],
            "권고": str,
        }
    """
    _load_model(use_adapter=use_adapter, adapter_path=adapter_path)

    system_text = _load_prompt("system_prompt.txt")
    template    = _select_template(output)
    user_text   = template.replace("{{SUMMARY}}", output.summary())

    image_path = output.result_image_path
    if image_path and Path(image_path).exists():
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        # OOM 방지: 학습 시 사용한 image_max_pixels 로 리사이즈
        max_pixels = getattr(CFG.model, "image_max_pixels", 200_704)
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.LANCZOS)
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
