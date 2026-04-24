# =============================================================================
# vlm/demo/app.py
# -----------------------------------------------------------------------------
# 기능:
#   Qwen3-VL-8B LoRA 모델을 사용한 돼지 도체 판정 Streamlit 데모.
#   도체 이미지와 측정값을 입력하면 한국어 판정 리포트를 생성합니다.
#
# 레이아웃 (3패널):
#   왼쪽  : 도체 이미지 + JSON 파일 업로드 / 샘플 선택
#   가운데: Qwen3-VL LoRA 판정 리포트 (3문장 요약 / 등급 근거 / 주의사항 / 권고)
#   오른쪽: 원본 측정값 테이블 + 등급 배지 + AI 검출 상태
#
# 동작 방법:
#   # 프로젝트 루트에서 실행
#   streamlit run vlm/demo/app.py
#
# 설정 (config.json):
#   paths.samples_dir    : 샘플 JSON 디렉터리 (사이드바 샘플 목록)
#   paths.lora_adapter   : LoRA 어댑터 경로 (없으면 베이스 모델로 추론)
#
# 전제 조건:
#   GPU 필수 (RTX 4090 권장, 최소 16GB VRAM)
#   프로젝트 루트에 config.json 존재
#
# 의존성:
#   streamlit, pillow
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vlm.schema.thema_pa_output import ThemaPAOutput, ErrorCode, BackboneSlope
from vlm.config import CFG

st.set_page_config(
    page_title="축산 VLM 판정 데모",
    page_icon="🐷",
    layout="wide",
)

GRADE_COLOR = {"1+": "🟢", "1": "🔵", "2": "🟡", "등외": "🔴"}
SAMPLE_DIR = ROOT / CFG.paths.samples_dir


@st.cache_resource(show_spinner="모델 로딩 중...")
def load_inference():
    from vlm.train import inference
    return inference


def _build_output(meta: dict, image_path: str | None) -> ThemaPAOutput:
    return ThemaPAOutput(
        carcass_no=meta["carcass_no"],
        slaughter_ymd=meta["slaughter_ymd"],
        backfat_average=meta["backfat_average"],
        multifidus_thk=meta["multifidus_thk"],
        body_length=meta.get("body_length", 0.0),
        body_width=meta.get("body_width", 0.0),
        body_weight=meta["body_weight"],
        gender=meta["gender"],
        grade=meta["grade"],
        error_code=ErrorCode(**meta.get("error_code", {})),
        backbone_slope=BackboneSlope(**meta.get("backbone_slope", {"has_large_slope": False, "threshold": None})),
        result_image_path=image_path,
    )


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("입력")

    sample_files = sorted(SAMPLE_DIR.glob("*.json")) if SAMPLE_DIR.exists() else []
    sample_names = ["직접 입력"] + [f.stem for f in sample_files]
    selected = st.selectbox("샘플 선택", sample_names)

    uploaded_json = st.file_uploader("JSON 업로드", type="json")
    uploaded_img = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

    run_btn = st.button("판정 실행", type="primary", use_container_width=True)

# ── 메타데이터 로드 ───────────────────────────────────────────────────────────
meta: dict | None = None
image_path: str | None = None

if uploaded_json:
    meta = json.load(uploaded_json)
elif selected != "직접 입력":
    path = SAMPLE_DIR / f"{selected}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("result_image_path") and Path(meta["result_image_path"]).exists():
        image_path = meta["result_image_path"]

if uploaded_img:
    tmp = ROOT / "vlm" / "data" / "_uploaded_image.jpg"
    tmp.write_bytes(uploaded_img.read())
    image_path = str(tmp)

# ── 3패널 레이아웃 ────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 1.6, 1])

with col_left:
    st.subheader("도체 이미지")
    if image_path and Path(image_path).exists():
        st.image(Image.open(image_path), use_container_width=True)
    else:
        st.info("이미지 없음")

    if meta:
        with st.expander("원본 JSON 보기"):
            st.json(meta)

with col_mid:
    st.subheader("판정 리포트")

    if run_btn and meta:
        if not meta:
            st.warning("JSON을 먼저 선택하거나 업로드하세요.")
        else:
            try:
                output = _build_output(meta, image_path)
            except Exception as e:
                st.error(f"입력 오류: {e}")
                st.stop()

            inference = load_inference()
            with st.spinner("추론 중..."):
                result = inference.generate_report(output)

            st.success("판정 완료")

            st.markdown("#### 3문장 요약")
            st.write(result.get("3문장_요약", ""))

            if result.get("비정상_근거"):
                st.markdown("#### 비정상 근거")
                st.warning(result["비정상_근거"])

            warnings = result.get("주의사항", [])
            if warnings:
                st.markdown("#### 주의사항")
                for w in warnings:
                    st.markdown(f"- {w}")

            st.markdown("#### 권고")
            st.info(result.get("권고", ""))

    elif not run_btn:
        st.info("왼쪽에서 샘플을 선택하고 '판정 실행'을 누르세요.")

with col_right:
    st.subheader("측정값")

    if meta:
        grade = meta.get("grade", "?")
        badge = GRADE_COLOR.get(grade, "⚪")
        st.markdown(f"## {badge} {grade} 등급")

        rows = [
            ("도체번호", meta.get("carcass_no", "")),
            ("도축일", meta.get("slaughter_ymd", "")),
            ("성별", {1: "암퇘지", 2: "수퇘지", 3: "거세"}.get(meta.get("gender"), "?")),
            ("도체중", f"{meta.get('body_weight', 0)} kg"),
            ("등지방 두께", f"{meta.get('backfat_average', 0)} mm"),
            ("뭇갈래근 두께", f"{meta.get('multifidus_thk', 0)} mm"),
            ("체장", f"{meta.get('body_length', 0)} cm"),
            ("체폭", f"{meta.get('body_width', 0)} cm"),
        ]
        for label, val in rows:
            st.metric(label, val)

        ec = meta.get("error_code", {})
        errors = [k for k, v in ec.items() if v == 1]
        st.markdown("#### AI 검출 상태")
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            st.success("✅ 모든 항목 정상")
    else:
        st.info("JSON을 선택하면 측정값이 표시됩니다.")
