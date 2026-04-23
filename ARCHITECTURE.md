# 한국어 축산 AI 판정 코파일럿 — 포트폴리오 아키텍처 초안

> 작성일: 2026-04-23  
> 기반 코드: `thema_pa` (YOLOv11 기반 돼지 도체 분석 시스템)

---

## 1. 전체 그림

```
┌─────────────────────────────────────────────────────────────────┐
│                        현장 카메라 / RFID                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 이미지 + 도체번호
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         thema_pa (기존)                          │
│  YOLOv11 seg → PostProcess → GradeCalc → DB / REST / MQTT       │
│                                                                  │
│  출력 JSON 예시:                                                  │
│  {                                                               │
│    "carcass_no": 3010,                                           │
│    "backfat_average": 22,        # mm                            │
│    "multifidus_thk": 48,         # mm                            │
│    "body_height": 920, "body_width": 340,                        │
│    "gender": 3,  "grade": "1+",                                  │
│    "error_code": {                                               │
│      "AI_BackFat_error": 0,                                      │
│      "AI_Backbone_error": 0,                                     │
│      "AI_HalfBone_error": 0,                                     │
│      "AI_multifidus_error": 0,                                   │
│      "AI_Outline_error": 0,                                      │
│      "pig_RightEntry": 0                                         │
│    },                                                            │
│    "backbone_slope": {"has_large_slope": false, ...},            │
│    "result_image_path": "D:/Image/20260423/AIS/jpg/xxx_ai.jpg"   │
│  }                                                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ structured JSON + result image
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│  PROJECT 1      │  │  PROJECT 3      │  │  PROJECT 2           │
│  QA Copilot     │  │  Multimodal API │  │  VLM Benchmark       │
│  (Explainability│  │  (Prod-ready)   │  │  (Research)          │
└────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘
         │                    │                       │
         └────────────────────┴───────────────────────┘
                               │
                        ▼ 공통 출력
              한국어 판정 리포트 / 대시보드
```

---

## 2. Project 1 — Explainable Livestock QA Copilot

### 목표
thema_pa의 구조화 JSON과 결과 이미지를 입력받아,  
**현장 작업자가 이해할 수 있는 한국어 판정 요약과 원인 분석**을 자동 생성한다.

### 핵심 입출력

```
입력: thema_pa JSON + result_image (optional)
출력: {
  "3문장_요약": "이 개체(도체번호 3010)는 거세 수컷이며 등지방 22mm, 뭇갈래근 48mm로 측정되어 1+등급으로 판정되었습니다. ...",
  "비정상_근거": null,   # error가 있을 때만 채움
  "주의사항": ["체폭/체장 측정 정상", "척추 기울기 정상"],
  "권고": "등지방 두께가 기준 범위 내에 있으나 상단에 근접합니다. 다음 개체 확인 권장."
}
```

### 컴포넌트

```
vlm/
├── schema/
│   └── thema_pa_output.py      # thema_pa JSON → Pydantic 모델
├── prompt/
│   ├── system_prompt.txt       # 도메인 지식 + 출력 형식 정의
│   ├── normal_case.txt         # 정상 판정 템플릿
│   ├── error_case.txt          # error_code 비정상 시 템플릿
│   └── failure_analysis.txt    # 등지방 추정 실패 원인 분석
├── report/
│   └── generator.py            # LLM 호출 → 리포트 생성
└── demo/
    └── app.py                  # Streamlit 3단 데모
                                # [원본 이미지 | 시각화 | 한국어 리포트]
```

### 프롬프트 설계 원칙

| error_code 상태 | 분기 프롬프트 | 생성 내용 |
|---|---|---|
| 모두 0 | normal_case | 정상 판정 요약 + 등급 근거 |
| AI_BackFat_error=1 | failure_analysis | 등지방 추정 실패 원인 + 재촬영 권고 |
| pig_RightEntry=1 | failure_analysis | 비정상 진입 감지 + 자세 보정 권고 |
| 그 외 error_code 비정상 | error_case | 해당 부위 측정 불가 원인 설명 |

### 4주차 마일스톤

- 1주차: thema_pa JSON 스키마 Pydantic으로 확정, 샘플 10건 수집
- 2주차: 프롬프트 3종 작성 + Claude API 연동, 리포트 생성기 구현
- 3주차: Streamlit 데모 (3단 레이아웃)
- 4주차: 실패 케이스 10건 이상 수동 검증 + 사례 문서화

---

## 3. Project 2 — Korean Livestock VLM Benchmark (LivestockVLM-Ko-Bench)

### 목표
thema_pa 출력과 현장 이미지를 이용해  
**도메인 특화 한국어 VLM 평가셋**을 정의하고, 재현 가능한 벤치마크로 만든다.

### 평가 태스크 3종

```
Task A — 이미지 기반 한국어 설명 (Image Captioning)
  입력: result_image (시각화된 등지방/윤곽선 overlay)
  정답: thema_pa가 계산한 수치 기반 레퍼런스 설명
  지표: ROUGE-L, BERTScore (한국어)

Task B — 진단 Reasoning (Chain-of-Thought QA)
  입력: JSON 수치 + 이미지
  질문: "이 개체가 2등급으로 판정된 이유를 단계적으로 설명하세요"
  평가: 수동 루브릭 (5점 척도 × 3기준: 정확성·완결성·현장 가독성)

Task C — 실패 케이스 해설 (Error Analysis)
  입력: error_code 비정상 + 실패 이미지
  질문: "등지방 추정에 실패한 원인과 현장 조치 방법을 설명하세요"
  평가: 원인 식별 정확도 (thema_pa 에러 코드 기준)
```

### 데이터셋 구조

```
benchmark/
├── samples/
│   ├── normal/          # error_code 모두 0                  (30건)
│   ├── backfat_fail/    # AI_BackFat_error=1               (15건)
│   ├── outline_fail/    # AI_Outline_error=1               (10건)
│   └── entry_fail/      # pig_RightEntry=1                 (10건)
├── annotations/
│   └── {sample_id}.json # 수동 검증 레퍼런스 답변
├── eval/
│   └── scorer.py        # ROUGE / BERTScore / 수동 루브릭 집계
└── report/
    └── auto_report.py   # 모델별 점수표 + 실패 케이스 시각화
```

### 벤치마크 설계 포인트 (포트폴리오 메시지)

> "도메인 특화 한국어 VLM 평가셋을 정의하고,  
> 기존 산업 CV 결과와 결합해 모델 성능을 재현 가능하게 측정했다."

- thema_pa 수치가 **ground truth** 역할 → 레이블 비용 Zero
- 실패 케이스 분류가 이미 error_code로 구조화되어 있어 태스크 C 자동 구성 가능
- GPT-4o / Claude / Gemini 비교 결과를 표로 제시

---

## 4. Project 3 — Factory-Ready Multimodal Inspection API

### 목표
thema_pa의 기존 REST/MQTT/DB 흐름 위에 VLM 설명 레이어를 추가하여  
**이미지 업로드 → CV 추론 → VLM 설명 → REST 응답 + 대시보드 게시**를  
end-to-end로 보여주는 운영용 데모를 구성한다.

### 시스템 흐름

```
[카메라 클라이언트]
        │  POST /analyze  (multipart: image + rfid)
        ▼
[FastAPI Gateway]  ← comm/rest_api.py 확장
        │
        ├─→ [thema_pa.analysis_pa()]  ← 기존 CV 파이프라인
        │         │ structured JSON
        │         ▼
        ├─→ [VLM Report Generator]   ← Project 1 report/generator.py
        │         │ 한국어 리포트
        │         ▼
        ├─→ [DB INSERT]              ← storage/db.py (기존)
        │
        ├─→ [MQTT publish]           ← infra/monitor_publisher.py (기존)
        │     topic: thematec/{site}/report  (신규 토픽)
        │
        └─→ HTTP Response:
              {
                "carcass_no": 3010,
                "grade": "1+",
                "backfat_average": 22,
                "report_ko": "...",    ← 신규
                "result_image_url": "..."
              }
```

### 신규 추가 파일

```
vlm/
└── api/
    ├── main.py          # FastAPI 엔트리포인트
    ├── router.py        # POST /analyze, GET /report/{carcass_no}
    └── middleware.py    # 요청 로깅 + 에러 핸들링
```

### 데모 시나리오

1. `virtual_cam_client.py` (기존) → 이미지 전송
2. FastAPI가 thema_pa + VLM 순서로 처리
3. Streamlit 대시보드에서 실시간 결과 확인
4. MQTT 구독 클라이언트가 한국어 리포트 수신

---

## 5. 세 프로젝트 연결 구조 (포트폴리오 뷰)

```
┌──────────────────────────────────────────────────────────────┐
│  PORTFOLIO STORY                                             │
│                                                              │
│  thema_pa ──────────────────────────────────────────────┐   │
│  (기존 산업 AI)                                          │   │
│                                                          │   │
│  Project 1: QA Copilot                                   │   │
│  → "기존 모델을 바꾸지 않고 한국어 설명 레이어 추가"    │   │
│  → 사업적 가치: 현장 작업자 교육비 절감, QA 자동화      │   │
│                                                          │   │
│  Project 2: VLM Benchmark                                │   │
│  → "도메인 특화 평가셋 설계 + 재현 가능한 측정"         │   │
│  → 연구적 가치: 한국어 축산 VLM 최초 벤치마크           │   │
│                                                          │   │
│  Project 3: Multimodal API                               │   │
│  → "CV + VLM + REST + MQTT end-to-end 통합 데모"        │   │
│  → 운영 가치: 실제 공장 시스템과 동일한 통신 구조       │◄──┘
└──────────────────────────────────────────────────────────────┘

공유 컴포넌트:
  vlm/schema/thema_pa_output.py  ← 세 프로젝트 모두 import
  vlm/report/generator.py        ← Project 1, 3 공유
  benchmark/samples/             ← Project 2, (Project 1 검증용)
```

---

## 6. 기술 스택 결정

| 레이어 | 선택 | 이유 |
|---|---|---|
| LLM/VLM | Claude claude-sonnet-4-6 | 한국어 품질, API 안정성 |
| 데모 UI | Streamlit | 빠른 프로토타이핑, 3단 레이아웃 |
| API 서버 | FastAPI | thema_pa와 동일 파이썬 생태계 |
| 스키마 | Pydantic v2 | thema_pa JSON → 타입 안전 변환 |
| 평가 | ROUGE-L + BERTScore (ko) | 한국어 NLG 표준 지표 |
| 메시징 | MQTT (HiveMQ, 기존) | infra/monitor_publisher.py 재활용 |

---

## 7. 주차별 실행 계획

### 1주차 — 기반 구축
- [ ] `vlm/schema/thema_pa_output.py` Pydantic 모델 작성
- [ ] thema_pa 샘플 JSON 10건 + 결과 이미지 수집
- [ ] 정상/실패 케이스 분류 기준 문서화

### 2주차 — 리포트 생성기
- [ ] `vlm/prompt/` 템플릿 3종 작성
- [ ] `vlm/report/generator.py` Claude API 연동
- [ ] 단위 테스트: 샘플 10건 리포트 품질 확인

### 3주차 — 데모 완성
- [ ] Streamlit 3단 데모 (`vlm/demo/app.py`)
- [ ] FastAPI 엔드포인트 (`vlm/api/main.py`)
- [ ] `virtual_cam_client.py` → API → 대시보드 end-to-end 테스트

### 4주차 — 벤치마크 + 문서화
- [ ] 샘플 50~100건 수집 및 수동 레퍼런스 작성
- [ ] `benchmark/eval/scorer.py` 구현
- [ ] 실패 케이스 10건 이상 사례 문서화
- [ ] README + 포트폴리오 설명 페이지 작성
