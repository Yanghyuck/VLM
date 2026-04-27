# FastAPI 엔드포인트 검증 결과

**실행 시각**: 2026-04-27 09:52:24
**서버 URL**: http://127.0.0.1:8000

## GET /v1/health

- 상태 코드: 200
- 응답: `{"status": "ready", "model_used": "lora", "adapter_exists": true}`

---

## POST /v1/report

**통과율**: 4/4

### ✅ vlm/schema/samples/normal_case.json

- HTTP: `200`
- 추론 시간: 22.91초
- 응답:

```json
{
  "summary": "도체번호 3010은 20260423 도축으로 1+ 등급 판정되었습니다. 등지방 22.4mm, 뭇갈래근 48.2mm, 도체중 87.3kg으로 측정되었습니다. 모든 AI 검출이 정상 완료되어 2026년 04월 23일 1+ 등급으로 판정되었습니다.",
  "grade_reason": null,
  "warnings": [],
  "recommendation": "1+ 등급으로 판정되었습니다. 정상 출하 처리하세요.",
  "model_used": "lora (22.91s)"
}
```

---

### ✅ vlm/schema/samples/backfat_error_case.json

- HTTP: `200`
- 추론 시간: 33.83초
- 응답:

```json
{
  "summary": "도체번호 3025은 등지방 검출 실패(척추, 등지방, 뭇갈래근)으로 인해 등외 등급 판정되었습니다. 모든 AI 검출이 정상 완료되지 않아 등급 신뢰도가 저하되었습니다. 즉시 재촬영 또는 수동 측정으로 판정 재검토가 필요합니다.",
  "grade_reason": "등지방 검출 실패(9.0mm) 및 척추/�갈래근 검출 오류로 AI 기반 판정 신뢰도가 0으로 하락하여 등외 판정이 불가피했습니다.",
  "warnings": [
    "재촬영 필요: Yes",
    "수동 측정 대상: 등지방, 뭇갈래근, 척추",
    "등급 이의 신청 가능: Yes"
  ],
  "recommendation": "즉시 재촬영 또는 수동 측정으로 판정 재검토하여 등급을 정확히 확정하십시오.",
  "model_used": "lora (33.82s)"
}
```

---

### ✅ vlm/schema/samples/entry_error_case.json

- HTTP: `200`
- 추론 시간: 31.3초
- 응답:

```json
{
  "summary": "도체번호 3041은 비정상 진입으로 등지방 검출이 실패하여 등외 등급으로 판정되었습니다. 등지방 9.0mm와 도체중 79.0kg으로 기준 범위 외에 해당됩니다. 판정 신뢰도가 저하되어 재확인이 필요합니다.",
  "grade_reason": "비정상 진입으로 인해 등지방 검출이 실패하여 판정 신뢰도가 0으로 하락함. 등외 판정은 검출 오류로 인해 정확도가 보장되지 않음.",
  "warnings": [
    "비정상 진입 재발 방지 필요",
    "등외 판정으로 등급 이의 신청 가능",
    "재촬영으로 신뢰도 복구 필요"
  ],
  "recommendation": "즉시 재촬영을 통해 검출 오류 복구 후 등급 재판정을 실시해야 합니다.",
  "model_used": "lora (31.3s)"
}
```

---

### ✅ vlm/schema/samples/sample_3473.json

- HTTP: `200`
- 추론 시간: 20.25초
- 응답:

```json
{
  "summary": "도체번호 3473은 20260422 도축으로 1+ 등급 판정되었습니다. 등지방 20.0mm, 뭇갈래근 15.3mm, 도체중 88.0kg으로 측정되었습니다. 모든 AI 검출이 정상 완료되어 1+ 등급 기준에 완전히 부합합니다.",
  "grade_reason": null,
  "warnings": [],
  "recommendation": "1+ 등급으로 판정되었습니다. 정상 출하 처리 바랍니다.",
  "model_used": "lora (20.25s)"
}
```

---

