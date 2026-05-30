# [AI팀 → 백엔드] 잔여 확인 2건 회신 (2026-05-30)

> 수신: 백엔드 · 작성: AI팀(윤석진) · 작성일 2026-05-30  
> 이전 문서: `backend_to_ai_reply_20260530.md`(잔여 확인 요청 2건)

---

## 1. 3-1 회신 — 비금융 피처 개수 주석 오기 수정 완료

`feature_engineering.py` 헤더 주석 및 `chatdart_ai.py` 주석의 **34개 → 32개** 정정 완료했습니다.

| 파일 | 수정 전 | 수정 후 |
|------|---------|---------|
| `feature_engineering.py` L6 | `FEATURE_COLUMNS (34개)` | `FEATURE_COLUMNS (32개)` |
| `chatdart_ai.py` 주석 | `비금융(34피처)` | `비금융(32피처)` |

기능 코드·리스트 순서 변경 없음. 주석만 실제 개수와 일치시켰습니다.

---

## 2. 3-2 회신 — 백엔드 확인 요청 3건

### 2-1. 모델 타입 비대칭 (LightGBM vs Ridge) — 의도된 것 맞음

| 항목 | 비금융 | 금융 |
|------|--------|------|
| 모델 | `LGBMRegressor` | `Ridge (alpha=10)` |
| 파일 크기 | ~531 KB | ~1 KB |
| 학습 샘플 | 454개 | **32개** |

**금융 모델이 Ridge인 이유**: DART `fnlttSinglAcnt` API가 금융 지주사 데이터를 2021년부터만 제공합니다.  
결과적으로 16개 금융사 × 2년 슬라이딩 윈도우 = **32 샘플**만 확보되었습니다.  
32개 샘플에 LightGBM을 적용하면 과적합 가능성이 높아 단순 선형 모델(Ridge)을 **잠정 baseline**으로 사용했습니다.  
(`train.py`에 `WARNING: 샘플 32개 < 50, Ridge Regression 사용` 로그가 명시되어 있습니다.)

→ **백엔드 기대치 기준**: 금융 모델은 현재 baseline 수준입니다. 대형사(KB·신한·하나 등) 예측은 방향성은 맞으나 ±1~2배 오차가 있을 수 있습니다. 이 점을 감안하여 시연 스크립트 및 UI 문구를 조정해 주시면 감사하겠습니다.

---

### 2-2. 재학습(표본 보강) 계획 — 현재 없음

현 시점에서 **추가 데이터 수집 및 모델 재학습 계획은 없습니다**.  
시연은 현재 출하된 모델(`model_fin.pkl`, `model_nonfin.pkl`)로 진행해 주시면 됩니다.

---

### 2-3. 출력 단위·입력 계약 유지 확인 — 확정

아래 계약은 재학습이 발생하더라도 **변경하지 않습니다**.

| 항목 | 비금융 | 금융 |
|------|--------|------|
| 입력 window | 5년 (`_0~4`) | **2년** (`_0~1`) |
| 입력 키 | `revenue, op, net, tl, equity, cash, ta` | `ta, tl, equity, net` |
| 피처 수 | 32개 | 9개 |
| 출력 | 영업이익률 (×revenue_4 역산) | 당기순이익 (KRW 원, 절대값) |
| 모델 파일명 | `models/model_nonfin.pkl` | `models/model_fin.pkl` |

파일명 포함 위 계약은 고정입니다. 변경이 생기면 사전에 공지하겠습니다.

---

## 3. 현재 레포 상태 (참고)

```
ai-server/
├── feature_engineering.py   비금융(32피처) + 금융(9피처/window=2)  ← 단일 권위 정의
├── dataset_builder.py        슬라이딩 윈도우 생성
├── train.py                  비금융 LightGBM + 금융 Ridge 통합 학습
├── predictor.py              모델 로드 + 예측 (A-세트 단독)
├── chatdart_ai.py            백엔드 공개 인터페이스
├── dart_collector.py         DART API 수집
├── models/
│   ├── model_nonfin.pkl      비금융 LightGBM (32피처, SMAPE 60.5%)
│   └── model_fin.pkl         금융 Ridge     ( 9피처, SMAPE 52.6%)
└── data/train.csv            872행 (비금융 808 + 금융 64)
```

이전 B-세트(`feature_engineering_fin.py`, `train_fin.py`, `dataset_builder_fin.py`)는 삭제 완료됐습니다.
