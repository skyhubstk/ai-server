"""
dataset_builder.py

재무제표 데이터를 슬라이딩 윈도우 방식으로 학습 샘플로 변환.

[개선]
  - 비금융 타깃: next_op_margin (영업이익률, 비율)
  - 금융 타깃:   next_net_income → next_roe (순이익/자본, 비율)
    · KB금융 5조 vs 교보증권 500억 scale 문제 해소
    · 비율 타깃으로 정규화 → LightGBM 학습 안정화
  - 금융 window=2 유지 (50샘플 유지)
  - 금융 피처 추가: sector_detail(업종), ins_liab(보험계약부채)

입력 DataFrame 필수 컬럼:
  company, year, is_financial,
  revenue, operating_profit, net_income,
  total_assets, total_liabilities, equity, cash

출력 (비금융):
  revenue_0~4, op_0~4, net_0~4, tl_0~4, equity_0~4, cash_0~4, ta_0~4
  next_op_margin         ← 예측 타깃 (영업이익률)
  next_operating_profit  ← 참고용

출력 (금융):
  ta_0~1, tl_0~1, equity_0~1, net_0~1, nii_0~1, llp_0~1, ins_liab_0~1
  sector_detail          ← 업종 (bank/insurance/securities)
  next_roe               ← 예측 타깃 (순이익/자본, 비율)
  next_net_income        ← 참고용 (역산: predicted_roe × equity)
"""

import numpy as np
import pandas as pd

WINDOW_NONFIN = 5   # 비금융: 5년 윈도우
WINDOW_FIN    = 2   # 금융  : 2년 윈도우 (4년 데이터 → 기업당 2샘플 확보)

# [ChatGPT6.pdf 3순위] 영업이익률 극단값 클리핑
# op_margin < -0.5 or > 0.5 구간은 분포 불안정 → SMAPE 폭주 방지
OP_MARGIN_CLIP_MIN = -0.5
OP_MARGIN_CLIP_MAX =  0.5


def make_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    비금융/금융을 분기하여 슬라이딩 윈도우 샘플 생성.
    is_financial 컬럼이 없으면 전체를 비금융으로 처리 (하위 호환).
    """
    if "is_financial" not in df.columns:
        df = df.copy()
        df["is_financial"] = 0

    nonfin_df = df[df["is_financial"] == 0].copy()
    fin_df    = df[df["is_financial"] == 1].copy()

    nonfin_samples = _make_nonfin_window(nonfin_df) if not nonfin_df.empty else pd.DataFrame()
    fin_samples    = _make_fin_window(fin_df)       if not fin_df.empty    else pd.DataFrame()

    return nonfin_samples, fin_samples


# ─────────────────────────────────────────────────────────────
# 비금융 윈도우
# target: next_op_margin = next_operating_profit / next_revenue
# ─────────────────────────────────────────────────────────────

def _make_nonfin_window(df: pd.DataFrame, window: int = WINDOW_NONFIN) -> pd.DataFrame:
    rows = []

    for company in df["company"].unique():
        temp = (
            df[df["company"] == company]
            .sort_values("year")
            .reset_index(drop=True)
        )
        if len(temp) < window + 1:
            continue

        for i in range(len(temp) - window):
            chunk  = temp.iloc[i: i + window]
            target = temp.iloc[i + window]

            next_op  = target.get("operating_profit")
            next_rev = target.get("revenue")

            # 타깃이 없으면 스킵
            if pd.isna(next_op) or pd.isna(next_rev):
                continue
            next_op  = float(next_op)
            next_rev = float(next_rev)

            # 영업이익률 (매출 0이면 스킵)
            if abs(next_rev) < 1.0:
                continue
            raw_margin     = next_op / next_rev
            # [ChatGPT6.pdf 3순위] 극단값 클리핑 (-0.5 ~ 0.5)
            next_op_margin = float(np.clip(raw_margin, OP_MARGIN_CLIP_MIN, OP_MARGIN_CLIP_MAX))

            row = {}
            for idx, rec in enumerate(chunk.itertuples(index=False)):
                row[f"revenue_{idx}"]  = _v(rec, "revenue")
                row[f"op_{idx}"]       = _v(rec, "operating_profit")
                row[f"net_{idx}"]      = _v(rec, "net_income")
                row[f"tl_{idx}"]       = _v(rec, "total_liabilities")
                row[f"equity_{idx}"]   = _v(rec, "equity")
                row[f"cash_{idx}"]     = _v(rec, "cash")
                row[f"ta_{idx}"]       = _v(rec, "total_assets")

            row["next_op_margin"]        = next_op_margin   # ← 예측 타깃
            row["next_operating_profit"] = next_op           # ← 역산 참고용
            row["next_revenue"]          = next_rev          # ← 역산 참고용
            rows.append(row)

    if not rows:
        raise ValueError(
            f"비금융 학습 샘플 0개. 각 기업에 최소 {window + 1}개년 데이터 필요."
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 금융 윈도우 (window=2, target=next_roe)
#
# 타깃 변경: next_net_income(절대값) → next_roe(순이익/자본, 비율)
#   · KB금융 5조 vs 교보증권 500억 scale 차이 해소
#   · 비금융 next_op_margin 과 대칭 구조
#   · next_net_income 은 역산 참고용으로 보존
# ROE 클리핑: 금융사 정상 ROE 범위 -0.3 ~ 0.3 (극단 손실 제외)
# ─────────────────────────────────────────────────────────────

ROE_CLIP_MIN = -0.30
ROE_CLIP_MAX =  0.30


def _make_fin_window(df: pd.DataFrame, window: int = WINDOW_FIN) -> pd.DataFrame:
    # 연간 데이터(사업보고서)만 사용 — 분기와 타깃 혼용 방지
    # report_type 컬럼이 없는 구버전 데이터도 그대로 통과
    if "report_type" in df.columns:
        df = df[df["report_type"] == "A"].copy()

    rows = []

    for company in df["company"].unique():
        temp = (
            df[df["company"] == company]
            .sort_values("year")
            .reset_index(drop=True)
        )
        if len(temp) < window + 1:
            continue

        # 업종 (sector_detail): 기업당 고정값
        sector_detail = ""
        if "sector_detail" in temp.columns:
            sector_detail = str(temp["sector_detail"].iloc[0])

        for i in range(len(temp) - window):
            chunk  = temp.iloc[i: i + window]
            target = temp.iloc[i + window]

            next_ni = target.get("net_income")
            if pd.isna(next_ni):
                continue
            next_ni = float(next_ni)

            # 타깃 연도의 자본 (ROE 분모)
            next_eq = _v(target, "equity")
            if next_eq <= 0:
                # 자본이 0 이하면 ROE 계산 불가 → 스킵
                continue
            raw_roe  = next_ni / next_eq
            next_roe = float(np.clip(raw_roe, ROE_CLIP_MIN, ROE_CLIP_MAX))

            row = {}
            for idx, rec in enumerate(chunk.itertuples(index=False)):
                row[f"ta_{idx}"]       = _v(rec, "total_assets")
                row[f"tl_{idx}"]       = _v(rec, "total_liabilities")
                row[f"equity_{idx}"]   = _v(rec, "equity")
                row[f"net_{idx}"]      = _v(rec, "net_income")
                # 은행 전용: 순이자이익, 대손충당금
                row[f"nii_{idx}"]      = _v(rec, "net_interest_income")
                row[f"llp_{idx}"]      = _v(rec, "loan_loss_provision")
                # 보험 전용: 보험계약부채
                row[f"ins_liab_{idx}"] = _v(rec, "insurance_liability")

            # 업종 원-핫 (기업 고정)
            row["sector_detail"] = sector_detail

            # 타깃
            row["next_roe"]        = next_roe   # ← 예측 타깃 (비율)
            row["next_net_income"] = next_ni    # ← 역산 참고용 (원)
            rows.append(row)

    if not rows:
        raise ValueError(
            f"금융 학습 샘플 0개. 각 기업에 최소 {window + 1}개년 연간 데이터 필요. "
            f"현재 연간(A) 데이터 최대 "
            f"{df.groupby('company').size().max() if not df.empty else 0}년."
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────

def _v(rec, col: str) -> float:
    """namedtuple에서 컬럼값 추출, 없으면 0.0"""
    try:
        val = getattr(rec, col)
    except AttributeError:
        return 0.0
    return float(val) if pd.notna(val) else 0.0
