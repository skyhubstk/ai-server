"""
dataset_builder.py

재무제표 데이터를 슬라이딩 윈도우 방식으로 학습 샘플로 변환.

[ChatGPT5.pdf 반영]
  - 비금융 타깃: 영업이익(절대값) → 영업이익률 next_op_margin = op / revenue
    · 삼성전자~소형주 scale 차이로 MAPE 폭주 → 비율 타깃으로 정규화
  - 금융 window: 3 → 2 (4년 데이터로 샘플 2배 확보: 16 → 32)
  - 비금융 next_operating_profit 참고용으로 함께 저장 (역산용)

입력 DataFrame 필수 컬럼:
  company, year, is_financial,
  revenue, operating_profit, net_income,
  total_assets, total_liabilities, equity, cash

출력 (비금융):
  revenue_0~4, op_0~4, net_0~4, tl_0~4, equity_0~4, cash_0~4, ta_0~4
  next_op_margin         ← 예측 타깃 (영업이익률)
  next_operating_profit  ← 참고용 (실제 이익 역산)

출력 (금융):
  ta_0~1, tl_0~1, equity_0~1, net_0~1
  next_net_income
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
# 금융 윈도우 (window=2, target=next_net_income)
# ─────────────────────────────────────────────────────────────

def _make_fin_window(df: pd.DataFrame, window: int = WINDOW_FIN) -> pd.DataFrame:
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

            next_ni = target.get("net_income")
            if pd.isna(next_ni):
                continue

            row = {}
            for idx, rec in enumerate(chunk.itertuples(index=False)):
                row[f"ta_{idx}"]     = _v(rec, "total_assets")
                row[f"tl_{idx}"]     = _v(rec, "total_liabilities")
                row[f"equity_{idx}"] = _v(rec, "equity")
                row[f"net_{idx}"]    = _v(rec, "net_income")

            row["next_net_income"] = float(next_ni)
            rows.append(row)

    if not rows:
        raise ValueError(
            f"금융 학습 샘플 0개. 각 기업에 최소 {window + 1}개년 데이터 필요. "
            f"현재 최대 {df.groupby('company').size().max()}년."
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
