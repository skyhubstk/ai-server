"""
dataset_builder.py

Dart 백엔드로부터 수신한 재무제표 데이터를
슬라이딩 윈도우 방식으로 학습 데이터셋(DataFrame)으로 변환.

입력 DataFrame 필수 컬럼:
  company, year, revenue, operating_profit, net_income, debt, equity, cash

출력 DataFrame 컬럼:
  revenue_0~4, op_0~4, net_0~4, debt_0~4, equity_0~4, cash_0~4,
  next_operating_profit   ← 예측 타깃 (다음 연도 영업이익)
"""

import pandas as pd


WINDOW = 5          # 입력으로 사용할 연도 수
REQUIRED_COLS = [
    "company", "year",
    "revenue", "operating_profit", "net_income",
    "debt", "equity", "cash",
]


def make_window(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    회사별로 연속 window 개 연도 → 다음 연도 영업이익 예측 샘플 생성.

    Parameters
    ----------
    df     : 원시 재무 데이터 DataFrame (REQUIRED_COLS 필요)
    window : 슬라이딩 윈도우 크기 (기본 5)

    Returns
    -------
    pd.DataFrame : 학습용 샘플 (피처 원시값 + 타깃)
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"입력 DataFrame에 필수 컬럼 누락: {missing}")

    rows = []

    for company in df["company"].unique():
        temp = df[df["company"] == company].sort_values("year").reset_index(drop=True)

        # window+1개 연도 미만이면 샘플 생성 불가
        if len(temp) < window + 1:
            continue

        for i in range(len(temp) - window):
            chunk  = temp.iloc[i : i + window]
            target = temp.iloc[i + window]

            row = {}
            for idx, rec in enumerate(chunk.itertuples(index=False)):
                row[f"revenue_{idx}"]  = rec.revenue
                row[f"op_{idx}"]       = rec.operating_profit
                row[f"net_{idx}"]      = rec.net_income
                row[f"debt_{idx}"]     = rec.debt
                row[f"equity_{idx}"]   = rec.equity
                row[f"cash_{idx}"]     = rec.cash

            row["next_operating_profit"] = target.operating_profit

            rows.append(row)

    if not rows:
        raise ValueError(
            "학습 샘플을 생성할 수 없습니다. "
            f"각 기업에 최소 {window + 1}개 연도 데이터가 필요합니다."
        )

    return pd.DataFrame(rows)
