"""
dataset_builder_fin.py  ─ 금융업 전용 슬라이딩 윈도우 데이터셋 빌더

비금융업과 달리 금융업은 매출액·영업이익이 없고
총자산·총부채·자기자본·당기순이익·현금만 수집 가능합니다.

입력 DataFrame 필수 컬럼:
  company, year, total_assets, total_liabilities, equity, net_income, cash

출력 DataFrame 컬럼:
  assets_0~4, liab_0~4, equity_0~4, ni_0~4, cash_0~4,
  next_net_income   ← 예측 타깃 (다음 연도 당기순이익)
"""

import pandas as pd


WINDOW_FIN = 5
REQUIRED_COLS_FIN = [
    "company", "year",
    "total_assets", "total_liabilities", "equity",
    "net_income", "cash",
]


def make_window_fin(df: pd.DataFrame, window: int = WINDOW_FIN) -> pd.DataFrame:
    """
    금융업 기업별로 연속 window 개 연도 → 다음 연도 순이익 예측 샘플 생성.

    Parameters
    ----------
    df     : 원시 금융 재무 데이터 DataFrame (REQUIRED_COLS_FIN 필요)
    window : 슬라이딩 윈도우 크기 (기본 5)

    Returns
    -------
    pd.DataFrame
        컬럼: assets_0~4, liab_0~4, equity_0~4, ni_0~4, cash_0~4,
               next_net_income
    """
    missing = [c for c in REQUIRED_COLS_FIN if c not in df.columns]
    if missing:
        raise ValueError(f"입력 DataFrame에 필수 컬럼 누락: {missing}")

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
            chunk  = temp.iloc[i : i + window]
            target = temp.iloc[i + window]

            row: dict = {}
            for idx, rec in enumerate(chunk.itertuples(index=False)):
                row[f"assets_{idx}"] = rec.total_assets
                row[f"liab_{idx}"]   = rec.total_liabilities
                row[f"equity_{idx}"] = rec.equity
                row[f"ni_{idx}"]     = rec.net_income
                row[f"cash_{idx}"]   = rec.cash

            row["next_net_income"] = target.net_income
            rows.append(row)

    if not rows:
        raise ValueError(
            "금융업 학습 샘플을 생성할 수 없습니다. "
            f"각 기업에 최소 {window + 1}개 연도 데이터가 필요합니다."
        )

    return pd.DataFrame(rows)
