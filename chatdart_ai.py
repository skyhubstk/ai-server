"""
chatdart_ai.py  ─ ChatDart AI 공개 인터페이스

AI 서버의 역할:
  1. DART 재무제표 수집 (dart_collector)
  2. 비금융업 / 금융업 모델 각각 학습
  3. 학습된 모델로 예측값 반환

GPT 요약 생성은 백엔드에서 담당합니다.

────────────────────────────────────────────────────────
백엔드 사용 예시:
  from chatdart_ai import predict_next_profit, train_from_records

  # 비금융업 예측 (영업이익)
  pred = predict_next_profit(data, sector="non_financial")

  # 금융업 예측 (당기순이익)
  pred = predict_next_profit(data, sector="financial")

  # 비금융업 학습
  train_from_records(records, sector="non_financial")

  # 금융업 학습
  train_from_records(records, sector="financial")
────────────────────────────────────────────────────────

단위 정책:
  DB / DART 원시 데이터는 모두 KRW(원) 단위.
  모델은 단위 변환을 수행하지 않으므로
  반환값 단위 = 입력 데이터 단위 = KRW(원).
"""

import logging
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from feature_engineering     import create_features
from feature_engineering_fin import create_features_fin
from predictor               import predict, reload_model, ModelNotFoundError
from train                   import run_training
from train_fin               import run_training_fin

__all__ = [
    "predict_next_profit",
    "train_from_records",
    "ModelNotFoundError",
]


def predict_next_profit(data: dict, sector: str = "non_financial") -> float:
    """
    5개 연도 원시 재무 데이터 → 다음 연도 예측값 반환.

    Parameters
    ----------
    data : dict
        비금융업(sector="non_financial"):
          revenue_0~4, op_0~4, net_0~4,
          debt_0~4 (=총부채), equity_0~4, cash_0~4
        금융업(sector="financial"):
          assets_0~4 (=총자산), liab_0~4 (=총부채),
          equity_0~4, ni_0~4 (=당기순이익), cash_0~4
        ※ _0=5년전 … _4=최근, 단위 KRW(원)

    sector : str
        "non_financial" (기본) → 다음 연도 영업이익 예측
        "financial"             → 다음 연도 당기순이익 예측

    Returns
    -------
    float : 예측값 (단위 = 입력과 동일, KRW)

    Raises
    ------
    ModelNotFoundError : 해당 sector 모델이 없을 때
    KeyError           : 필수 키가 누락되었을 때
    ValueError         : sector 값이 잘못되었을 때
    """
    if sector == "financial":
        features = create_features_fin(data)
    elif sector == "non_financial":
        features = create_features(data)
    else:
        raise ValueError(f"sector 는 'non_financial' 또는 'financial' 이어야 합니다. 입력값: {sector!r}")

    return predict(features, sector=sector)


def train_from_records(
    records:  list,
    sector:   str  = "non_financial",
    save_csv: bool = True,
) -> dict:
    """
    재무제표 레코드 리스트로 모델을 재학습하고 저장.

    Parameters
    ----------
    records : list[dict]
        비금융업: company, year, revenue, operating_profit,
                  net_income, debt, equity, cash
        금융업:   company, year, total_assets, total_liabilities,
                  equity, net_income, cash
        기업별 최소 6개 연도 이상 필요.

    sector : str
        "non_financial" → data/train.csv + models/model.pkl
        "financial"     → data/train_financial.csv + models/model_financial.pkl

    save_csv : bool
        True 이면 기존 CSV 에 누적 저장 후 학습 (기본값 True).

    Returns
    -------
    dict : {"mae", "mape", "n_samples", "best_iter", "model_path"}
    """
    if sector not in ("non_financial", "financial"):
        raise ValueError(f"sector 는 'non_financial' 또는 'financial' 이어야 합니다. 입력값: {sector!r}")

    csv_path = (
        "data/train_financial.csv" if sector == "financial"
        else "data/train.csv"
    )
    new_df = pd.DataFrame(records)

    if save_csv:
        os.makedirs("data", exist_ok=True)
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            raw_df = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["company", "year"])
                .sort_values(["company", "year"])
                .reset_index(drop=True)
            )
        else:
            raw_df = new_df
        raw_df.to_csv(csv_path, index=False)
    else:
        raw_df = new_df

    if sector == "financial":
        result = run_training_fin(raw_df=raw_df)
    else:
        result = run_training(raw_df=raw_df)

    reload_model(sector=sector)
    return result
