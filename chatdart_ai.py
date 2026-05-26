"""
chatdart_ai.py — ChatDart AI 공개 인터페이스

AI 서버의 역할:
  1. 재무제표 데이터로 모델 학습
  2. 학습된 모델로 다음 연도 영업이익 예측값 반환

GPT 프롬프트 구성 및 요약 생성은 백엔드에서 담당합니다.

사용 예시 (백엔드 Python 코드):
    from chatdart_ai import predict_next_profit, train_from_records

    # ── 예측 ───────────────────────────────────────────
    prediction = predict_next_profit({
        "revenue_0": 1_100_000, ..., "revenue_4": 1_500_000,
        "op_0":  90_000,        ..., "op_4":     155_000,
        "net_0":  68_000,       ..., "net_4":    115_000,
        "debt_0": 290_000,      ..., "debt_4":   260_000,
        "equity_0": 550_000,    ..., "equity_4": 750_000,
        "cash_0":  55_000,      ..., "cash_4":   100_000,
    })
    # prediction = 178_500.0  (단위: 입력 데이터와 동일)

    # ── 모델 재학습 ─────────────────────────────────────
    result = train_from_records([
        {"company": "삼성전자", "year": 2018, "revenue": 2_000_000,
         "operating_profit": 300_000, "net_income": 220_000,
         "debt": 500_000, "equity": 1_200_000, "cash": 100_000},
        ...  # 기업별 최소 6개 연도 이상 필요
    ])
    # result = {"mae": ..., "mape": ..., "n_samples": ..., "best_iter": ...}
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

from feature_engineering import create_features
from predictor import predict, reload_model, ModelNotFoundError
from train import run_training

__all__ = [
    "predict_next_profit",
    "train_from_records",
    "ModelNotFoundError",
]


def predict_next_profit(data: dict) -> float:
    """
    5개 연도 원시 재무 데이터 → 다음 연도 영업이익 예측값 반환.

    ※ 단위 주의 ※
      반환값의 단위는 입력 데이터의 단위와 완전히 동일합니다.
      모델은 단위 변환을 일절 수행하지 않으며, 학습 시 사용한
      train_from_records()의 레코드 단위를 그대로 따릅니다.

      예시)
        - 입력을 백만원 단위로 학습/예측했다면 → 반환값도 백만원
        - 입력을 억원 단위로 학습/예측했다면  → 반환값도 억원

      따라서 백엔드에서 단위를 통일한 뒤 사용하는 것을 권장합니다.

    Parameters
    ----------
    data : dict
        필수 키:
          revenue_0~4  매출액 (5개 연도, _0=5년전 … _4=최근)
          op_0~4       영업이익
          net_0~4      순이익
          debt_0~4     부채총계
          equity_0~4   자본총계
          cash_0~4     현금및현금성자산
        모든 값은 동일 단위여야 합니다.

    Returns
    -------
    float
        다음 연도 예측 영업이익.
        단위는 입력 데이터(op_0~4)와 동일합니다.

    Raises
    ------
    ModelNotFoundError
        모델 파일이 없을 때 (train_from_records() 먼저 호출 필요)
    KeyError
        필수 키가 data에 없을 때
    """
    features = create_features(data)
    return predict(features)


def train_from_records(
    records: list,
    save_csv: bool = True,
    csv_path: str = "data/train.csv",
) -> dict:
    """
    재무제표 레코드 리스트로 모델을 재학습하고 저장.

    Parameters
    ----------
    records : list[dict]
        재무제표 레코드 목록. 각 dict 필수 키:
          company, year, revenue, operating_profit,
          net_income, debt, equity, cash
        기업별 최소 6개 연도 이상 필요.

    save_csv : bool
        True 이면 csv_path 에 누적 저장 후 학습 (기본값 True).
        False 이면 전달된 records 만으로 학습.

    csv_path : str
        누적 저장에 사용할 CSV 경로 (기본값 "data/train.csv").

    Returns
    -------
    dict
        {
          "mae":        float,  # CV 평균 MAE
          "mape":       float,  # CV 평균 MAPE (%)
          "n_samples":  int,    # 학습 샘플 수
          "best_iter":  int,    # 최적 트리 수
          "model_path": str,    # 저장된 모델 경로
        }
    """
    new_df = pd.DataFrame(records)

    if save_csv:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

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

    result = run_training(raw_df=raw_df)
    reload_model()
    return result
