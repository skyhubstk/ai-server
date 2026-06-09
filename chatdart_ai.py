"""
chatdart_ai.py  ─ ChatDart AI 공개 인터페이스

AI 서버의 역할:
  1. DART 재무제표 수집 (dart_collector)
  2. 비금융 / 금융 모델 각각 학습
  3. 학습된 모델로 예측값 반환

GPT 요약 생성은 백엔드에서 담당합니다.

────────────────────────────────────────────────────────
백엔드 사용 예시:
  from chatdart_ai import predict_next_profit, train_from_records

  # 비금융 예측 → 영업이익률(비율) 반환 (× revenue_4 → 영업이익)
  margin = predict_next_profit(data, sector="non_financial")

  # 금융 예측 → 역산 당기순이익(KRW 원) 반환
  # 내부 동작: LightGBM → ROE 예측 → ROE × equity_1 = 순이익
  net_income = predict_next_profit(data, sector="financial")

  # 학습
  train_from_records(records, sector="non_financial")
  train_from_records(records, sector="financial")
────────────────────────────────────────────────────────

피처 구성:
  비금융: feature_engineering.py (32피처, window=5)
  금융  : feature_engineering.py (16피처, window=2)
    필수  ta_0~1, tl_0~1, equity_0~1, net_0~1
    은행  nii_0~1, llp_0~1
    보험  ins_liab_0~1
    업종  sector_detail (bank/insurance/securities)

단위 정책:
  DB / DART 원시 데이터는 모두 KRW(원) 단위.
  비금융 반환값: 영업이익률 (무차원 비율, 0~1)
  금융  반환값: 역산 당기순이익 (KRW 원, = predicted_roe × equity_1)
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

# ── A-세트(권위 정의) import ────────────────────────────────────
# feature_engineering.py 단일 파일: 비금융(32피처/window=5) + 금융(16피처/window=2)
from feature_engineering import create_features, create_features_fin
from predictor           import predict_from_raw, reload_model, ModelNotFoundError
from train               import run_training

__all__ = [
    "predict_next_profit",
    "train_from_records",
    "ModelNotFoundError",
]


def predict_next_profit(data: dict, sector: str = "non_financial") -> float:
    """
    원시 재무 데이터 dict → 다음 연도 예측값 반환.

    Parameters
    ----------
    data : dict
        비금융(sector="non_financial"):
          revenue_0~4  매출액 (5개 연도, _0=5년전 … _4=최근)
          op_0~4       영업이익
          net_0~4      당기순이익
          tl_0~4       부채총계 (total_liabilities)
          equity_0~4   자본총계
          cash_0~4     현금및현금성자산
          ta_0~4       총자산 (없으면 tl+equity 자동 보완)
          ※ 단위 KRW(원)

        금융(sector="financial"):          ← window=2
          ta_0~1         총자산 (_0=1년전, _1=최근)
          tl_0~1         부채총계
          equity_0~1     자본총계
          net_0~1        당기순이익
          nii_0~1        순이자이익 (은행; 없으면 0)
          llp_0~1        대손충당금 (은행; 없으면 0)
          ins_liab_0~1   보험계약부채 (보험사; 없으면 0)
          sector_detail  업종 ("bank"/"insurance"/"securities"; 없으면 "")
          ※ 단위 KRW(원)

    sector : str
        "non_financial" (기본) → 영업이익률(0~1) 반환
                                  역산: 반환값 × data["revenue_4"] = 영업이익(원)
        "financial"             → 역산 당기순이익(KRW 원) 반환
                                  내부: LightGBM → ROE 예측 → ROE × equity_1

    Returns
    -------
    float
        비금융: 영업이익률 (예: 0.11 = 11%)
        금융  : 역산 당기순이익 (원) = predicted_roe × equity_1

    Raises
    ------
    ModelNotFoundError : 해당 sector 모델 파일 없음
    ValueError         : sector 값 오류
    """
    if sector not in ("non_financial", "financial"):
        raise ValueError(
            f"sector 는 'non_financial' 또는 'financial' 이어야 합니다. 입력값: {sector!r}"
        )

    result = predict_from_raw(data, sector=sector)

    if sector == "financial":
        return result["predicted_net_income"]
    else:
        # 백엔드가 역산: pred_margin × revenue_4 = 영업이익(원)
        return result["predicted_op_margin"]


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
        비금융: company, year, revenue, operating_profit,
                net_income, total_liabilities (또는 debt), equity, cash
                기업별 최소 6개 연도 이상

        금융  : company, year, is_financial=1,
                total_assets, total_liabilities, equity, net_income,
                net_interest_income (은행; 없으면 0),
                loan_loss_provision  (은행; 없으면 0),
                insurance_liability  (보험; 없으면 0),
                sector_detail        ("bank"/"insurance"/"securities"),
                report_type          ("A": 연간 데이터만 학습에 사용)
                기업별 최소 3개 연도 이상 (window=2 + 타깃 1년)

    sector : str
        "non_financial" → models/model_nonfin.pkl
        "financial"     → models/model_fin.pkl

    save_csv : bool
        True 이면 data/train.csv 에 누적 저장 후 학습 (기본값 True).

    Returns
    -------
    dict : {"mae", "smape", "n_samples", "model_path", ...}
    """
    if sector not in ("non_financial", "financial"):
        raise ValueError(
            f"sector 는 'non_financial' 또는 'financial' 이어야 합니다. 입력값: {sector!r}"
        )

    new_df = pd.DataFrame(records)

    # ── 컬럼 정규화 ──────────────────────────────────────────
    # 구버전 "debt" 키 → "total_liabilities" 변환
    if "debt" in new_df.columns and "total_liabilities" not in new_df.columns:
        new_df = new_df.rename(columns={"debt": "total_liabilities"})

    # train.py는 is_financial 컬럼으로 분기
    new_df["is_financial"] = 1 if sector == "financial" else 0

    # 금융 레코드에 비금융 컬럼 없어도 호환되도록 NaN 채움
    if sector == "financial":
        for col in ("revenue", "operating_profit"):
            if col not in new_df.columns:
                new_df[col] = None

    # ── CSV 누적 저장 ─────────────────────────────────────────
    csv_path = "data/train.csv"
    if save_csv:
        os.makedirs("data", exist_ok=True)
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            # report_type 포함 dedup: 연간(A)·분기(Q3) 데이터가 같은 company+year라도
            # 다른 행이므로 report_type까지 키에 포함 (없는 구버전 데이터는 그냥 통과)
            dedup_cols = ["company", "year"]
            if "report_type" in pd.concat([new_df, existing], ignore_index=True).columns:
                dedup_cols.append("report_type")
            raw_df = (
                pd.concat([new_df, existing], ignore_index=True)
                .drop_duplicates(subset=dedup_cols)
                .sort_values(["company", "year"])
                .reset_index(drop=True)
            )
        else:
            raw_df = new_df
        raw_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        raw_df = new_df

    # ── 학습 (train.py 통합 파이프라인) ──────────────────────
    result = run_training(raw_df=raw_df)
    reload_model(sector=sector)

    # 해당 섹터 결과만 반환
    sector_key = "fin" if sector == "financial" else "nonfin"
    return result.get(sector_key, result)
