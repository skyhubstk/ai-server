"""
predictor.py  ─ 모델 로드 및 예측 (비금융 / 금융 분리)

[타깃 정의]
  - 비금융: next_op_margin (영업이익률) → 역산: pred_margin * revenue_4 = 영업이익
  - 금융  : next_net_income (순이익 절대값, KRW 원)

[backend_to_ai_repo_review_20260529.md 반영]
  - feature_engineering_fin.py(37피처/window=5) 제거 완료
    → feature_engineering.py 단일 파일: 비금융(34피처) + 금융(9피처/window=2)
  - 모델 경로: model.pkl/model_financial.pkl → model_nonfin.pkl/model_fin.pkl
  - 비금융 반환값: 영업이익률(0~1), 금융 반환값: 순이익(KRW 절대값)

사용:
  from predictor import predict, predict_from_raw, reload_model, ModelNotFoundError
"""

import os
import logging
import pandas as pd
from joblib import load

from feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_FIN,
    create_features,
    create_features_fin,
)

log = logging.getLogger(__name__)

# ── 모델 경로 (train.py와 동일하게 맞춤) ──────────────────────
MODEL_PATH_NON_FIN = os.getenv("MODEL_PATH",     "models/model_nonfin.pkl")
MODEL_PATH_FIN     = os.getenv("MODEL_PATH_FIN", "models/model_fin.pkl")

_model_non_fin = None
_model_fin     = None


class ModelNotFoundError(FileNotFoundError):
    """모델 파일이 없을 때 발생하는 예외"""
    pass


# ── 지연 로딩 ──────────────────────────────────────────────────

def _load(path: str):
    if not os.path.exists(path):
        raise ModelNotFoundError(
            f"모델 파일({path})이 없습니다. "
            "먼저 python train.py 를 실행하세요."
        )
    log.info("모델 로드: %s", path)
    return load(path)


def _get_model_non_fin():
    global _model_non_fin
    if _model_non_fin is None:
        _model_non_fin = _load(MODEL_PATH_NON_FIN)
    return _model_non_fin


def _get_model_fin():
    global _model_fin
    if _model_fin is None:
        _model_fin = _load(MODEL_PATH_FIN)
    return _model_fin


# ── 저수준 예측 (피처 dict → 예측값) ──────────────────────────

def predict(features: dict, sector: str = "non_financial") -> float:
    """
    피처 dict → 다음 연도 예측값 반환.

    Parameters
    ----------
    features : create_features() 또는 create_features_fin() 반환값
    sector   : "non_financial" (기본) → 영업이익률(0~1 비율) 반환
               "financial"           → 순이익(KRW 원, 절대값) 반환

    Returns
    -------
    float
        비금융: 영업이익률 (예: 0.12 = 12%)
        금융  : 순이익 (원)
    """
    if sector == "financial":
        model   = _get_model_fin()
        columns = FEATURE_COLUMNS_FIN
    else:
        model   = _get_model_non_fin()
        columns = FEATURE_COLUMNS

    x    = pd.DataFrame([features], columns=columns)
    pred = float(model.predict(x)[0])
    log.info("[%s] 예측 결과: %.6f", sector, pred)
    return pred


# ── 고수준 예측 (원시 재무 데이터 → 결과 dict) ─────────────────

def predict_from_raw(raw: dict, sector: str = "non_financial") -> dict:
    """
    원시 재무 데이터 dict → 피처 생성 → 예측 결과 dict.

    Parameters
    ----------
    raw    : 비금융: {revenue_0~4, op_0~4, net_0~4, tl_0~4, equity_0~4, cash_0~4, ta_0~4}
             금융  : {ta_0~1, tl_0~1, equity_0~1, net_0~1}  (window=2)
    sector : "non_financial" | "financial"

    Returns
    -------
    dict
        비금융: {
          "predicted_op_margin":  float,   # 예측 영업이익률 (0~1)
          "predicted_op_profit":  float,   # 역산 영업이익 (원) = margin × revenue_4
          "base_revenue":         float,   # 기준 매출액 (revenue_4)
        }
        금융: {
          "predicted_net_income": float,   # 예측 순이익 (원)
        }
    """
    if sector == "financial":
        feats         = create_features_fin(raw)
        pred          = predict(feats, sector="financial")
        return {
            "predicted_net_income": pred,
        }
    else:
        feats         = create_features(raw)
        pred_margin   = predict(feats, sector="non_financial")
        base_revenue  = float(raw.get("revenue_4", 0.0) or 0.0)
        pred_op_profit = pred_margin * base_revenue
        return {
            "predicted_op_margin":  pred_margin,
            "predicted_op_profit":  pred_op_profit,
            "base_revenue":         base_revenue,
        }


# ── 모델 캐시 초기화 ────────────────────────────────────────────

def reload_model(sector: str = "non_financial") -> None:
    """학습 완료 후 해당 sector 모델 캐시 초기화 → 다음 호출 시 재로드."""
    global _model_non_fin, _model_fin
    if sector == "financial":
        _model_fin = None
        log.info("[금융] 모델 캐시 초기화")
    else:
        _model_non_fin = None
        log.info("[비금융] 모델 캐시 초기화")
