"""
train.py

[ChatGPT5.pdf 반영]
  1. 비금융 타깃: next_operating_profit → next_op_margin (영업이익률)
     · 삼성전자(50조)~소형주(100억) scale 차이 → 비율로 정규화 → MAPE 안정화
  2. 금융 모델: LightGBM → Ridge Regression (샘플 부족 시 단순 모델이 안전)
     · 샘플 < MIN_FIN_SAMPLES(50): 경고 출력 후 Ridge 사용
  3. 평가지표: MAPE 단독 → MAE + SMAPE 병행

비금융: models/model_nonfin.pkl
금융  : models/model_fin.pkl
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from dataset_builder import make_window
from feature_engineering import (
    create_features_df,     FEATURE_COLUMNS,
    create_features_fin_df, FEATURE_COLUMNS_FIN,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────
DATA_PATH    = "data/train.csv"
MODEL_DIR    = "models"
MODEL_NONFIN = os.path.join(MODEL_DIR, "model_nonfin.pkl")
MODEL_FIN    = os.path.join(MODEL_DIR, "model_fin.pkl")

# 금융 샘플 최소 기준 (미만이면 Ridge 사용, 경고 로그)
MIN_FIN_SAMPLES = 50

# ── 비금융 하이퍼파라미터 (LightGBM) ──
LGBM_PARAMS = dict(
    n_estimators      = 2000,
    learning_rate     = 0.01,
    max_depth         = 4,
    num_leaves        = 15,
    min_child_samples = 5,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)
EARLY_STOPPING_ROUNDS = 30
N_CV_NONFIN           = 5
N_CV_FIN              = 2


# ── 평가지표 ──────────────────────────────────

def _smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Symmetric MAPE — 실제값 0 근처에서도 폭주 없음.
    SMAPE = mean(|actual-pred| / ((|actual|+|pred|)/2)) * 100
    분모가 0이면 해당 항목은 0으로 처리.
    """
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    ratio = np.where(denom < 1e-9, 0.0, np.abs(actual - pred) / denom)
    return float(np.mean(ratio) * 100)


# ── 비금융 CV ──────────────────────────────────

def _run_cv_nonfin(X: pd.DataFrame, y: np.ndarray) -> tuple:
    tscv = TimeSeriesSplit(n_splits=N_CV_NONFIN)
    cv_mae, cv_smape, best_iters = [], [], []

    log.info("[비금융] TimeSeriesSplit %d-Fold CV (타깃: 영업이익률)", N_CV_NONFIN)
    for fold, (tr_i, val_i) in enumerate(tscv.split(X), 1):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(
            X.iloc[tr_i], y[tr_i],
            eval_set=[(X.iloc[val_i], y[val_i])],
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(period=-1),
            ],
        )
        best_iters.append(getattr(m, "best_iteration_", LGBM_PARAMS["n_estimators"]))

        preds = m.predict(X.iloc[val_i])
        mae   = mean_absolute_error(y[val_i], preds)
        smp   = _smape(y[val_i], preds)
        cv_mae.append(mae); cv_smape.append(smp)
        log.info("  Fold %d | MAE=%.4f | SMAPE=%.2f%%", fold, mae, smp)

    log.info("[비금융] CV 평균 MAE=%.4f  SMAPE=%.2f%%",
             np.mean(cv_mae), np.mean(cv_smape))
    return cv_mae, cv_smape, best_iters


# ── 금융 CV (Ridge) ──────────────────────────────────

def _run_cv_fin(X: pd.DataFrame, y: np.ndarray) -> tuple:
    n = len(X)
    if n < MIN_FIN_SAMPLES:
        log.warning("[금융] 샘플 %d개 < %d → Ridge Regression 사용 (과적합 방지)",
                    n, MIN_FIN_SAMPLES)
    else:
        log.info("[금융] 샘플 %d개 → Ridge Regression", n)

    tscv = TimeSeriesSplit(n_splits=N_CV_FIN)
    cv_mae, cv_smape = [], []

    log.info("[금융] TimeSeriesSplit %d-Fold CV (타깃: 순이익)", N_CV_FIN)
    for fold, (tr_i, val_i) in enumerate(tscv.split(X), 1):
        m = Ridge(alpha=10.0)
        m.fit(X.iloc[tr_i], y[tr_i])
        preds = m.predict(X.iloc[val_i])
        mae   = mean_absolute_error(y[val_i], preds)
        smp   = _smape(y[val_i], preds)
        cv_mae.append(mae); cv_smape.append(smp)
        log.info("  Fold %d | MAE=%.4g | SMAPE=%.2f%%", fold, mae, smp)

    log.info("[금융] CV 평균 MAE=%.4g  SMAPE=%.2f%%",
             np.mean(cv_mae), np.mean(cv_smape))
    return cv_mae, cv_smape


# ── 메인 ──────────────────────────────────────

def run_training(raw_df: pd.DataFrame = None) -> dict:
    """
    비금융/금융 분기 학습 파이프라인.

    Returns
    -------
    dict : {"nonfin": {...}, "fin": {...}}
    """
    if raw_df is None:
        log.info("학습 데이터 로드: %s", DATA_PATH)
        raw_df = pd.read_csv(DATA_PATH)
    log.info("원시 데이터 shape: %s", raw_df.shape)

    log.info("슬라이딩 윈도우 샘플 생성 중...")
    nonfin_window, fin_window = make_window(raw_df)
    log.info("비금융 샘플: %d행  /  금융 샘플: %d행",
             len(nonfin_window), len(fin_window))

    os.makedirs(MODEL_DIR, exist_ok=True)
    result = {}

    # ── A. 비금융 ──────────────────────────────
    if not nonfin_window.empty:
        log.info("=" * 55)
        log.info("비금융 모델 학습 (LightGBM / 타깃=영업이익률)")
        X_nf = create_features_df(nonfin_window)
        y_nf = nonfin_window["next_op_margin"].values

        cv_mae, cv_smape, best_iters = _run_cv_nonfin(X_nf, y_nf)

        best_iter    = max(int(np.mean(best_iters)), 50)
        final_params = {**LGBM_PARAMS, "n_estimators": best_iter}
        model_nf     = LGBMRegressor(**final_params)
        model_nf.fit(X_nf, y_nf)
        dump(model_nf, MODEL_NONFIN)
        log.info("비금융 모델 저장: %s (trees=%d)", MODEL_NONFIN, best_iter)

        result["nonfin"] = {
            "mae":        round(float(np.mean(cv_mae)),   6),
            "smape":      round(float(np.mean(cv_smape)), 2),
            "n_samples":  int(len(X_nf)),
            "model_path": MODEL_NONFIN,
            "best_iter":  best_iter,
        }
    else:
        log.warning("비금융 샘플 없음 — 모델 학습 스킵")
        result["nonfin"] = {}

    # ── B. 금융 ──────────────────────────────
    if not fin_window.empty:
        log.info("=" * 55)
        log.info("금융 모델 학습 (Ridge / 타깃=순이익)")
        X_fin = create_features_fin_df(fin_window)
        y_fin = fin_window["next_net_income"].values

        cv_mae, cv_smape = _run_cv_fin(X_fin, y_fin)

        # 최종 Ridge 모델
        model_fin = Ridge(alpha=10.0)
        model_fin.fit(X_fin, y_fin)
        dump(model_fin, MODEL_FIN)
        log.info("금융 모델 저장: %s (Ridge alpha=10)", MODEL_FIN)

        result["fin"] = {
            "mae":        round(float(np.mean(cv_mae)),   2),
            "smape":      round(float(np.mean(cv_smape)), 2),
            "n_samples":  int(len(X_fin)),
            "model_path": MODEL_FIN,
            "model_type": "Ridge",
        }
    else:
        log.warning("금융 샘플 없음 — 모델 학습 스킵")
        result["fin"] = {}

    log.info("=" * 55)
    log.info("전체 학습 완료: %s", result)
    return result


if __name__ == "__main__":
    run_training()
