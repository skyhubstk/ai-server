"""
train_fin.py  ─ 금융업 전용 모델 학습 파이프라인

학습 파이프라인:
  1. data/train_financial.csv 로드
  2. dataset_builder_fin.make_window_fin() → 슬라이딩 윈도우 샘플 생성
  3. feature_engineering_fin.create_features_fin_df() → 37개 파생 피처
  4. TimeSeriesSplit 교차검증 (과거→미래 순서 보존)
  5. LightGBM 학습 (Early Stopping + 정규화)
  6. models/model_financial.pkl 저장

실행:
  python train_fin.py

예측 대상: 다음 연도 당기순이익 (next_net_income)
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from dataset_builder_fin import make_window_fin
from feature_engineering_fin import create_features_fin_df, FEATURE_COLUMNS_FIN

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────
DATA_PATH_FIN  = "data/train_financial.csv"
MODEL_DIR      = "models"
MODEL_PATH_FIN = os.path.join(MODEL_DIR, "model_financial.pkl")

# ── 하이퍼파라미터 ────────────────────────────────────────
# 금융업 특성:
#   - 비금융업보다 기업 수가 적어 샘플이 더 적음 → 강한 정규화
#   - 이익 변동이 경기·금리에 민감하므로 낮은 learning_rate 유지
#   - 피처(37개)가 비금융업(32개)보다 많아 colsample_bytree 더 낮춤
LGBM_PARAMS_FIN = dict(
    n_estimators      = 2000,
    learning_rate     = 0.01,
    max_depth         = 4,
    num_leaves        = 15,
    min_child_samples = 5,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,    # 피처 수 더 많아 비율 낮춤
    reg_alpha         = 0.2,    # L1 강화
    reg_lambda        = 1.5,    # L2 강화
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)
EARLY_STOPPING_ROUNDS = 50
N_CV_SPLITS           = 5


def run_training_fin(raw_df: pd.DataFrame = None) -> dict:
    """
    금융업 학습 파이프라인 실행.

    Parameters
    ----------
    raw_df : 이미 로드된 DataFrame (None 이면 DATA_PATH_FIN 에서 읽음)

    Returns
    -------
    dict : {"mae", "mape", "n_samples", "model_path", "best_iter"}
    """
    # ── 1. 데이터 로드 ──
    if raw_df is None:
        log.info("[금융업] 학습 데이터 로드: %s", DATA_PATH_FIN)
        raw_df = pd.read_csv(DATA_PATH_FIN)
    log.info("[금융업] 원시 데이터 shape: %s", raw_df.shape)

    # ── 2. 슬라이딩 윈도우 샘플 생성 ──
    log.info("[금융업] 슬라이딩 윈도우 샘플 생성 중...")
    window_df = make_window_fin(raw_df)
    log.info("[금융업] 윈도우 샘플 수: %d", len(window_df))

    # ── 3. 파생 피처 생성 ──
    log.info("[금융업] 피처 엔지니어링 중...")
    X = create_features_fin_df(window_df)
    y = window_df["next_net_income"].values
    log.info("[금융업] 피처 수: %d", X.shape[1])

    # ── 4. TimeSeriesSplit 교차검증 ──
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_mae, cv_mape = [], []
    best_iterations = []   # 백엔드 역제안 ②: 모든 fold best_iter 누적

    log.info("[금융업] TimeSeriesSplit %d-Fold CV 시작...", N_CV_SPLITS)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cv = LGBMRegressor(**LGBM_PARAMS_FIN)
        model_cv.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(period=-1),
            ],
        )

        # 역제안 ②: fold별 best_iteration 누적
        best_iterations.append(
            getattr(model_cv, "best_iteration_", LGBM_PARAMS_FIN["n_estimators"])
        )

        preds     = model_cv.predict(X_val)
        fold_mae  = mean_absolute_error(y_val, preds)
        fold_mape = mean_absolute_percentage_error(y_val, preds)
        cv_mae.append(fold_mae)
        cv_mape.append(fold_mape)
        log.info("  Fold %d | MAE=%.2f | MAPE=%.2f%%",
                 fold, fold_mae, fold_mape * 100)

    log.info("[금융업] CV 평균 MAE=%.2f  MAPE=%.2f%%",
             np.mean(cv_mae), np.mean(cv_mape) * 100)

    # ── 5. 전체 데이터로 최종 모델 학습 ──
    # 역제안 ②: 전체 fold 평균 best_iteration 사용
    best_iter = max(int(np.mean(best_iterations)), 100)
    log.info("[금융업] 전체 데이터로 최종 모델 학습 중... (trees=%d)", best_iter)

    final_params = {**LGBM_PARAMS_FIN, "n_estimators": best_iter}
    final_model  = LGBMRegressor(**final_params)
    final_model.fit(X, y)

    # ── 6. 모델 저장 ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(final_model, MODEL_PATH_FIN)
    log.info("[금융업] 모델 저장 완료: %s", MODEL_PATH_FIN)

    result = {
        "mae":        round(float(np.mean(cv_mae)), 2),
        "mape":       round(float(np.mean(cv_mape)) * 100, 2),
        "n_samples":  int(len(X)),
        "model_path": MODEL_PATH_FIN,
        "best_iter":  best_iter,
    }
    log.info("[금융업] 학습 결과: %s", result)
    return result


if __name__ == "__main__":
    run_training_fin()
