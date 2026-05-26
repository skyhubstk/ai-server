"""
train.py

학습 파이프라인:
  1. data/train.csv 로드
  2. dataset_builder.make_window() → 슬라이딩 윈도우 샘플 생성
  3. feature_engineering.create_features_df() → 파생 피처 생성
  4. TimeSeriesSplit 교차검증으로 과거→미래 순서 유지
  5. LightGBM 학습 (Early Stopping + 정규화 하이퍼파라미터)
  6. models/model.pkl 저장

실행:
  python train.py
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

from dataset_builder import make_window
from feature_engineering import create_features_df, FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────
DATA_PATH  = "data/train.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# ── 하이퍼파라미터 ──────────────────────────────
# • n_estimators 충분히 크게 → early stopping 으로 실제 최적 트리 수 결정
# • learning_rate 낮게 (0.01) → 과적합 방지, 일반화 성능 향상
# • num_leaves 작게 (15) → max_depth=4 기준 2^4-1, 재무 데이터 샘플 부족에 대응
# • subsample/colsample_bytree: 배깅 + 피처 샘플링으로 분산 감소
# • reg_alpha(L1) + reg_lambda(L2): 가중치 정규화
LGBM_PARAMS = dict(
    n_estimators       = 2000,
    learning_rate      = 0.01,
    max_depth          = 4,
    num_leaves         = 15,
    min_child_samples  = 5,       # 리프 최소 샘플 수 (소규모 데이터 대응)
    subsample          = 0.8,     # 행 배깅 비율
    subsample_freq     = 1,
    colsample_bytree   = 0.8,     # 컬럼 샘플링 비율
    reg_alpha          = 0.1,     # L1 정규화
    reg_lambda         = 1.0,     # L2 정규화
    random_state       = 42,
    n_jobs             = -1,
    verbose            = -1,
)
EARLY_STOPPING_ROUNDS = 50
N_CV_SPLITS           = 5       # TimeSeriesSplit 분할 수


def run_training(raw_df: pd.DataFrame = None) -> dict:
    """
    학습 파이프라인 실행.

    Parameters
    ----------
    raw_df : 이미 로드된 DataFrame (None 이면 DATA_PATH 에서 읽음)

    Returns
    -------
    dict : {"mae": float, "mape": float, "n_samples": int, "model_path": str}
    """
    # ── 1. 데이터 로드 ──
    if raw_df is None:
        log.info("학습 데이터 로드: %s", DATA_PATH)
        raw_df = pd.read_csv(DATA_PATH)
    log.info("원시 데이터 shape: %s", raw_df.shape)

    # ── 2. 슬라이딩 윈도우 샘플 생성 ──
    log.info("슬라이딩 윈도우 샘플 생성 중...")
    window_df = make_window(raw_df)
    log.info("윈도우 샘플 수: %d", len(window_df))

    # ── 3. 파생 피처 생성 ──
    log.info("피처 엔지니어링 중...")
    X = create_features_df(window_df)
    y = window_df["next_operating_profit"].values
    log.info("피처 수: %d", X.shape[1])

    # ── 4. TimeSeriesSplit 교차검증 ──
    # random split 대신 시간 순서를 보존하는 분할 사용
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_mae, cv_mape = [], []

    log.info("TimeSeriesSplit %d-Fold CV 시작...", N_CV_SPLITS)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cv = LGBMRegressor(**LGBM_PARAMS)
        model_cv.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(period=-1),
            ],
        )

        preds = model_cv.predict(X_val)
        fold_mae  = mean_absolute_error(y_val, preds)
        fold_mape = mean_absolute_percentage_error(y_val, preds)
        cv_mae.append(fold_mae)
        cv_mape.append(fold_mape)
        log.info("  Fold %d | MAE=%.2f | MAPE=%.2f%%",
                 fold, fold_mae, fold_mape * 100)

    log.info("CV 평균 MAE=%.2f  MAPE=%.2f%%",
             np.mean(cv_mae), np.mean(cv_mape) * 100)

    # ── 5. 전체 데이터로 최종 모델 학습 ──
    log.info("전체 데이터로 최종 모델 학습 중...")
    # CV 평균 best iteration 추정
    best_iter = int(np.mean([
        getattr(m, "best_iteration_", LGBM_PARAMS["n_estimators"])
        for m in [model_cv]          # 마지막 fold 모델 기준
    ]))
    best_iter = max(best_iter, 100)

    final_params = {**LGBM_PARAMS, "n_estimators": best_iter}
    final_model  = LGBMRegressor(**final_params)
    final_model.fit(X, y)

    # ── 6. 모델 저장 ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(final_model, MODEL_PATH)
    log.info("모델 저장 완료: %s  (trees=%d)", MODEL_PATH, best_iter)

    result = {
        "mae":         round(float(np.mean(cv_mae)), 2),
        "mape":        round(float(np.mean(cv_mape)) * 100, 2),
        "n_samples":   int(len(X)),
        "model_path":  MODEL_PATH,
        "best_iter":   best_iter,
    }
    log.info("학습 결과: %s", result)
    return result


if __name__ == "__main__":
    run_training()
