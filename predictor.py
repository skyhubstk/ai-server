"""
predictor.py

저장된 LightGBM 모델을 로드하여 영업이익 예측.
- 서버 기동 시 모델이 없어도 크래시 없이 지연 로딩(lazy load)
- FEATURE_COLUMNS 순서를 보장하여 train/inference 불일치 방지
- FastAPI 의존성 없음 — 백엔드에서 직접 import 가능
"""

import os
import logging
import pandas as pd
from joblib import load

from feature_engineering import FEATURE_COLUMNS

log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

_model = None   # 지연 로드용 캐시


class ModelNotFoundError(FileNotFoundError):
    """모델 파일이 존재하지 않을 때 발생하는 예외"""
    pass


def _get_model():
    """모델 지연 로드 (첫 호출 시 1회만 디스크에서 읽음)"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise ModelNotFoundError(
                f"모델 파일({MODEL_PATH})이 없습니다. "
                "train.run_training() 을 먼저 실행하여 모델을 학습시켜 주세요."
            )
        log.info("모델 로드: %s", MODEL_PATH)
        _model = load(MODEL_PATH)
    return _model


def predict(features: dict) -> float:
    """
    피처 dict → 다음 연도 영업이익 예측값 반환.

    Parameters
    ----------
    features : create_features() 반환값 (FEATURE_COLUMNS 키 포함)

    Returns
    -------
    float
        다음 연도 예측 영업이익.
        단위는 학습 데이터(train_from_records)에서 사용한 단위와 동일하며,
        모델 내부에서 단위 변환을 수행하지 않습니다.
    """
    model = _get_model()

    # FEATURE_COLUMNS 순서로 DataFrame 생성 (컬럼 순서 불일치 방지)
    x = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    pred = float(model.predict(x)[0])
    log.info("예측 결과: %.2f", pred)
    return pred


def reload_model():
    """모델 파일이 갱신된 후 강제 재로드 (학습 완료 후 호출)"""
    global _model
    _model = None
    log.info("모델 캐시 초기화 → 다음 예측 시 재로드됩니다.")
