"""
feature_engineering_fin.py  ─ 금융업(은행·보험·증권) 전용 피처 엔지니어링

금융업 특성:
  - 매출액·영업이익·매출원가·판관비 계정이 없거나 의미가 다름
  - 이자수익·수수료수익이 주 수익원
  - 총자산·부채·자기자본·순이익·현금만 공통 비교 가능

입력 원시 계정과목 (5개 연도: _0=5년전 … _4=최근):
  assets_0~4    총자산
  liab_0~4      총부채
  equity_0~4    자기자본
  ni_0~4        당기순이익
  cash_0~4      현금및현금성자산

생성 파생 피처 (37개):
  [수익성]    roe_0~4, roa_0~4
  [자본구조]  leverage_0~4, equity_ratio_0~4
  [성장성]    ni_growth_1~4, asset_growth_1~4, ni_cagr, asset_cagr
  [안정성]    cash_to_assets_4, equity_multiplier_4
  [추세/변동] roe_trend, roe_std, ni_growth_std
  [규모]      log_total_assets_4

예측 대상: 다음 연도 당기순이익 (next_net_income)
단위: 학습 데이터와 동일 단위(원, KRW) — 변환 없음
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 금융업 모델 학습/추론 공통 컬럼 순서 (절대 변경 금지)
# ─────────────────────────────────────────────────────────────
FEATURE_COLUMNS_FIN = [
    # 수익성 (Profitability)
    "roe_0", "roe_1", "roe_2", "roe_3", "roe_4",          # ROE = 순이익/자기자본
    "roa_0", "roa_1", "roa_2", "roa_3", "roa_4",          # ROA = 순이익/총자산
    # 자본구조 (Capital Structure)
    "leverage_0", "leverage_1", "leverage_2", "leverage_3", "leverage_4",      # 부채/총자산
    "equity_ratio_0", "equity_ratio_1", "equity_ratio_2", "equity_ratio_3", "equity_ratio_4",  # 자기자본/총자산
    # 성장성 (Growth)
    "ni_growth_1", "ni_growth_2", "ni_growth_3", "ni_growth_4",        # 순이익 YoY 성장률
    "asset_growth_1", "asset_growth_2", "asset_growth_3", "asset_growth_4",    # 총자산 YoY 성장률
    "ni_cagr",                                             # 순이익 4년 CAGR
    "asset_cagr",                                          # 총자산 4년 CAGR
    # 안정성·유동성 (Stability & Liquidity)
    "cash_to_assets_4",                                    # 현금/총자산
    "equity_multiplier_4",                                 # 총자산/자기자본 (레버리지 배수)
    # 추세/변동성 (Trend & Volatility)
    "roe_trend",                                           # roe_4 - roe_0 (방향성)
    "roe_std",                                             # ROE 표준편차 (안정성)
    "ni_growth_std",                                       # 순이익 성장률 표준편차
    # 규모 (Scale)
    "log_total_assets_4",                                  # log(총자산_최근)
]


# ─────────────────────────────────────────────────────────────
# 내부 유틸리티 함수 (백엔드 역제안 ① 반영 — np.divide 방식)
# ─────────────────────────────────────────────────────────────

def _safe_div(num, denom, fill: float = 0.0) -> float:
    """
    0 나누기·inf·NaN 방지 나눗셈.

    백엔드 역제안 ①: np.where 는 양쪽 식을 모두 평가하여
    Python scalar 0/0 시 ZeroDivisionError가 발생할 수 있음.
    np.divide(out=, where=) 방식은 마스킹된 위치를 아예 평가하지 않아 안전.
    """
    numerator   = np.asarray(num,   dtype=float)
    denominator = np.asarray(denom, dtype=float)
    result = np.divide(
        numerator, denominator,
        out=np.full_like(numerator, fill, dtype=float),
        where=np.abs(denominator) >= 1e-9,
    )
    return float(result) if result.ndim == 0 else result


def _safe_log(x, fill: float = 0.0) -> float:
    """음수/0 방지 log 변환"""
    val = float(x)
    return float(np.log(val)) if val > 0 else fill


def _growth_rate(new, old, fill: float = 0.0) -> float:
    """전기 대비 성장률"""
    return _safe_div(new - old, abs(float(old)), fill)


def _cagr(end, start, years: int, fill: float = 0.0) -> float:
    """연평균 성장률(CAGR) — 음수값은 CAGR 정의 불능이므로 fill 반환"""
    s, e = float(start), float(end)
    if s <= 0 or e <= 0:
        return fill
    return float((e / s) ** (1.0 / years) - 1)


# ─────────────────────────────────────────────────────────────
# 핵심 함수
# ─────────────────────────────────────────────────────────────

def create_features_fin(row: dict) -> dict:
    """
    금융업 원시 재무 데이터 dict → 파생 피처 dict 반환.

    Parameters
    ----------
    row : dict
        필수 키:
          assets_0~4   총자산 (5개 연도, _0=5년전 … _4=최근)
          liab_0~4     총부채
          equity_0~4   자기자본
          ni_0~4       당기순이익
          cash_0~4     현금및현금성자산
        모든 값은 동일 단위(KRW) 여야 합니다.

    Returns
    -------
    dict : FEATURE_COLUMNS_FIN 순서로 정렬된 피처 dict
    """
    ta  = [float(row[f"assets_{i}"])  for i in range(5)]
    tl  = [float(row[f"liab_{i}"])    for i in range(5)]
    eq  = [float(row[f"equity_{i}"])  for i in range(5)]
    ni  = [float(row[f"ni_{i}"])      for i in range(5)]
    cas = [float(row[f"cash_{i}"])    for i in range(5)]

    feats: dict = {}

    # ── 수익성 ──────────────────────────────────────────────
    for i in range(5):
        feats[f"roe_{i}"] = _safe_div(ni[i], eq[i])
        feats[f"roa_{i}"] = _safe_div(ni[i], ta[i])

    # ── 자본구조 ────────────────────────────────────────────
    for i in range(5):
        feats[f"leverage_{i}"]    = _safe_div(tl[i], ta[i])
        feats[f"equity_ratio_{i}"] = _safe_div(eq[i], ta[i])

    # ── 성장성 ──────────────────────────────────────────────
    for i in range(1, 5):
        feats[f"ni_growth_{i}"]    = _growth_rate(ni[i],  ni[i - 1])
        feats[f"asset_growth_{i}"] = _growth_rate(ta[i],  ta[i - 1])

    feats["ni_cagr"]    = _cagr(ni[4],  ni[0],  years=4)
    feats["asset_cagr"] = _cagr(ta[4],  ta[0],  years=4)

    # ── 안정성·유동성 ────────────────────────────────────────
    feats["cash_to_assets_4"]    = _safe_div(cas[4], ta[4])
    feats["equity_multiplier_4"] = _safe_div(ta[4],  eq[4])

    # ── 추세/변동성 ──────────────────────────────────────────
    roes       = [feats[f"roe_{i}"]        for i in range(5)]
    ni_growths = [feats[f"ni_growth_{i}"]  for i in range(1, 5)]

    feats["roe_trend"]     = roes[4] - roes[0]
    feats["roe_std"]       = float(np.std(roes))
    feats["ni_growth_std"] = float(np.std(ni_growths))

    # ── 규모 ────────────────────────────────────────────────
    feats["log_total_assets_4"] = _safe_log(ta[4])

    return {k: feats[k] for k in FEATURE_COLUMNS_FIN}


def create_features_fin_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 전체에 create_features_fin 적용 → 피처 DataFrame 반환"""
    rows = [create_features_fin(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS_FIN)
