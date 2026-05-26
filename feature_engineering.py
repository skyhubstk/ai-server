"""
feature_engineering.py

학습(train.py)과 추론(predictor.py) 양쪽에서 동일하게 사용.

입력 원시 계정과목 (5개 연도: _0=5년전 … _4=최근):
  revenue_0~4      매출액
  op_0~4           영업이익
  net_0~4          순이익
  debt_0~4         부채총계
  equity_0~4       자본총계
  cash_0~4         현금및현금성자산

생성 파생 피처:
  [수익성]  op_margin_0~4, net_margin_0~4
  [성장성]  revenue_growth_1~4, op_growth_1~4, revenue_cagr, op_cagr
  [안정성]  debt_ratio_4, equity_ratio_4, debt_to_equity_4
  [현금흐름] cash_to_revenue_4, cash_to_debt_4
  [자본수익] roe_4, roa_4
  [추세/변동] op_margin_trend, op_margin_std, revenue_growth_std
  [규모]    log_revenue_4, log_assets_4
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 모델 학습/추론 공통 컬럼 순서 (절대 변경 금지)
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    # 수익성 (Profitability)
    "op_margin_0", "op_margin_1", "op_margin_2", "op_margin_3", "op_margin_4",
    "net_margin_0", "net_margin_1", "net_margin_2", "net_margin_3", "net_margin_4",
    # 성장성 (Growth)
    "revenue_growth_1", "revenue_growth_2", "revenue_growth_3", "revenue_growth_4",
    "op_growth_1", "op_growth_2", "op_growth_3", "op_growth_4",
    "revenue_cagr", "op_cagr",
    # 안정성 (Leverage)
    "debt_ratio_4", "equity_ratio_4", "debt_to_equity_4",
    # 현금흐름
    "cash_to_revenue_4", "cash_to_debt_4",
    # 자본수익률
    "roe_4", "roa_4",
    # 추세/변동성
    "op_margin_trend", "op_margin_std", "revenue_growth_std",
    # 규모 (log-scale)
    "log_revenue_4", "log_assets_4",
]


def _safe_div(num, denom, fill=0.0):
    """0 나누기 방지 및 inf 처리"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(denom) < 1e-9, fill, num / denom
        )
    return float(result) if np.ndim(result) == 0 else result


def _safe_log(x, fill=0.0):
    """음수/0 방지 log 변환"""
    val = float(x)
    return np.log(val) if val > 0 else fill


def _growth_rate(new, old, fill=0.0):
    """전기 대비 성장률"""
    return _safe_div(new - old, abs(old), fill)


def _cagr(end, start, years, fill=0.0):
    """연평균 성장률(CAGR)"""
    if start <= 0 or end <= 0:
        return fill
    return float((end / start) ** (1.0 / years) - 1)


# ──────────────────────────────────────────────
# 핵심 함수
# ──────────────────────────────────────────────

def create_features(row: dict) -> dict:
    """
    원시 재무 데이터 dict → 파생 피처 dict 반환.
    row 는 {revenue_0~4, op_0~4, net_0~4, debt_0~4, equity_0~4, cash_0~4} 를 포함해야 함.
    반환 dict 의 키 순서는 FEATURE_COLUMNS 와 동일.
    """
    r = [row[f"revenue_{i}"] for i in range(5)]
    op = [row[f"op_{i}"]      for i in range(5)]
    net = [row[f"net_{i}"]    for i in range(5)]
    debt = [row[f"debt_{i}"]  for i in range(5)]
    eq = [row[f"equity_{i}"]  for i in range(5)]
    cash = [row[f"cash_{i}"]  for i in range(5)]

    total_assets = [debt[i] + eq[i] for i in range(5)]

    feats = {}

    # ── 수익성 ──────────────────────────────────
    for i in range(5):
        feats[f"op_margin_{i}"]  = _safe_div(op[i],  r[i])
        feats[f"net_margin_{i}"] = _safe_div(net[i], r[i])

    # ── 성장성 ──────────────────────────────────
    for i in range(1, 5):
        feats[f"revenue_growth_{i}"] = _growth_rate(r[i],  r[i-1])
        feats[f"op_growth_{i}"]      = _growth_rate(op[i], op[i-1])

    feats["revenue_cagr"] = _cagr(r[4],  r[0], years=4)
    feats["op_cagr"]      = _cagr(op[4], op[0], years=4)

    # ── 안정성 ──────────────────────────────────
    feats["debt_ratio_4"]      = _safe_div(debt[4], total_assets[4])
    feats["equity_ratio_4"]    = _safe_div(eq[4],   total_assets[4])
    feats["debt_to_equity_4"]  = _safe_div(debt[4], eq[4])

    # ── 현금흐름 ──────────────────────────────────
    feats["cash_to_revenue_4"] = _safe_div(cash[4], r[4])
    feats["cash_to_debt_4"]    = _safe_div(cash[4], debt[4])

    # ── 자본수익률 ──────────────────────────────────
    feats["roe_4"] = _safe_div(net[4], eq[4])
    feats["roa_4"] = _safe_div(net[4], total_assets[4])

    # ── 추세/변동성 ──────────────────────────────────
    op_margins = [feats[f"op_margin_{i}"] for i in range(5)]
    rev_growths = [feats[f"revenue_growth_{i}"] for i in range(1, 5)]

    feats["op_margin_trend"] = op_margins[4] - op_margins[0]
    feats["op_margin_std"]   = float(np.std(op_margins))
    feats["revenue_growth_std"] = float(np.std(rev_growths))

    # ── 규모 (log-scale) ──────────────────────────────────
    feats["log_revenue_4"] = _safe_log(r[4])
    feats["log_assets_4"]  = _safe_log(total_assets[4])

    # FEATURE_COLUMNS 순서로 정렬하여 반환
    return {k: feats[k] for k in FEATURE_COLUMNS}


def create_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 전체에 create_features 적용 → 피처 DataFrame 반환"""
    rows = [create_features(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
