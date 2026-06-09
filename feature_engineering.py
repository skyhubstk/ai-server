"""
feature_engineering.py

학습(train.py)과 추론(predictor.py) 양쪽에서 동일하게 사용.

[비금융 피처] FEATURE_COLUMNS (32개)
  입력: revenue_0~4, op_0~4, net_0~4, tl_0~4, equity_0~4, cash_0~4, ta_0~4
  생성: op_margin, net_margin, 성장률, CAGR, 부채비율, ROE, ROA, 추세, 규모

[금융 피처] FEATURE_COLUMNS_FIN (16개) — window=2 기준
  입력(필수):    ta_0~1, tl_0~1, equity_0~1, net_0~1
  입력(은행):    nii_0~1 (순이자이익), llp_0~1 (대손충당금)  — 없으면 0.0
  입력(보험):    ins_liab_0~1 (보험계약부채)                 — 없으면 0.0
  입력(업종):    sector_detail (bank/insurance/securities)   — 없으면 ""
  생성:
    건전성    leverage_ratio_1, equity_ratio_1
    수익성    roe_1, roa_1
    성장성    asset_growth, equity_growth, net_growth
    규모      log_assets_1
    ROE추세   roe_trend  (= roe_1 - roe_0)
    은행      nii_ratio_1, llp_ratio_1, nii_growth
    보험      ins_liab_ratio_1
    업종원-핫 is_bank, is_insurance, is_securities

  금융사: revenue/operating_profit 없음 → 자산·자본·순이익 기반 피처만 사용
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 비금융 피처 컬럼 (절대 순서 변경 금지)
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    # 수익성
    "op_margin_0", "op_margin_1", "op_margin_2", "op_margin_3", "op_margin_4",
    "net_margin_0", "net_margin_1", "net_margin_2", "net_margin_3", "net_margin_4",
    # 성장성
    "revenue_growth_1", "revenue_growth_2", "revenue_growth_3", "revenue_growth_4",
    "op_growth_1", "op_growth_2", "op_growth_3", "op_growth_4",
    "revenue_cagr", "op_cagr",
    # 안정성
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

# ──────────────────────────────────────────────
# 금융 피처 컬럼 — window=2 (idx: 0=과거, 1=최근)
# ──────────────────────────────────────────────
FEATURE_COLUMNS_FIN = [
    # ── 건전성 ──────────────────────────────
    "leverage_ratio_1",   # 부채/자산 (레버리지, 최근)
    "equity_ratio_1",     # 자본/자산 (자본건전성, 최근)
    # ── 수익성 ──────────────────────────────
    "roe_1",              # 순이익/자본 (자본수익률, 최근)
    "roa_1",              # 순이익/자산 (자산수익률, 최근)
    # ── 성장성 ──────────────────────────────
    "asset_growth",       # 총자산 성장률 (0→1년)
    "equity_growth",      # 자본 성장률 (0→1년)
    "net_growth",         # 순이익 성장률 (0→1년)
    # ── 규모 ────────────────────────────────
    "log_assets_1",       # log(총자산)
    # ── ROE 변화 ────────────────────────────
    "roe_trend",          # roe_1 - roe_0 (수익성 방향성)
    # ── 은행 전용 ────────────────────────────
    "nii_ratio_1",        # 순이자이익/총자산 (NIM 근사)
    "llp_ratio_1",        # |대손충당금|/총자산 (건전성)
    "nii_growth",         # 순이자이익 성장률 (0→1년)
    # ── 보험 전용 ────────────────────────────
    "ins_liab_ratio_1",   # 보험계약부채/총자산 (보험사 부채구조)
    # ── 업종 원-핫 (LightGBM 트리 분기용) ───
    "is_bank",            # 은행지주 여부 (1/0)
    "is_insurance",       # 보험사 여부 (1/0)
    "is_securities",      # 증권사 여부 (1/0)
]


# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def _safe_div(num, denom, fill: float = 0.0) -> float:
    numerator   = np.asarray(num,   dtype=float)
    denominator = np.asarray(denom, dtype=float)
    result = np.divide(
        numerator, denominator,
        out=np.full_like(numerator, fill, dtype=float),
        where=np.abs(denominator) >= 1e-9,
    )
    return float(result) if result.ndim == 0 else result


def _safe_log(x, fill=0.0):
    val = float(x)
    return np.log(val) if val > 0 else fill


def _growth_rate(new, old, fill=0.0):
    return _safe_div(new - old, abs(old), fill)


def _cagr(end, start, years, fill=0.0):
    if start <= 0 or end <= 0:
        return fill
    return float((end / start) ** (1.0 / years) - 1)


# ──────────────────────────────────────────────
# 비금융 피처 생성
# ──────────────────────────────────────────────

def create_features(row: dict) -> dict:
    """
    비금융 원시 재무 데이터 dict → 파생 피처 dict.
    row: {revenue_0~4, op_0~4, net_0~4, tl_0~4, equity_0~4, cash_0~4, ta_0~4}
    """
    r    = [row.get(f"revenue_{i}", 0.0) or 0.0 for i in range(5)]
    op   = [row.get(f"op_{i}",      0.0) or 0.0 for i in range(5)]
    net  = [row.get(f"net_{i}",     0.0) or 0.0 for i in range(5)]
    tl   = [row.get(f"tl_{i}",      0.0) or 0.0 for i in range(5)]
    eq   = [row.get(f"equity_{i}",  0.0) or 0.0 for i in range(5)]
    cash = [row.get(f"cash_{i}",    0.0) or 0.0 for i in range(5)]
    ta   = [row.get(f"ta_{i}",      0.0) or 0.0 for i in range(5)]

    total_assets = [ta[i] if ta[i] > 0 else (tl[i] + eq[i]) for i in range(5)]

    feats = {}

    for i in range(5):
        feats[f"op_margin_{i}"]  = _safe_div(op[i],  r[i])
        feats[f"net_margin_{i}"] = _safe_div(net[i], r[i])

    for i in range(1, 5):
        feats[f"revenue_growth_{i}"] = _growth_rate(r[i],  r[i-1])
        feats[f"op_growth_{i}"]      = _growth_rate(op[i], op[i-1])

    feats["revenue_cagr"] = _cagr(r[4],  r[0], years=4)
    feats["op_cagr"]      = _cagr(op[4], op[0], years=4)

    feats["debt_ratio_4"]     = _safe_div(tl[4], total_assets[4])
    feats["equity_ratio_4"]   = _safe_div(eq[4], total_assets[4])
    feats["debt_to_equity_4"] = _safe_div(tl[4], eq[4])

    feats["cash_to_revenue_4"] = _safe_div(cash[4], r[4])
    feats["cash_to_debt_4"]    = _safe_div(cash[4], tl[4])

    feats["roe_4"] = _safe_div(net[4], eq[4])
    feats["roa_4"] = _safe_div(net[4], total_assets[4])

    op_margins  = [feats[f"op_margin_{i}"] for i in range(5)]
    rev_growths = [feats[f"revenue_growth_{i}"] for i in range(1, 5)]

    feats["op_margin_trend"]    = op_margins[4] - op_margins[0]
    feats["op_margin_std"]      = float(np.std(op_margins))
    feats["revenue_growth_std"] = float(np.std(rev_growths))

    feats["log_revenue_4"] = _safe_log(r[4])
    feats["log_assets_4"]  = _safe_log(total_assets[4])

    return {k: feats[k] for k in FEATURE_COLUMNS}


def create_features_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = [create_features(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# ──────────────────────────────────────────────
# 금융 피처 생성 — window=2 (idx 0~1)
# ──────────────────────────────────────────────

def create_features_fin(row: dict) -> dict:
    """
    금융사 원시 재무 데이터 dict → 파생 피처 dict.

    입력 키:
      ta_0~1, tl_0~1, equity_0~1, net_0~1   (기본)
      nii_0~1, llp_0~1                       (은행 전용, 없으면 0)
      ins_liab_0~1                           (보험 전용, 없으면 0)
      sector_detail                          (업종: bank/insurance/securities)

    window=2 (idx: 0=과거, 1=최근)
    """
    ta       = [row.get(f"ta_{i}",       0.0) or 0.0 for i in range(2)]
    tl       = [row.get(f"tl_{i}",       0.0) or 0.0 for i in range(2)]
    eq       = [row.get(f"equity_{i}",   0.0) or 0.0 for i in range(2)]
    net      = [row.get(f"net_{i}",      0.0) or 0.0 for i in range(2)]
    nii      = [row.get(f"nii_{i}",      0.0) or 0.0 for i in range(2)]
    llp      = [row.get(f"llp_{i}",      0.0) or 0.0 for i in range(2)]
    ins_liab = [row.get(f"ins_liab_{i}", 0.0) or 0.0 for i in range(2)]
    sector   = str(row.get("sector_detail", "") or "")

    feats = {}

    # 건전성
    feats["leverage_ratio_1"] = _safe_div(tl[1], ta[1])
    feats["equity_ratio_1"]   = _safe_div(eq[1], ta[1])

    # 수익성
    feats["roe_1"] = _safe_div(net[1], eq[1])
    feats["roa_1"] = _safe_div(net[1], ta[1])

    # 성장률 (0→1년)
    feats["asset_growth"]  = _growth_rate(ta[1],  ta[0])
    feats["equity_growth"] = _growth_rate(eq[1],  eq[0])
    feats["net_growth"]    = _growth_rate(net[1], net[0])

    # 규모
    feats["log_assets_1"] = _safe_log(ta[1])

    # ROE 추세 (roe_1 - roe_0)
    roe_0 = _safe_div(net[0], eq[0])
    feats["roe_trend"] = feats["roe_1"] - roe_0

    # 은행 전용
    feats["nii_ratio_1"] = _safe_div(nii[1], ta[1])
    feats["llp_ratio_1"] = _safe_div(abs(llp[1]), ta[1])
    feats["nii_growth"]  = _growth_rate(nii[1], nii[0])

    # 보험 전용
    feats["ins_liab_ratio_1"] = _safe_div(ins_liab[1], ta[1])

    # 업종 원-핫 (LightGBM 트리가 업종별로 분기 가능하도록)
    feats["is_bank"]       = 1.0 if sector == "bank"       else 0.0
    feats["is_insurance"]  = 1.0 if sector == "insurance"  else 0.0
    feats["is_securities"] = 1.0 if sector == "securities" else 0.0

    return {k: feats[k] for k in FEATURE_COLUMNS_FIN}


def create_features_fin_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = [create_features_fin(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS_FIN)
