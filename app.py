"""
app.py  ─ ChatDart AI 동작 확인용 데모 스크립트

[사용 흐름]
  1. python train.py     → 모델 학습 (model_nonfin.pkl / model_fin.pkl 생성)
  2. python app.py       → 예측 동작 확인

[예측 반환값]
  비금융: predicted_op_margin (영업이익률, 비율) + 역산 영업이익(원)
  금융  : predicted_roe (ROE 비율) + 역산 순이익(원) + base_equity
"""

import logging

from predictor import predict_from_raw, ModelNotFoundError

logging.basicConfig(level=logging.WARNING)  # 데모 시 INFO 로그 숨김

# ─────────────────────────────────────────────────────────────
# 비금융 샘플 데이터 (윈도우 5년: 2020~2024)
# 키: revenue_0~4, op_0~4, net_0~4, tl_0~4, equity_0~4, cash_0~4, ta_0~4
#     (0=가장 오래된 연도, 4=가장 최근 연도)
# ─────────────────────────────────────────────────────────────
NON_FIN_RAW = {
    # 매출액 (원)
    "revenue_0": 1_050_000_000_000,
    "revenue_1": 1_200_000_000_000,
    "revenue_2": 1_350_000_000_000,
    "revenue_3": 1_500_000_000_000,
    "revenue_4": 1_650_000_000_000,
    # 영업이익 (원)
    "op_0":  75_000_000_000,
    "op_1": 110_000_000_000,
    "op_2": 130_000_000_000,
    "op_3": 155_000_000_000,
    "op_4": 178_000_000_000,
    # 당기순이익 (원)
    "net_0":  55_000_000_000,
    "net_1":  82_000_000_000,
    "net_2":  98_000_000_000,
    "net_3": 115_000_000_000,
    "net_4": 133_000_000_000,
    # 부채총계 total_liabilities (※ "debt_*" → "tl_*" 수정)
    "tl_0": 310_000_000_000,
    "tl_1": 280_000_000_000,
    "tl_2": 270_000_000_000,
    "tl_3": 260_000_000_000,
    "tl_4": 250_000_000_000,
    # 자본총계 (원)
    "equity_0": 540_000_000_000,
    "equity_1": 600_000_000_000,
    "equity_2": 670_000_000_000,
    "equity_3": 750_000_000_000,
    "equity_4": 850_000_000_000,
    # 현금및현금성자산 (원)
    "cash_0":  48_000_000_000,
    "cash_1":  70_000_000_000,
    "cash_2":  85_000_000_000,
    "cash_3": 100_000_000_000,
    "cash_4": 120_000_000_000,
    # 총자산 (tl + equity 로 대체 가능, 없으면 0)
    "ta_0": 850_000_000_000,
    "ta_1": 880_000_000_000,
    "ta_2": 940_000_000_000,
    "ta_3": 1_010_000_000_000,
    "ta_4": 1_100_000_000_000,
}

# ─────────────────────────────────────────────────────────────
# 금융 샘플 데이터 (윈도우 2년: 2023~2024) ← window=2 기준
# 키(필수): ta_0~1, tl_0~1, equity_0~1, net_0~1
#           (0=과거 1년, 1=최근 1년)
# 키(은행): nii_0~1 (순이자이익), llp_0~1 (대손충당금)
# 키(보험): ins_liab_0~1 (보험계약부채)
# 키(업종): sector_detail (bank | insurance | securities)
# ─────────────────────────────────────────────────────────────
FIN_RAW = {
    # 총자산 (원)
    "ta_0": 255_000_000_000_000,   # 2023
    "ta_1": 270_000_000_000_000,   # 2024
    # 부채총계 (원)
    "tl_0": 232_000_000_000_000,
    "tl_1": 245_000_000_000_000,
    # 자본총계 (원)
    "equity_0": 23_000_000_000_000,
    "equity_1": 25_000_000_000_000,
    # 당기순이익 (원)
    "net_0": 2_000_000_000_000,
    "net_1": 2_200_000_000_000,
    # 은행 전용: 순이자이익, 대손충당금 (비은행은 0 또는 생략)
    "nii_0": 3_500_000_000_000,
    "nii_1": 3_700_000_000_000,
    "llp_0":  300_000_000_000,
    "llp_1":  280_000_000_000,
    # 보험 전용: 보험계약부채 (비보험사는 0 또는 생략)
    "ins_liab_0": 0,
    "ins_liab_1": 0,
    # 업종: bank / insurance / securities
    "sector_detail": "bank",
}


def main():
    sep = "=" * 65
    print(sep)
    print("ChatDart AI  --  비금융 / 금융 분리 예측 데모")
    print(sep)

    # ── 비금융 예측 ──────────────────────────────────────────
    print("\n[비금융] 다음 연도 영업이익 예측...")
    try:
        result = predict_from_raw(NON_FIN_RAW, sector="non_financial")
        margin = result["predicted_op_margin"]
        profit = result["predicted_op_profit"]
        rev    = result["base_revenue"]
        print(f"  기준 매출액       : {rev:>20,.0f} 원")
        print(f"  예측 영업이익률   : {margin:>20.4f}  ({margin*100:.2f}%)")
        print(f"  예측 영업이익(역산): {profit:>20,.0f} 원")
    except ModelNotFoundError as e:
        print(f"  [오류] {e}")
        print("  → python train.py 를 먼저 실행하세요.")

    # ── 금융 예측 ────────────────────────────────────────────
    print("\n[금융]   다음 연도 ROE / 당기순이익 예측...")
    try:
        result = predict_from_raw(FIN_RAW, sector="financial")
        roe    = result["predicted_roe"]          # ROE 비율 (예측 타깃)
        ni     = result["predicted_net_income"]   # 역산 순이익 = roe × equity_1
        equity = result["base_equity"]            # 기준 자본 (equity_1)
        print(f"  기준 자본(equity_1) : {equity:>20,.0f} 원")
        print(f"  예측 ROE            : {roe:>20.4f}  ({roe*100:.2f}%)")
        print(f"  예측 순이익(역산)   : {ni:>20,.0f} 원")
    except ModelNotFoundError as e:
        print(f"  [오류] {e}")
        print("  → python train.py 를 먼저 실행하세요.")

    print()
    print("  ※ 실제 서비스: 백엔드에서 predict_from_raw() 결과를 GPT에 전달하여 해설 생성")
    print(sep)


if __name__ == "__main__":
    main()
