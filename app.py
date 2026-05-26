"""
app.py — 백엔드 통합 사용 예제

실제 백엔드에서는 아래처럼 chatdart_ai 모듈을 직접 import하여 사용합니다.

    from chatdart_ai import predict_next_profit, train_from_records

이 파일은 동작 확인용 샘플 스크립트입니다.
실행: python app.py
"""

from chatdart_ai import predict_next_profit, train_from_records, ModelNotFoundError

# ── 샘플 재무제표 데이터 (단위: 백만원) ──────────────────────────────
SAMPLE_RECORDS = [
    {"company": "샘플기업", "year": 2018, "revenue": 1_000_000, "operating_profit":  80_000, "net_income":  60_000, "debt": 300_000, "equity": 500_000, "cash":  50_000},
    {"company": "샘플기업", "year": 2019, "revenue": 1_100_000, "operating_profit":  90_000, "net_income":  68_000, "debt": 290_000, "equity": 550_000, "cash":  55_000},
    {"company": "샘플기업", "year": 2020, "revenue": 1_050_000, "operating_profit":  75_000, "net_income":  55_000, "debt": 310_000, "equity": 540_000, "cash":  48_000},
    {"company": "샘플기업", "year": 2021, "revenue": 1_200_000, "operating_profit": 110_000, "net_income":  82_000, "debt": 280_000, "equity": 600_000, "cash":  70_000},
    {"company": "샘플기업", "year": 2022, "revenue": 1_350_000, "operating_profit": 130_000, "net_income":  98_000, "debt": 270_000, "equity": 670_000, "cash":  85_000},
    {"company": "샘플기업", "year": 2023, "revenue": 1_500_000, "operating_profit": 155_000, "net_income": 115_000, "debt": 260_000, "equity": 750_000, "cash": 100_000},
]

# 분석 요청용 입력 (가장 최근 5개 연도 = 2019~2023)
SAMPLE_INPUT = {
    "revenue_0": 1_100_000, "revenue_1": 1_050_000, "revenue_2": 1_200_000,
    "revenue_3": 1_350_000, "revenue_4": 1_500_000,
    "op_0":  90_000, "op_1":  75_000, "op_2": 110_000,
    "op_3": 130_000, "op_4": 155_000,
    "net_0":  68_000, "net_1":  55_000, "net_2":  82_000,
    "net_3":  98_000, "net_4": 115_000,
    "debt_0": 290_000, "debt_1": 310_000, "debt_2": 280_000,
    "debt_3": 270_000, "debt_4": 260_000,
    "equity_0": 550_000, "equity_1": 540_000, "equity_2": 600_000,
    "equity_3": 670_000, "equity_4": 750_000,
    "cash_0": 55_000, "cash_1": 48_000, "cash_2": 70_000,
    "cash_3": 85_000, "cash_4": 100_000,
}


def main():
    print("=" * 60)
    print("ChatDart AI — 사용 예제")
    print("=" * 60)

    # ── 1단계: 모델 학습 ──
    print("\n[1] 모델 학습 중...")
    result = train_from_records(SAMPLE_RECORDS, save_csv=True)
    print(f"    MAE     : {result['mae']:,.0f}")
    print(f"    MAPE    : {result['mape']:.2f}%")
    print(f"    샘플 수  : {result['n_samples']}")
    print(f"    트리 수  : {result['best_iter']}")

    # ── 2단계: 예측 ──
    print("\n[2] 다음 연도 영업이익 예측 중...")
    try:
        pred = predict_next_profit(SAMPLE_INPUT)
        print(f"\n    예측 영업이익 : {pred:,.0f} 백만원")
        print("\n    ※ GPT 요약은 백엔드에서 이 예측값을 활용하여 생성합니다.")
    except ModelNotFoundError as e:
        print(f"    [오류] {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
