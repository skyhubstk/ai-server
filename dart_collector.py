"""
dart_collector.py  ─ DART OpenAPI 재무제표 수집 모듈

[전략]
  - 대기업·중견기업 ~90개 종목코드를 하드코딩하여 안정적으로 수집
  - 연도 범위: 2010 ~ 2024 (15개년)
  - 기업당 최대 API 호출: 15년 × 최대 2번(CFS→OFS) = 30번

[chatgpt4.txt 반영] 데이터 수집은 합치고, 학습만 분기
  → 단일 CSV: data/train.csv (is_financial 컬럼으로 구분)
  → 금융사: is_financial=1, revenue/operating_profit=NaN, total_assets 사용
  → 비금융: is_financial=0, revenue/operating_profit 사용, total_assets=equity+debt

통합 스키마:
  company, year, is_financial,
  revenue, operating_profit, net_income,
  total_assets, total_liabilities, equity, cash

필수 환경 변수:
  DART_API_KEY  DART OpenAPI 인증키 (https://opendart.fss.or.kr)

사용:
  python dart_collector.py                           # 대기업 전체 수집
  python dart_collector.py --limit 10               # 앞 10개만 테스트
  python dart_collector.py --codes 105560 055550    # 특정 종목만
"""

import io
import json
import os
import time
import logging
import argparse
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 대기업·중견기업 종목코드 (KOSPI 주요 기업)
# 소형사·미상장 제외, DART에 2010년 이후 데이터 있는 기업만
# ─────────────────────────────────────────────────────────────
TARGET_STOCK_CODES: list = [
    # ── 반도체·전자 ──
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "009150",  # 삼성전기
    "006400",  # 삼성SDI
    "034220",  # LG디스플레이
    "011070",  # LG이노텍
    "066570",  # LG전자
    "035420",  # NAVER
    "035720",  # 카카오
    "018260",  # 삼성SDS
    # ── 자동차·부품 ──
    "005380",  # 현대자동차
    "000270",  # 기아
    "012330",  # 현대모비스
    "086280",  # 현대글로비스
    "161390",  # 한국타이어앤테크놀로지
    # ── 화학·소재 ──
    "051910",  # LG화학
    "011170",  # 롯데케미칼
    "009830",  # 한화솔루션
    "004800",  # 효성
    "010130",  # 고려아연
    "003670",  # 포스코퓨처엠
    # ── 철강·중공업 ──
    "005490",  # POSCO홀딩스
    "004020",  # 현대제철
    "010140",  # 삼성중공업
    "009540",  # 한국조선해양
    "034020",  # 두산에너빌리티
    "329180",  # HD현대중공업
    "267260",  # HD현대일렉트릭
    # ── 에너지·유틸리티 ──
    "015760",  # 한국전력
    "036460",  # 한국가스공사
    "096770",  # SK이노베이션
    "010950",  # S-Oil
    # ── 통신 ──
    "017670",  # SK텔레콤
    "030200",  # KT
    "033780",  # KT&G
    # ── 바이오·제약 ──
    "207940",  # 삼성바이오로직스
    "068270",  # 셀트리온
    "128940",  # 한미약품
    "000100",  # 유한양행
    "003090",  # 대웅제약
    # ── 유통·소비재 ──
    "139480",  # 이마트
    "023530",  # 롯데쇼핑
    "282330",  # BGF리테일
    "090430",  # 아모레퍼시픽
    "008770",  # 호텔신라
    "005180",  # 빙그레
    "000080",  # 하이트진로
    "271560",  # 오리온
    "021240",  # 코웨이
    "069960",  # 현대백화점
    # ── 건설·인프라 ──
    "000720",  # 현대건설
    "006360",  # GS건설
    # ── 항공·물류 ──
    "003490",  # 대한항공
    "000120",  # CJ대한통운
    "180640",  # 한진칼
    # ── 방산·항공우주 ──
    "047810",  # 한국항공우주(KAI)
    "012450",  # 한화에어로스페이스
    # ── 게임·엔터 ──
    "036570",  # 엔씨소프트
    "251270",  # 넷마블
    "259960",  # 크래프톤
    "352820",  # 하이브
    # ── 지주·복합 ──
    "028260",  # 삼성물산
    "034730",  # SK
    "078930",  # GS
    "006260",  # LS
    "002380",  # KCC
    "097950",  # CJ제일제당
    "035760",  # CJ ENM
    "047050",  # 포스코인터내셔널
    "103140",  # 풍산
    "011790",  # SKC
    # ── 금융 (은행지주) ──
    "105560",  # KB금융
    "055550",  # 신한지주
    "086790",  # 하나금융지주
    "316140",  # 우리금융지주
    "138040",  # 메리츠금융지주
    # ── 금융 (보험) ──
    "000810",  # 삼성화재해상보험
    "032830",  # 삼성생명
    "005830",  # DB손해보험
    "001450",  # 현대해상
    "000370",  # 한화손해보험
    # ── 금융 (증권) ──
    "006800",  # 미래에셋증권
    "071050",  # 한국금융지주
    "005940",  # NH투자증권
    "016360",  # 삼성증권
    "039490",  # 키움증권
    "003530",  # 대신증권
]

# ─────────────────────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────────────────────
START_YEAR    = 2010
END_YEAR      = 2024    # _latest_fiscal_year()로 자동 설정

MIN_REVENUE   = 1_000_000_000    # 10억원 (대형사라면 통과)
MIN_ASSETS    = 100_000_000_000  # 1000억원 (금융사)

CORP_CACHE_PATH  = "data/.corp_cache.json"
CORP_CACHE_TTL_H = 24

DART_API_BASE = "https://opendart.fss.or.kr/api"
REPRT_CODE_A  = "11011"   # 사업보고서
FS_DIV_CFS    = "CFS"
FS_DIV_OFS    = "OFS"

# fnlttSinglAcnt 고정 (모든 기업 안정적, 호출량 최소화)
FS_ENDPOINT   = f"{DART_API_BASE}/fnlttSinglAcnt.json"

RETRY_COUNT   = 3
RETRY_DELAY   = 3
REQUEST_DELAY = 0.5    # 기업당 딜레이 (대기업 목록이므로 0.5초로도 충분)

# ─────────────────────────────────────────────────────────────
# [ChatGPT6.pdf 2순위] 업종(sector) 매핑
# 비금융 내 섹터별 분리 준비 → train.csv에 sector 컬럼 추가
# ─────────────────────────────────────────────────────────────
SECTOR_MAP: dict = {
    # IT/전자
    "005930": "IT",        "000660": "IT",        "009150": "IT",
    "006400": "IT",        "034220": "IT",        "011070": "IT",
    "066570": "IT",        "018260": "IT",
    # 인터넷/게임
    "035420": "인터넷",    "035720": "인터넷",
    "036570": "게임",      "251270": "게임",
    "259960": "게임",
    # 자동차
    "005380": "자동차",    "000270": "자동차",    "012330": "자동차",
    "086280": "자동차",    "161390": "자동차",
    # 화학/소재
    "051910": "화학",      "011170": "화학",      "009830": "화학",
    "004800": "화학",      "010130": "소재",      "003670": "소재",
    "011790": "화학",
    # 철강/중공업
    "005490": "철강",      "004020": "철강",      "010140": "중공업",
    "009540": "중공업",    "034020": "중공업",    "329180": "중공업",
    "267260": "중공업",
    # 에너지/유틸리티
    "015760": "에너지",    "036460": "에너지",    "096770": "에너지",
    "010950": "에너지",
    # 통신
    "017670": "통신",      "030200": "통신",      "033780": "통신",
    # 바이오/제약
    "207940": "바이오",    "068270": "바이오",    "128940": "제약",
    "000100": "제약",      "003090": "제약",
    # 유통/소비재
    "139480": "유통",      "023530": "유통",      "282330": "유통",
    "090430": "소비재",    "008770": "소비재",    "005180": "소비재",
    "000080": "소비재",    "271560": "소비재",    "021240": "소비재",
    "069960": "유통",
    # 건설
    "000720": "건설",      "006360": "건설",
    # 항공/물류
    "003490": "항공",      "000120": "물류",      "180640": "항공",
    # 방산
    "047810": "방산",      "012450": "방산",
    # 엔터
    "352820": "엔터",
    # 지주/복합
    "028260": "지주",      "034730": "지주",      "078930": "지주",
    "006260": "지주",      "002380": "지주",      "097950": "지주",
    "035760": "지주",      "047050": "지주",      "103140": "소재",
    # 금융(은행)
    "105560": "금융_은행", "055550": "금융_은행", "086790": "금융_은행",
    "316140": "금융_은행", "138040": "금융_은행",
    # 금융(보험)
    "000810": "금융_보험", "032830": "금융_보험", "005830": "금융_보험",
    "001450": "금융_보험", "000370": "금융_보험",
    # 금융(증권)
    "006800": "금융_증권", "071050": "금융_증권", "005940": "금융_증권",
    "016360": "금융_증권", "039490": "금융_증권", "003530": "금융_증권",
}

# 금융사 종목코드 집합 (회사명 키워드 방식 대체)
# → 이름에 "보험"/"금융" 없어도 정확히 판별 (현대해상·신한지주·삼성생명 오분류 방지)
FINANCIAL_STOCK_CODES: set = {
    # 은행지주
    "105560",  # KB금융
    "055550",  # 신한지주
    "086790",  # 하나금융지주
    "316140",  # 우리금융지주
    "138040",  # 메리츠금융지주
    # 보험
    "000810",  # 삼성화재해상보험
    "032830",  # 삼성생명
    "005830",  # DB손해보험
    "001450",  # 현대해상
    "000370",  # 한화손해보험
    # 증권
    "006800",  # 미래에셋증권
    "071050",  # 한국금융지주
    "005940",  # NH투자증권
    "016360",  # 삼성증권
    "039490",  # 키움증권
    "003530",  # 한화투자증권
}

# ─────────────────────────────────────────────────────────────
# 계정과목 매핑
# ─────────────────────────────────────────────────────────────
ALL_CONCEPT_MAP: dict = {
    "ifrs-full_Revenue":                                       "revenue",
    "ifrs-full_SalesRevenueNet":                               "revenue",
    "ifrs-full_RevenueFromContractWithCustomer":               "revenue",
    "ifrs-full_InterestIncome":                                "revenue",
    "dart_GrossOperatingRevenues":                             "revenue",
    "ifrs-full_CostOfSales":                                   "cost_of_sales",
    "ifrs-full_GrossProfit":                                   "gross_profit",
    "ifrs-full_SellingGeneralAndAdministrativeExpenses":       "sga",
    "dart_SellingGeneralAdministrativeExpenses":               "sga",
    "dart_OperatingExpenses":                                  "sga",
    "dart_OperatingIncomeLoss":                                "operating_profit",
    "k-ifrs_OperatingIncomeLoss":                             "operating_profit",
    "ifrs-full_OperatingIncomeLoss":                          "operating_profit",
    "ifrs-full_ProfitLoss":                                    "net_income",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent":        "net_income",
    "dart_ProfitLoss":                                         "net_income",
    "ifrs-full_Assets":                                        "total_assets",
    "ifrs-full_Liabilities":                                   "total_liabilities",
    "ifrs-full_Equity":                                        "equity",
    "ifrs-full_EquityAttributableToOwnersOfParent":            "equity",
    "ifrs-full_CashAndCashEquivalents":                        "cash",
}

ALL_NM_MAP: dict = {
    "매출액":                       "revenue",
    "수익(매출액)":                  "revenue",
    "영업수익":                      "revenue",
    "영업수익(매출액)":              "revenue",
    "매출":                         "revenue",
    "순매출액":                      "revenue",
    "보험영업수익":                  "revenue",
    "이자수익":                      "revenue",
    "순영업수익":                    "revenue",
    "영업총수익":                    "revenue",
    "수수료수익":                    "revenue",
    "수익":                         "revenue",
    "매출원가":                      "cost_of_sales",
    "매출총이익":                    "gross_profit",
    "판매비와관리비":                 "sga",
    "판매비와 관리비":                "sga",
    "판관비":                        "sga",
    "영업이익":                      "operating_profit",
    "영업이익(손실)":                 "operating_profit",
    "영업손익":                      "operating_profit",
    "당기순이익":                    "net_income",
    "당기순이익(손실)":              "net_income",
    "당기순손익":                    "net_income",
    "당기순이익(당기순손실)":         "net_income",
    "지배기업 소유주 지분 순이익":    "net_income",
    "지배지분순이익":                "net_income",
    "지배주주지분순이익":            "net_income",
    "지배기업주주지분순이익":        "net_income",
    "자산총계":                      "total_assets",
    "총자산":                        "total_assets",
    "부채총계":                      "total_liabilities",
    "총부채":                        "total_liabilities",
    "자본총계":                      "equity",
    "총자본":                        "equity",
    "지배기업 소유주지분":           "equity",
    "현금및현금성자산":              "cash",
    "현금 및 현금성자산":            "cash",
}


# ─────────────────────────────────────────────────────────────
# 기업 목록 캐시 (corpCode.xml → stock_code → corp_code 변환용)
# ─────────────────────────────────────────────────────────────

def _load_corp_list(api_key: str) -> list:
    cache: dict = {}
    if os.path.exists(CORP_CACHE_PATH):
        try:
            with open(CORP_CACHE_PATH, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    cached_at_str = cache.get("cached_at", "2000-01-01")
    try:
        cached_at = datetime.fromisoformat(cached_at_str)
    except ValueError:
        cached_at = datetime(2000, 1, 1)

    if cache.get("corps") and datetime.now() - cached_at < timedelta(hours=CORP_CACHE_TTL_H):
        corps = cache["corps"]
        log.info("기업 목록 캐시 로드: %d개 (캐시: %s)",
                 len(corps), cached_at.strftime("%Y-%m-%d %H:%M"))
        return corps

    log.info("corpCode.xml 다운로드 중...")
    resp = requests.get(f"{DART_API_BASE}/corpCode.xml",
                        params={"crtfc_key": api_key}, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

    root = ET.fromstring(xml_bytes)
    corps = [
        {
            "corp_code":  (item.findtext("corp_code")  or "").strip(),
            "corp_name":  (item.findtext("corp_name")  or "").strip(),
            "stock_code": (item.findtext("stock_code") or "").strip(),
        }
        for item in root.findall("list")
        if (item.findtext("stock_code") or "").strip()
    ]
    log.info("corpCode.xml 파싱 완료: 상장 법인 %d개", len(corps))

    os.makedirs(os.path.dirname(CORP_CACHE_PATH), exist_ok=True)
    with open(CORP_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"corps": corps, "cached_at": datetime.now().isoformat()},
                  f, ensure_ascii=False)
    return corps


# ─────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────

def _parse_amount(amount_str: str) -> Optional[int]:
    s = str(amount_str).replace(",", "").strip()
    if not s or s in ("-", "—", "", "nan", "None", "N/A"):
        return None
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _fill_missing_non_fin(data: dict) -> dict:
    d = dict(data)
    r   = d.get("revenue")          or 0
    cos = d.get("cost_of_sales")    or 0
    gp  = d.get("gross_profit")     or 0
    op  = d.get("operating_profit") or 0
    if not gp and r and cos:
        d["gross_profit"] = r - cos
        gp = d["gross_profit"]
    if not d.get("sga"):
        if gp and op:
            d["sga"] = gp - op
        elif r and op:
            d["sga"] = r - op
    return d


# ─────────────────────────────────────────────────────────────
# DART API
# ─────────────────────────────────────────────────────────────

def _fetch_fs(corp_code: str, year: int, api_key: str,
              fs_div: str = FS_DIV_CFS) -> list:
    params = {
        "crtfc_key": api_key, "corp_code": corp_code,
        "bsns_year": str(year), "reprt_code": REPRT_CODE_A, "fs_div": fs_div,
    }
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(FS_ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "000":
                return data.get("list", [])
            if status == "020":
                log.warning("DART 호출 제한(020) corp=%s year=%d fs=%s — 10초 대기",
                            corp_code, year, fs_div)
                time.sleep(10)
                continue
            if status == "013":
                log.debug("  데이터없음(013) corp=%s year=%d fs=%s", corp_code, year, fs_div)
            else:
                log.info("  DART status=%s msg=%s corp=%s year=%d fs=%s",
                         status, data.get("message", ""), corp_code, year, fs_div)
            return []
        except requests.exceptions.RequestException as exc:
            log.warning("API 호출 실패 [%d/%d] corp=%s year=%d: %s",
                        attempt, RETRY_COUNT, corp_code, year, exc)
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
    return []


def _parse_items(items: list, year: int,
                 corp_name: str = "", diag: bool = False) -> dict:
    result: dict = {year: {}, year - 1: {}, year - 2: {}}
    matched: set = set()
    unmatched: list = []

    for item in items:
        a_id = str(item.get("account_id", "")).strip()
        a_nm = str(item.get("account_nm", "")).strip()
        key  = ALL_CONCEPT_MAP.get(a_id) or ALL_NM_MAP.get(a_nm)
        if not key:
            if diag and a_nm and len(unmatched) < 6:
                unmatched.append(f"  id={a_id!r} nm={a_nm!r}")
            continue
        matched.add(key)
        for col, yr in [("thstrm_amount", year),
                         ("frmtrm_amount", year - 1),
                         ("bfefrmtrm_amount", year - 2)]:
            val = _parse_amount(item.get(col))
            if val is not None and yr in result and key not in result[yr]:
                result[yr][key] = val

    if diag:
        if matched:
            log.info("  [진단] %s 매핑: %s", corp_name, sorted(matched))
        else:
            log.warning("  [진단] %s 매핑 없음 (항목 %d개)", corp_name, len(items))
            for s in unmatched:
                log.warning(s)
    return result


def _collect_one(corp_code: str, corp_name: str, years: list,
                 api_key: str, diag: bool = False) -> dict:
    """연도별 직접 호출: CFS 우선, 없으면 OFS."""
    extracted  = {y: {} for y in years}
    _diag_done = [False]

    for year in sorted(years):
        items: list = []
        for fs_div in (FS_DIV_CFS, FS_DIV_OFS):
            items = _fetch_fs(corp_code, year, api_key, fs_div)
            if items:
                break

        if not items:
            continue

        run_diag = diag and not _diag_done[0]
        if run_diag:
            _diag_done[0] = True

        parsed = _parse_items(items, year, corp_name, run_diag)
        for yr, fields in parsed.items():
            if yr not in extracted:
                continue
            for k, v in fields.items():
                if k not in extracted[yr]:
                    extracted[yr][k] = v

    return extracted


# ─────────────────────────────────────────────────────────────
# 메인 수집
# ─────────────────────────────────────────────────────────────

def collect_all(
    stock_codes: Optional[list] = None,
    limit: Optional[int]        = None,
    start_year: int             = START_YEAR,
    end_year: int               = END_YEAR,
    save_dir: str               = "data",
    diag: bool                  = False,
) -> dict:
    """
    대기업·중견기업 재무제표 수집.

    기본 대상: TARGET_STOCK_CODES (하드코딩 ~80개 대기업)
    --codes   : 지정 종목만
    --limit   : TARGET_STOCK_CODES 앞 N개
    연도 범위  : start_year ~ end_year (기본 2010~2024)
    """
    api_key = os.getenv("DART_API_KEY", "")
    if not api_key or api_key == "your_dart_api_key_here":
        raise ValueError(
            "DART_API_KEY 가 설정되지 않았습니다.\n"
            ".env 에서 DART_API_KEY=실제인증키 로 수정하세요."
        )

    years = list(range(start_year, end_year + 1))
    log.info("수집 연도 범위: %d ~ %d (%d개년)", years[0], years[-1], len(years))

    all_corps = _load_corp_list(api_key)
    corp_map  = {c["stock_code"]: c for c in all_corps}

    # 수집 대상 결정
    if stock_codes:
        target_codes = stock_codes
    else:
        target_codes = TARGET_STOCK_CODES

    if limit:
        target_codes = target_codes[:limit]

    candidates = []
    for code in target_codes:
        info = corp_map.get(code)
        if info:
            candidates.append(info)
        else:
            log.warning("종목코드 %s: corpCode.xml 에 없음 → 스킵", code)

    log.info("수집 대상: %d개 기업 × %d개년 = 최대 %d회 API 호출",
             len(candidates), len(years), len(candidates) * len(years) * 2)

    # chatgpt4.txt: 단일 CSV + is_financial 컬럼 통합 스키마
    # company, year, is_financial, revenue, operating_profit, net_income,
    # total_assets, total_liabilities, equity, cash
    all_rows: list = []

    for i, corp in enumerate(candidates, 1):
        corp_code  = corp["corp_code"]
        corp_name  = corp["corp_name"]
        stock_code = corp["stock_code"]
        # 종목코드 집합으로 금융사 판별 (회사명 키워드 방식 오분류 해결)
        is_fin = stock_code in FINANCIAL_STOCK_CODES

        log.info("[%d/%d] %s (%s) [%s]",
                 i, len(candidates), corp_name, corp["stock_code"],
                 "금융" if is_fin else "비금융")

        try:
            extracted = _collect_one(corp_code, corp_name, years, api_key, diag=diag)
        except Exception as exc:
            log.warning("  %s 수집 예외: %s", corp_name, exc)
            extracted = {}

        saved_years = 0
        for year in years:
            year_data = extracted.get(year, {})
            if not year_data:
                continue

            base = {
                "company":      corp_name,
                "year":         year,
                "is_financial": 1 if is_fin else 0,
                "sector":       SECTOR_MAP.get(stock_code, "기타"),
            }

            if is_fin:
                ta  = year_data.get("total_assets")
                eq  = year_data.get("equity")
                ni  = year_data.get("net_income")
                if ta is None or eq is None or ni is None:
                    continue
                if abs(ta) < MIN_ASSETS:
                    continue
                all_rows.append({
                    **base,
                    "revenue":           None,   # 금융업: 매출 개념 없음
                    "operating_profit":  None,
                    "net_income":        ni,
                    "total_assets":      ta,
                    "total_liabilities": year_data.get("total_liabilities"),
                    "equity":            eq,
                    "cash":              year_data.get("cash"),
                })
                saved_years += 1
            else:
                year_data = _fill_missing_non_fin(year_data)
                revenue   = year_data.get("revenue")
                op        = year_data.get("operating_profit")
                if revenue is None:
                    log.debug("  %s %d: revenue=None", corp_name, year)
                    continue
                if abs(revenue) < MIN_REVENUE:
                    continue
                tl = year_data.get("total_liabilities")
                eq = year_data.get("equity")
                all_rows.append({
                    **base,
                    "revenue":           revenue,
                    "operating_profit":  op if op is not None else 0,
                    "net_income":        year_data.get("net_income"),
                    "total_assets":      (tl + eq) if tl and eq else None,
                    "total_liabilities": tl,
                    "equity":            eq,
                    "cash":              year_data.get("cash"),
                })
                saved_years += 1

        log.info("  → %d/%d개년 저장", saved_years, len(years))
        time.sleep(REQUEST_DELAY)

    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train.csv")

    _save_csv(all_rows, train_path, ["company", "year"])

    n_fin    = sum(1 for r in all_rows if r.get("is_financial") == 1)
    n_nonfin = sum(1 for r in all_rows if r.get("is_financial") == 0)

    if os.path.exists(train_path):
        total = len(pd.read_csv(train_path))
    else:
        total = 0

    log.info("수집 완료 — 비금융 %d행 / 금융 %d행 / 합계 %d행 → %s",
             n_nonfin, n_fin, total, train_path)
    return {"n_non_financial": n_nonfin, "n_financial": n_fin}


def _save_csv(rows: list, path: str, dedup_keys: list) -> None:
    if not rows:
        log.warning("저장할 데이터 없음: %s", path)
        return
    new_df = pd.DataFrame(rows)

    # 스키마 불일치 감지: is_financial 컬럼 없는 구버전 파일은 리셋
    if os.path.exists(path):
        existing = pd.read_csv(path)
        new_cols = set(new_df.columns)
        old_cols = set(existing.columns)
        if "is_financial" in new_cols and "is_financial" not in old_cols:
            log.warning("기존 %s 스키마 불일치(구버전) → 새 데이터로 교체", path)
            combined = new_df
        else:
            combined = (
                pd.concat([new_df, existing], ignore_index=True)  # new 우선
                .drop_duplicates(subset=dedup_keys, keep="first")
                .sort_values(dedup_keys)
                .reset_index(drop=True)
            )
    else:
        combined = new_df

    combined.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("저장 완료: %s (%d행)", path, len(combined))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="DART 재무제표 수집기 (대기업·중견기업)")
    parser.add_argument("--codes", nargs="*", default=None,
                        help="종목코드 직접 지정 (예: 005930 000660)")
    parser.add_argument("--limit", type=int, default=None,
                        help="대기업 목록에서 앞 N개만 수집")
    parser.add_argument("--start-year", type=int, default=START_YEAR,
                        help=f"수집 시작 연도 (기본: {START_YEAR})")
    parser.add_argument("--end-year", type=int, default=END_YEAR,
                        help=f"수집 종료 연도 (기본: {END_YEAR})")
    parser.add_argument("--save-dir", default="data",
                        help="CSV 저장 디렉터리 (기본: data)")
    parser.add_argument("--diag", action="store_true",
                        help="계정과목 매핑 진단 로그")
    args = parser.parse_args()

    result = collect_all(
        stock_codes=args.codes,
        limit=args.limit,
        start_year=args.start_year,
        end_year=args.end_year,
        save_dir=args.save_dir,
        diag=args.diag,
    )
    print(f"\n수집 결과: 비금융 {result['n_non_financial']}행 / "
          f"금융 {result['n_financial']}행")


if __name__ == "__main__":
    main()
