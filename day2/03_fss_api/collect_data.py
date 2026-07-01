"""
KOSPI200 종목 주가 데이터 수집 스크립트 (공공데이터포털 FSS API 버전)
data/kospi200_list.csv → data/stock_prices.csv

FSS API 특성: 종목별 기간 조회, 최대 100건/페이지
데이터 제공 범위: 2000년 이후
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

API_KEY    = os.getenv("FSS_API_KEY", "")
BASE_URL   = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
LIST_CSV   = os.path.join(DATA_DIR, "kospi200_list.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "stock_prices.csv")
START_DATE = "20000101"
PAGE_ROWS  = 100

NUM_COLS = ["clpr", "vs", "fltRt", "mkp", "hipr", "lopr", "trqu", "trPrc", "lstgStCnt", "mrktTotAmt"]


def fetch_stock_range(ticker: str, begin: str, end: str) -> pd.DataFrame:
    """FSS API - 종목별 기간 주가 조회 (페이징 처리)"""
    all_items = []
    page = 1

    while True:
        params = {
            "serviceKey": API_KEY,
            "numOfRows":  str(PAGE_ROWS),
            "pageNo":     str(page),
            "resultType": "json",
            "beginBasDt": begin,
            "endBasDt":   end,
            "likeSrtnCd": ticker,
        }
        resp = requests.get(
            f"{BASE_URL}/getStockPriceInfo",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()

        body  = resp.json().get("response", {}).get("body", {})
        total = int(body.get("totalCount", 0))
        items = body.get("items", {})

        if not items or total == 0:
            break

        item_list = items.get("item", [])
        if isinstance(item_list, dict):
            item_list = [item_list]

        all_items.extend(item_list)

        if len(all_items) >= total:
            break

        page += 1
        time.sleep(0.1)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    if "srtnCd" in df.columns:
        df["srtnCd"] = df["srtnCd"].astype(str).str.zfill(6)

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    if not API_KEY:
        print(".env 파일에 FSS_API_KEY 를 입력하세요")
        sys.exit(1)

    if not os.path.exists(LIST_CSV):
        print("data/kospi200_list.csv 파일이 없습니다.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    # KOSPI200 종목 리스트
    stocks = pd.read_csv(LIST_CSV, encoding="utf-8-sig", dtype=str)
    stocks["srtnCd"] = stocks["srtnCd"].astype(str).str.zfill(6)
    print(f"KOSPI200 종목 수: {len(stocks)}개")

    # 기존 데이터 로드
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig", dtype={"srtnCd": str, "basDt": str})
        existing["srtnCd"] = existing["srtnCd"].astype(str).str.zfill(6)
        existing_keys = set(zip(existing["basDt"], existing["srtnCd"]))
        existing_by_code = existing.groupby("srtnCd")["basDt"].apply(set).to_dict()
        print(f"기존 데이터 {len(existing):,}건 로드 "
              f"(범위: {existing['basDt'].min()} ~ {existing['basDt'].max()})")
    else:
        existing = pd.DataFrame()
        existing_keys = set()
        existing_by_code = {}
        print("기존 데이터 없음. 전체 수집 시작")

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    print(f"수집 대상 기간: {START_DATE} ~ {yesterday}\n")

    new_frames = []
    total_stocks = len(stocks)

    for i, row in enumerate(stocks.itertuples(index=False), start=1):
        code = row.srtnCd
        name = row.itmsNm

        owned = existing_by_code.get(code, set())

        # 수집이 필요한 구간 계산 (누락 구간만)
        fetch_ranges = []
        if not owned:
            fetch_ranges.append((START_DATE, yesterday))
        else:
            min_owned = min(owned)
            max_owned = max(owned)
            if min_owned > START_DATE:
                prev = (datetime.strptime(min_owned, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
                fetch_ranges.append((START_DATE, prev))
            if max_owned < yesterday:
                nxt = (datetime.strptime(max_owned, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
                fetch_ranges.append((nxt, yesterday))

        if not fetch_ranges:
            print(f"[{i}/{total_stocks}] {name}({code}) - 최신 상태")
            continue

        range_desc = ", ".join(f"{b}~{e}" for b, e in fetch_ranges)
        print(f"[{i}/{total_stocks}] {name}({code}) 조회 중... ({range_desc})", end=" ", flush=True)

        stock_new = []
        try:
            for begin, end in fetch_ranges:
                df = fetch_stock_range(code, begin, end)
                if df.empty:
                    continue
                # 이미 보유한 (날짜, 종목) 제거
                mask = pd.Series(
                    list(zip(df["basDt"], df["srtnCd"]))
                ).isin(existing_keys).values
                df = df[~mask]
                if not df.empty:
                    stock_new.append(df)

            if not stock_new:
                print("신규 없음")
            else:
                combined = pd.concat(stock_new, ignore_index=True)
                new_frames.append(combined)
                existing_keys.update(zip(combined["basDt"], combined["srtnCd"]))
                print(f"{len(combined)}건 수집")

        except Exception as e:
            print(f"오류: {e}")

        time.sleep(0.3)

    if not new_frames:
        print("\n새로운 데이터가 없습니다.")
        return

    # 기존 + 신규 합치기
    all_frames = ([existing] if not existing.empty else []) + new_frames
    result = pd.concat(all_frames, ignore_index=True)
    result["srtnCd"] = result["srtnCd"].astype(str).str.zfill(6)
    result = result.drop_duplicates(subset=["basDt", "srtnCd"])
    result = result.sort_values(["srtnCd", "basDt"]).reset_index(drop=True)

    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n완료: 총 {len(result):,}건 → {OUTPUT_CSV} 저장")


if __name__ == "__main__":
    main()
