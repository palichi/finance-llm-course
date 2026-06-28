"""
pykrx를 사용해 KOSPI 200 구성종목을 data/data.csv로 저장하는 스크립트
make_kospi200.py의 입력 파일(data/data.csv)을 생성한다.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT    = os.path.join(DATA_DIR, "data.csv")


def get_recent_business_day() -> str:
    """오늘 또는 가장 최근 영업일을 YYYYMMDD 형식으로 반환."""
    date = datetime.today()
    # 토요일(5), 일요일(6)이면 금요일로 되돌림
    while date.weekday() >= 5:
        date -= timedelta(days=1)
    return date.strftime("%Y%m%d")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    base_date = get_recent_business_day()
    print(f"기준일: {base_date}")

    # KOSPI 200 구성종목 조회
    tickers = stock.get_index_portfolio_deposit_file("1028", date=base_date)
    print(f"조회된 종목 수: {len(tickers)}")

    # 종목명 조회
    rows = []
    for ticker in tickers:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"종목코드": ticker, "종목명": name})

    df = pd.DataFrame(rows)
    df["종목코드"] = df["종목코드"].str.zfill(6)

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT}  ({len(df)}개 종목)")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
