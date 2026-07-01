"""
코스피 전 종목 일별 거래내역 수집 → 단일 CSV 저장
"""
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from krx_client import stock

START_DATE = "20140101"
OUTPUT_PATH = "kospi_all_history.csv"


def fetch_ticker_history(ticker: str, name: str, start: str, end: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv(start, end, ticker)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(columns={"날짜": "날짜", "시가": "시가", "고가": "고가",
                        "저가": "저가", "종가": "종가", "거래량": "거래량", "등락률": "등락률"}, inplace=True)
    df.insert(1, "종목코드", ticker)
    df.insert(2, "종목명", name)
    return df


def main():
    end_date = datetime.today().strftime("%Y%m%d")

    print("▶ 코스피 종목 리스트 조회 중...")
    tickers = stock.get_market_ticker_list(end_date, market="KOSPI")
    names = {t: stock.get_market_ticker_name(t) for t in tickers}
    print(f"  총 {len(tickers)}개 종목\n")

    results = []
    failed = []

    for ticker in tqdm(tickers, desc="수집 중", unit="종목"):
        name = names[ticker]
        try:
            df = fetch_ticker_history(ticker, name, START_DATE, end_date)
            if not df.empty:
                results.append(df)
        except Exception as e:
            failed.append((ticker, name, str(e)))
        time.sleep(0.1)  # 서버 부하 방지

    print(f"\n▶ 수집 완료: {len(tickers) - len(failed)}개 성공, {len(failed)}개 실패")

    if failed:
        print("  실패 종목:")
        for t, n, e in failed:
            print(f"    {t} {n}: {e}")

    print(f"\n▶ 데이터 병합 중...")
    all_df = pd.concat(results, ignore_index=True)
    all_df.sort_values(["날짜", "종목코드"], inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    print(f"  총 {len(all_df):,}행 ({all_df['종목코드'].nunique()}개 종목)")
    print(f"  기간: {all_df['날짜'].min().date()} ~ {all_df['날짜'].max().date()}")

    all_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n▶ 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
