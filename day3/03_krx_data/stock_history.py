from krx_client import stock
import pandas as pd
from datetime import datetime


def get_stock_history(ticker: str, start: str = "20140101", end: str = None) -> pd.DataFrame:
    """
    KRX 공식 API로 종목의 일별 OHLCV를 가져옵니다.
    ticker: 종목코드 (예: '005930')
    start:  시작일 'YYYYMMDD' (KRX 제공 최초: 2014-04-07)
    end:    종료일 'YYYYMMDD', 기본값은 오늘
    """
    if end is None:
        end = datetime.today().strftime("%Y%m%d")

    df = stock.get_market_ohlcv(start, end, ticker)
    df.index.name = "날짜"
    return df


if __name__ == "__main__":
    TICKER = "005930"

    name = stock.get_market_ticker_name(TICKER)
    print(f"{name}({TICKER}) 일별 거래내역 조회 중 (KRX 공식 API)...")

    df = get_stock_history(TICKER)

    print(f"\n기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"총 {len(df):,}거래일\n")

    print("=== 최초 5일 ===")
    print(df.head())
    print("\n=== 최근 5일 ===")
    print(df.tail())

    today = datetime.today().strftime("%Y%m%d")
    output_path = f"{TICKER}_{name}_history_{today}.csv"
    df.to_csv(output_path, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {output_path}")
