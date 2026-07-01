from krx_client import stock
import pandas as pd
from datetime import datetime, timedelta


def get_recent_business_day() -> str:
    today = datetime.today()
    for delta in range(7):
        day = today - timedelta(days=delta)
        if day.weekday() < 5:  # 월~금
            return day.strftime("%Y%m%d")


def get_kospi_list(date: str = None) -> pd.DataFrame:
    """
    KRX 공식 API로 코스피 종목 리스트를 가져옵니다.
    date: 'YYYYMMDD' 형식. 기본값은 최근 영업일.
    """
    if date is None:
        date = get_recent_business_day()

    tickers = stock.get_market_ticker_list(date, market="KOSPI")

    rows = []
    for ticker in tickers:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"종목코드": ticker, "종목명": name})

    df = pd.DataFrame(rows)
    return df, date


if __name__ == "__main__":
    print("코스피 종목 리스트 조회 중 (KRX 공식 API)...")
    df, date = get_kospi_list()

    print(f"기준일: {date}")
    print(f"총 {len(df)}개 종목\n")
    print(df.to_string(index=False))

    output_path = f"kospi_list_{date}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {output_path}")
