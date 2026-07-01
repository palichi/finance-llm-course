"""
Yahoo Finance에서 주식 OHLCV 데이터를 다운로드하여 data/ 폴더에 CSV로 저장.
실행: python download_data.py
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# 종목코드 → Yahoo Finance 티커 매핑 (KRX 종목은 .KS 또는 .KQ 접미사)
STOCKS = {
    "005930": ("삼성전자",  "005930.KS"),
    "000660": ("SK하이닉스", "000660.KS"),
    "035420": ("NAVER",    "035420.KS"),
    "035720": ("카카오",    "035720.KS"),
    "005380": ("현대차",    "005380.KS"),
}

DAYS = 400   # MA60 계산 여유분 포함해 충분한 기간
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download(yf_ticker: str, name: str) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=DAYS)
    raw   = yf.download(yf_ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if raw.empty:
        print(f"  [경고] {name}({yf_ticker}) 데이터 없음")
        return pd.DataFrame()

    # yfinance 0.2+ 는 MultiIndex 반환 가능 → 단일 레벨로 정리
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "date"
    df.columns    = ["open", "high", "low", "close", "volume"]
    df = df.dropna().reset_index()
    df["date"]   = pd.to_datetime(df["date"]).dt.date
    # 한국 주식 가격은 원 단위 정수
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].round(0).astype(int)
    df["volume"] = df["volume"].astype(int)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for code, (name, yf_ticker) in STOCKS.items():
        print(f"다운로드: {name} ({code}) ← {yf_ticker}")
        df = download(yf_ticker, name)
        if df.empty:
            continue
        path = os.path.join(DATA_DIR, f"{code}.csv")
        df.to_csv(path, index=False)
        print(f"  저장: {path}  ({len(df)}행, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")

    print("\n완료. data/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
