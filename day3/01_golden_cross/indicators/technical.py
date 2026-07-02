"""
기술지표 계산 모듈.

입력 DataFrame은 금융위원회 공공데이터 원본 컬럼명(basDt, srtnCd, clpr ...)을
그대로 받으며, 함수 내부에서 표준명(date, ticker, close ...)으로 일괄 변환한다.
이후 모든 모듈은 표준명만 참조한다.

원본 → 표준 컬럼 매핑 (COLUMN_MAP):
    basDt  → date    (int YYYYMMDD → datetime)
    srtnCd → ticker
    itmsNm → name
    mkp    → open
    hipr   → high
    lopr   → low
    clpr   → close
    trqu   → volume

종목별로 그룹화해서 독립적으로 계산하므로 경계 오염 없음.
각 종목의 데이터 시작 시점이 다를 수 있음(신규상장 등) — 종목 간 날짜를
억지로 맞추지 않고 해당 종목의 첫 행부터 지표를 계산한다.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 컬럼 표준화
# ---------------------------------------------------------------------------

COLUMN_MAP: dict[str, str] = {
    # 금융위원회 API 원본 컬럼명
    "basDt" : "date",
    "srtnCd": "ticker",
    "itmsNm": "name",
    "mkp"   : "open",
    "hipr"  : "high",
    "lopr"  : "low",
    "clpr"  : "close",
    "trqu"  : "volume",
    # 한국어 컬럼명 (stock_prices.csv 신규 포맷)
    "날짜"   : "date",
    "종목코드" : "ticker",
    "종목명"  : "name",
    "시가"   : "open",
    "고가"   : "high",
    "저가"   : "low",
    "종가"   : "close",
    "거래량"  : "volume",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 컬럼명 → 표준명 변환. 없는 컬럼은 조용히 건너뜀.
    date 컬럼이 YYYYMMDD(정수/8자리 문자열) 또는 YYYY-MM-DD 모두 처리.
    """
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        date_str = df["date"].astype(str).str.strip()
        if date_str.str.match(r"^\d{8}$").all():
            df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
        else:
            df["date"] = pd.to_datetime(date_str)
    return df


# ---------------------------------------------------------------------------
# 내부 헬퍼 (단일 종목 Series 대상)
# ---------------------------------------------------------------------------

def _sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def _ema(close: pd.Series, span: int) -> pd.Series:
    # adjust=False → Wilder/EMA 계열과 동일한 재귀식
    return close.ewm(span=span, adjust=False).mean()


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's smoothing RSI.
    - ewm(alpha=1/period, adjust=False) 로 시드 없이 처음부터 지수 스무딩.
    - 분모(평균 손실) == 0 이면 RSI = 100 처리.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)   # 분모 0 → NaN
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)           # 완전 상승 구간 → 100
    return rsi


def _cross_flag(sma_fast: pd.Series, sma_slow: pd.Series,
                window: int, direction: str) -> pd.Series:
    """
    직전 window 캔들 내에 골든(direction='up') 또는 데드(direction='down')
    크로스가 발생했으면 1, 아니면 0.

    크로스 조건 (해당 바 기준):
        up   : 현재 fast > slow  AND  전일 fast <= slow
        down : 현재 fast < slow  AND  전일 fast >= slow
    """
    if direction == "up":
        cross = ((sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))).astype(int)
    else:
        cross = ((sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))).astype(int)

    return cross.rolling(window, min_periods=1).max().astype(int)


# ---------------------------------------------------------------------------
# 단일 종목 계산 (내부용, 표준명 컬럼 전제)
# ---------------------------------------------------------------------------

def _compute_single(df: pd.DataFrame, cross_window: int) -> pd.DataFrame:
    close = df["close"].astype(float)   # 표준명 사용

    sma5  = _sma(close, 5)
    sma20 = _sma(close, 20)
    sma60 = _sma(close, 60)
    ema20 = _ema(close, 20)
    rsi14 = _rsi_wilder(close, 14)

    df = df.copy()
    df["sma5"]  = sma5
    df["sma20"] = sma20
    df["sma60"] = sma60
    df["ema20"] = ema20

    df["golden_flag"] = _cross_flag(sma5, sma20, cross_window, "up")
    df["dead_flag"]   = _cross_flag(sma5, sma20, cross_window, "down")

    df["disparity20"]          = (close / sma20) * 100
    df["disparity20_centered"] = df["disparity20"] - 100

    df["rsi14"]      = rsi14
    df["rsi14_norm"] = (rsi14 - 50) / 50

    return df


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def compute_indicators(
    df          : pd.DataFrame,
    cross_window: int = 5,
    nan_policy  : str = "keep",
) -> pd.DataFrame:
    """
    종목별로 기술지표를 계산해 추가한 DataFrame을 반환한다.

    Parameters
    ----------
    df : pd.DataFrame
        금융위원회 원본 컬럼(basDt, srtnCd, clpr ...) 또는
        이미 표준화된 컬럼(date, ticker, close ...) 모두 허용.
        date(basDt) 오름차순 정렬 전제.
    cross_window : int
        골든/데드 크로스 플래그를 몇 캔들 동안 유지할지 (기본 5).
    nan_policy : {"keep", "drop", "backfill"}
        초기 NaN 구간 처리 방식.
        - "keep"     : 그대로 둠 (기본)
        - "drop"     : sma60 · rsi14 기준으로 NaN 행 제거
        - "backfill" : 뒤 값으로 앞을 채움 (bfill)

    Returns
    -------
    pd.DataFrame
        표준 컬럼(date, ticker, close ...) +
        sma5/20/60, ema20, golden_flag, dead_flag,
        disparity20, disparity20_centered, rsi14, rsi14_norm
    """
    if df.empty:
        return df.copy()

    df = _standardize_columns(df)

    parts = [
        _compute_single(group.sort_values("date"), cross_window)
        for _, group in df.groupby("ticker", sort=False)
    ]
    result = pd.concat(parts, ignore_index=False)

    if nan_policy == "drop":
        result = result.dropna(subset=["sma60", "rsi14"])
    elif nan_policy == "backfill":
        indicator_cols = [
            "sma5", "sma20", "sma60", "ema20",
            "golden_flag", "dead_flag",
            "disparity20", "disparity20_centered",
            "rsi14", "rsi14_norm",
        ]
        result[indicator_cols] = result[indicator_cols].bfill()

    return result.reset_index(drop=True)
