"""
로컬 CSV 기반 주식 캔들차트 & 기술지표 Streamlit 앱
API 불필요 — data/ 폴더의 CSV 파일을 직접 읽습니다.
실행: streamlit run app_local.py
데이터 갱신: python download_data.py
"""

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

POPULAR_STOCKS = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "NAVER": "035420",
    "카카오": "035720",
    "현대차": "005380",
}


# ── 데이터 로드 ────────────────────────────────────────────────────────────

@st.cache_data
def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def available_tickers() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return [f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]


# ── 기술지표 계산 ──────────────────────────────────────────────────────────

def calc_ma(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    for p in periods:
        df[f"MA{p}"] = df["close"].rolling(p).mean()
    return df


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── 차트 빌더 ──────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, show_ma: dict, indicator: str) -> go.Figure:
    n_rows      = 3 if indicator != "없음" else 2
    row_heights = [0.55, 0.15, 0.30] if n_rows == 3 else [0.70, 0.30]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    up_color   = "#FF3B3B"
    down_color = "#0066FF"
    candle_colors = [up_color if c >= o else down_color for o, c in zip(df["open"], df["close"])]

    # 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
            increasing_fillcolor=up_color,
            decreasing_fillcolor=down_color,
            name="캔들",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 이동평균선
    ma_styles = {
        "MA5":  ("MA5",  "#FFD700"),
        "MA20": ("MA20", "#00CC44"),
        "MA60": ("MA60", "#9B59B6"),
    }
    for key, (col, color) in ma_styles.items():
        if show_ma.get(key) and col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[col], name=key,
                           line=dict(color=color, width=1.2)),
                row=1, col=1,
            )

    # 거래량
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"],
               marker_color=candle_colors, name="거래량", showlegend=False),
        row=2, col=1,
    )

    # 보조지표
    if indicator == "RSI" and n_rows == 3:
        rsi = calc_rsi(df["close"])
        fig.add_trace(
            go.Scatter(x=df["date"], y=rsi, name="RSI(14)",
                       line=dict(color="#F39C12", width=1.5)),
            row=3, col=1,
        )
        for level, color in [(70, "rgba(255,59,59,0.4)"), (30, "rgba(0,102,255,0.4)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)

    elif indicator == "MACD" and n_rows == 3:
        macd_line, signal_line, histogram = calc_macd(df["close"])
        hist_colors = [up_color if v >= 0 else down_color for v in histogram]
        fig.add_trace(
            go.Bar(x=df["date"], y=histogram, name="히스토그램",
                   marker_color=hist_colors, showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=macd_line, name="MACD",
                       line=dict(color="#F39C12", width=1.3)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=signal_line, name="Signal",
                       line=dict(color="#E74C3C", width=1.3)),
            row=3, col=1,
        )

    fig.update_layout(
        height=700,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    for i in range(1, n_rows + 1):
        fig.update_yaxes(gridcolor="#2A2A2A", row=i, col=1)
        fig.update_xaxes(gridcolor="#2A2A2A", row=i, col=1)

    return fig


# ── 기술적 분석 요약 ────────────────────────────────────────────────────────

def show_summary(df: pd.DataFrame) -> None:
    rsi = calc_rsi(df["close"]).iloc[-1]
    macd_line, signal_line, _ = calc_macd(df["close"])
    last_macd   = macd_line.iloc[-1]
    last_signal = signal_line.iloc[-1]
    prev_macd   = macd_line.iloc[-2] if len(macd_line) > 1 else last_macd
    prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else last_signal

    close = df["close"].iloc[-1]
    ma5   = df["MA5"].iloc[-1]  if "MA5"  in df.columns else None
    ma20  = df["MA20"].iloc[-1] if "MA20" in df.columns else None
    ma60  = df["MA60"].iloc[-1] if "MA60" in df.columns else None

    if rsi >= 70:
        rsi_signal, rsi_color = "과매수 ⚠️", "#FF3B3B"
    elif rsi <= 30:
        rsi_signal, rsi_color = "과매도 ⚠️", "#0066FF"
    else:
        rsi_signal, rsi_color = "중립", "#888888"

    valid_mas = {k: v for k, v in {"MA5": ma5, "MA20": ma20, "MA60": ma60}.items() if v is not None}
    if len(valid_mas) >= 2:
        values = list(valid_mas.values())
        if all(values[i] > values[i + 1] for i in range(len(values) - 1)):
            ma_signal, ma_color = "정배열 📈", "#FF3B3B"
        elif all(values[i] < values[i + 1] for i in range(len(values) - 1)):
            ma_signal, ma_color = "역배열 📉", "#0066FF"
        else:
            ma_signal, ma_color = "혼조", "#888888"
    else:
        ma_signal, ma_color = "-", "#888888"

    golden = (prev_macd <= prev_signal) and (last_macd > last_signal)
    dead   = (prev_macd >= prev_signal) and (last_macd < last_signal)
    if golden:
        macd_signal, macd_color = "골든크로스 🌟", "#FF3B3B"
    elif dead:
        macd_signal, macd_color = "데드크로스 ☠️", "#0066FF"
    elif last_macd > last_signal:
        macd_signal, macd_color = "MACD > Signal", "#FF8C00"
    else:
        macd_signal, macd_color = "MACD < Signal", "#888888"

    st.markdown("### 기술적 분석 요약")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("RSI (14일)", f"{rsi:.1f}")
        st.markdown(f"<span style='color:{rsi_color};font-weight:bold'>{rsi_signal}</span>",
                    unsafe_allow_html=True)
    with c2:
        st.metric("이동평균 배열", "")
        st.markdown(f"<span style='color:{ma_color};font-weight:bold'>{ma_signal}</span>",
                    unsafe_allow_html=True)
    with c3:
        st.metric("MACD 방향", "")
        st.markdown(f"<span style='color:{macd_color};font-weight:bold'>{macd_signal}</span>",
                    unsafe_allow_html=True)

    # 마지막 가격 표시
    st.divider()
    st.markdown(f"**최근 종가:** {close:,.0f}원")


# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="📊 주식 차트 분석 (로컬)", layout="wide")
st.title("📊 주식 차트 분석 (로컬 데이터)")

tickers = available_tickers()

if not tickers:
    st.error("❌ data/ 폴더에 CSV 파일이 없습니다. 먼저 `python download_data.py`를 실행하세요.")
    st.stop()

# ── 사이드바 ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("종목 선택")

    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "005930"

    ticker_input = st.text_input("종목코드", value=st.session_state["ticker"], max_chars=6)

    st.markdown("**인기 종목**")
    cols = st.columns(2)
    for i, (name, code) in enumerate(POPULAR_STOCKS.items()):
        disabled = code not in tickers
        label    = name if not disabled else f"{name} (미다운)"
        if cols[i % 2].button(label, key=f"btn_{code}",
                              use_container_width=True, disabled=disabled):
            st.session_state["ticker"] = code
            st.rerun()

    st.divider()

    period_map   = {"60일": 60, "120일": 120, "250일": 250}
    period_label = st.selectbox("조회 기간", list(period_map.keys()), index=1)
    period_days  = period_map[period_label]

    st.divider()

    st.subheader("이동평균선")
    show_ma5  = st.checkbox("MA5  (노란색)", value=True)
    show_ma20 = st.checkbox("MA20 (초록색)", value=True)
    show_ma60 = st.checkbox("MA60 (보라색)", value=False)

    st.divider()

    st.subheader("보조지표")
    indicator = st.radio("보조지표 선택", ["없음", "RSI", "MACD"], index=0)

    st.divider()
    if st.button("데이터 캐시 초기화"):
        st.cache_data.clear()
        st.rerun()

# ── 데이터 로드 & 렌더링 ──────────────────────────────────────────────────

ticker = st.session_state.get("ticker", ticker_input) or ticker_input
if ticker_input != st.session_state.get("ticker"):
    ticker = ticker_input

if ticker not in tickers:
    st.warning(f"⚠️ {ticker}.csv 파일이 없습니다. `python download_data.py`로 데이터를 받아주세요.")
    st.stop()

df = load_ohlcv(ticker)
if df.empty:
    st.info("조회된 데이터가 없습니다.")
    st.stop()

# MA 계산은 전체 기간으로, 화면 표시는 period_days로 자름
df = calc_ma(df, [5, 20, 60])
df_display = df.tail(period_days).reset_index(drop=True)

# 데이터 최신화 일자 표시
csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
mtime    = date.fromtimestamp(os.path.getmtime(csv_path))
st.caption(
    f"**{ticker}** | "
    f"{df_display['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
    f"{df_display['date'].iloc[-1].strftime('%Y-%m-%d')} | "
    f"{len(df_display)}거래일 | 파일 갱신: {mtime}"
)

show_ma = {"MA5": show_ma5, "MA20": show_ma20, "MA60": show_ma60}
fig = build_chart(df_display, show_ma, indicator)
st.plotly_chart(fig, use_container_width=True)

st.divider()
show_summary(df_display)
