"""
PPO 매매 판단 대시보드.

실행:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH  = ROOT / "../../day2/03_fss_api/data/stock_prices.csv"
MODEL_PATH = ROOT / "models/20260701_221527/best_model.zip"
LOOKBACK_CHART = 252  # 차트에 표시할 최근 N일 (약 1년)

st.set_page_config(page_title="PPO 매매 판단", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# 캐시 리소스 (1회 로드)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_raw_df():
    import pandas as pd
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig",
                     dtype={"srtnCd": str, "종목코드": str})
    code_col = "srtnCd" if "srtnCd" in df.columns else "종목코드"
    name_col = "itmsNm" if "itmsNm" in df.columns else "종목명"
    df[code_col] = df[code_col].astype(str).str.zfill(6)
    df.attrs["code_col"] = code_col
    df.attrs["name_col"] = name_col
    return df


@st.cache_resource
def _load_model():
    from stable_baselines3 import PPO
    return PPO.load(str(MODEL_PATH))


# ---------------------------------------------------------------------------
# 종목 검색
# ---------------------------------------------------------------------------

def _search_ticker(query: str, raw_df) -> list[tuple[str, str]]:
    """쿼리로 (ticker, name) 목록 반환. 6자리 숫자면 코드 직접 매칭."""
    q = query.strip()
    if not q:
        return []
    code_col = raw_df.attrs.get("code_col", "srtnCd")
    name_col = raw_df.attrs.get("name_col", "itmsNm")
    if q.isdigit():
        q = q.zfill(6)
        rows = raw_df[raw_df[code_col] == q]
        return [(r[code_col], r[name_col]) for _, r in rows.iterrows()]
    mask = raw_df[name_col].str.contains(q, na=False)
    dedup = raw_df[mask].drop_duplicates(code_col)
    return [(r[code_col], r[name_col]) for _, r in dedup.iterrows()]


# ---------------------------------------------------------------------------
# 추론 (종목 바뀔 때만 실행)
# ---------------------------------------------------------------------------

def _run_pipeline(ticker: str):
    from inference.predict import predict
    from explain.rule_based import explain as rb_explain
    from explain.llm_explainer import generate_explanation

    result    = predict(ticker, model_path=MODEL_PATH, data_path=DATA_PATH)
    explain_r = rb_explain(result)

    # RAG (corpus 있으면 사용, 없으면 빈 리스트)
    rag_results = []
    try:
        from explain.rag_retriever import retrieve
        rag_results = retrieve(explain_r, ticker=ticker)
    except Exception:
        pass

    explanation = generate_explanation(explain_r, rag_results)

    return result, explain_r, rag_results, explanation


# ---------------------------------------------------------------------------
# 차트: 최근 N일 종가 + 이동평균 + 골든/데드크로스 마커
# ---------------------------------------------------------------------------

def _plot_chart(ticker: str, raw_df, n: int = LOOKBACK_CHART):
    import plotly.graph_objects as go
    from indicators.technical import compute_indicators
    import warnings

    code_col = raw_df.attrs.get("code_col", "srtnCd")
    grp = raw_df[raw_df[code_col] == ticker].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = compute_indicators(grp, nan_policy="drop")

    if df.empty:
        return None

    df = df.tail(n).copy()

    fig = go.Figure()

    # 종가
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        name="종가", line=dict(color="#1f77b4", width=2),
    ))
    # SMA5
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sma5"],
        name="SMA5", line=dict(color="#ff7f0e", width=1, dash="dot"),
    ))
    # SMA20
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sma20"],
        name="SMA20", line=dict(color="#2ca02c", width=1, dash="dash"),
    ))
    # SMA60
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sma60"],
        name="SMA60", line=dict(color="#d62728", width=1, dash="longdash"),
    ))

    # 골든크로스 마커 (매수 시그널)
    golden = df[df["golden_flag"] == 1]
    if not golden.empty:
        fig.add_trace(go.Scatter(
            x=golden["date"], y=golden["close"],
            mode="markers", name="골든크로스(매수 시그널)",
            marker=dict(symbol="triangle-up", size=12, color="red",
                        line=dict(color="darkred", width=1)),
        ))

    # 데드크로스 마커 (매도 시그널)
    dead = df[df["dead_flag"] == 1]
    if not dead.empty:
        fig.add_trace(go.Scatter(
            x=dead["date"], y=dead["close"],
            mode="markers", name="데드크로스(매도 시그널)",
            marker=dict(symbol="triangle-down", size=12, color="blue",
                        line=dict(color="darkblue", width=1)),
        ))

    fig.update_layout(
        title=f"최근 {n}일 종가 + 이동평균",
        xaxis_title="날짜",
        yaxis_title="가격(원)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    # ── 모델 존재 확인 ────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        st.error(
            "학습된 모델이 없습니다. "
            "`train/run_training.py`를 먼저 실행해주세요."
        )
        st.stop()

    raw_df = _load_raw_df()

    st.title("📈 PPO 매매 판단 대시보드")

    # ── 종목 입력 ─────────────────────────────────────────────────────
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query = st.text_input(
            "종목코드 또는 종목명 입력",
            placeholder="예: 005930  /  삼성전자",
            key="query_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("분석", use_container_width=True)

    # 검색 결과 처리
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None
    if "selected_name" not in st.session_state:
        st.session_state["selected_name"] = None
    if "pipeline_cache" not in st.session_state:
        st.session_state["pipeline_cache"] = {}

    if search_clicked and query:
        matches = _search_ticker(query, raw_df)
        if not matches:
            st.warning(f"'{query}'에 해당하는 종목을 찾을 수 없습니다.")
        elif len(matches) == 1:
            st.session_state["selected_ticker"] = matches[0][0]
            st.session_state["selected_name"]   = matches[0][1]
        else:
            # 복수 매칭 → selectbox
            options = [f"{t}  {n}" for t, n in matches]
            chosen  = st.selectbox("여러 종목이 검색됐습니다. 선택하세요:", options)
            if chosen:
                idx = options.index(chosen)
                st.session_state["selected_ticker"] = matches[idx][0]
                st.session_state["selected_name"]   = matches[idx][1]

    ticker = st.session_state["selected_ticker"]
    name   = st.session_state["selected_name"]

    if not ticker:
        st.info("종목코드(6자리) 또는 종목명을 입력하고 '분석' 버튼을 누르세요.")
        st.markdown(
            "<div style='position:fixed;bottom:0;left:0;right:0;"
            "background:#f0f2f6;padding:8px 24px;font-size:12px;color:#555;'>"
            "이 결과는 과거 데이터 기반 모델의 참고 지표이며 투자 자문이 아닙니다. "
            "실제 매매는 본인 판단과 책임 하에 결정하세요.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── 추론 (종목 바뀔 때만) ─────────────────────────────────────────
    cache = st.session_state["pipeline_cache"]
    if ticker not in cache:
        with st.spinner(f"{ticker} ({name}) 분석 중..."):
            try:
                cache[ticker] = _run_pipeline(ticker)
            except ValueError as e:
                st.error(f"분석 실패: {e}")
                return

    result, explain_r, rag_results, explanation = cache[ticker]

    # ── 결과 출력 ─────────────────────────────────────────────────────
    st.markdown(f"### {ticker}  {name}")

    # 큰 글씨 액션 (색상 구분)
    ACTION_STYLE = {
        "BUY" : ("매수", "#28a745", "white"),
        "HOLD": ("유보", "#ffc107", "black"),
        "SELL": ("매도", "#dc3545", "white"),
    }
    act_ko, bg, fg = ACTION_STYLE.get(result.action_name, ("?", "#888", "white"))
    st.markdown(
        f"<div style='font-size:48px;font-weight:bold;"
        f"background:{bg};color:{fg};display:inline-block;"
        f"padding:8px 32px;border-radius:12px;margin-bottom:12px;'>"
        f"{act_ko}</div>",
        unsafe_allow_html=True,
    )

    # action_probs 막대그래프
    import plotly.graph_objects as go

    fig_prob = go.Figure(go.Bar(
        x=[result.action_probs["BUY"],
           result.action_probs["HOLD"],
           result.action_probs["SELL"]],
        y=["매수", "유보", "매도"],
        orientation="h",
        marker_color=["#28a745", "#ffc107", "#dc3545"],
        text=[f"{p:.0%}" for p in [
            result.action_probs["BUY"],
            result.action_probs["HOLD"],
            result.action_probs["SELL"],
        ]],
        textposition="outside",
    ))
    fig_prob.update_layout(
        height=180, margin=dict(t=10, b=10, l=60, r=60),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        showlegend=False,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # 판단 근거 (LLM 설명)
    st.markdown("#### 판단 근거")
    st.write(explanation.text)
    if not explanation.used_llm:
        st.caption(f"_수치 템플릿 사용 ({explanation.reason})_")

    # 세부 지표 expander
    with st.expander("세부 지표 보기"):
        ind = result.indicators
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI14",       f"{ind.get('rsi14', 0):.1f}")
        c2.metric("이격도20",    f"{ind.get('disparity20', 100):.1f}")
        c3.metric("SMA5",        f"{ind.get('sma5', 0):,.0f}")
        c4.metric("SMA20",       f"{ind.get('sma20', 0):,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("SMA60",       f"{ind.get('sma60', 0):,.0f}")
        c6.metric("EMA20",       f"{ind.get('ema20', 0):,.0f}")
        c7.metric("골든크로스",  "발생" if explain_r.golden_flag else "없음")
        c8.metric("데드크로스",  "발생" if explain_r.dead_flag   else "없음")

        if explain_r.low_confidence:
            st.warning(
                f"⚠️ 확신도 낮음 — 1위·2위 확률 차이 "
                f"{abs(explain_r.top1_prob - explain_r.top2_prob):.0%}"
            )

    # 종가 + 이동평균 + 시그널 마커 차트
    fig_chart = _plot_chart(ticker, raw_df)
    if fig_chart:
        st.plotly_chart(fig_chart, use_container_width=True)

    # 참고 과거 사례 (RAG)
    if rag_results:
        with st.expander(f"참고 과거 사례 ({len(rag_results)}건)"):
            for i, r in enumerate(rag_results, 1):
                meta = r.metadata
                act  = {0: "매도", 1: "유보", 2: "매수"}.get(meta.get("ppo_action", -1), "?")
                r5   = meta.get("ret5",  -999)
                r10  = meta.get("ret10", -999)
                st.markdown(
                    f"**{i}. {meta.get('ticker','?')} · {meta.get('date','?')}**  "
                    f"PPO: {act} / 5일 후: {r5:.1f}% / 10일 후: {r10:.1f}%"
                    if r5 != -999 else
                    f"**{i}. {meta.get('ticker','?')} · {meta.get('date','?')}**  PPO: {act}"
                )
                st.caption(r.card_text[:200] + "…")

    # 하단 고정 안내
    st.markdown(
        "<div style='position:fixed;bottom:0;left:0;right:0;"
        "background:#f0f2f6;padding:8px 24px;font-size:12px;color:#555;"
        "border-top:1px solid #ddd;z-index:999;'>"
        "이 결과는 과거 데이터 기반 모델의 참고 지표이며 투자 자문이 아닙니다. "
        "실제 매매는 본인 판단과 책임 하에 결정하세요.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
