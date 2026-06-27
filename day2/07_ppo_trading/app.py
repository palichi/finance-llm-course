"""
app.py — PPO 주가 예측 챗봇 (Streamlit)

실행:
    streamlit run app.py

흐름 (순서 고정):
    1단계 PPO  → 매수/매도/유보 결정  (핵심, 변경 불가)
    2단계 RAG  → 결정 근거 문서 검색  (보조, 설명 전용)
    3단계 GPT  → PPO 결정 + RAG 문서 → 자연어 응답 생성
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# predict_stock.py가 같은 디렉터리에 있으므로 path 추가
sys.path.insert(0, str(Path(__file__).parent))
from predict_stock import predict_ppo, search_rag

load_dotenv(Path(__file__).parent / "../../.env")

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODELS_DIR   = Path(__file__).parent / "models"
PRICES_CSV   = Path(__file__).parent / "../03_fss_api/data/stock_prices.csv"
ACTION_EMOJI = {"매수": "📈", "매도": "📉", "유보": "⏸️"}
ACTION_COLOR = {"매수": "green",  "매도": "red",  "유보": "gray"}

# ── 종목 매핑 (srtnCd → itmsNm, itmsNm → srtnCd 6자리) ──────────────────────
@st.cache_data
def load_ticker_map() -> tuple[dict[str, str], dict[str, str]]:
    """모델 파일이 있는 종목만 매핑에 포함."""
    available = {p.stem.replace("_ppo", "") for p in MODELS_DIR.glob("*_ppo.zip")}
    df = pd.read_csv(PRICES_CSV, encoding="utf-8-sig", usecols=["srtnCd", "itmsNm"])
    df["srtnCd"] = df["srtnCd"].astype(str).str.zfill(6)
    df = df.drop_duplicates("srtnCd")
    df = df[df["srtnCd"].isin(available)]
    code_to_name = dict(zip(df["srtnCd"], df["itmsNm"]))
    name_to_code = {v: k for k, v in code_to_name.items()}
    return code_to_name, name_to_code

CODE_TO_NAME, NAME_TO_CODE = load_ticker_map()

# ── 입력 파싱: 종목코드 or 종목명 추출 ────────────────────────────────────────

def resolve_ticker(user_input: str) -> tuple[str | None, str | None]:
    """(ticker, name) 반환. 인식 불가 시 (None, None)."""
    text = user_input.strip()

    # 6자리 숫자 코드 우선
    m = re.search(r"\b(\d{6})\b", text)
    if m:
        code = m.group(1)
        if code in CODE_TO_NAME:
            return code, CODE_TO_NAME[code]
        return None, None

    # 종목명 완전 일치
    for name, code in NAME_TO_CODE.items():
        if name in text:
            return code, name

    # 부분 일치 (2글자 이상)
    for name, code in NAME_TO_CODE.items():
        if len(name) >= 2 and name[:2] in text:
            return code, name

    return None, None


# ── GPT 응답 생성 (PPO 결정을 변경하지 않고 설명만) ──────────────────────────

def build_response(
    openai_client: OpenAI,
    ticker: str,
    name: str,
    action_label: str,
    ref_date: str,
    close_price: float,
    rag_docs: list[str],
) -> str:
    docs_text = ""
    if rag_docs:
        docs_text = "\n".join(f"[문서 {i+1}] {d[:300]}" for i, d in enumerate(rag_docs))

    system_prompt = (
        "당신은 주식 투자 어시스턴트입니다. "
        "PPO 모델이 이미 내린 결정(매수/매도/유보)을 사용자에게 친절하게 설명하는 역할만 합니다. "
        "RAG 검색 문서는 PPO 결정의 근거를 찾아 설명하는 데만 사용하세요. "
        "절대로 PPO의 결정을 바꾸거나 반박하지 마세요."
    )

    user_content = f"""
아래는 PPO 모델이 내린 최종 결정입니다. 이 결정은 확정되어 있으며 변경할 수 없습니다.

■ 종목: {name} ({ticker})
■ 기준일: {ref_date}
■ 기준 종가: {int(close_price):,}원
■ PPO 결정: {action_label}

{"■ 참고 문서 (설명 보조용):\n" + docs_text if docs_text else "■ 참고 문서: 없음"}

위 정보를 바탕으로:
1. PPO가 '{action_label}' 결정을 내렸음을 먼저 명확히 전달하세요.
2. 참고 문서가 있다면 '{action_label}' 결정을 뒷받침하는 내용을 찾아 간략히 설명하세요.
3. 참고 문서가 없으면 기술적 분석 관점에서 일반적인 설명만 덧붙이세요.
4. 투자는 본인 판단임을 마지막에 짧게 안내하세요.
"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content


# ── Streamlit 앱 ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="📊 PPO 주식 예측 챗봇", layout="wide")
st.title("📊 PPO 주식 예측 챗봇")
st.caption("PPO 모델이 먼저 결정 → RAG가 근거 설명 | 종목코드 또는 종목명으로 질문하세요")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    show_rag = st.toggle("RAG 근거 문서 표시", value=True)
    show_steps = st.toggle("단계별 처리 과정 표시", value=True)

    st.markdown("---")
    st.subheader("💡 예시 질문")
    examples = [
        "삼성전자 어때?",
        "005930 예측해줘",
        "SK하이닉스 지금 사도 돼?",
        "현대차 매수해야 할까?",
        "LG에너지솔루션 전망은?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_query"] = ex

    st.markdown("---")
    st.subheader("📋 지원 종목 수")
    st.metric("모델 보유 종목", f"{len(CODE_TO_NAME)}개")

# ── 채팅 히스토리 초기화 ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! PPO 기반 주식 방향 예측 챗봇입니다. 🤖\n\n"
                "**종목코드**(예: `005930`) 또는 **종목명**(예: `삼성전자`)을 포함해서 "
                "질문해 주세요.\n\n"
                "> ⚠️ 이 예측은 강화학습 모델의 학습 결과이며, 실제 투자 조언이 아닙니다."
            ),
        }
    ]

# ── 기존 메시지 렌더링 ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 입력 처리 ─────────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("종목코드 또는 종목명으로 질문하세요...") or pending

if user_input:
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        ticker, name = resolve_ticker(user_input)

        if ticker is None:
            reply = (
                "❓ 종목을 인식하지 못했습니다.\n\n"
                "**6자리 종목코드**(예: `005930`)나 "
                "**종목명**(예: `삼성전자`)을 포함해서 다시 질문해 주세요."
            )
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            # ── 1단계: PPO 예측 ────────────────────────────────────────────
            if show_steps:
                with st.status("⚙️ 처리 중...", expanded=True) as status:
                    st.write(f"🤖 **[1단계]** PPO 모델 예측 중... (`{ticker}` {name})")
                    try:
                        action, action_label, ref_date, close_price = predict_ppo(ticker)
                    except SystemExit:
                        status.update(label="❌ 오류", state="error")
                        reply = f"❌ `{ticker}` ({name}) 모델 또는 데이터 파일을 찾을 수 없습니다."
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.stop()

                    # ── 2단계: RAG 검색 ────────────────────────────────────
                    st.write(f"🔍 **[2단계]** RAG 근거 문서 검색 중...")
                    rag_docs: list[str] = []
                    if show_rag:
                        query = f"{ticker} {name} 주가 {action_label} 근거"
                        rag_docs = search_rag(ticker, query)

                    # ── 3단계: GPT 응답 생성 ───────────────────────────────
                    st.write("💬 **[3단계]** 응답 생성 중...")
                    gpt_reply = build_response(
                        openai_client, ticker, name,
                        action_label, ref_date, close_price, rag_docs,
                    )
                    status.update(label="✅ 완료", state="complete", expanded=False)
            else:
                try:
                    action, action_label, ref_date, close_price = predict_ppo(ticker)
                except SystemExit:
                    reply = f"❌ `{ticker}` ({name}) 모델 또는 데이터 파일을 찾을 수 없습니다."
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()

                rag_docs = []
                if show_rag:
                    query = f"{ticker} {name} 주가 {action_label} 근거"
                    rag_docs = search_rag(ticker, query)

                gpt_reply = build_response(
                    openai_client, ticker, name,
                    action_label, ref_date, close_price, rag_docs,
                )

            # ── 결과 표시 ───────────────────────────────────────────────────
            emoji  = ACTION_EMOJI[action_label]
            color  = ACTION_COLOR[action_label]

            # PPO 결정 카드 (항상 최상단)
            st.markdown(
                f"### {emoji} PPO 예측 결과\n"
                f"| 항목 | 내용 |\n"
                f"|------|------|\n"
                f"| 종목 | **{name}** (`{ticker}`) |\n"
                f"| 기준일 | {ref_date} |\n"
                f"| 기준 종가 | {int(close_price):,}원 |\n"
                f"| **결정** | :{color}[**{action_label}**] |"
            )

            # RAG 문서 (접을 수 있게)
            if rag_docs and show_rag:
                with st.expander(f"📄 RAG 참고 문서 ({len(rag_docs)}건)"):
                    for i, doc in enumerate(rag_docs, 1):
                        st.markdown(f"**[{i}]** {doc[:300]}{'…' if len(doc) > 300 else ''}")

            # GPT 설명
            st.markdown("---")
            st.markdown(gpt_reply)

            # 메시지 저장 (카드 + 설명 통합)
            saved = (
                f"### {emoji} [{name} ({ticker})] PPO 결정: **{action_label}**\n"
                f"- 기준일: {ref_date} | 종가: {int(close_price):,}원\n\n"
                f"{gpt_reply}"
            )
            st.session_state.messages.append({"role": "assistant", "content": saved})
