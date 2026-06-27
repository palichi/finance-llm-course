"""
predict_stock.py — PPO 예측 + RAG 설명 (선택)

실행:
    python predict_stock.py --ticker 005930
    python predict_stock.py --ticker 005930 --explain
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

load_dotenv(Path(__file__).parent / "../../.env")

from stock_direction_env import WINDOW_SIZE

# ─── 경로 ──────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
MODELS_DIR        = BASE_DIR / "models"
DATA_DIR          = BASE_DIR / "../03_fss_api/data/ppo_ready"
FINETUNED_EMB_DIR = BASE_DIR / "finetuned_embedding"
CHROMA_DIR        = BASE_DIR / "../04_chromadb/chroma_db_text"
CHROMA_COLLECTION = "stock_text_documents"

ACTION_LABEL = {0: "매수", 1: "매도", 2: "유보"}


# ─── 데이터 로딩 ────────────────────────────────────────────────────

def load_recent_ohlcv(ticker: str) -> tuple[pd.DataFrame, np.ndarray]:
    """최근 WINDOW_SIZE(20)일 OHLCV 반환 — (원본 df 행, ohlcv ndarray)."""
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        print(f"[오류] 데이터 파일이 없습니다: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={"basdt": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        print(f"[오류] 데이터 부족: {len(df)}행 (최소 {WINDOW_SIZE}행 필요)")
        sys.exit(1)

    recent = df.tail(WINDOW_SIZE).reset_index(drop=True)
    ohlcv = recent[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    return recent, ohlcv


# ─── State 구성 (환경과 동일한 전처리) ─────────────────────────────

def build_obs(ohlcv: np.ndarray) -> np.ndarray:
    """stock_direction_env._get_obs() 와 동일: 열별 min-max 정규화 후 flatten."""
    window = ohlcv.copy()
    col_min = window.min(axis=0)
    col_max = window.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    window = (window - col_min) / col_range
    return window.flatten().astype(np.float32)


# ─── 1단계: PPO 예측 ────────────────────────────────────────────────

def predict_ppo(ticker: str) -> tuple[int, str, str, float]:
    """(action, action_label, ref_date, close_price) 반환."""
    model_path = MODELS_DIR / f"{ticker}_ppo.zip"
    if not model_path.exists():
        print(f"[오류] 모델 파일이 없습니다: {model_path}")
        sys.exit(1)

    model = PPO.load(str(model_path))
    recent_df, ohlcv = load_recent_ohlcv(ticker)

    obs = build_obs(ohlcv)
    action = int(model.predict(obs, deterministic=True)[0])

    ref_date = str(recent_df["date"].iloc[-1])
    close_price = float(recent_df["close"].iloc[-1])

    return action, ACTION_LABEL[action], ref_date, close_price


# ─── 2단계: RAG 설명 검색 ───────────────────────────────────────────

def search_rag(ticker: str, query: str) -> list[str]:
    """ChromaDB에서 ticker 필터로 top-3 문서 반환. 실패 시 빈 리스트."""
    try:
        import chromadb
        from openai import OpenAI

        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ChromaDB ticker는 선행 0 없이 저장됨 (srtnCd를 int→str 변환한 값)
        chroma_ticker = str(int(ticker))

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_collection(CHROMA_COLLECTION)

        resp = openai_client.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        query_vec = resp.data[0].embedding

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            where={"ticker": chroma_ticker},
        )

        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]

    except Exception as e:
        print(f"[경고] RAG 검색 실패 (PPO 결과에 영향 없음): {e}")
        return []


# ─── 3단계: 출력 조립 ───────────────────────────────────────────────

def print_result(
    ticker: str,
    action: int,
    action_label: str,
    ref_date: str,
    close_price: float,
    docs: list[str],
) -> None:
    print(f"\n{'=' * 50}")
    print(f"[{ticker}] 예측 결과: {action_label}")
    print(f"{'=' * 50}")
    print(f"  기준일  : {ref_date}")
    print(f"  종가    : {int(close_price):,}원")

    if docs:
        print(f"\n[참고 근거 — RAG 검색 결과]")
        for i, doc in enumerate(docs, 1):
            preview = doc[:200].replace("\n", " ")
            print(f"  {i}. {preview}{'...' if len(doc) > 200 else ''}")

    print()


# ─── main ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 주가 방향 예측")
    parser.add_argument("--ticker", required=True, help="종목코드 (예: 005930)")
    parser.add_argument("--explain", action="store_true", help="RAG 설명 포함")
    args = parser.parse_args()

    ticker = args.ticker

    # 1단계 — PPO 예측 (핵심, 필수)
    action, action_label, ref_date, close_price = predict_ppo(ticker)

    # 2단계 — RAG 검색 (보조, --explain 시에만)
    docs: list[str] = []
    if args.explain:
        query = f"{ticker} 주가 {action_label} 근거"
        docs = search_rag(ticker, query)

    # 3단계 — 출력
    print_result(ticker, action, action_label, ref_date, close_price, docs)


if __name__ == "__main__":
    main()
