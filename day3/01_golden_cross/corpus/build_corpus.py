#!/usr/bin/env python
"""
STEP 7-A: Corpus 구축 스크립트.

지표 사례 카드를 생성하고, sentence-transformers 임베딩 후 ChromaDB에 저장한다.
BM25 인덱스도 pickle로 별도 저장한다.

흐름:
  1. stock_prices.csv 로드 → 종목별 지표 계산
  2. 사례 카드 텍스트 생성 (ticker, date, RSI, disparity, cross 플래그, 5/10일 수익률)
  3. 임베딩(sentence-transformers) → ChromaDB upsert
  4. BM25 인덱스 구축 → pickle 저장
  5. 이미 있는 종목·날짜는 스킵 (증분 업데이트)

실행:
    python -m corpus.build_corpus                          # 전체
    python -m corpus.build_corpus --ticker 005930          # 단일 종목
    python -m corpus.build_corpus --max_stocks 5           # 최대 N종목 테스트
    python -m corpus.build_corpus --val_only               # val 구간만

뉴스/공시:
    현재는 DART Open API 키가 없으면 "관련 뉴스 없음" / "관련 공시 없음"으로
    명시한다. 키 준비 후 DART_API_KEY 환경변수를 설정하면 자동으로 수집한다.

사후 수익률(5일/10일):
    이 값은 사례 카드의 "기록" 용도이며, 학습 데이터에 사용되지 않는다.
    PPO가 이미 결정을 내린 뒤의 사후 정보이므로 데이터 누설이 아니다.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators.technical import compute_indicators

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------

CORPUS_DIR   = ROOT / "corpus"
CHROMA_DIR   = CORPUS_DIR / "chroma_db"
BM25_PATH    = CORPUS_DIR / "bm25_index.pkl"
NEWS_CACHE   = CORPUS_DIR / "news_cache"
DATA_PATH    = ROOT / "../../day2/03_fss_api/data/stock_prices.csv"
MODEL_PATH   = ROOT / "models/20260701_221527/best_model.zip"
VAL_DATE     = "20240101"

EMBED_MODEL  = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 한국어 특화 SBERT
COLLECTION   = "trading_cases"


# ---------------------------------------------------------------------------
# 임베딩 모델 (모듈 로드 시 1회 초기화)
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ---------------------------------------------------------------------------
# ChromaDB 클라이언트 (1회 초기화)
# ---------------------------------------------------------------------------

_chroma_client = None
_collection    = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection    = _chroma_client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# 뉴스/공시 (키 없을 때 빈 값)
# ---------------------------------------------------------------------------

def _fetch_news(ticker: str, date_str: str) -> str:
    """DART API 키가 있으면 공시를 가져오고, 없으면 '관련 뉴스 없음' 반환."""
    dart_key = os.getenv("DART_API_KEY", "")
    if not dart_key:
        return "관련 공시 없음"

    # 캐시 확인
    NEWS_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = NEWS_CACHE / f"{ticker}_{date_str.replace('-','')}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data.get("summary", "관련 공시 없음")

    try:
        import requests
        date_no_dash = date_str.replace("-", "")
        url = "https://opendart.fss.or.kr/api/list.json"
        params = {
            "crtfc_key": dart_key,
            "stock_code": ticker,
            "bgn_de": date_no_dash,
            "end_de": date_no_dash,
            "page_no": 1,
            "page_count": 5,
        }
        resp = requests.get(url, params=params, timeout=5)
        items = resp.json().get("list", [])
        if items:
            titles = " / ".join(it.get("report_nm", "") for it in items[:3])
            summary = f"공시: {titles}"
        else:
            summary = "관련 공시 없음"
        cache_file.write_text(
            json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8"
        )
        return summary
    except Exception:
        return "관련 공시 없음"


# ---------------------------------------------------------------------------
# 사례 카드 텍스트 생성
# ---------------------------------------------------------------------------

def _action_name(a: int) -> str:
    return {0: "매도", 1: "유보", 2: "매수"}.get(a, "?")


def _build_card(
    ticker  : str,
    name    : str,
    date_str: str,
    row     : pd.Series,
    action  : int,
    prob    : float,
    ret5    : Optional[float],
    ret10   : Optional[float],
    news    : str,
) -> str:
    rsi_zone = (
        "과매수" if row["rsi14"] >= 70 else
        "과매도" if row["rsi14"] <= 30 else
        "중립"
    )
    disp_zone = (
        "과열" if row["disparity20"] >= 105 else
        "침체" if row["disparity20"] <= 95 else
        "중립"
    )
    golden = "있음" if row["golden_flag"] else "없음"
    dead   = "있음" if row["dead_flag"]   else "없음"
    r5  = f"{ret5:.2f}%" if ret5  is not None else "N/A"
    r10 = f"{ret10:.2f}%" if ret10 is not None else "N/A"

    return (
        f"{ticker}({name}) {date_str}: "
        f"RSI14={row['rsi14']:.1f}({rsi_zone}), "
        f"이격도20={row['disparity20']:.1f}({disp_zone}), "
        f"골든크로스={golden}, 데드크로스={dead}, "
        f"당시 PPO 판단={_action_name(action)}(확률 {prob:.0%}), "
        f"{news}. "
        f"이후 5일 수익률={r5}, 10일 수익률={r10}"
    )


# ---------------------------------------------------------------------------
# 사후 수익률 계산 (사례 카드용 기록 — 학습에 미사용)
# ---------------------------------------------------------------------------

def _future_return(df: pd.DataFrame, idx: int, days: int) -> Optional[float]:
    if idx + days >= len(df):
        return None
    entry = df.iloc[idx]["close"]
    exit_ = df.iloc[idx + days]["close"]
    if entry == 0:
        return None
    return (exit_ - entry) / entry * 100.0


# ---------------------------------------------------------------------------
# PPO 추론 (모델이 없으면 action=-1 반환)
# ---------------------------------------------------------------------------

def _infer_action(
    df       : pd.DataFrame,
    idx      : int,
    model,
    lookback : int = 20,
) -> tuple[int, float]:
    """df의 idx 위치(lookback 포함)로 PPO action 추론."""
    if model is None or idx < lookback:
        return -1, 0.0

    w = df.iloc[idx - lookback + 1 : idx + 1].copy()
    close = w["close"].to_numpy(dtype=np.float64)
    sma5  = w["sma5"].to_numpy(dtype=np.float64)
    sma20 = w["sma20"].to_numpy(dtype=np.float64)
    sma60 = w["sma60"].to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        sma5_d  = np.where(sma5  > 0, (close / sma5  - 1) * 100, 0.0)
        sma20_d = np.where(sma20 > 0, (close / sma20 - 1) * 100, 0.0)
        sma60_d = np.where(sma60 > 0, (close / sma60 - 1) * 100, 0.0)

    obs = np.column_stack([
        sma5_d, sma20_d, sma60_d,
        w["golden_flag"].to_numpy(dtype=np.float64),
        w["dead_flag"].to_numpy(dtype=np.float64),
        w["disparity20_centered"].to_numpy(dtype=np.float64),
        w["rsi14_norm"].to_numpy(dtype=np.float64),
        np.zeros(lookback),  # position=0 가정
        np.zeros(lookback),  # unrealized_pnl=0 가정
    ]).astype(np.float32)

    import torch
    device = model.policy.device
    obs_t  = torch.tensor(obs[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        dist  = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()
    action = int(probs.argmax())
    return action, float(probs[action])


# ---------------------------------------------------------------------------
# BM25 인덱스 구축
# ---------------------------------------------------------------------------

def _build_bm25(cards: list[str]) -> object:
    from rank_bm25 import BM25Okapi
    tokenized = [re.findall(r"[가-힣a-zA-Z0-9]+", c) for c in cards]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# 메인 빌드 루틴
# ---------------------------------------------------------------------------

def build_corpus(
    data_path  : Path = DATA_PATH,
    model_path : Path = MODEL_PATH,
    ticker     : str | None = None,
    max_stocks : int | None = None,
    val_only   : bool = False,
) -> None:
    coll = _get_collection()
    embed = _get_embed_model()

    # ── PPO 모델 로드 ──────────────────────────────────────────────────
    ppo_model = None
    if model_path.exists():
        from stable_baselines3 import PPO
        ppo_model = PPO.load(str(model_path))
        print(f"[모델] {model_path}")
    else:
        print(f"[경고] 모델 없음({model_path}) — action='미학습'으로 표기")

    # ── 데이터 로드 ────────────────────────────────────────────────────
    raw = pd.read_csv(data_path, encoding="utf-8-sig",
                      dtype={"srtnCd": str, "종목코드": str})
    code_col = "srtnCd" if "srtnCd" in raw.columns else "종목코드"
    name_col = "itmsNm" if "itmsNm" in raw.columns else "종목명"
    raw[code_col] = raw[code_col].str.zfill(6)
    codes = [ticker] if ticker else sorted(raw[code_col].unique())
    if max_stocks:
        codes = codes[:max_stocks]

    # ── 이미 저장된 doc_id 조회 ───────────────────────────────────────
    existing_ids: set[str] = set()
    try:
        existing_ids = set(coll.get()["ids"])
    except Exception:
        pass
    print(f"[corpus] 기존 {len(existing_ids)}건 → 증분 업데이트")

    # ── 전체 카드 수집 (BM25 재구축용) ────────────────────────────────
    all_cards: list[str] = []
    all_ids  : list[str] = []

    for ci, code in enumerate(codes):
        grp  = raw[raw[code_col] == code].copy()
        name = grp[name_col].iloc[0] if name_col in grp.columns else code
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = compute_indicators(grp, nan_policy="drop")
        if df.empty or len(df) < 20:
            continue

        if val_only:
            date_int = df["date"].dt.strftime("%Y%m%d").astype(int)
            df = df[date_int >= int(VAL_DATE)].reset_index(drop=True)
            if len(df) < 20:
                continue

        new_docs  : list[str] = []
        new_embs  : list[list[float]] = []
        new_metas : list[dict] = []
        new_ids   : list[str]  = []

        for idx, row in df.iterrows():
            if idx < 20:
                continue
            date_str = row["date"].strftime("%Y-%m-%d")
            doc_id   = f"{code}_{date_str}"

            if doc_id in existing_ids:
                all_cards.append("")  # placeholder
                all_ids.append(doc_id)
                continue

            action, prob = _infer_action(df, idx, ppo_model)
            ret5         = _future_return(df, idx, 5)
            ret10        = _future_return(df, idx, 10)
            news         = _fetch_news(code, date_str)
            card         = _build_card(
                code, str(name), date_str, row,
                action, prob, ret5, ret10, news,
            )
            emb = embed.encode(card, convert_to_numpy=True).tolist()

            meta = {
                "ticker"      : code,
                "name"        : str(name),
                "date"        : date_str,
                "rsi14"       : float(row["rsi14"]),
                "disparity20" : float(row["disparity20"]),
                "golden_flag" : int(row["golden_flag"]),
                "dead_flag"   : int(row["dead_flag"]),
                "ppo_action"  : action,
                "ppo_prob"    : float(prob),
                "ret5"        : float(ret5)  if ret5  is not None else -999.0,
                "ret10"       : float(ret10) if ret10 is not None else -999.0,
            }

            new_docs.append(card)
            new_embs.append(emb)
            new_metas.append(meta)
            new_ids.append(doc_id)
            existing_ids.add(doc_id)

            all_cards.append(card)
            all_ids.append(doc_id)

        # ChromaDB upsert (배치)
        if new_docs:
            coll.upsert(
                documents  = new_docs,
                embeddings = new_embs,
                metadatas  = new_metas,
                ids        = new_ids,
            )
        if (ci + 1) % 10 == 0 or ci == len(codes) - 1:
            print(f"  [{ci+1}/{len(codes)}] {code}: +{len(new_docs)}건 저장")

    # ── BM25 재구축 ────────────────────────────────────────────────────
    # 기존 corpus 전체 텍스트를 ChromaDB에서 가져와서 재구축
    # (ChromaDB get()은 ids를 include에 넣지 않아도 항상 반환)
    all_stored  = coll.get(include=["documents"])
    stored_docs = all_stored["documents"]
    stored_ids  = all_stored["ids"]

    if stored_docs:
        bm25 = _build_bm25(stored_docs)
        BM25_PATH.write_bytes(pickle.dumps({"bm25": bm25, "ids": stored_ids, "docs": stored_docs}))
        print(f"[BM25] {len(stored_docs)}건 인덱스 저장 → {BM25_PATH}")

    total = coll.count()
    print(f"\n[완료] ChromaDB: {total}건 / BM25: {len(stored_docs) if stored_docs else 0}건")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 사례 카드 corpus 구축")
    parser.add_argument("--ticker",     default=None, help="단일 종목코드")
    parser.add_argument("--max_stocks", type=int, default=None, help="최대 종목 수 (테스트용)")
    parser.add_argument("--val_only",   action="store_true", help="val 구간(20260101 이후)만")
    parser.add_argument("--data",  default=str(DATA_PATH),  help="stock_prices.csv 경로")
    parser.add_argument("--model", default=str(MODEL_PATH), help="PPO 모델 .zip 경로")
    args = parser.parse_args()

    build_corpus(
        data_path  = Path(args.data),
        model_path = Path(args.model),
        ticker     = args.ticker,
        max_stocks = args.max_stocks,
        val_only   = args.val_only,
    )


if __name__ == "__main__":
    main()
