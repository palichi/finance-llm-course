"""
STEP 7-B: Advanced RAG 검색 모듈.

4가지 기법을 조합:
  1. Query Expansion  — rule_based 결과에서 조건별 질의 생성 (규칙 템플릿)
  2. Hybrid Search    — ChromaDB 벡터 검색 + BM25 키워드 검색 (RRF 결합)
  3. Metadata Filter  — RSI/disparity 범위, 자기참조 방지, fallback 단계 완화
  4. Reranking        — 코사인 유사도로 최종 재정렬, Top-K 반환

query_expansion 전략:
  규칙 기반 템플릿 사용 (vs LLM 확장):
  ─ 이유: 매 추론마다 호출되므로 속도·비용이 중요하다.
    LLM 확장은 다양성은 높지만 latency ≥500ms + API 비용이 추가된다.
    규칙 템플릿으로도 "RSI 과매수 매수 사례", "골든크로스 직후 유보" 등
    핵심 조건을 충분히 커버할 수 있다. 향후 LLM으로 교체 가능한 구조로 작성.

hybrid_search 결합 방식 — RRF (Reciprocal Rank Fusion) 채택:
  ─ weighted sum 대비 RRF의 장점:
    벡터/BM25 점수의 스케일 차이를 정규화할 필요가 없다.
    순위 기반이므로 이상치(outlier) 점수에 강건하다.

reranking 방식:
  cross-encoder는 정밀하지만 (BAAI/bge-reranker 등) 초기화 비용이 크고
  매 호출마다 n_candidates × 1회 모델 forward가 필요하다.
  임베딩 코사인 유사도 재계산은 이미 임베딩된 corpus 벡터를 재사용하므로
  추가 forward 없이 빠르다. 이 구현에서는 코사인 유사도 재계산을 채택.
  cross-encoder로 교체하려면 _rerank() 함수만 바꾸면 된다.

사용법:
    from explain.rule_based import ExplainResult
    from explain.rag_retriever import retrieve, RAGResult

    results = retrieve(explain_result, ticker="005930")
    for r in results:
        print(r.card_text, r.score)
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from explain.rule_based import ExplainResult

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_DIR  = ROOT / "corpus"
CHROMA_DIR  = CORPUS_DIR / "chroma_db"
BM25_PATH   = CORPUS_DIR / "bm25_index.pkl"
EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
COLLECTION  = "trading_cases"

DEFAULT_TOP_K    = 5
DEFAULT_CANDS    = 20   # reranking 전 후보 수
RSI_RANGE        = 10.0  # ±N 이내 RSI 필터
DISP_RANGE       = 5.0   # ±N 이내 disparity 필터
MIN_CANDIDATES   = 3     # 필터 후 최소 후보 수


# ---------------------------------------------------------------------------
# 반환 타입
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    doc_id      : str
    card_text   : str
    score       : float          # 최종 관련도 점수 (높을수록 관련)
    metadata    : dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 모듈 수준 싱글턴 (매 호출마다 재로딩 방지)
# ---------------------------------------------------------------------------

_embed_model  = None
_chroma_coll  = None
_bm25_bundle  = None   # {"bm25": BM25Okapi, "ids": [...], "docs": [...]}


def _get_embed():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_collection():
    global _chroma_coll
    if _chroma_coll is None:
        import chromadb
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_coll = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_coll


def _get_bm25():
    global _bm25_bundle
    if _bm25_bundle is None and BM25_PATH.exists():
        _bm25_bundle = pickle.loads(BM25_PATH.read_bytes())
    return _bm25_bundle


# ---------------------------------------------------------------------------
# 1. Query Expansion (규칙 기반 템플릿)
# ---------------------------------------------------------------------------

def _expand_queries(er: "ExplainResult") -> list[str]:
    """
    활성화된 지표 조건을 별도 검색 질의로 확장.
    규칙 템플릿 방식 채택 — 속도·비용 우선.
    """
    queries: list[str] = []

    # 기본 질의: 종목 + 행동
    action_ko = {"BUY": "매수", "HOLD": "유보", "SELL": "매도"}
    action_str = action_ko.get(er.top1_action, er.top1_action)
    queries.append(f"RSI {er.rsi14:.0f} 이격도 {er.disparity20:.0f} {action_str} 사례")

    # RSI 조건
    if er.rsi_zone == "OVERBOUGHT":
        queries.append(f"RSI 과매수 {action_str} 사례")
    elif er.rsi_zone == "OVERSOLD":
        queries.append(f"RSI 과매도 {action_str} 사례")

    # 이격도 조건
    if er.disparity_zone == "OVERHEATED":
        queries.append(f"이격도 과열 {action_str} 사례")
    elif er.disparity_zone == "DEPRESSED":
        queries.append(f"이격도 침체 {action_str} 사례")

    # 크로스 조건
    if er.golden_flag:
        queries.append(f"골든크로스 직후 {action_str} 사례")
    if er.dead_flag:
        queries.append(f"데드크로스 직후 {action_str} 사례")

    # shaping 발동 조건
    if er.shaping_rsi_buy_penalty:
        queries.append("RSI 과매수 매수 패널티 사례")
    if er.shaping_golden_hold_bonus:
        queries.append("골든크로스 보유 유지 보너스 사례")
    if er.shaping_disparity_buy_penalty:
        queries.append("이격도 과열 신규 매수 패널티 사례")

    # 중복 제거, 최대 5개
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:5]


# ---------------------------------------------------------------------------
# 2. Hybrid Search (벡터 + BM25, RRF 결합)
# ---------------------------------------------------------------------------

def _vector_search(query: str, n: int) -> list[tuple[str, float]]:
    """ChromaDB 벡터 검색 → [(doc_id, distance)] 반환."""
    coll  = _get_collection()
    embed = _get_embed()

    if coll.count() == 0:
        return []

    emb = embed.encode(query, convert_to_numpy=True).tolist()
    res = coll.query(
        query_embeddings=[emb],
        n_results=min(n, coll.count()),
        include=["distances"],
    )
    ids  = res["ids"][0]
    dists= res["distances"][0]  # cosine distance (0=identical)
    return [(doc_id, float(dist)) for doc_id, dist in zip(ids, dists)]


def _bm25_search(query: str, n: int) -> list[tuple[str, float]]:
    """BM25 키워드 검색 → [(doc_id, score)] 반환."""
    bundle = _get_bm25()
    if bundle is None:
        return []
    bm25   = bundle["bm25"]
    ids    = bundle["ids"]
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [(ids[i], float(s)) for i, s in ranked]


def _rrf(
    vec_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion.
    벡터 검색: distance → rank (작을수록 높은 순위)
    BM25: score → rank (클수록 높은 순위)
    """
    fused: dict[str, float] = {}

    # 벡터: distance 오름차순 정렬 후 rank 부여
    vec_sorted = sorted(vec_results, key=lambda x: x[1])
    for rank, (doc_id, _) in enumerate(vec_sorted, 1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)

    # BM25: score 내림차순 정렬 후 rank 부여
    bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
    for rank, (doc_id, _) in enumerate(bm25_sorted, 1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)

    return fused  # {doc_id: rrf_score}


def _hybrid_search(query: str, n: int) -> dict[str, float]:
    """벡터 + BM25 RRF 결합 → {doc_id: rrf_score}."""
    vec  = _vector_search(query, n * 2)
    bm25 = _bm25_search(query, n * 2)
    return _rrf(vec, bm25)


# ---------------------------------------------------------------------------
# 3. Metadata Filtering
# ---------------------------------------------------------------------------

def _metadata_filter(
    candidates  : dict[str, float],
    er          : "ExplainResult",
    current_date: str,
    same_ticker : str | None,
    rsi_range   : float,
    disp_range  : float,
) -> dict[str, float]:
    """
    RSI/disparity 범위 필터 + 자기참조 방지.
    후보가 MIN_CANDIDATES 미만이면 필터를 단계적으로 완화.
    """
    coll = _get_collection()
    if coll.count() == 0 or not candidates:
        return candidates

    # doc_id → metadata 일괄 조회
    ids_to_fetch = list(candidates.keys())
    fetched      = coll.get(ids=ids_to_fetch, include=["metadatas"])
    meta_map: dict[str, dict] = {
        doc_id: meta
        for doc_id, meta in zip(fetched["ids"], fetched["metadatas"])
    }

    def _apply(rsi_r: float, disp_r: float) -> dict[str, float]:
        result = {}
        for doc_id, score in candidates.items():
            meta = meta_map.get(doc_id, {})
            date = meta.get("date", "")

            # 자기참조 방지: 너무 최근(같은 날~20일 이내)
            if date >= current_date:
                continue

            # RSI 범위 필터
            doc_rsi = meta.get("rsi14", er.rsi14)
            if abs(doc_rsi - er.rsi14) > rsi_r:
                continue

            # Disparity 범위 필터
            doc_disp = meta.get("disparity20", er.disparity20)
            if abs(doc_disp - er.disparity20) > disp_r:
                continue

            result[doc_id] = score
        return result

    # 1차 필터
    filtered = _apply(rsi_range, disp_range)

    # fallback: 범위 완화
    if len(filtered) < MIN_CANDIDATES:
        filtered = _apply(rsi_range * 2, disp_range * 2)
    if len(filtered) < MIN_CANDIDATES:
        filtered = _apply(rsi_range * 4, disp_range * 4)
    if len(filtered) < MIN_CANDIDATES:
        # 범위 필터 제거, 자기참조 방지만 유지
        filtered = {
            doc_id: score
            for doc_id, score in candidates.items()
            if meta_map.get(doc_id, {}).get("date", "") < current_date
        }

    return filtered


# ---------------------------------------------------------------------------
# 4. Reranking (임베딩 코사인 유사도)
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _rerank(
    candidates : dict[str, float],
    query      : str,
    top_k      : int,
    same_ticker: str | None = None,
) -> list[RAGResult]:
    """
    쿼리 임베딩과 각 사례 카드 임베딩의 코사인 유사도로 재정렬.
    same_ticker 지정 시 같은 종목을 우선 배치하고 나머지로 빈 슬롯을 채운다.
    ChromaDB에서 embeddings를 가져와 재사용 — 추가 모델 forward 없음.
    """
    if not candidates:
        return []

    coll  = _get_collection()
    embed = _get_embed()

    ids_list = list(candidates.keys())
    try:
        fetched = coll.get(
            ids=ids_list,
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception:
        return []

    q_emb = embed.encode(query, convert_to_numpy=True)

    scored: list[RAGResult] = []
    for doc_id, doc, meta, doc_emb in zip(
        fetched["ids"],
        fetched["documents"],
        fetched["metadatas"],
        fetched["embeddings"],
    ):
        sim   = _cosine(q_emb, np.array(doc_emb))
        rrf   = candidates.get(doc_id, 0.0)
        score = 0.6 * sim + 0.4 * rrf * 100
        scored.append(RAGResult(
            doc_id   = doc_id,
            card_text= doc,
            score    = score,
            metadata = meta,
        ))

    scored.sort(key=lambda x: x.score, reverse=True)

    # 같은 종목 우선: same_ticker 결과 먼저, 나머지로 빈 슬롯 채움
    if same_ticker:
        same = [r for r in scored if r.metadata.get("ticker") == same_ticker]
        others = [r for r in scored if r.metadata.get("ticker") != same_ticker]
        scored = same + others

    return scored[:top_k]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def retrieve(
    er          : "ExplainResult",
    ticker      : str | None = None,
    top_k       : int   = DEFAULT_TOP_K,
    n_candidates: int   = DEFAULT_CANDS,
    rsi_range   : float = RSI_RANGE,
    disp_range  : float = DISP_RANGE,
) -> list[RAGResult]:
    """
    Advanced RAG 검색 — 4기법 조합.

    Parameters
    ----------
    er : ExplainResult
        explain.rule_based.explain() 반환값.
    ticker : str | None
        현재 종목코드. None이면 자기참조 방지 완화.
    top_k : int
        최종 반환 사례 수.
    n_candidates : int
        reranking 전 후보 수.
    rsi_range : float
        RSI ±범위 필터 (기본 ±10).
    disp_range : float
        disparity ±범위 필터 (기본 ±5).

    Returns
    -------
    list[RAGResult]
        재정렬된 Top-K 사례 (빈 corpus면 빈 리스트).
    """
    coll = _get_collection()
    if coll.count() == 0:
        return []

    # ── 1. Query Expansion ────────────────────────────────────────────
    queries = _expand_queries(er)

    # ── 2. Hybrid Search (각 질의 결과 합산) ──────────────────────────
    fused: dict[str, float] = {}
    for q in queries:
        results = _hybrid_search(q, n_candidates)
        for doc_id, score in results.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + score

    if not fused:
        return []

    # ── 3. Metadata Filtering ─────────────────────────────────────────
    filtered = _metadata_filter(
        candidates   = fused,
        er           = er,
        current_date = er.date,
        same_ticker  = ticker,
        rsi_range    = rsi_range,
        disp_range   = disp_range,
    )

    if not filtered:
        return []

    # ── 4. Reranking (대표 질의 기준, 같은 종목 우선) ────────────────
    main_query = queries[0]
    return _rerank(filtered, main_query, top_k, same_ticker=ticker)
