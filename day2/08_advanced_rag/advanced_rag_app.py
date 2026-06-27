import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv("../../.env")

import chromadb
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚀 Advanced RAG 실습", layout="wide")
st.title("🚀 Advanced RAG 실습")
st.caption("기본 RAG vs Advanced RAG(HyDE + Ensemble + Parent Document + Reranking) 비교")

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = "../04_chromadb/chroma_db"
COLLECTION_NAME = "stock_data"
EMBED_MODEL = "text-embedding-3-small"

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)

@st.cache_resource
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

collection = get_chroma_collection()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Advanced RAG 기법 선택")
use_hyde      = st.sidebar.checkbox("HyDE (가상 문서 생성)")
use_ensemble  = st.sidebar.checkbox("Ensemble Retriever (BM25 + 시맨틱)")
use_parent    = st.sidebar.checkbox("Parent Document Retriever")
use_rerank    = st.sidebar.checkbox("Reranking (재순위화)")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 예시 질문")
example_questions = [
    "삼성전자 12월 주가는?",
    "거래량이 가장 많았던 종목은?",
    "최근 반도체 업종 동향은?",
    "RSI 지표가 뭐야?",
    "코스피 지수 전망 알려줘",
]
for q in example_questions:
    if st.sidebar.button(q):
        st.session_state["query_input"] = q

# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def chroma_search(query_text: str, k: int = 5) -> list[Document]:
    embedding = embed_text(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=k)
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=doc, metadata=meta or {}))
    return docs


def fetch_all_docs() -> list[Document]:
    """Fetch all documents from ChromaDB for BM25 index."""
    results = collection.get(include=["documents", "metadatas"])
    docs = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=doc, metadata=meta or {}))
    return docs


def generate_answer(question: str, context_docs: list[Document]) -> str:
    context = "\n\n".join(d.page_content for d in context_docs)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 한국 주식 시장 전문가입니다. "
                    "제공된 컨텍스트를 바탕으로 질문에 답변해 주세요. "
                    "컨텍스트에 없는 내용은 '정보가 부족합니다'라고 답하세요."
                ),
            },
            {
                "role": "user",
                "content": f"컨텍스트:\n{context}\n\n질문: {question}",
            },
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def generate_hyde_doc(question: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "한국 주식 시장에 관한 질문에 대해 "
                    "마치 실제 답변처럼 가상의 답변을 작성해 주세요. "
                    "사실 여부와 관계없이 그럴듯한 답변을 생성하세요."
                ),
            },
            {"role": "user", "content": f"질문: {question}\n\n가상의 답변을 작성해줘:"},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


def parent_expand(child_docs: list[Document]) -> list[Document]:
    """
    Simulate parent document retrieval by re-fetching a larger chunk
    from the same source text using a 2000-char parent splitter.
    We use all ChromaDB docs that share the same metadata source as the child.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=50)

    all_docs = fetch_all_docs()

    # Group all docs by source metadata (fall back to treating each doc as own parent)
    # Build a map: child content → parent chunk
    child_to_parent: dict[str, str] = {}
    for doc in all_docs:
        # Treat each ChromaDB document as a "parent" and split it into children
        parents = parent_splitter.split_documents([doc])
        for parent in parents:
            children = child_splitter.split_documents([parent])
            for child in children:
                child_to_parent[child.page_content.strip()] = parent.page_content

    expanded = []
    seen = set()
    for child in child_docs:
        parent_text = child_to_parent.get(child.page_content.strip())
        if parent_text and parent_text not in seen:
            seen.add(parent_text)
            expanded.append(Document(page_content=parent_text, metadata=child.metadata))
        elif child.page_content not in seen:
            seen.add(child.page_content)
            expanded.append(child)
    return expanded


def rerank_docs(question: str, docs: list[Document], top_k: int = 3):
    encoder = get_cross_encoder()
    pairs = [(question, d.page_content) for d in docs]
    scores = encoder.predict(pairs).tolist()
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return ranked, [d for _, d in ranked[:top_k]]


# ── Main UI ───────────────────────────────────────────────────────────────────
query_input = st.text_input(
    "질문을 입력하세요",
    value=st.session_state.get("query_input", ""),
    placeholder="예) 삼성전자 12월 주가는?",
    key="query_input",
)
run_btn = st.button("🔍 비교 실행", type="primary", use_container_width=True)

if run_btn and query_input.strip():
    question = query_input.strip()

    col_basic, col_adv = st.columns(2)

    # ── 기본 RAG ──────────────────────────────────────────────────────────────
    basic_docs: list[Document] = []
    basic_answer = ""
    with col_basic:
        st.subheader("🔍 기본 RAG")
        with st.spinner("검색 중..."):
            t0 = time.time()
            try:
                basic_docs = chroma_search(question, k=5)
                basic_answer = generate_answer(question, basic_docs)
                basic_elapsed = time.time() - t0
                st.write(basic_answer)
                with st.expander("📄 검색된 문서"):
                    for i, d in enumerate(basic_docs, 1):
                        st.markdown(f"**[{i}]** {d.page_content[:300]}…")
                st.caption(f"⏱ 응답시간: {basic_elapsed:.1f}초")
            except Exception as e:
                st.error(f"기본 RAG 오류: {e}")

    # ── Advanced RAG ──────────────────────────────────────────────────────────
    adv_docs: list[Document] = []
    adv_answer = ""
    with col_adv:
        st.subheader("🚀 Advanced RAG")

        applied_techniques: list[str] = []
        with st.spinner("Advanced RAG 파이프라인 실행 중..."):
            t0 = time.time()
            try:
                search_query = question

                # 1) HyDE
                if use_hyde:
                    applied_techniques.append("HyDE")
                    hyde_doc = generate_hyde_doc(question)
                    search_query = hyde_doc
                    with st.expander("💭 생성된 가상 문서"):
                        st.write(hyde_doc)

                # 2) Retrieve (Ensemble or Semantic)
                if use_ensemble:
                    applied_techniques.append("Ensemble Retriever")
                    all_docs = fetch_all_docs()
                    bm25 = BM25Retriever.from_documents(all_docs, k=5)
                    chroma_retriever_docs = chroma_search(search_query, k=5)

                    class _SimpleRetriever:
                        def __init__(self, docs):
                            self._docs = docs
                        def get_relevant_documents(self, q):
                            return self._docs
                        def invoke(self, q):
                            return self._docs

                    semantic_retriever = _SimpleRetriever(chroma_retriever_docs)
                    ensemble = EnsembleRetriever(
                        retrievers=[bm25, semantic_retriever],
                        weights=[0.5, 0.5],
                    )
                    adv_docs = ensemble.invoke(search_query)
                else:
                    adv_docs = chroma_search(search_query, k=5)

                # 3) Parent Document
                if use_parent:
                    applied_techniques.append("Parent Document Retriever")
                    adv_docs = parent_expand(adv_docs)

                # 4) Reranking
                if use_rerank:
                    applied_techniques.append("Reranking")
                    ranked_with_scores, adv_docs = rerank_docs(question, adv_docs, top_k=3)
                    with st.expander("📊 재정렬 점수"):
                        st.markdown("**재정렬 전 → 후 순서 비교**")
                        before_texts = [d.page_content[:80] for _, d in ranked_with_scores]
                        after_texts  = [d.page_content[:80] for d in adv_docs]
                        rows = []
                        for rank, (score, doc) in enumerate(ranked_with_scores, 1):
                            rows.append({
                                "순위": rank,
                                "점수": f"{score:.4f}",
                                "문서 미리보기": doc.page_content[:80] + "…",
                            })
                        import pandas as pd
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                adv_answer = generate_answer(question, adv_docs)
                adv_elapsed = time.time() - t0

                # Badge display
                if applied_techniques:
                    badge_html = " ".join(
                        f'<span style="background:#1f77b4;color:white;padding:2px 10px;'
                        f'border-radius:12px;font-size:0.85em;margin-right:4px">{t}</span>'
                        for t in applied_techniques
                    )
                    st.markdown(f"🛠 **적용된 기법:** {badge_html}", unsafe_allow_html=True)
                else:
                    st.info("선택된 기법 없음 — 기본 RAG와 동일하게 동작합니다.")

                st.write(adv_answer)
                with st.expander("📄 검색된 문서"):
                    for i, d in enumerate(adv_docs, 1):
                        st.markdown(f"**[{i}]** {d.page_content[:300]}…")
                st.caption(f"⏱ 응답시간: {adv_elapsed:.1f}초")

            except Exception as e:
                st.error(f"Advanced RAG 오류: {e}")

    # ── 비교 분석 섹션 ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "📊 **같은 질문, 기본 RAG와 Advanced RAG의 검색 결과가 어떻게 달랐나요?**"
    )
    if basic_docs and adv_docs:
        basic_set = {d.page_content for d in basic_docs}
        adv_set   = {d.page_content for d in adv_docs}
        if basic_set != adv_set:
            st.warning("🔍 **문서 차이 발견** — 두 방식이 서로 다른 문서를 가져왔습니다!")
            only_basic = basic_set - adv_set
            only_adv   = adv_set - basic_set
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**기본 RAG에만 있는 문서:**")
                for t in list(only_basic)[:3]:
                    st.markdown(f"- {t[:120]}…")
            with c2:
                st.markdown("**Advanced RAG에만 있는 문서:**")
                for t in list(only_adv)[:3]:
                    st.markdown(f"- {t[:120]}…")
        else:
            st.success("두 방식이 동일한 문서를 검색했습니다.")

elif run_btn:
    st.warning("질문을 입력해 주세요.")
