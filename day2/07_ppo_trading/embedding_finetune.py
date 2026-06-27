"""
한국 주식 도메인 특화 임베딩 모델 Fine-tuning
베이스 모델: BAAI/bge-m3
데이터 소스: ChromaDB "stock_text_documents"
"""

import os
import time
import sys
import random
from pathlib import Path

# 메모리 단편화 방지 — 학습 전 환경변수 설정 (torch import 전에 위치해야 함)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import chromadb
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

# ─── 설정 ────────────────────────────────────────────────────────────────────
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "../04_chromadb/chroma_db_text"
COLLECTION_NAME = "stock_text_documents"
OUTPUT_DIR = "./finetuned_embedding"
EPOCHS = 3
BATCH_SIZE = 8             # MiniLM(118M) — 8GB GPU에서 여유롭게 가능
GRAD_ACCUM = 1             # 유효 배치 크기 = BATCH_SIZE × GRAD_ACCUM = 8
SAMPLE_SIZE = 500          # ChromaDB에서 가져올 문서 수 (전체 30만 개 중 샘플)
MIN_PAIRS = 20             # 최소 학습 쌍 수


# ─── 1. 디바이스 확인 ─────────────────────────────────────────────────────────
def check_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[디바이스] GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[디바이스] GPU 없음 → CPU 사용")
        print("[경고] CPU 학습은 GPU 대비 10~50배 오래 걸릴 수 있습니다. 잠시 기다려주세요.")
    return device


# ─── 2. ChromaDB에서 문서 로딩 ────────────────────────────────────────────────
def load_documents() -> tuple[list[str], list[dict]]:
    chroma_path = Path(__file__).parent / CHROMA_PATH
    if not chroma_path.exists():
        print(f"[오류] ChromaDB 경로를 찾을 수 없습니다: {chroma_path.resolve()}")
        print("  → '../04_chromadb/chroma_db_text' 디렉터리가 존재하는지 확인하세요.")
        sys.exit(1)

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
    except Exception as e:
        print(f"[오류] ChromaDB 연결 실패: {e}")
        sys.exit(1)

    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        available = [c.name for c in client.list_collections()]
        print(f"[오류] 컬렉션 '{COLLECTION_NAME}'을(를) 찾을 수 없습니다.")
        print(f"  → 사용 가능한 컬렉션: {available}")
        sys.exit(1)

    total = col.count()
    print(f"[데이터] 컬렉션 '{COLLECTION_NAME}' 총 문서 수: {total:,}")

    # 전체에서 고르게 샘플링 (offset 기반)
    step = max(1, total // SAMPLE_SIZE)
    offsets = list(range(0, min(total, step * SAMPLE_SIZE), step))
    random.shuffle(offsets)

    documents, metadatas = [], []
    batch = 100
    for start in range(0, len(offsets), batch):
        chunk_offsets = offsets[start:start + batch]
        for offset in chunk_offsets:
            res = col.get(limit=1, offset=offset, include=["documents", "metadatas"])
            if res["documents"]:
                documents.append(res["documents"][0])
                metadatas.append(res["metadatas"][0])

    print(f"[데이터] 샘플링된 문서 수: {len(documents)}")
    return documents, metadatas


# ─── 3. (질문, 정답문서) 쌍 자동 생성 ───────────────────────────────────────
def build_pairs(documents: list[str], metadatas: list[dict]) -> tuple[list[str], list[str]]:
    """메타데이터 기반으로 다양한 질문 패턴을 생성한다."""

    def parse_date(date_str: str) -> tuple[str, str, str]:
        """'20240102' → ('2024', '1', '2')"""
        try:
            y, m, d = date_str[:4], str(int(date_str[4:6])), str(int(date_str[6:8]))
        except Exception:
            y, m, d = "?", "?", "?"
        return y, m, d

    # 질문 템플릿 — 실제 메타데이터 값을 채워 넣는다
    def make_queries(doc: str, meta: dict) -> list[str]:
        name = meta.get("name", "")
        ticker = meta.get("ticker", "")
        market = meta.get("market", "")
        date_str = meta.get("date", "00000000")
        year, month, day = parse_date(date_str)

        templates = [
            f"{name} {year}년 {month}월 {day}일 주가는?",
            f"{name}({ticker}) 종가 정보",
            f"{market} 종목 {name} {month}월 주가",
            f"{year}년 {month}월 {day}일 {name} 주가 데이터",
            f"{name} 거래량과 시가총액",
            f"{name} 주식 {year}년 {month}월 시세",
            f"{ticker} 종목 {month}월 {day}일 주가",
        ]
        return templates

    anchors, positives = [], []
    used: set[int] = set()

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        queries = make_queries(doc, meta)
        for q in queries:
            anchors.append(q)
            positives.append(doc)
        used.add(i)

    # 거래량/시가총액 특화 쌍 추가
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        name = meta.get("name", "")
        if "거래량" in doc:
            anchors.append(f"{name} 거래량이 가장 많았던 날은?")
            positives.append(doc)
        if "시가총액" in doc:
            anchors.append(f"{name} 시가총액은 얼마인가요?")
            positives.append(doc)

    # 최소 쌍 수 보장 (중복 허용)
    while len(anchors) < MIN_PAIRS and documents:
        idx = random.randint(0, len(documents) - 1)
        meta = metadatas[idx]
        name = meta.get("name", "")
        month = parse_date(meta.get("date", "00000000"))[1]
        anchors.append(f"{name} {month}월 주가 정보를 알려줘")
        positives.append(documents[idx])

    print(f"[학습데이터] 생성된 (질문, 문서) 쌍: {len(anchors)}")
    return anchors, positives


# ─── 4. 학습 전후 비교 ───────────────────────────────────────────────────────
EVAL_QUERIES = [
    "삼성전자 2024년 12월 주가는?",
    "거래량이 가장 많았던 날의 KOSPI 종목 주가",
    "SK하이닉스 종가와 시가총액 정보",
    "현대차 2023년 상반기 주가 시세",
    "KOSDAQ 종목 2024년 거래량 데이터",
]


def embed_corpus(model: SentenceTransformer, corpus: list[str]) -> torch.Tensor:
    """코퍼스 전체를 한 번만 임베딩하고 CPU 텐서로 반환 (모델 디바이스 무관)."""
    emb = model.encode(corpus, batch_size=16, show_progress_bar=True,
                       convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu()


def top3_results(query: str, corpus: list[str], corpus_emb: torch.Tensor,
                 model: SentenceTransformer) -> list[tuple[float, str]]:
    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = torch.nn.functional.cosine_similarity(q_emb.cpu().unsqueeze(0), corpus_emb)
    top_idx = scores.topk(3).indices.tolist()
    return [(round(scores[i].item(), 4), corpus[i][:80] + "…") for i in top_idx]


def compare_models(base_model: SentenceTransformer, ft_model: SentenceTransformer,
                   corpus: list[str]) -> None:
    print("\n" + "=" * 70)
    print("  학습 전후 검색 결과 비교 (Top-3)")
    print("=" * 70)

    print("\n[베이스 모델] 코퍼스 임베딩 중...")
    base_emb = embed_corpus(base_model, corpus)
    print("[Fine-tuned 모델] 코퍼스 임베딩 중...")
    ft_emb = embed_corpus(ft_model, corpus)

    for q in EVAL_QUERIES:
        print(f"\n질문: {q}")
        print(f"{'─'*65}")
        base_res = top3_results(q, corpus, base_emb, base_model)
        ft_res = top3_results(q, corpus, ft_emb, ft_model)

        print(f"  {'순위':<4} {'베이스 모델':^35} {'Fine-tuned 모델':^35}")
        print(f"  {'─'*4} {'─'*35} {'─'*35}")
        for rank, (br, fr) in enumerate(zip(base_res, ft_res), 1):
            b_score, b_doc = br
            f_score, f_doc = fr
            print(f"  {rank:<4} [{b_score:.4f}] {b_doc[:28]:<28}  [{f_score:.4f}] {f_doc[:28]}")


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    start_total = time.time()

    # 1) 디바이스
    device = check_device()

    # 2) 베이스 모델 로딩 — 비교용이므로 CPU에 올려 GPU VRAM 절약
    print(f"\n[기능1] 베이스 모델 로딩: {BASE_MODEL}")
    print("  (베이스 모델은 비교용 → CPU, 학습 모델만 GPU 사용)")
    base_model = SentenceTransformer(BASE_MODEL, device="cpu")
    print("  → 베이스 모델 로딩 완료")

    # 3) ChromaDB 데이터 로딩
    print(f"\n[기능2] ChromaDB 데이터 로딩 중...")
    documents, metadatas = load_documents()

    # 4) 학습 쌍 생성
    anchors, positives = build_pairs(documents, metadatas)

    # 5) HuggingFace Dataset 구성
    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    print(f"[학습데이터] Dataset 크기: {len(dataset)}")

    # 6) Fine-tuning
    print(f"\n[기능3] Contrastive Learning 학습 시작 (epoch={EPOCHS}, batch={BATCH_SIZE})")
    print(f"  손실함수: MultipleNegativesRankingLoss")

    # 학습용 모델은 별도 인스턴스 (베이스 모델을 비교용으로 유지)
    ft_model = SentenceTransformer(BASE_MODEL, device=device)
    loss = MultipleNegativesRankingLoss(ft_model)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,             # 활성화 메모리 대폭 절감 (속도 약간 감소)
        warmup_steps=0.1,                        # float → warmup ratio (Transformers v5+)
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        logging_steps=50,
        save_strategy="no",
    )

    trainer = SentenceTransformerTrainer(
        model=ft_model,
        args=train_args,
        train_dataset=dataset,
        loss=loss,
    )

    # 학습 직전 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  GPU 여유 메모리: {free_gb:.1f} GB")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"  → 학습 완료 ({elapsed:.1f}초)")

    # 7) 학습 전후 비교 (코퍼스: 샘플링된 문서 사용)
    print(f"\n[기능4] 학습 전후 검색 품질 비교")
    compare_models(base_model, ft_model, documents[:200])

    # 8) 모델 저장
    print(f"\n[기능5] Fine-tuned 모델 저장: {OUTPUT_DIR}")
    ft_model.save(OUTPUT_DIR)
    print(f"  → 저장 완료")

    total_elapsed = time.time() - start_total
    print(f"\n[완료] 전체 소요 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")


if __name__ == "__main__":
    main()
