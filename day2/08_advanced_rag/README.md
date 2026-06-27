# 📁 Day 2 · 08 · Advanced RAG — HyDE / Reranking / Ensemble / Parent Document

## 이 실습에서 배우는 것
- 기본 RAG(검색 증강 생성)가 실무에서 부딪히는 4가지 한계 체감
- 각 한계를 해결하는 Advanced RAG 기법 4가지 이해
- Pre-Retrieval → Retrieval → Post-Retrieval 파이프라인 구조 이해
- 기본 RAG와 Advanced RAG를 같은 화면에서 직접 비교

---

## 📂 이 폴더의 파일 구성

```
day2/08_advanced_rag/
└── advanced_rag_app.py      ← 실습에서 Claude Code로 직접 만들기
```

---

## 🔍 기본 RAG의 한계 — 왜 Advanced RAG가 필요한가

기본 RAG(질문 → 검색 → 답변)는 단순하지만, 실무 데이터에서는 아래 4가지 문제가 자주 발생합니다.

| 문제 | 설명 | 해결 기법 |
|------|------|----------|
| 질문-문서 불일치 | 질문 형태와 문서 형태가 다름 | **HyDE** |
| 검색 품질 저하 | Top-k 결과 중 관련 없는 문서 포함 | **Reranking** |
| 키워드 누락 | 의미는 같지만 단어가 다른 경우 | **Ensemble Retriever** |
| 문맥 손실 | 작은 청크로 인한 맥락 부족 | **Parent Document Retriever** |

---

## 📊 Advanced RAG 기법 체계

```
Advanced RAG
├── Pre-Retrieval (검색 전)
│   ├── HyDE (가상 문서 생성)
│   └── Query Expansion (쿼리 확장)
├── Retrieval (검색)
│   ├── Ensemble Retriever (BM25 + 시맨틱)
│   └── Parent Document Retriever
└── Post-Retrieval (검색 후)
    ├── Reranking (재순위화)
    └── Context Compression (컨텍스트 압축)
```

---

## 비유로 이해하기

```
HyDE                      = 질문을 그대로 던지지 않고, "이런 답이 나올 것 같다"는
                            가상의 답안을 먼저 써본 뒤 그 답안과 비슷한 문서를 찾는 사람

Ensemble Retriever        = 도서관에서 "키워드 검색대"와 "의미 기반 사서" 둘 다에게
                            동시에 물어보고 결과를 합치는 방식

Parent Document Retriever = 문장 한 줄만 보지 않고, 그 문장이 속한 문단 전체를
                            함께 가져다주는 사서

Reranking                 = 1차로 후보 10권을 뽑아온 뒤, 더 꼼꼼히 읽을 줄 아는
                            전문가가 다시 순서를 매겨주는 단계
```

---

## 기법별 핵심 비교표

| 기법 | 적용 단계 | 해결하는 문제 | 비용/속도 영향 |
|------|----------|--------------|---------------|
| HyDE | Pre-Retrieval | 질문과 문서의 표현 차이 | LLM 호출 1회 추가 (속도↓) |
| Ensemble Retriever | Retrieval | 키워드 검색과 의미 검색의 사각지대 | 검색 2배 (속도 소폭↓) |
| Parent Document Retriever | Retrieval | 작은 청크의 문맥 부족 | 저장 용량↑ (속도 영향 적음) |
| Reranking | Post-Retrieval | Top-k 안의 관련 없는 문서 | Rerank 모델 호출 추가 (속도↓) |

---

## 🖥 실습 순서

---

### 사전 확인 — 기본 RAG가 먼저 준비되어 있어야 합니다

```bash
# ① ChromaDB 확인 (day2/04_chromadb 실습 완료 후)
cd day2/04_chromadb
python -c "
import chromadb
db  = chromadb.PersistentClient(path='chroma_db')
col = db.get_collection('stock_data')
print(f'✅ 벡터 DB 연결됨: {col.count():,}건')
"
cd ../08_advanced_rag

# ② OpenAI API 확인
cd ../01_openai
python hello_openai.py
cd ../08_advanced_rag
```

> day2/08_comparison(기본 RAG vs 일반 LLM vs Fine-tuning) 실습을 먼저 완료했다면
> 이번 실습이 훨씬 수월합니다. 같은 ChromaDB(`../04_chromadb/chroma_db`)를 재사용합니다.

---

### 필요 패키지 설치

```bash
pip install langchain langchain-openai langchain-community rank_bm25 cohere
```

> Reranking은 Cohere Rerank API 또는 로컬 cross-encoder 모델 둘 다 가능합니다.
> 이번 실습은 무료로 바로 실습할 수 있는 **로컬 cross-encoder 방식**을 기본으로 합니다.

```bash
pip install sentence-transformers
```

---

### STEP 1 · 폴더 이동

```bash
cd day2/08_advanced_rag
```

---

### STEP 2 · Claude Code 실행

```bash
claude
```

---

### STEP 3 · [Claude Code 실습] advanced_rag_app.py 만들기

아래 프롬프트를 **그대로 복사해서 붙여넣기** 하세요:

```
day2/08_advanced_rag/advanced_rag_app.py 파일을 만들어줘.

기본 RAG와 Advanced RAG(HyDE + Ensemble + Parent Document + Reranking)를
같은 화면에서 비교하는 Streamlit 웹앱이야.

기본 설정:
  페이지 제목: "🚀 Advanced RAG 실습"
  레이아웃: wide
  .env 파일: ../../.env 경로에서 읽기
  ChromaDB 경로: ../04_chromadb/chroma_db, 컬렉션명: stock_data

사이드바 구성:
  적용할 기법을 체크박스로 선택할 수 있게 만들어줘:
    ☐ HyDE (가상 문서 생성)
    ☐ Ensemble Retriever (BM25 + 시맨틱)
    ☐ Parent Document Retriever
    ☐ Reranking (재순위화)
  체크박스 조합에 따라 파이프라인이 동적으로 바뀌어야 함
  예시 질문 버튼 5개도 사이드바에 추가:
    "삼성전자 12월 주가는?"
    "거래량이 가장 많았던 종목은?"
    "최근 반도체 업종 동향은?"
    "RSI 지표가 뭐야?"
    "코스피 지수 전망 알려줘"

메인 화면 — 2개 컬럼으로 좌우 비교:

[왼쪽 컬럼] 🔍 기본 RAG
  질문 임베딩 → ChromaDB 유사도 검색 (k=5) → 검색 결과 + 질문을 GPT에 전달
  "📄 검색된 문서" expander로 결과 표시
  하단에 "⏱ 응답시간: N초" 표시

[오른쪽 컬럼] 🚀 Advanced RAG
  사이드바에서 체크한 기법만 순서대로 적용:

  1) HyDE 체크 시:
     - 질문을 바로 검색하지 않고, 먼저 GPT에게
       "이 질문에 대한 가상의 답변을 작성해줘"라고 요청
     - 생성된 가상 답변을 임베딩해서 검색 쿼리로 사용
     - "💭 생성된 가상 문서" expander로 가상 답변 표시

  2) Ensemble Retriever 체크 시:
     - BM25Retriever(키워드 기반) + ChromaDB 시맨틱 검색을 함께 사용
     - langchain의 EnsembleRetriever로 두 결과를 weights=[0.5, 0.5]로 결합
     - 체크 안 하면 시맨틱 검색만 사용

  3) Parent Document Retriever 체크 시:
     - 작은 청크(child, 500자)로 검색하되
     - 검색된 청크가 속한 원본 문단(parent, 2000자)을 함께 가져와서 컨텍스트로 사용
     - parent_splitter / child_splitter 둘 다 정의
     - 체크 안 하면 검색된 청크만 그대로 사용

  4) Reranking 체크 시:
     - sentence-transformers의 cross-encoder 모델
       (cross-encoder/ms-marco-MiniLM-L-6-v2) 로드
     - 검색된 후보 문서들을 질문과 함께 cross-encoder에 넣어 점수 재계산
     - 점수 높은 순으로 재정렬 후 상위 3개만 최종 사용
     - "📊 재정렬 점수" expander로 재정렬 전/후 순서 비교 표시

  최종적으로 선택된 기법을 모두 거친 컨텍스트 + 질문을 GPT에 전달해서 답변 생성
  "🛠 적용된 기법" 으로 현재 체크된 기법 목록을 답변 위에 배지(badge) 형태로 표시
  하단에 "⏱ 응답시간: N초" 표시

공통 UI:
  상단에 질문 입력창 (st.text_input) + 비교 실행 버튼
  두 컬럼 답변을 나란히 배치 (st.columns(2))
  각 방식의 API 호출 오류는 해당 컬럼에만 st.error 표시
  (한 방식이 실패해도 나머지는 정상 작동)

하단 비교 분석 섹션:
  "📊 같은 질문, 기본 RAG와 Advanced RAG의 검색 결과가 어떻게 달랐나요?" 안내 텍스트
  두 방식이 가져온 문서가 다를 경우 "문서 차이 발견" 하이라이트 표시

기타 조건:
  체크박스를 하나도 선택하지 않으면 오른쪽 컬럼은 기본 RAG와 동일하게 동작
  (Advanced RAG가 "기능을 추가하는 방식"임을 보여주기 위함)

파일명: advanced_rag_app.py
```

---

### STEP 4 · advanced_rag_app.py 실행

```bash
streamlit run advanced_rag_app.py
```

---

### STEP 5 · 기법별 비교 테스트

#### 테스트 1 · HyDE 효과 확인 (질문-문서 불일치 해결)
```
체크: HyDE만 ON
질문: "요즘 잘 나가는 종목 뭐야?"

예상 결과:
🔍 기본 RAG      → 질문 자체("잘 나가는 종목")로 검색 → 문서와 표현이 달라 검색 품질 낮음
🚀 Advanced RAG  → "최근 거래량과 상승률이 높은 종목은 ○○이다" 같은
                  가상 답변을 먼저 만들고, 그 답변과 비슷한 실제 문서를 검색
                  → 더 정확한 문서 매칭
```

#### 테스트 2 · Ensemble Retriever 효과 확인 (키워드 누락 해결)
```
체크: Ensemble Retriever만 ON
질문: "SK하이닉스 거래량 폭증"

예상 결과:
🔍 기본 RAG      → "거래량 폭증"이라는 표현이 문서에 없으면 시맨틱 검색만으로 놓칠 수 있음
🚀 Advanced RAG  → BM25가 "거래량", "SK하이닉스" 키워드를 직접 매칭해서 보완
                  → 시맨틱 검색이 놓친 문서까지 함께 검색됨
```

#### 테스트 3 · Parent Document Retriever 효과 확인 (문맥 손실 해결)
```
체크: Parent Document Retriever만 ON
질문: "삼성전자 실적 발표 이후 시장 반응은?"

예상 결과:
🔍 기본 RAG      → 작은 청크 하나만 가져와서 앞뒤 맥락이 끊긴 답변
🚀 Advanced RAG  → 해당 청크가 포함된 문단 전체를 가져와서
                  더 맥락이 풍부한 답변
```

#### 테스트 4 · Reranking 효과 확인 (검색 품질 저하 해결)
```
체크: Reranking만 ON
질문: "반도체 업종 동향"

예상 결과:
🔍 기본 RAG      → Top-5 중 일부는 질문과 관련성이 낮은 문서 포함 가능
🚀 Advanced RAG  → cross-encoder가 질문-문서 쌍을 직접 채점해서
                  관련성 높은 순으로 재정렬 → 상위 3개만 사용
```

#### 테스트 5 · 전체 기법 동시 적용
```
체크: 4개 모두 ON
질문: "최근 반도체 업종 동향은?"

예상 결과:
파이프라인: HyDE로 가상 답변 생성 → Ensemble로 후보 확장 검색
          → Parent Document로 문맥 확장 → Reranking으로 최종 정제
응답 시간이 기본 RAG보다 늘어나지만, 답변의 근거 문서 품질이 향상됨을 확인
```

---

### STEP 6 · 추가 실습 — 파이프라인 확장

**미션 1 · Query Expansion 추가**
```
advanced_rag_app.py 에 Query Expansion 체크박스를 추가해줘.
원래 질문을 GPT로 3가지 유사 질문으로 확장해서
각각 검색한 결과를 합치는 방식으로 구현해줘.
```

**미션 2 · Context Compression 추가**
```
advanced_rag_app.py 에 Context Compression 체크박스를 추가해줘.
검색된 문서에서 질문과 직접 관련 없는 문장은 GPT로 걸러내고
핵심 문장만 남겨서 컨텍스트 길이를 줄이는 기능을 구현해줘.
```

**미션 3 · 기법별 비용/속도 트레이드오프 대시보드**
```
advanced_rag_app.py 사이드바에
지금까지 테스트한 기법 조합별로
평균 응답시간과 누적 API 호출 횟수를
표로 누적 표시해줘.
```

---

## 💡 이 실습의 핵심 포인트

```
Advanced RAG는 "기본 RAG를 대체"하는 것이 아니라
"기본 RAG의 약점을 보완하는 부품을 추가"하는 것입니다.

Pre-Retrieval (검색 전)  → 질문 자체를 더 좋은 검색 쿼리로 바꾸는 단계
Retrieval (검색)         → 검색 방식과 범위를 넓히고 깊게 만드는 단계
Post-Retrieval (검색 후) → 가져온 결과 중 진짜 필요한 것만 골라내는 단계

모든 기법을 항상 쓸 필요는 없고,
어떤 문제(질문-문서 불일치 / 키워드 누락 / 문맥 손실 / 검색 품질 저하)가
실제로 발생하는지 먼저 확인한 뒤, 맞는 기법만 골라 쓰는 것이 실무 접근법입니다.
```

---

## ❓ 자주 묻는 질문

**Q. HyDE를 켜면 응답이 왜 느려지나요?**
```
HyDE는 검색 전에 GPT를 한 번 더 호출해서 가상 문서를 만들기 때문입니다.
"검색 1번 + LLM 호출 1번(가상 문서) + LLM 호출 1번(최종 답변)" 구조라
기본 RAG보다 LLM 호출이 1회 더 늘어납니다.
```

**Q. Ensemble Retriever의 weights는 어떻게 정하나요?**
```
weights=[0.5, 0.5]는 BM25와 시맨틱 검색을 동등하게 반영하는 기본값입니다.
키워드(종목명, 숫자)가 중요한 질문이 많다면 BM25 비중을 높이고(예: [0.7, 0.3]),
의미·문맥이 중요한 질문이 많다면 시맨틱 비중을 높이세요(예: [0.3, 0.7]).
```

**Q. Reranking 모델을 매번 로드하면 느린데요?**
```
실습에서는 이해를 위해 매번 로드하지만, 실제 서비스에서는
st.cache_resource로 cross-encoder 모델을 한 번만 로드해서 재사용합니다.
```

**Q. 4가지 기법을 다 적용해도 답변이 비슷해요**
```
day2/08_comparison 실습과 마찬가지로, 추상적인 질문보다
구체적인 날짜·종목명·수치를 포함한 질문이 기법 간 차이를 더 잘 보여줍니다.
또한 ChromaDB에 저장된 데이터량이 적으면 차이가 작게 느껴질 수 있습니다.
```

**Q. 모든 기법을 항상 켜두고 쓰면 안 되나요?**
```
가능하지만 비용·속도 부담이 커집니다 (LLM 호출 증가 + Rerank 모델 호출 증가).
실무에서는 보통 데이터 특성에 맞는 기법 1~2개만 선택해서 적용합니다.
```