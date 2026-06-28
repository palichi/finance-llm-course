# 📁 Day 3 · 02 · KùzuDB로 배우는 Graph RAG

## 이 실습에서 배우는 것
- KùzuDB 설치와 기본 동작 원리 이해 (서버 없이 동작하는 임베디드 그래프 DB)
- Cypher 쿼리 언어로 노드·관계 만들고 조회하기
- CSV 데이터를 그래프로 대량 적재하기 (COPY FROM)
- LangChain과 결합해서 자연어 질문 → Cypher 자동 생성 → 답변까지 이어지는 Graph RAG 체인 만들기

---

## 📂 이 폴더의 파일 구성

```
day3/02_graph_rag/
├── 01_basic_kuzu.py        ← STEP 4에서 직접 만들기 (스키마/노드/관계/조회)
├── 02_csv_import.py        ← STEP 6에서 직접 만들기 (CSV → 그래프)
├── 03_graph_rag_chain.py   ← STEP 8에서 직접 만들기 (LangChain 연동)
└── data/
    ├── pest.csv            ← 실습용 샘플 데이터 (병해충)
    ├── crop.csv            ← 실습용 샘플 데이터 (작물)
    └── infects.csv          ← 관계 데이터 (병해충 → 작물)
```

---

## 🔍 KùzuDB란

KùzuDB는 **서버를 띄울 필요 없이, 파일 하나(폴더 하나)가 곧 데이터베이스가 되는 그래프 DB**입니다.

```
일반적인 DB 사용 흐름           KùzuDB 사용 흐름
계정 생성 → 서버 접속 정보       파일 경로 지정
입력 → 인증 → 쿼리 실행         → 바로 쿼리 실행

설치 / 접속 절차가 여러 단계     pip install 한 줄 + 경로 지정으로 끝
```

쿼리 언어는 **Cypher**를 사용합니다. "MATCH ... RETURN ..." 형태로
"이런 모양의 노드와 관계를 찾아서 보여줘"라고 선언적으로 작성합니다.

---

## 비유로 이해하기

```
KùzuDB = 내 책상 위에 책장을 직접 두고 내가 바로 찾아보는 것
       (서버 접속, 인증 절차 없음. 폴더 하나가 곧 책장 전체)

Cypher 쿼리 = "빨간 표지의 책들 중에서, 저자가 OOO인 책만 찾아줘"
            라고 사서에게 말하듯 패턴으로 질문하는 것
```

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| Database | `kuzu.Database("./경로")` — 폴더 경로 하나가 DB 전체 |
| Connection | 해당 Database에 쿼리를 보내는 통로 (`kuzu.Connection(db)`) |
| NODE TABLE | 그래프의 점(노드) 종류를 정의하는 스키마 (예: Pest, Crop) |
| REL TABLE | 노드 사이의 관계(엣지)를 정의하는 스키마 (예: Infects) |
| Cypher | 그래프를 조회·생성·수정하는 쿼리 언어 |
| COPY FROM | CSV 파일을 통째로 그래프에 적재하는 명령 |

---

## 🖥 실습 순서

---

### STEP 1 · 사전 확인

```bash
python --version    # 3.10 이상 필요
```

---

### STEP 2 · 필요 패키지 설치

```bash
pip install kuzu pandas
```

LangChain 연동까지 실습하려면 (STEP 8부터 필요):

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

> ⚠️ **참고**: 독립 패키지였던 `langchain-kuzu`는 현재 archived(유지보수 종료) 상태입니다.
> 이번 실습은 `langchain_community.graphs.kuzu_graph.KuzuGraph` 와
> `langchain_community.chains.graph_qa.kuzu.KuzuQAChain` 을 사용합니다.
> `langchain-community` 안에 이미 포함되어 있어 별도 설치가 필요 없습니다.

설치 확인:

```bash
python -c "import kuzu; print(kuzu.__version__)"
```

버전 번호가 출력되면 정상 설치된 것입니다.

---

### STEP 3 · 폴더 이동 및 Claude Code 실행

```bash
cd day3/02_graph_rag
claude
```

---

### STEP 4 · [Claude Code 실습] 01_basic_kuzu.py 만들기

가장 먼저 **KùzuDB가 어떻게 동작하는지** 손으로 익히는 단계입니다.
아래 프롬프트를 **그대로 복사해서 붙여넣기** 하세요:

```
day3/02_graph_rag/01_basic_kuzu.py 파일을 만들어줘.

KùzuDB의 가장 기본적인 사용법을 익히는 스크립트야.
주제는 "병해충(Pest) - 작물(Crop)" 관계로 만들어줘.

1) kuzu.Database("./kuzu_db") 로 로컬 DB 생성
   (./kuzu_db 폴더가 곧 데이터베이스 파일임을 주석으로 설명)

2) 스키마 생성:
   - NODE TABLE Pest(name STRING, PRIMARY KEY(name))
   - NODE TABLE Crop(name STRING, PRIMARY KEY(name))
   - REL TABLE Infects(FROM Pest TO Crop, severity STRING)
     (severity는 "낮음"/"보통"/"심함" 같은 값)

3) 노드와 관계를 Cypher로 직접 5개 이상 생성:
   예) 진딧물, 응애, 노균병 같은 병해충 3종
       사과나무, 배나무, 장미 같은 작물 3종
       각각 INFECTS 관계로 연결 (severity 포함)

4) 조회 쿼리 3개를 실행하고 결과를 print:
   - 특정 병해충이 영향을 주는 작물 전체 조회
   - 특정 작물에 영향을 주는 병해충 전체 조회
   - severity가 "심함"인 관계만 필터링해서 조회

5) 각 단계마다 한글 주석으로
   "지금 이 코드가 무엇을 하는지" 설명을 달아줘
   (별도 서버 접속 코드 없이 바로 동작한다는 점을 강조하는 주석 포함)

파일명: 01_basic_kuzu.py
```

---

### STEP 5 · 01_basic_kuzu.py 실행

```bash
python 01_basic_kuzu.py
```

실행 후 `kuzu_db` 라는 폴더가 생성된 걸 확인하세요. **이 폴더 자체가 데이터베이스입니다.**

```bash
ls -la kuzu_db/
```

> 💡 이 폴더를 통째로 복사하거나 압축하면 그래프 DB 전체를 옮길 수 있습니다.
> 다른 컴퓨터에서 작업을 이어가려면 이 폴더만 그대로 가져가면 됩니다.

---

### STEP 6 · [Claude Code 실습] CSV로 실제 데이터 불러오기

손으로 5개씩 만드는 건 실습용이고, 실무에서는 CSV로 대량 데이터를 한 번에 넣습니다.
아래 프롬프트를 **그대로 복사해서 붙여넣기** 하세요:

```
day3/02_graph_rag/02_csv_import.py 파일을 만들어줘.
같은 폴더에 data/pest.csv, data/crop.csv, data/infects.csv 도 함께 만들어줘.

data/pest.csv 컬럼: name, category
  (category 예: 해충, 병원균, 잡초)
  최소 8개 행 (예: 진딧물,해충 / 노균병,병원균 / 응애,해충 등
  실제 한국 농업에서 흔한 병해충 이름으로 채워줘)

data/crop.csv 컬럼: name, type
  (type 예: 과수, 화훼, 채소)
  최소 6개 행 (사과나무, 장미, 배추 등)

data/infects.csv 컬럼: pest_name, crop_name, severity
  pest.csv와 crop.csv에 있는 이름들을 조합해서
  최소 10개의 관계 행을 만들어줘 (severity: 낮음/보통/심함 중 하나)

02_csv_import.py 내용:
1) kuzu.Database("./kuzu_db_csv") 로 새 DB 생성
   (01번 실습과 분리된 별도 DB)

2) 스키마 생성:
   - NODE TABLE Pest(name STRING, category STRING, PRIMARY KEY(name))
   - NODE TABLE Crop(name STRING, type STRING, PRIMARY KEY(name))
   - REL TABLE Infects(FROM Pest TO Crop, severity STRING)

3) COPY FROM 으로 CSV 3개를 한 번에 적재
   (conn.execute('COPY Pest FROM "data/pest.csv" (header=true)') 형태)

4) 적재 확인을 위해:
   - 전체 Pest 개수, 전체 Crop 개수, 전체 Infects 관계 개수를 각각 조회해서 print
   - category가 "해충"인 Pest 중 severity가 "심함"인 Crop을 함께 조회해서 표 형태로 print
     (conn.execute(...).get_as_df() 사용해서 pandas DataFrame으로 출력)

5) 각 단계 한글 주석 포함

파일명: 02_csv_import.py
```

---

### STEP 7 · 02_csv_import.py 실행

```bash
python 02_csv_import.py
```

---

### STEP 8 · [Claude Code 실습] LangChain 연동 — 자연어로 그래프 질문하기

여기서부터가 진짜 **Graph RAG**입니다. 자연어 질문을 LLM이 Cypher로 변환하고, 그 결과를 다시 자연어 답변으로 만들어줍니다.
아래 프롬프트를 **그대로 복사해서 붙여넣기** 하세요:

```
day3/02_graph_rag/03_graph_rag_chain.py 파일을 만들어줘.

02_csv_import.py 에서 만든 ./kuzu_db_csv 를 그대로 사용해서
자연어 질문 → Cypher 자동 생성 → 답변까지 이어지는
Graph RAG 체인을 만드는 스크립트야.

기본 설정:
  .env 파일: ../../.env 경로에서 OPENAI_API_KEY 읽기 (python-dotenv 사용)

구현 내용:
1) kuzu.Database("./kuzu_db_csv") 로 기존 DB 연결 (새로 만들지 않음)

2) from langchain_community.graphs import KuzuGraph
   graph = KuzuGraph(db, allow_dangerous_requests=True)
   graph.refresh_schema() 로 스키마 갱신

3) from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
   from langchain_openai import ChatOpenAI
   KuzuQAChain.from_llm() 으로 체인 생성
     - llm: ChatOpenAI(model="gpt-4o-mini", temperature=0)
     - graph: 위에서 만든 graph
     - verbose=True (생성된 Cypher를 콘솔에 보여주기 위함)
     - allow_dangerous_requests=True

4) while 루프로 터미널에서 계속 질문을 입력받기:
   - 사용자가 한글로 질문 입력 (예: "진딧물이 영향을 주는 작물은 뭐야?")
   - chain.invoke(질문) 실행
   - 생성된 Cypher 쿼리와 최종 답변을 구분해서 출력
     "🔧 생성된 Cypher: ..."
     "💬 답변: ..."
   - "exit" 입력 시 종료

5) 에러 처리:
   - Cypher 생성에 실패하거나 스키마에 없는 내용을 물으면
     "스키마에서 답을 찾을 수 없습니다. 다른 질문을 시도해보세요." 출력
   - try-except로 전체 감싸기

6) 한글 주석으로 각 단계 설명
   (특히 KuzuQAChain이 내부적으로
   "질문 → Cypher 생성 → 그래프 실행 → 결과를 자연어로 재구성"
   4단계를 거친다는 것을 주석으로 설명)

파일명: 03_graph_rag_chain.py
```

---

### STEP 9 · 03_graph_rag_chain.py 실행 및 질문 테스트

```bash
python 03_graph_rag_chain.py
```

#### 테스트 질문 예시

```
질문 1: "진딧물이 영향을 주는 작물은 뭐야?"
예상 동작: Pest(name='진딧물')-[:Infects]->(Crop) 형태의 Cypher 생성
         → 관련 작물 목록을 자연어로 답변

질문 2: "심함 단계로 피해를 주는 병해충은 뭐가 있어?"
예상 동작: severity = '심함' 조건이 포함된 Cypher 생성
         → 해당 병해충 목록 답변

질문 3: "사과나무에 영향을 주는 해충 종류는?"
예상 동작: Crop(name='사과나무')와 Pest(category='해충') 조건이
         결합된 Cypher 생성

질문 4 (의도적으로 스키마 밖 질문): "오늘 날씨 어때?"
예상 동작: 스키마에 없는 내용이므로 답을 찾지 못한다는 안내가 나와야 정상
         (그래프에 없는 걸 지어내지 않는 것이 RAG의 핵심)
```

---

### STEP 10 · 추가 실습 — 기능 확장

**미션 1 · 응답 시간 측정**
```
03_graph_rag_chain.py 에 응답 시간을 측정하는 코드를 추가해줘.
질문마다 "Cypher 생성 시간"과 "전체 응답 시간"을 따로 측정해서
출력해줘.
```

**미션 2 · 더 큰 데이터셋으로 확장**
```
data/pest.csv, crop.csv, infects.csv 를 각각 100개 행으로 늘려서
랜덤 생성하는 스크립트를 만들어줘.
적재 시간과 조회 속도가 데이터 양에 따라 어떻게 변하는지 측정해줘.
```

**미션 3 · 벡터 검색 결합 (KùzuDB 내장 vector index)**
```
03_graph_rag_chain.py 를 확장해서, 병해충 설명(description) 컬럼을
임베딩한 뒤 KùzuDB의 벡터 인덱스 기능을 사용해
"증상이 비슷한 병해충 찾기" 기능을 추가해줘.
```

**미션 4 · Streamlit 웹앱으로 만들기**
```
03_graph_rag_chain.py 를 기반으로
streamlit_app.py 를 만들어줘.
질문 입력창 + 답변 표시 + "생성된 Cypher 보기" expander로
구성된 웹앱으로 만들어줘.
```

---

## 💡 이 실습의 핵심 포인트

```
KùzuDB의 핵심은 "설치와 시작이 가볍다"는 것입니다.

별도 서버를 띄우거나 계정을 만들 필요 없이
pip install kuzu 한 줄 + 폴더 경로 지정만으로
그래프 데이터베이스를 바로 사용할 수 있습니다.

Pre-Retrieval (질문 이해) → Retrieval (그래프 검색) → Generation (답변 생성)
이라는 RAG의 큰 흐름은 동일하지만,
KùzuDB + LangChain의 KuzuQAChain을 쓰면
"질문 → Cypher 자동 생성 → 그래프 조회 → 자연어 답변"
이 한 번의 chain.invoke() 호출로 끝납니다.
```

---

## ❓ 자주 묻는 질문

**Q. `langchain-kuzu` 패키지를 따로 설치해야 하나요?**
```
아닙니다. 독립 패키지였던 langchain-kuzu는 현재 archived 상태이며,
이 실습은 langchain-community 안에 포함된
langchain_community.graphs.kuzu_graph.KuzuGraph 를 사용합니다.
별도 설치 없이 langchain-community 만 설치하면 됩니다.
```

**Q. kuzu_db 폴더를 지우고 다시 만들면 안 되나요?**
```
가능합니다. KùzuDB는 폴더 하나가 곧 데이터베이스이므로,
해당 폴더를 삭제(rm -rf kuzu_db)하고 스크립트를 다시 실행하면
깨끗한 상태에서 새로 시작됩니다.
```

**Q. KuzuQAChain이 이상한 Cypher를 생성해요**
```
KuzuQAChain은 그래프의 스키마(테이블/컬럼 이름)를 보고 Cypher를 생성하므로,
질문에 등장하는 단어가 스키마의 실제 값(예: '진딧물', '심함')과
정확히 일치하지 않으면 빈 결과나 잘못된 Cypher가 나올 수 있습니다.
질문을 스키마에 있는 표현과 비슷하게 바꿔서 다시 시도해보세요.
verbose=True 로 설정해두면 어떤 Cypher가 생성됐는지 바로 확인할 수 있어
디버깅이 쉽습니다.
```

**Q. 데이터가 많아지면 느려지지 않나요?**
```
KùzuDB는 컬럼형 저장 구조를 사용해서 대량 데이터 조회·분석에
최적화되어 있습니다. 실습 수준의 데이터(수백~수만 건)에서는
체감되는 속도 저하 없이 사용할 수 있습니다.
```

**Q. 여러 프로그램이 같은 kuzu_db 폴더에 동시에 접근하면 어떻게 되나요?**
```
KùzuDB는 한 프로세스가 DB 파일을 점유하는 임베디드 구조입니다.
같은 DB 폴더를 여러 스크립트에서 동시에 열면 충돌이 날 수 있으므로,
한 번에 하나의 스크립트/프로세스만 접근하도록 사용하는 것이 안전합니다.
```
