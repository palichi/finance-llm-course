# 🌳 식물 병충해 진단 RAG 시스템 — 프로젝트 설계서

> 바이브 코딩(Claude Code) 작업 지시를 위한 마스터 문서
> 작성 목적: 이 문서를 그대로 Claude Code에 붙여넣거나, 단계별 섹션을 순서대로 지시하면 시스템이 구축되도록 설계함

---

## 0. 프로젝트 한 줄 요약

사용자가 **나무/작물 이름 + 증상**을 입력하면, 공공데이터(국가병해충 정보)를 기반으로 한 RAG가 **발병 원인 분석 + 방제(처방) 안내**를 생성해주는 웹 서비스를 구축한다.

| 항목 | 내용 |
|---|---|
| 대상 | 수목(조경수), 농작물, 화훼 등 **모든 식물**의 병해충 |
| 핵심 기능 | 증상 기반 원인 진단, 방제법 추천, 근거 출처 표시 |
| 개발 방식 | 바이브 코딩 (Claude Code 단계별 프롬프트 지시) |
| 사업 목적 | 조경기사 경력 + AI Agent 기술을 결합한 포트폴리오 겸 실제 서비스 |

---

## 1. 데이터 소스 (공공데이터)

### 1-1. 국가농작물병해충관리시스템 (NCPMS, 농촌진흥청)
- **공식 사이트**: ncpms.rda.go.kr
- **공공데이터포털 등록명**: "농촌진흥청_작물 병해충 검색 서비스" (data.go.kr 검색)
- 제공 정보: 작물별/병해충별 **발생환경, 증상설명, 방제방법, 병원체명, 대상작물명**, 피해사진 URL
- 직접 API 방식: `http://ncpms.rda.go.kr/npmsAPI/service?apiKey={키}&serviceCode={SVC코드}&...`
- **주의**: 작물 중심 DB이므로 조경수(소나무, 단풍나무 등 정원수)는 별도 보강 필요 → 1-2, 1-3과 결합

### 1-2. 농사로 (nongsaro.go.kr)
- 작목별 농업기술정보, 병해충 방제정보 Open API 제공
- 조경수/임목 관련 자료는 산림청 데이터와 교차 검증 필요

### 1-3. 산림청 / 국립산림과학원 (보강 데이터, 조경수 특화)
- "산림병해충 정보시스템", "수목진료" 관련 공공데이터포털 자료 검색 필요
- 소나무재선충병, 흰불나방 등 **조경수 특화 병해충**은 NCPMS만으로 커버 안 되는 경우가 많으므로 필수 보강

### 1-4. 데이터 수집 시 Claude Code에게 줄 작업
> 공공데이터포털(data.go.kr) API 키 발급은 **사람이 직접 신청**해야 함 (즉시 자동승인이 아닌 경우 있음). Claude Code는 발급받은 키로 수집 스크립트만 작성.

---

## 2. 전체 아키텍처 설계

```
[사용자 입력: 식물명 + 증상 텍스트]
        │
        ▼
[Query Understanding] ── 식물명 정규화 (학명/속명 매핑), 증상 키워드 추출
        │
        ▼
[Hybrid Retrieval]
   ├─ Vector Search (ChromaDB, bge-m3 임베딩)  : 증상 설명 유사도 검색
   └─ Graph Search (KùzuDB)                    : 식물-병해충-증상-방제법 관계 탐색
        │
        ▼
[Reranking] (선택, bge-reranker)
        │
        ▼
[LLM 답변 생성] (LangGraph 기반 Agent)
   ├─ 1차: 가능한 병해충 후보 N개 제시 (확률/근거 포함)
   ├─ 2차: 추가 질문 필요 시 사용자에게 되묻기 (예: "잎 뒷면도 변색되었나요?")
   └─ 3차: 최종 진단 + 방제법 + 출처 인용
        │
        ▼
[FastAPI 응답] → 프론트엔드 (Streamlit 또는 React)
```

### 왜 Graph RAG를 병행하는가
병해충 진단은 **"식물 종 → 병해충 종 → 증상 → 환경조건 → 방제법"** 관계가 명확한 그래프 구조다.
단순 벡터 유사도만으로는 "소나무 + 갈변" 같은 모호한 질의에서 오답 후보가 섞이기 쉬움.
→ KùzuDB로 관계 그래프를 만들고, 벡터 검색으로 후보를 좁힌 뒤 그래프에서 관계를 검증하는 **하이브리드 방식** 권장 (귀하의 PPO+RAG 프로젝트와 동일한 설계 철학 재사용 가능).

---

## 3. 그래프 스키마 설계 (KùzuDB)

### 노드(Node) 타입
- `Plant` (식물): name, scientific_name, category(수목/작물/화훼), family
- `Pest` (병해충): name, type(병/해충/생리장해), pathogen
- `Symptom` (증상): description, affected_part(잎/줄기/뿌리/열매)
- `Treatment` (방제법): method, chemical_name, application_timing, organic_yn
- `Environment` (발생환경): season, temperature_range, humidity_condition

### 관계(Edge) 타입
- `(Plant)-[AFFECTED_BY]->(Pest)`
- `(Pest)-[CAUSES]->(Symptom)`
- `(Pest)-[TREATED_BY]->(Treatment)`
- `(Pest)-[OCCURS_IN]->(Environment)`

---

## 4. 기술 스택

| 영역 | 선택 기술 | 비고 |
|---|---|---|
| 백엔드 | FastAPI | 기존 보유 역량 활용 |
| Agent 오케스트레이션 | LangGraph | 다단계 질의-재질의 흐름 구현 |
| Vector DB | ChromaDB | 로컬/경량, 기존 프로젝트와 동일 |
| Graph DB | KùzuDB | Neo4j 무료 티어 한계 회피 (기존 의사결정 재사용) |
| 임베딩 | BAAI/bge-m3 (메모리 제약 시 ko-sroberta-multitask) | 기존 환경 설정 재사용 |
| 프론트엔드 | Streamlit (MVP) → React (정식 버전) | 빠른 검증 우선 |
| 개발 도구 | Claude Code | 본 문서 기반 프롬프트로 단계별 작업 |

---

## 5. 프로젝트 폴더 구조 (Claude Code에게 1차로 생성 요청)

```
plant-pest-rag/
├── data/
│   ├── raw/                  # NCPMS, 농사로 원본 수집 데이터
│   ├── processed/             # 정제된 JSON (식물-병해충-증상-방제)
│   └── graph/                 # KùzuDB 적재용 csv/json
├── src/
│   ├── ingestion/              # 공공데이터 수집 스크립트
│   ├── graph_builder/          # KùzuDB 스키마 생성/적재
│   ├── retrieval/               # Vector + Graph 하이브리드 검색
│   ├── agent/                   # LangGraph 에이전트 (진단 플로우)
│   └── api/                      # FastAPI 라우터
├── app/                         # Streamlit 프론트엔드
├── tests/
├── notebooks/                   # 탐색/검증용 Jupyter
├── .env.example
├── requirements.txt
└── README.md
```

---

## 6. Claude Code 단계별 작업 지시 프롬프트

> 아래 프롬프트를 **순서대로 하나씩** Claude Code에 입력하세요. 한 번에 다 시키지 말고 단계별 완료 확인 후 다음 단계로 진행하는 것을 권장합니다 (바이브 코딩 핵심 원칙).

### Phase 1. 프로젝트 초기화
```
plant-pest-rag 라는 이름의 프로젝트를 생성해줘.
폴더 구조는 다음과 같이 만들어줘: [위 5번 폴더 구조 붙여넣기]
Python 3.11 가상환경(venv)을 설정하고, requirements.txt에 다음 패키지를 포함해줘:
fastapi, uvicorn, langchain, langgraph, chromadb, kuzu, sentence-transformers,
streamlit, python-dotenv, requests, beautifulsoup4, pydantic
.env.example 파일도 만들어서 API 키들(NCPMS_API_KEY, ANTHROPIC_API_KEY 등)을 placeholder로 넣어줘.
```

### Phase 2. 공공데이터 수집 스크립트
```
src/ingestion/ 폴더에 NCPMS Open API(국가농작물병해충관리시스템)에서
작물별/병해충별 데이터를 수집하는 스크립트를 작성해줘.
- 입력: apiKey, serviceCode
- 수집 항목: 작물명, 병해충명, 병원체명, 증상설명, 발생환경, 방제방법, 사진URL
- XML/JSON 응답을 파싱해서 data/raw/ 에 JSON으로 저장
- API 호출 실패 시 재시도 로직과 rate limit 고려
- 한글 인코딩 문제 없도록 처리해줘
※ API 키는 아직 없으니 일단 mock 응답으로 동작 확인 가능하게 만들어줘
```

### Phase 3. 데이터 정제 및 통합 스키마 변환
```
data/raw/ 의 원본 데이터를 읽어서, 다음 통합 스키마의 JSON으로 변환하는
스크립트를 src/ingestion/normalize.py 에 작성해줘.
스키마: { plant_name, scientific_name, category, pest_name, pest_type,
pathogen, symptoms: [], affected_parts: [], environment_conditions: [],
treatments: [{method, chemical_name, timing, organic_yn}], source, source_url }
결과는 data/processed/plant_pest_unified.json 으로 저장해줘.
```

### Phase 4. Graph DB 구축
```
src/graph_builder/ 에 KùzuDB 스키마 생성 및 데이터 적재 스크립트를 작성해줘.
노드: Plant, Pest, Symptom, Treatment, Environment
관계: AFFECTED_BY, CAUSES, TREATED_BY, OCCURS_IN
(스키마 상세는 본 설계서 3번 섹션 참고)
data/processed/plant_pest_unified.json 을 읽어서 그래프에 적재하는 스크립트도 작성해줘.
적재 후 노드/관계 개수를 출력하는 검증 스크립트도 함께 만들어줘.
```

### Phase 5. 벡터 DB 구축
```
src/retrieval/vector_store.py 를 작성해줘.
- ChromaDB에 plant_pest_unified.json의 증상 설명(symptoms)을 임베딩해서 저장
- 임베딩 모델: BAAI/bge-m3 (실패 시 jhgan/ko-sroberta-multitask로 폴백)
- 메타데이터로 plant_name, pest_name, source를 함께 저장
- 컬렉션명은 'plant_pest_symptoms'로 통일
```

### Phase 6. 하이브리드 검색 로직
```
src/retrieval/hybrid_search.py 를 작성해줘.
입력: 식물명(선택), 증상 텍스트
처리:
1) 벡터 검색으로 유사 증상 top-k 후보 추출
2) 후보의 pest_name으로 KùzuDB 그래프 조회 → 관련 Plant, Treatment, Environment 확인
3) 식물명이 입력된 경우 그래프에서 해당 Plant와 AFFECTED_BY 관계가 있는 Pest만 필터링
4) 최종 후보 리스트(병해충명, 신뢰도, 근거 증상, 방제법, 출처)를 반환
```

### Phase 7. LangGraph 진단 에이전트
```
src/agent/diagnosis_agent.py 를 LangGraph로 작성해줘.
플로우:
1) 사용자 입력(식물명+증상) 수신
2) hybrid_search 호출
3) 후보가 명확하면(신뢰도 높고 1개 우세) → 바로 진단 결과 생성 노드로
4) 후보가 모호하면(여러 개 비슷한 점수) → 추가 질문 생성 노드로 분기
   (예: "잎 앞면입니까 뒷면입니까?", "발생 시기가 언제입니까?")
5) 사용자 추가 답변을 받아 재검색 → 최종 진단
6) 최종 출력 형식: { 추정원인, 신뢰도, 증상근거, 방제법(화학적/유기농 구분), 출처 }
시스템 프롬프트에는 "확실하지 않은 경우 추정이라고 명시하고, 농약 사용시 반드시
전문가 상담 권고 문구를 포함하라"는 안전장치를 넣어줘.
```

### Phase 8. FastAPI 백엔드
```
src/api/main.py 에 FastAPI 앱을 만들어줘.
엔드포인트:
- POST /diagnose : {plant_name, symptom_text} → 진단 결과 또는 추가질문
- POST /diagnose/followup : {session_id, answer} → 후속 진단
- GET /plants : 등록된 식물 목록 조회
- GET /health
세션 관리는 간단히 메모리 dict로 시작하고, 추후 Redis로 교체 가능하게 구조화해줘.
```

### Phase 9. Streamlit MVP 프론트엔드
```
app/main.py 에 Streamlit 앱을 작성해줘.
- 식물명 입력 + 증상 텍스트 입력 폼
- "진단하기" 버튼 클릭 시 FastAPI /diagnose 호출
- 결과를 카드 형태로 표시 (병해충명, 신뢰도, 방제법, 출처 링크)
- 추가질문이 오면 답변 입력 UI로 자연스럽게 전환
- 색상은 단조롭게(모노톤) 구성해줘
```

### Phase 10. 테스트 및 검증
```
tests/ 에 다음 테스트를 작성해줘.
- 데이터 적재 검증 (그래프 노드/관계 수, 벡터 컬렉션 문서 수)
- hybrid_search 단위 테스트 (샘플 증상 5개로 검증)
- 알려진 케이스 검증: "소나무 + 잎 갈변 + 송진 누출" → 소나무재선충병 추정이 나오는지
- API 통합 테스트 (TestClient 사용)
```

---

## 7. 향후 확장 (사업화 단계)

| 단계 | 내용 |
|---|---|
| MVP 검증 후 | 이미지 업로드 기반 1차 스크리닝 (병해충 사진 → 후보 좁히기) 추가 검토 |
| 데이터 보강 | 산림청/지자체 조경수 병해충 데이터 추가 수집, 사용자 피드백 학습 |
| 신뢰성 강화 | 출처 표기 의무화, 전문가 검수 루프(Human-in-the-loop) |
| 수익 모델 | 조경업체/관리사무소 대상 B2B API, 개인 사용자 대상 구독형 |

---

## 8. 작업 시 주의사항 (Claude Code 지시 시 공통 적용)

1. **API 키는 절대 코드에 하드코딩하지 말 것** — 항상 .env로 분리
2. 공공데이터 API는 호출 제한이 있을 수 있으므로 캐싱 로직 필수
3. 농약/방제법 안내는 **의료/법률 정보 수준의 책임**이 따르므로, 모든 답변에 "참고용이며 정확한 진단은 농업기술센터/수목진료 전문가 상담 권장" 문구 고정 삽입
4. 각 Phase 완료 후 반드시 결과를 확인하고 다음 Phase로 진행 (한번에 전체 구현 지시 금지)
5. 그래프/벡터 DB 스키마는 Phase 3에서 확정 후 임의 변경하지 말 것 (일관성 유지)

---

## 9. 즉시 해야 할 일 (사람이 직접)

- [ ] 공공데이터포털(data.go.kr)에서 NCPMS API 키 신청
- [ ] 농사로(nongsaro.go.kr) Open API 키 신청
- [ ] 산림청 산림병해충 관련 공공데이터 검색 및 신청 (조경수 보강용)
- [ ] 위 키 발급 완료 후 Phase 2 프롬프트의 mock 부분을 실제 키로 교체 지시
