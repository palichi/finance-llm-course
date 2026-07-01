# 📁 Day 1 · 03 · Streamlit 핵심 기능 — KIS API 연동 실습

## 이 실습에서 배우는 것
- Streamlit의 다양한 UI 컴포넌트 (위젯, 레이아웃, 상태관리)
- `st.session_state`로 데이터 유지하기
- KIS OpenAPI 데이터를 여러 Streamlit 기능으로 시각화하기
- Claude Code로 기능을 점진적으로 확장하는 방법

---

## 사전 확인 — KIS API 연결이 되어 있어야 합니다

```bash
cd day1/02_kis_api
python kis_client.py
# ✅ Token 발급 성공 이 나오면 OK
cd ../03_streamlit
```

---

## 🖥 실습 순서

---

### STEP 1 · 폴더 이동

```bash
cd day1/03_streamlit
```

```bash
ls
# README.md 만 있으면 정상
```

---

### STEP 2 · Claude Code 실행

**새 터미널 탭 열기** (Ctrl + Shift + `)

```bash
claude
```

---

### STEP 3 · [Claude Code 실습] app.py 만들기

아래 프롬프트를 **그대로 복사해서 붙여넣기** 하세요:

```
day1/03_streamlit/app.py 파일을 만들어줘.

KIS OpenAPI 모의투자 데이터를 가지고
Streamlit의 다양한 핵심 기능을 연습하는 웹앱이야.

기본 설정:
  .env 파일: ../../.env 경로에서 읽기
  KIS 모의투자 Base URL: https://openapivts.koreainvestment.com:29443
  페이지 제목: "🧩 KIS 모의투자 트레이딩 앱"
  st.set_page_config(layout="wide") 사용

기능 1 · 멀티 탭 구조 (st.tabs)
  탭1: "현재가 조회"
  탭2: "관심종목 리스트"
  탭3: "잔고 조회"
  탭4: "위젯 갤러리"

기능 2 · 탭1 - 현재가 조회
  st.columns 로 좌우 2분할
  좌측: 종목코드 입력창 + 조회 버튼
  우측: 현재가, 전일대비(색상: 상승 빨강/하락 파랑), 등락률
  TR: FHKST01010100

기능 3 · 탭2 - 관심종목 리스트 (session_state 활용)
  종목코드 입력 후 "추가" 버튼 → st.session_state 리스트에 저장
  st.data_editor 로 관심종목 표 보여주기 (행 삭제 가능하게)
  "전체 현재가 새로고침" 버튼 → 리스트의 모든 종목 현재가 일괄 조회
  각 종목 행에 삭제 버튼(🗑) 추가

기능 4 · 탭3 - 잔고 조회
  TR: VTTC8434R (모의투자 잔고조회)
  보유종목을 st.dataframe 으로 표시
  총 평가금액, 총 매입금액, 총 평가손익을 st.metric 3개로 카드 표시
  평가손익은 양수면 빨간색(delta_color="normal"), 음수면 파란색

기능 5 · 탭4 - 위젯 갤러리 (Streamlit 기능 학습용)
  st.slider 로 조회 기간(일) 선택 → 선택값 출력
  st.selectbox 로 시장 구분 선택 (코스피/코스닥)
  st.radio 로 정렬 기준 선택 (현재가/등락률/거래량)
  st.expander 로 "API 요청 원본 JSON 보기" 펼치기/접기
  st.toast 로 "조회 완료!" 알림 띄우기 (조회 버튼 클릭 시)
  st.spinner 로 API 호출 중 로딩 표시

기능 6 · 사이드바
  st.sidebar 에 "API 연결 상태" 표시 (✅/❌)
  st.sidebar 에 마지막 조회 시각 표시
  st.sidebar 에 "전체 새로고침" 버튼

기타 조건:
  API 오류 시 st.error 로 안내
  데이터 없을 때 st.info 로 안내
  session_state 키 충돌 없도록 명확한 변수명 사용
  파일명: app.py
```

---

### STEP 4 · app.py 실행 확인

Claude Code가 파일을 만들면 **기존 터미널로 돌아가서** 실행:

```bash
streamlit run app.py
```

브라우저에서 확인:

```
1. 탭1에서 삼성전자(005930) 조회 → 현재가/전일대비 색상 확인
2. 탭2에서 관심종목 2~3개 추가 → 표에 잘 쌓이는지 확인
3. "전체 현재가 새로고침" 클릭 → 한 번에 갱신되는지 확인
4. 탭3에서 잔고 조회 → metric 카드 3개 확인
5. 탭4에서 슬라이더/셀렉트박스/라디오 조작 → 값이 바뀌는지 확인
6. expander 펼쳐서 JSON 원본 확인
```

---

### STEP 5 · 추가 실습 — Claude Code 트레이딩 스킬 활용

이 단계부터는 Claude Code의 **트레이딩 스킬**을 사용합니다.
스킬이 등록되어 있으면 KIS API 호출 규칙, TR_ID 패턴, 에러 처리 방식을
Claude가 자동으로 참고해서 더 정확한 코드를 만들어 줍니다.

```bash
claude
```

```
/skills 목록을 확인해줘. 트레이딩 관련 스킬이 있으면
그 스킬을 참고해서 day1/03_streamlit/app.py 에
"실시간 자동 새로고침" 기능을 추가해줘.

조건:
- st.session_state 로 자동 새로고침 ON/OFF 상태 관리
- 사이드바 체크박스로 켜고 끄기
- 켜져 있을 때 30초마다 탭1의 현재가를 자동 갱신
- streamlit-autorefresh 패키지 또는 st.rerun() 활용
```

> 💡 트레이딩 스킬이 아직 없다면, Claude Code에 다음과 같이 물어보세요:
> `현재 사용 가능한 스킬 목록을 보여줘`

**미션 1 · 다중 종목 비교 차트**
```
app.py 탭1 옆에 "종목 비교" 기능을 추가해줘.
멀티셀렉트(st.multiselect)로 종목 3개까지 선택하면
st.bar_chart 로 현재가를 나란히 비교해줘.
```

**미션 2 · 폼(Form)으로 입력 묶기**
```
탭1의 종목코드 입력과 조회 버튼을
st.form 으로 감싸서 Enter 키로도 제출되게 해줘.
```

**미션 3 · 캐싱 적용**
```
현재가 조회 함수에 @st.cache_data(ttl=10) 를 적용해서
10초 안에 같은 종목을 다시 조회하면
API를 호출하지 않고 캐시된 값을 보여주게 해줘.
```

---

## 📐 이번 실습에서 다루는 Streamlit 핵심 개념

```
컴포넌트            용도
──────────────────────────────────────
st.tabs            화면을 탭으로 분리
st.columns         좌우 레이아웃 분할
st.session_state   페이지 새로고침에도 값 유지
st.data_editor     표 형태 데이터 편집(추가/삭제)
st.metric          숫자 + 증감 표시 카드
st.expander        내용 펼치기/접기
st.toast           짧은 알림 메시지
st.spinner         로딩 중 표시
st.cache_data      함수 결과 캐싱(중복 호출 방지)
st.form            여러 입력을 한 번에 제출
```

### session_state 기본 패턴
```python
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if st.button("추가"):
    st.session_state.watchlist.append(ticker_code)
```

---

## ❓ 자주 묻는 질문

**Q. 탭을 전환하면 입력했던 값이 사라져요**
```
원인: session_state 없이 일반 변수로만 값을 저장한 경우
해결: Claude Code 에 요청
"이 입력값을 session_state 에 저장해서
 탭을 옮겨도 유지되게 해줘."
```

**Q. 관심종목 리스트에 같은 종목이 중복으로 추가돼요**
```
Claude Code 에 요청:
"watchlist 에 추가할 때 이미 있는 종목코드면
 중복 추가되지 않게 막아줘."
```

**Q. 자동 새로고침을 켜두면 너무 자주 API를 호출해요**
```
원인: 새로고침 주기가 너무 짧거나 캐시가 없는 경우
해결: 새로고침 주기를 늘리거나(60초 이상)
     @st.cache_data(ttl=...) 를 함께 적용
```

**Q. "트레이딩 스킬을 찾을 수 없다"고 나와요**
```
스킬이 아직 설치되어 있지 않은 환경입니다.
스킬 없이도 동일한 결과를 얻을 수 있도록
프롬프트에 KIS API 규칙(Base URL, TR_ID, 인증 방식)을
직접 명시해서 요청하면 됩니다.
```
