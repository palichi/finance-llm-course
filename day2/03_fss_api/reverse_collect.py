"""
KOSPI200 종목 주가 데이터 수집 스크립트 (과거 방향 수집)
data/kospi200_list.csv → data/stock_prices.csv

수집 방향: 종목별 "현재 보유한 가장 오래된 날짜 - 1일"부터 과거로 거슬러
           500건씩 수집. API 구간에 데이터가 진짜로 없을 때만 completed.csv에
           기록하고 다음 실행부터 건너뜀.

사용법:
  python reverse_collect.py               # 일반 실행
  python reverse_collect.py --reset       # completed.csv 초기화 후 전 종목 재수집
"""

import argparse
import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 기능 10: ../../.env 에서 API Key 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

API_KEY   = os.getenv("FSS_API_KEY", "")
BASE_URL  = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
LIST_CSV  = os.path.join(DATA_DIR, "kospi200_list.csv")

OUTPUT_CSV    = os.path.join(DATA_DIR, "stock_prices.csv")
COMPLETED_CSV = os.path.join(DATA_DIR, "completed.csv")

BATCH_SIZE     = 500   # 종목당 1회 실행에 수집할 목표 건수
PAGE_ROWS      = 100   # API 1페이지당 요청 건수
BEGIN_PROBE    = "19990101"  # 쿼리 시작(과거) 경계 — 대부분 종목의 실제 최초 상장일보다 앞
SANITY_TICKER  = "005930"    # API 정상 여부 확인용 기준 종목 (삼성전자)


def get_stock_price_until(ticker: str, end_date: str, target_rows: int) -> pd.DataFrame:
    """
    종목의 주가 데이터를 과거 방향으로 수집한다.

    end_date 이전(end_date 포함) 데이터를 최신순으로 내려받아,
    target_rows 건을 채울 때까지 페이지를 넘기며 누적한다.
    API가 더 이상 데이터를 주지 않으면(0건) 그 사실을 반환값에 함께 알려준다.

    Returns:
        (df, exhausted): df는 수집된 데이터(최대 target_rows건),
                          exhausted는 "이 종목의 과거 데이터가 더 없음"을 의미하는 bool
    """
    all_items = []
    page = 1
    exhausted = False

    # beginBasDt를 아주 과거로 넉넉히 잡고, endBasDt만 "현재 갖고 있는 최초일자-1일"로 고정.
    # 그 구간 안에서 최신순(=end_date에 가까운 순)으로 내려오는 데이터를 모아
    # target_rows에 도달하면 멈춘다. (이 API는 정렬 순서를 보장하지 않을 수 있어
    # 최종적으로 우리가 직접 날짜 내림차순 정렬 후 상위 target_rows만 취한다.)
    begin_probe = BEGIN_PROBE

    while len(all_items) < target_rows:
        params = {
            "serviceKey": API_KEY,
            "numOfRows":  str(PAGE_ROWS),
            "pageNo":     str(page),
            "resultType": "json",
            "beginBasDt": begin_probe,
            "endBasDt":   end_date,
            "likeSrtnCd": ticker,
        }
        resp = requests.get(
            f"{BASE_URL}/getStockPriceInfo",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()

        body  = resp.json().get("response", {}).get("body", {})
        total = int(body.get("totalCount", 0))
        items = body.get("items", {})

        if not items or total == 0:
            # 이 구간엔 데이터가 전혀 없음 → 과거 데이터 소진
            exhausted = True
            break

        item_list = items.get("item", [])
        if isinstance(item_list, dict):   # 1건일 때 dict로 옴
            item_list = [item_list]

        all_items.extend(item_list)

        # 이번 페이지까지 합쳐서 전체 구간(totalCount)을 다 받았는지 확인.
        # totalCount가 target_rows보다 작다면, 이 종목은 이 구간에서
        # 가져올 수 있는 데이터를 전부 가져온 것 → 다음 페이지가 없으므로 소진 처리.
        if len(all_items) >= total:
            if total < target_rows:
                exhausted = True
            break

        page += 1
        time.sleep(0.2)   # 페이지 간 부하 방지

    if not all_items:
        return pd.DataFrame(), exhausted

    df = pd.DataFrame(all_items)

    # 기능 3: srtnCd 앞자리 0 보존
    df["srtnCd"] = df["srtnCd"].astype(str).str.zfill(6)

    num_cols = ["clpr", "vs", "mkp", "hipr", "lopr", "trqu", "trPrc", "lstgStCnt", "mrktTotAmt"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 최신 날짜부터 내림차순으로 정렬한 뒤, 목표 건수만큼만 취함
    # (begin_probe~end_date 구간이 target_rows보다 큰 데이터를 모두 끌고 올 수 있으므로
    #  실제로 필요한 만큼만 자른다)
    df = df.sort_values("basDt", ascending=False).reset_index(drop=True)
    if len(df) > target_rows:
        df = df.iloc[:target_rows].copy()
        # 정확히 target_rows를 채웠다면, 그 이전에 더 있을 수도 있으니
        # exhausted를 False로 유지 (다음 실행에서 더 가져올 여지가 있음)
        exhausted = False

    df = df.sort_values("basDt").reset_index(drop=True)
    return df, exhausted


def load_completed() -> set:
    """완료(더 이상 과거 데이터 없음) 처리된 종목코드 집합을 로드"""
    if not os.path.exists(COMPLETED_CSV):
        return set()
    completed = pd.read_csv(COMPLETED_CSV, encoding="utf-8-sig", dtype=str)
    if "srtnCd" not in completed.columns:
        return set()
    return set(completed["srtnCd"].astype(str).str.zfill(6))


def append_completed(code: str, name: str):
    """완료 종목을 data/completed.csv에 추가 기록"""
    row = pd.DataFrame([{
        "srtnCd": code,
        "itmsNm": name,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    if os.path.exists(COMPLETED_CSV):
        row.to_csv(COMPLETED_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(COMPLETED_CSV, mode="w", header=True, index=False, encoding="utf-8-sig")


def test_api() -> bool:
    """
    API 키가 유효하고 서버가 정상 응답하는지 확인.
    삼성전자(005930) 최근 1건 조회로 빠르게 검증.
    """
    params = {
        "serviceKey": API_KEY,
        "numOfRows":  "1",
        "pageNo":     "1",
        "resultType": "json",
        "beginBasDt": "20240101",
        "endBasDt":   "20240110",
        "srtnCd":     SANITY_TICKER,
    }
    try:
        resp = requests.get(
            f"{BASE_URL}/getStockPriceInfo",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json().get("response", {}).get("body", {})
        return int(body.get("totalCount", 0)) > 0
    except Exception as e:
        print(f"[API 점검 실패] {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="completed.csv를 삭제하고 전 종목을 다시 수집 대상에 포함"
    )
    args = parser.parse_args()

    # API Key 확인
    if not API_KEY:
        print("❌ .env 파일에 FSS_API_KEY 를 입력하세요")
        sys.exit(1)

    # ── [수정 3] 시작 전 API 정상 동작 확인 ────────────────────────────
    print("[점검] API 키 및 서버 응답 확인 중...")
    if not test_api():
        print("❌ API 응답 이상 — 키 만료·호출 한도·네트워크 문제를 확인하세요.")
        print("   (오늘 호출 한도를 초과했다면 내일 다시 실행하세요)")
        sys.exit(1)
    print("✅ API 정상\n")

    # 종목 리스트 파일 확인
    if not os.path.exists(LIST_CSV):
        print("❌ data/kospi200_list.csv 파일이 없습니다.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── --reset 플래그: completed.csv 초기화 ────────────────────────────
    if args.reset and os.path.exists(COMPLETED_CSV):
        os.remove(COMPLETED_CSV)
        print("🔄 completed.csv 초기화 완료 — 전 종목 재수집 대상\n")

    # 종목 리스트 읽기
    stocks = pd.read_csv(LIST_CSV, encoding="utf-8-sig", dtype=str)
    stocks["srtnCd"] = stocks["srtnCd"].astype(str).str.zfill(6)

    # 진짜로 완료된 종목만 건너뜀
    completed_codes = load_completed()
    if completed_codes:
        before = len(stocks)
        stocks = stocks[~stocks["srtnCd"].isin(completed_codes)].reset_index(drop=True)
        skipped = before - len(stocks)
        if skipped:
            print(f"과거 데이터 수집이 완료된 종목 {skipped}개는 건너뜁니다.")

    total_stocks = len(stocks)
    if total_stocks == 0:
        print("수집할 종목이 없습니다 (전부 완료 처리됨).")
        return

    # 기존 데이터 로드
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig", dtype={"srtnCd": str})
        existing["srtnCd"] = existing["srtnCd"].astype(str).str.zfill(6)
    else:
        existing = pd.DataFrame()

    all_frames = [existing] if not existing.empty else []
    failed     = []
    newly_done = []

    for i, row in enumerate(stocks.itertuples(index=False), start=1):
        code = row.srtnCd
        name = row.itmsNm

        print(f"[{i}/{total_stocks}] {name}({code}) 과거 데이터 수집 중...")

        # ── [수정 1] end_date 결정 ─────────────────────────────────────
        # 기존 데이터가 있으면: 보유 중인 가장 오래된 날짜 - 1일 (과거 방향)
        # 기존 데이터가 없으면: 오늘 (전체 이력을 처음부터 역순으로 수집)
        if not existing.empty and code in existing["srtnCd"].values:
            earliest_owned = existing[existing["srtnCd"] == code]["basDt"].astype(str).min()
            end_date = (
                datetime.strptime(earliest_owned, "%Y%m%d") - timedelta(days=1)
            ).strftime("%Y%m%d")
        else:
            end_date = datetime.today().strftime("%Y%m%d")  # 오늘부터 과거로

        try:
            df, exhausted = get_stock_price_until(code, end_date, BATCH_SIZE)

            # ── [수정 2] df.empty → 완료 처리 금지 ──────────────────────
            # API 오류·호출 한도 초과도 0건을 반환하므로, 빈 결과는
            # "과거 소진"이 아닌 "이번 호출 실패"로 처리해 다음 실행에서 재시도.
            if df.empty:
                print(f"  ⚠️  {name}: 0건 수신 — API 오류 또는 해당 구간 데이터 없음 "
                      f"(end_date={end_date}) → 다음 실행에서 재시도")
                failed.append(code)
                continue

            all_frames.append(df)
            print(f"  ✅ {name}: {len(df)}건 수집 (endBasDt={end_date})")

            # exhausted=True AND df가 non-empty 일 때만 진짜 과거 소진
            if exhausted:
                print(f"  📌 {name}: 더 이상 과거 데이터 없음 → 완료 처리")
                append_completed(code, name)
                newly_done.append(code)

        except Exception as e:
            print(f"  ⚠️  {name}: 오류({e}) → 다음 실행에서 재시도")
            failed.append(code)

        time.sleep(0.5)

    if not all_frames:
        print("\n수집된 데이터가 없습니다.")
        return

    # 전체 합치기 + 중복 제거
    result = pd.concat(all_frames, ignore_index=True)
    result["srtnCd"] = result["srtnCd"].astype(str).str.zfill(6)
    result = result.drop_duplicates(subset=["srtnCd", "basDt"])
    result = result.sort_values(["srtnCd", "basDt"]).reset_index(drop=True)

    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료: 총 {len(result):,}건 → {OUTPUT_CSV}")

    if newly_done:
        print(f"📌 완료 처리 종목 {len(newly_done)}개: {', '.join(newly_done)}")
    if failed:
        print(f"⚠️  이번 실행 실패 종목 {len(failed)}개 (다음 실행에서 자동 재시도): "
              f"{', '.join(failed)}")


if __name__ == "__main__":
    main()
