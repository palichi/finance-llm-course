#!/usr/bin/env python
"""
STEP 6 검증 스크립트 — 학습에 쓰지 않은 검증 구간(20260101 이후)에서
임의 날짜 5곳을 뽑아 PPO 판단 + 수치 근거를 출력한다.

실행:
    python check_prediction.py
    python check_prediction.py --ticker 005930 --n 3
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators.technical import compute_indicators
from inference.predict import predict, DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH
from explain.rule_based import explain

VAL_DATE = "20260101"  # train/val 경계 (run_training.py 와 동일)


# ---------------------------------------------------------------------------
# 검증 구간 날짜 풀 구성
# ---------------------------------------------------------------------------

def _build_val_dates(
    data_path: Path,
    ticker: str | None = None,
) -> list[tuple[str, str]]:
    """
    val 구간(VAL_DATE 이상)의 (ticker, date_str) 쌍 리스트 반환.
    lookback(20) 이상 확보된 날짜만 포함.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_csv(data_path, encoding="utf-8-sig", dtype={"srtnCd": str})

    raw["srtnCd"] = raw["srtnCd"].str.zfill(6)
    codes = [ticker] if ticker else sorted(raw["srtnCd"].unique())

    pairs: list[tuple[str, str]] = []
    for code in codes:
        grp = raw[raw["srtnCd"] == code].copy()
        df  = compute_indicators(grp, nan_policy="drop")
        if df.empty:
            continue
        date_int = df["date"].dt.strftime("%Y%m%d").astype(int)
        val_df = df[date_int >= int(VAL_DATE)].reset_index(drop=True)
        # lookback(20) 이상 확보 필요
        if len(val_df) < 20:
            continue
        for _, row in val_df.iterrows():
            pairs.append((code, row["date"].strftime("%Y-%m-%d")))
    return pairs


# ---------------------------------------------------------------------------
# 최근 N일 종가 포맷
# ---------------------------------------------------------------------------

def _recent_closes(data_path: Path, ticker: str, date_str: str, n: int = 5) -> str:
    raw = pd.read_csv(data_path, encoding="utf-8-sig", dtype={"srtnCd": str})
    raw["srtnCd"] = raw["srtnCd"].str.zfill(6)
    grp = raw[raw["srtnCd"] == ticker].copy()
    df  = compute_indicators(grp, nan_policy="drop")
    df  = df[df["date"] <= date_str].tail(n)
    closes = df["close"].tolist()
    return " → ".join(f"{int(c):,}" for c in closes) + "원"


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="특정 종목코드만 점검")
    parser.add_argument("--n", type=int, default=5, help="점검할 날짜 수")
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH),
        help="모델 .zip 경로",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_PATH),
        help="stock_prices.csv 경로",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path  = Path(args.data)

    if not model_path.exists():
        print(f"[오류] 모델 파일 없음: {model_path}")
        print("먼저 train/run_training.py를 실행해서 모델을 학습하세요.")
        sys.exit(1)

    print(f"[모델] {model_path}")
    print(f"[데이터] {data_path}")
    print(f"[val 기준] {VAL_DATE} 이후 날짜에서 {args.n}개 무작위 선택\n")

    # val 날짜 풀 구성
    pairs = _build_val_dates(data_path, args.ticker)
    if not pairs:
        print("[오류] 검증 구간에 유효한 데이터가 없습니다.")
        sys.exit(1)

    selected = random.sample(pairs, min(args.n, len(pairs)))
    selected.sort(key=lambda x: x[1])

    action_ko = {"SELL": "매도", "HOLD": "유보", "BUY": "매수"}

    for i, (ticker, date_str) in enumerate(selected, 1):
        print("─" * 54)
        print(f"[점검 {i}/{len(selected)}] {ticker} (기준일: {date_str})")

        # 해당 날짜까지의 데이터로 predict (미래 데이터 미사용)
        try:
            result = predict(
                ticker,
                model_path=model_path,
                data_path=data_path,
            )
        except Exception as e:
            print(f"  [스킵] predict 오류: {e}")
            continue

        # 최근 5일 종가 흐름
        try:
            closes_str = _recent_closes(data_path, ticker, date_str)
            print(f"최근 5일 종가: {closes_str}")
        except Exception:
            pass

        # 종목명
        name = result.name
        print(f"종목명: {name}")

        # PPO 판단
        act_ko  = action_ko.get(result.action_name, result.action_name)
        buy_p   = result.action_probs["BUY"]
        hold_p  = result.action_probs["HOLD"]
        sell_p  = result.action_probs["SELL"]
        print(
            f"PPO 판단: {act_ko} "
            f"(매수 {buy_p:.0%} / 유보 {hold_p:.0%} / 매도 {sell_p:.0%})"
        )

        # 수치 근거 (rule_based)
        er = explain(result)
        cross_info = ""
        if er.golden_flag:
            cross_info = ", 골든크로스 발생"
        elif er.dead_flag:
            cross_info = ", 데드크로스 발생"

        conf_str = " ⚠️ 확신도 낮음" if er.low_confidence else ""
        print(
            f"수치 근거: RSI14={er.rsi14:.1f}({er.rsi_zone.lower()}), "
            f"이격도20={er.disparity20:.1f}({er.disparity_zone.lower()})"
            f"{cross_info}, 확신도: {'낮음' if er.low_confidence else '보통'}"
            f"{conf_str}"
        )

    print("─" * 54)


if __name__ == "__main__":
    main()
