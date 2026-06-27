"""
check_prediction.py — PPO 모델 판단 점검 스크립트

RAG/ChromaDB 없이 순수 PPO 모델만으로 판단을 확인합니다.
검증 구간(뒤쪽 20%)에서 20일 구간 5곳을 무작위로 뽑아
모델 예측과 실제 2일 후 결과를 비교합니다.

사용법:
    python check_prediction.py --ticker 005930
"""

import argparse
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from stock_direction_env import WINDOW_SIZE, LOOKAHEAD, TRAIN_RATIO

DATA_DIR   = "../03_fss_api/data/ppo_ready"
MODELS_DIR = "./models"
NUM_SAMPLES = 5

ACTION_LABEL = {0: "매수", 1: "매도", 2: "유보"}
ACTION_TO_DIRECTION = {0: "up", 1: "down", 2: "neutral"}
DIRECTION_LABEL = {"up": "상승", "down": "하락", "neutral": "변화없음"}


def load_csv(ticker: str) -> pd.DataFrame:
    path = Path(DATA_DIR) / f"{ticker}.csv"
    if not path.exists():
        print(f"데이터 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={"basdt": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_threshold(df: pd.DataFrame) -> float:
    closes = df["close"].values.astype(float)
    if len(closes) < 4:
        return 1.5
    returns_2d = (closes[LOOKAHEAD:] - closes[:-LOOKAHEAD]) / closes[:-LOOKAHEAD] * 100.0
    return float(np.clip(np.std(returns_2d), 0.5, 5.0))


def build_obs(ohlcv: np.ndarray, start: int) -> np.ndarray:
    """stock_direction_env._get_obs() 와 동일한 전처리."""
    window = ohlcv[start : start + WINDOW_SIZE].copy()
    col_min = window.min(axis=0)
    col_max = window.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    window = (window - col_min) / col_range
    return window.flatten().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="PPO 모델 판단 점검")
    parser.add_argument("--ticker", required=True, help="종목코드 (예: 005930)")
    parser.add_argument("--seed", type=int, default=None, help="무작위 시드 (재현용)")
    args = parser.parse_args()

    ticker = args.ticker
    if args.seed is not None:
        random.seed(args.seed)

    # ── 모델 로딩 ──────────────────────────────────────────────────
    model_path = Path(MODELS_DIR) / f"{ticker}_ppo.zip"
    if not model_path.exists():
        print(f"모델 파일이 없습니다: {model_path}")
        print("먼저 train_ppo_direction.py 를 실행하세요")
        sys.exit(1)

    model = PPO.load(str(model_path))
    print(f"모델 로딩 완료: {model_path}")

    # ── 데이터 로딩 및 검증 구간 분리 ─────────────────────────────
    df = load_csv(ticker)
    threshold = compute_threshold(df)

    split = int(len(df) * TRAIN_RATIO)
    val_df = df.iloc[split:].reset_index(drop=True)

    ohlcv  = val_df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    closes = val_df["close"].values.astype(np.float64)
    dates  = val_df["date"].astype(str).values

    print(f"전체 데이터: {len(df)}행 | 검증 구간: {len(val_df)}행 | 임계값 α = {threshold:.2f}%\n")

    # ── 유효 시작 인덱스 샘플링 ────────────────────────────────────
    # 필요 인덱스: start + WINDOW_SIZE - 1 + LOOKAHEAD < len(val_df)
    max_start = len(val_df) - WINDOW_SIZE - LOOKAHEAD
    if max_start < NUM_SAMPLES:
        print(f"검증 데이터 부족: {len(val_df)}행 (최소 {WINDOW_SIZE + LOOKAHEAD + NUM_SAMPLES}행 필요)")
        sys.exit(1)

    sampled = sorted(random.sample(range(0, max_start + 1), NUM_SAMPLES))

    # ── 점검 루프 ──────────────────────────────────────────────────
    hits = 0
    for i, start in enumerate(sampled, 1):
        date_from = dates[start]
        date_to   = dates[start + WINDOW_SIZE - 1]

        obs    = build_obs(ohlcv, start)
        action = int(model.predict(obs, deterministic=True)[0])

        ppo_label     = ACTION_LABEL[action]
        predicted_dir = ACTION_TO_DIRECTION[action]

        current_close = closes[start + WINDOW_SIZE - 1]
        future_close  = closes[start + WINDOW_SIZE - 1 + LOOKAHEAD]
        pct_change    = (future_close - current_close) / current_close * 100.0

        if pct_change >= threshold:
            actual_dir = "up"
        elif pct_change <= -threshold:
            actual_dir = "down"
        else:
            actual_dir = "neutral"

        is_hit = (predicted_dir == actual_dir)
        if is_hit:
            hits += 1

        print("──────────────────────────────────────")
        print(f"[점검 {i}/{NUM_SAMPLES}] {ticker} ({date_from} ~ {date_to} 구간)")
        print()

        # 20일 종가 흐름 표
        window_rows = val_df.iloc[start : start + WINDOW_SIZE]
        print(f"  {'날짜':>10}  {'종가':>10}  {'전일대비':>10}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*10}")

        close_list  = []
        prev_close_ = None
        for _, row in window_rows.iterrows():
            c = int(row["close"])
            close_list.append(c)
            if prev_close_ is None:
                change_str = "-"
            else:
                diff = c - prev_close_
                change_str = f"{'+' if diff >= 0 else ''}{diff:,}"
            print(f"  {str(row['date']):>10}  {c:>10,}  {change_str:>10}")
            prev_close_ = c

        # 종가 흐름 한줄 요약
        if len(close_list) > 4:
            flow = (
                " → ".join(f"{p:,}" for p in close_list[:2])
                + " → ... → "
                + " → ".join(f"{p:,}" for p in close_list[-2:])
            )
        else:
            flow = " → ".join(f"{p:,}" for p in close_list)

        print(f"\n  최근 20일 종가 흐름: {flow}원")
        print(f"  PPO 판단: {ppo_label}")

        direction_kor = DIRECTION_LABEL[actual_dir]
        hit_mark      = "✅ 적중" if is_hit else "❌ 빗나감"
        print(
            f"  실제 2일 후 결과: {direction_kor} "
            f"({int(future_close):,}원, {pct_change:+.2f}%) → {hit_mark}"
        )

    print("──────────────────────────────────────")
    print(f"\n{NUM_SAMPLES}건 중 {hits}건 적중 (적중률 {hits / NUM_SAMPLES * 100:.0f}%)")


if __name__ == "__main__":
    main()
