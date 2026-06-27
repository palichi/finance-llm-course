"""
train_ppo_direction.py — 종목별 PPO 주가 방향 예측 학습 스크립트

보상 정책: Option A (stock_direction_env.py 참조)
  변화 없음 + 매수/매도 → -2 | 변화 없음 + 유보 → +5
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from stock_direction_env import (
    StockDirectionEnv,
    ACTION_TO_DIRECTION,
    WINDOW_SIZE,
    EPISODE_STEPS,
    LOOKAHEAD,
    TRAIN_RATIO,
)

# ─────────────────── 하이퍼파라미터 ────────────────────
# 관찰/에피소드 구조 (stock_direction_env.py 와 동기화)
WINDOW_SIZE_     = WINDOW_SIZE    # 20일
EPISODE_STEPS_   = EPISODE_STEPS  # 20 스텝
LOOKAHEAD_       = LOOKAHEAD      # 2일 후 예측
TRAIN_RATIO_     = TRAIN_RATIO    # 학습 데이터 비율 0.8

# 임계값 α: 자동 계산 사용 (compute_threshold 함수 참조)
# 수동으로 고정하려면 AUTO_THRESHOLD = False 후 FIXED_THRESHOLD_PCT 지정
AUTO_THRESHOLD      = True
FIXED_THRESHOLD_PCT = 1.5   # AUTO_THRESHOLD=False 일 때 사용

# PPO 학습 파라미터
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS       = 512     # 롤아웃 버퍼 크기
PPO_BATCH_SIZE    = 64
PPO_N_EPOCHS      = 10
PPO_GAMMA         = 0.99
PPO_ENT_COEF      = 0.01    # 탐험 장려 엔트로피 계수
PPO_TOTAL_STEPS   = 100_000  # 종목당 총 학습 스텝

# 경로
DATA_DIR   = "../03_fss_api/data/ppo_ready"
MODELS_DIR = "./models"

# 로그 출력 주기 (에피소드 단위)
LOG_INTERVAL_EPISODES = 50
# ────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)

ACTION_LABEL = {0: "Buy ", 1: "Sell", 2: "Hold"}


# ── 유틸리티 ──────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    """BOM 처리 포함 CSV 로드 및 컬럼 정규화."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.lower().strip() for c in df.columns]
    # FSS API 필드명 basdt → date
    df = df.rename(columns={"basdt": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_threshold(df: pd.DataFrame) -> float:
    """
    종목별 2일 수익률 표준편차를 기준으로 임계값 α 자동 산정.
    1σ 수준을 사용 — 중간 강도의 변동을 '의미 있는 방향'으로 간주.
    결과를 [0.5%, 5.0%] 범위로 클리핑.
    """
    closes = df["close"].values.astype(float)
    if len(closes) < 4:
        return FIXED_THRESHOLD_PCT
    returns_2d = (closes[LOOKAHEAD:] - closes[:-LOOKAHEAD]) / closes[:-LOOKAHEAD] * 100.0
    std = float(np.std(returns_2d))
    return float(np.clip(std, 0.5, 5.0))


# ── 콜백 ──────────────────────────────────────────────────────────

class EpisodeLogCallback(BaseCallback):
    """에피소드 완료마다 누적 리턴을 추적하고 주기적으로 출력."""

    def __init__(self, ticker: str, log_interval: int = LOG_INTERVAL_EPISODES):
        super().__init__(verbose=0)
        self.ticker       = ticker
        self.log_interval = log_interval
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0
        self._ep_count  = 0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self._ep_count += 1
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            if self._ep_count % self.log_interval == 0:
                recent_avg = np.mean(self.episode_rewards[-self.log_interval :])
                print(
                    f"    [{self.ticker}] ep {self._ep_count:5d} | "
                    f"최근 {self.log_interval}ep 평균 리턴: {recent_avg:+.2f}"
                )
        return True


# ── 검증 ──────────────────────────────────────────────────────────

def validate(model: PPO, val_env: StockDirectionEnv, ticker: str) -> tuple[float, dict, int]:
    """
    검증 구간의 모든 에피소드를 순서대로 실행해 행동 분포와 적중률 측정.
    에피소드 시작 위치는 무작위가 아닌 순서대로(reset_at) 사용해 재현성 보장.
    """
    action_counts: dict[int, int] = defaultdict(int)
    correct = 0
    total   = 0

    for ep_start in val_env.episode_starts:
        obs, _ = val_env.reset_at(ep_start)
        for _ in range(EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = val_env.step(int(action))

            action_counts[int(action)] += 1
            predicted = ACTION_TO_DIRECTION[int(action)]
            actual    = info["actual_direction"]
            if predicted == actual:
                correct += 1
            total += 1

            if terminated or truncated:
                break

    total    = max(total, 1)
    accuracy = correct / total * 100.0

    print(f"\n  [{ticker}] 검증 결과 ({total} 스텝, {len(val_env.episode_starts)} 에피소드)")
    for a in range(3):
        cnt = action_counts[a]
        pct = cnt / total * 100.0
        print(f"    {ACTION_LABEL[a]}: {cnt:5d}회 ({pct:5.1f}%)")
    print(f"    Accuracy: {accuracy:.1f}%  (예측 방향 일치율)")

    return accuracy, dict(action_counts), total


# ── 종목 단위 학습 ────────────────────────────────────────────────

def train_ticker(csv_path: str) -> dict:
    ticker = Path(csv_path).stem
    print(f"\n{'='*60}")
    print(f" 종목: {ticker}  |  파일: {Path(csv_path).name}")
    print(f"{'='*60}")

    df = load_csv(csv_path)

    if len(df) < 100:
        print(f"  [SKIP] 데이터 부족 ({len(df)}행, 최소 100행 필요)")
        return {"ticker": ticker, "status": "skipped", "avg_return": None, "accuracy": None, "threshold": None}

    # 임계값 α 산정
    threshold = compute_threshold(df) if AUTO_THRESHOLD else FIXED_THRESHOLD_PCT
    print(f"  전체 데이터: {len(df)}행 | 임계값 α = {threshold:.2f}% | "
          f"학습 {int(len(df)*TRAIN_RATIO)}행 / 검증 {len(df)-int(len(df)*TRAIN_RATIO)}행")

    # 환경 생성
    try:
        train_env = StockDirectionEnv(df, mode="train", threshold_pct=threshold)
        val_env   = StockDirectionEnv(df, mode="val",   threshold_pct=threshold)
    except ValueError as e:
        print(f"  [SKIP] 환경 생성 실패: {e}")
        return {"ticker": ticker, "status": "skipped", "avg_return": None, "accuracy": None, "threshold": threshold}

    print(f"  에피소드 수 — 학습: {len(train_env.episode_starts)}, 검증: {len(val_env.episode_starts)}")

    # PPO 모델 생성
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = PPO_LEARNING_RATE,
        n_steps       = PPO_N_STEPS,
        batch_size    = PPO_BATCH_SIZE,
        n_epochs      = PPO_N_EPOCHS,
        gamma         = PPO_GAMMA,
        ent_coef      = PPO_ENT_COEF,
        device        = "cpu",   # MlpPolicy는 CPU가 더 효율적
        verbose       = 0,
    )

    callback = EpisodeLogCallback(ticker=ticker)
    print(f"  PPO 학습 시작 (총 {PPO_TOTAL_STEPS:,} 스텝) ...")
    model.learn(total_timesteps=PPO_TOTAL_STEPS, callback=callback)

    # 모델 저장
    save_path = os.path.join(MODELS_DIR, f"{ticker}_ppo")
    model.save(save_path)
    print(f"  모델 저장: {save_path}.zip")

    # 학습 리턴 요약
    ep_rewards = callback.episode_rewards
    avg_return = float(np.mean(ep_rewards[-LOG_INTERVAL_EPISODES:])) if ep_rewards else 0.0
    print(f"  최종 {LOG_INTERVAL_EPISODES}ep 평균 리턴: {avg_return:+.2f}")

    # 검증
    accuracy, action_counts, total_steps = validate(model, val_env, ticker)

    return {
        "ticker"    : ticker,
        "status"    : "done",
        "avg_return": round(avg_return, 2),
        "accuracy"  : round(accuracy, 1),
        "threshold" : round(threshold, 2),
        "n_train_ep": len(train_env.episode_starts),
        "n_val_ep"  : len(val_env.episode_starts),
    }


# ── 메인 ──────────────────────────────────────────────────────────

def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        print(f"CSV 파일을 찾을 수 없습니다: {DATA_DIR}")
        return

    print(f"총 {len(csv_files)}개 종목 순차 학습")
    print(f"[보상 정책] Option A — 가격 변화 없음 + 매수/매도 → -2, 유보 → +5")
    print(f"[임계값]   AUTO_THRESHOLD={AUTO_THRESHOLD} "
          f"(종목별 2일 수익률 σ 기반 자동 산정)\n")

    results = []
    for csv_path in csv_files:
        result = train_ticker(csv_path)
        results.append(result)

    # ── 최종 요약 표 ─────────────────────────────────────────────
    done    = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"\n{'='*70}")
    print(" 학습 결과 요약")
    print(f"{'='*70}")
    header = f"{'종목코드':>10}  {'α(%)':>6}  {'학습ep':>6}  {'검증ep':>6}  {'평균리턴':>10}  {'적중률':>8}"
    print(header)
    print("-" * 70)
    for r in done:
        print(
            f"{r['ticker']:>10}  {r['threshold']:>6.2f}  "
            f"{r['n_train_ep']:>6}  {r['n_val_ep']:>6}  "
            f"{r['avg_return']:>+10.2f}  {r['accuracy']:>7.1f}%"
        )
    print("-" * 70)
    if done:
        avg_acc    = np.mean([r["accuracy"]  for r in done])
        avg_return = np.mean([r["avg_return"] for r in done])
        print(f"{'전체 평균':>10}  {'':>6}  {'':>6}  {'':>6}  {avg_return:>+10.2f}  {avg_acc:>7.1f}%")
    if skipped:
        print(f"\n[건너뜀] {len(skipped)}개 종목: {[r['ticker'] for r in skipped]}")
    print(f"\n완료: {len(done)}/{len(results)} 종목 | 모델 저장 위치: {MODELS_DIR}/")


if __name__ == "__main__":
    main()
