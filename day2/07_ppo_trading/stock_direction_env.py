"""
stock_direction_env.py — 종목별 PPO 주가 방향 예측 Gymnasium 환경

보상 정책 선택: Option A
  실제 가격 변화가 NEUTRAL(임계값 이내)일 때:
    - 매수(Buy) / 매도(Sell) 선택 → -2  (근거 없는 방향성 배팅에 소폭 패널티)
    - 유보(Hold)              선택 → +5  (중립 상황의 정확한 예측에 소폭 보상)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ─────────────────── 하이퍼파라미터 ────────────────────
WINDOW_SIZE    = 20    # 관찰 윈도우 (일)
EPISODE_STEPS  = 20    # 에피소드당 예측 스텝 수
LOOKAHEAD      = 2     # 예측 시계 (일)
THRESHOLD_PCT  = 1.5   # 가격 변동률 임계값 α (%)
TRAIN_RATIO    = 0.8   # 학습 데이터 비율
# ────────────────────────────────────────────────────────

# (실제방향, 행동) → 보상 매핑
# 실제방향: 'up' | 'down' | 'neutral'
# 행동: 0=매수(Buy), 1=매도(Sell), 2=유보(Hold)
REWARD_TABLE: dict[tuple[str, int], float] = {
    ("up",      0): +10.0,   # 상승 예측 맞음
    ("up",      1): -10.0,   # 상승 예측 틀림
    ("up",      2):   0.0,   # 상승 기회 미활용
    ("down",    0): -10.0,   # 하락 예측 틀림
    ("down",    1): +10.0,   # 하락 예측 맞음
    ("down",    2):  -5.0,   # 손실 회피 실패
    ("neutral", 0):  -2.0,   # Option A: 불필요한 매수 배팅
    ("neutral", 1):  -2.0,   # Option A: 불필요한 매도 배팅
    ("neutral", 2):  +5.0,   # Option A: 중립 상황 정확히 유보
}

# 행동 인덱스 → 예측 방향 (적중률 계산용)
ACTION_TO_DIRECTION = {0: "up", 1: "down", 2: "neutral"}


class StockDirectionEnv(gym.Env):
    """
    종목별 주가 방향 예측 강화학습 환경.

    State:  직전 WINDOW_SIZE일의 [open, high, low, close, volume]
            → min-max 정규화 후 flatten → shape (WINDOW_SIZE * 5,)
    Action: Discrete(3) — 0=매수, 1=매도, 2=유보
    Episode: EPISODE_STEPS 스텝 후 종료
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",
        threshold_pct: float = THRESHOLD_PCT,
        seed: int = 42,
    ):
        super().__init__()
        assert mode in ("train", "val"), "mode는 'train' 또는 'val'이어야 합니다"

        n = len(df)
        split = int(n * TRAIN_RATIO)

        raw = df.iloc[:split] if mode == "train" else df.iloc[split:]
        self.data = raw.reset_index(drop=True)

        self.threshold_pct = threshold_pct
        self.closes = self.data["close"].values.astype(np.float64)
        self.ohlcv  = self.data[["open", "high", "low", "close", "volume"]].values.astype(np.float64)

        # ── 유효 에피소드 시작 인덱스 목록 구성 ──────────────────────
        # 에피소드 시작 ep_start에서 마지막 스텝(s = EPISODE_STEPS-1)의 요구 인덱스:
        #   윈도우 끝:  ep_start + s + WINDOW_SIZE - 1
        #   2일 후가:   ep_start + s + WINDOW_SIZE - 1 + LOOKAHEAD
        #             = ep_start + (EPISODE_STEPS-1) + WINDOW_SIZE - 1 + LOOKAHEAD
        #             = ep_start + 41  (20+19+2)
        max_need = WINDOW_SIZE + EPISODE_STEPS - 1 + LOOKAHEAD  # = 41
        n_data = len(self.data)

        # 비중복(non-overlapping) 에피소드 경계: 0, 20, 40, ...
        self.episode_starts = [
            s for s in range(0, n_data - max_need, EPISODE_STEPS)
        ]

        if not self.episode_starts:
            raise ValueError(
                f"데이터가 부족합니다. 최소 {max_need + 1}행 필요, 현재 {n_data}행"
            )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(WINDOW_SIZE * 5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)
        self.ep_start      = self.episode_starts[0]
        self.step_count    = 0
        self.episode_return = 0.0
        self.ep_log: list[dict] = []

    # ── 내부 메서드 ───────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx    = self.ep_start + self.step_count
        window = self.ohlcv[idx : idx + WINDOW_SIZE].copy()

        # 열(컬럼)별 min-max 정규화
        col_min   = window.min(axis=0)
        col_max   = window.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0   # 상수 열의 분모 0 방지
        window = (window - col_min) / col_range
        return window.flatten().astype(np.float32)

    # ── Public API ────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 에피소드 시작 위치 무작위 선택 (학습 다양성 확보)
        self.ep_start      = int(self._rng.choice(self.episode_starts))
        self.step_count    = 0
        self.episode_return = 0.0
        self.ep_log        = []
        return self._get_obs(), {}

    def reset_at(self, ep_start: int):
        """검증 시 특정 에피소드 시작 위치를 직접 지정."""
        assert ep_start in self.episode_starts, f"유효하지 않은 ep_start: {ep_start}"
        self.ep_start      = ep_start
        self.step_count    = 0
        self.episode_return = 0.0
        self.ep_log        = []
        return self._get_obs(), {}

    def step(self, action: int):
        idx = self.ep_start + self.step_count

        # 현재가: 윈도우 마지막 날 종가 (기준가 정의 — 2일 후 변동률 산정의 분모)
        current_close = self.closes[idx + WINDOW_SIZE - 1]
        # 2일 후 종가
        future_close  = self.closes[idx + WINDOW_SIZE - 1 + LOOKAHEAD]

        pct_change = (future_close - current_close) / current_close * 100.0

        if pct_change >= self.threshold_pct:
            actual_direction = "up"
        elif pct_change <= -self.threshold_pct:
            actual_direction = "down"
        else:
            actual_direction = "neutral"

        reward = REWARD_TABLE[(actual_direction, int(action))]
        self.episode_return += reward
        self.step_count     += 1
        terminated           = self.step_count >= EPISODE_STEPS

        self.ep_log.append(
            {
                "step"            : self.step_count,
                "action"          : int(action),
                "actual_direction": actual_direction,
                "pct_change"      : round(pct_change, 4),
                "reward"          : reward,
            }
        )

        obs  = self._get_obs() if not terminated else np.zeros(WINDOW_SIZE * 5, dtype=np.float32)
        info = {
            "actual_direction": actual_direction,
            "pct_change"      : pct_change,
            "episode_return"  : self.episode_return,
            "ep_log"          : self.ep_log if terminated else [],
        }
        return obs, reward, terminated, False, info

    def render(self):
        pass
