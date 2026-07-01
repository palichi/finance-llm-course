"""
gymnasium TradingEnv — 단일 종목 에피소드 기반 PPO 학습 환경.

설계 원칙:
- 환경 자체는 단일 종목 DataFrame을 전달받아 에피소드를 구성한다.
- reset(options={"code": ...}) 으로 외부(train 모듈)가 종목을 주입한다.
  지정하지 않으면 stock_data에서 랜덤 선택.
- 종목 간 포지션/손익 경계는 reset() 호출로만 분리된다.
- position / unrealized_pnl은 lookback 창 전체에 브로드캐스트한다.

무효 행동 처리: 무시 + 작은 페널티(-0.01, config화)

짧은 종목 처리:
- __init__ 시점에 len(df) < lookback 인 종목을 자동으로 제외하고 경고.
- 유효 종목이 하나도 없으면 ValueError.
- 이렇게 하면 reset()에서 항상 유효 종목만 선택된다.

state 벡터 (9차원, lookback=20 → shape (20, 9)):
  [SMA5_disp, SMA20_disp, SMA60_disp, golden_flag, dead_flag,
   disparity20_centered, rsi14_norm, position, unrealized_pnl]
  * SMAx_disp = (close/SMAx - 1) * 100  (%)

입력 DataFrame은 indicators.technical.compute_indicators() 결과를 전제로 함.
컬럼명은 표준명(date, ticker, close, sma5, ...) 사용.

action: Discrete(3) — 0=매도, 1=유보, 2=매수

거래 모델: 전액 매수/전량 매도, 거래비용 양방향 적용
  equity = cash + shares * current_close  (초기 equity = 1.0)

episode 종료:
  1. 데이터 끝 도달
  2. 자산이 초기의 ruin_ratio 이하로 하락
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from train.reward import RewardShaper

LOOKBACK    = 20
N_FEATURES  = 9
ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY  = 2


class TradingEnv(gym.Env):
    """
    단일 종목 주식 매매 강화학습 환경.

    Parameters
    ----------
    stock_data : dict[str, pd.DataFrame]
        {종목코드: compute_indicators() 결과 DataFrame}.
        NaN 없는 상태로 전달할 것 (nan_policy="drop" 권장).
        lookback보다 짧은 종목은 자동으로 제외된다.
    cfg : dict | None
        설정값. 기본값은 _CFG_DEFAULTS 참조.
    render_mode : str | None
        "human" 이면 render() 시 콘솔 출력.
    """

    metadata = {"render_modes": ["human"]}

    _CFG_DEFAULTS: dict = {
        "lookback"       : LOOKBACK,
        "tx_cost"        : 0.003,
        "ruin_ratio"     : 0.7,
        "invalid_penalty": -0.01,
    }

    def __init__(
        self,
        stock_data    : dict[str, pd.DataFrame],
        cfg           : dict | None = None,
        reward_shaper : RewardShaper | None = None,
        render_mode   : str | None = "human",
    ) -> None:
        super().__init__()

        self._cfg        = {**self._CFG_DEFAULTS, **(cfg or {})}
        self.render_mode = render_mode

        self._lookback    : int   = int(self._cfg["lookback"])
        self._tx_cost     : float = float(self._cfg["tx_cost"])
        self._ruin_ratio  : float = float(self._cfg["ruin_ratio"])
        self._invalid_pen : float = float(self._cfg["invalid_penalty"])

        # lookback보다 짧은 종목 필터링
        valid   = {c: d for c, d in stock_data.items() if len(d) >= self._lookback}
        skipped = set(stock_data) - set(valid)
        if skipped:
            warnings.warn(
                f"데이터 부족(lookback={self._lookback}) 종목 제외: {skipped}",
                stacklevel=2,
            )
        if not valid:
            raise ValueError(
                f"유효 종목 없음: 모든 종목의 행 수가 lookback({self._lookback})보다 짧습니다."
            )
        self._stock_data    = valid
        self._reward_shaper = reward_shaper

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._lookback, N_FEATURES),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # episode 상태 (reset()에서 초기화)
        self._df            : pd.DataFrame | None = None
        self._code          : str | None = None
        self._cursor        : int   = 0
        self._position      : int   = 0
        self._entry_price   : float = 0.0
        self._shares        : float = 0.0
        self._cash          : float = 1.0
        self._equity        : float = 1.0
        self._initial_eq    : float = 1.0
        self._unrealized_pnl: float = 0.0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed   : int | None  = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        에피소드 초기화.

        Parameters
        ----------
        options : dict | None
            {"code": "000660"} 로 종목 지정. 없으면 랜덤 선택.

        Returns
        -------
        obs  : ndarray, shape (lookback, 9)
        info : {"code", "start_idx", "date"}
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "code" in options:
            code = options["code"]
            if code not in self._stock_data:
                raise KeyError(f"종목 '{code}'가 stock_data에 없음")
            self._code = code
        else:
            codes      = list(self._stock_data.keys())
            self._code = codes[int(self._rng.integers(len(codes)))]

        self._df = self._stock_data[self._code].reset_index(drop=True)

        self._cursor = self._lookback - 1

        self._position       = 0
        self._entry_price    = 0.0
        self._shares         = 0.0
        self._cash           = 1.0
        self._equity         = 1.0
        self._initial_eq     = 1.0
        self._unrealized_pnl = 0.0

        if self._reward_shaper is not None:
            self._reward_shaper.reset_episode()

        obs  = self._get_obs()
        date = self._df["date"].iloc[self._cursor]
        info = {
            "code"      : self._code,
            "start_idx" : self._cursor,
            "date"      : date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        한 스텝 실행.

        거래 흐름:
          1. 현재 cursor 행의 close 가격으로 행동 실행
          2. 포트폴리오 가치 재계산 → 보상 산출
          3. cursor 전진
          4. 종료 조건 판단 후 다음 obs 반환
        """
        if self._df is None:
            raise RuntimeError("reset()을 먼저 호출하세요")

        row             = self._df.iloc[self._cursor]
        price           = float(row["close"])
        prev_equity     = self._equity
        position_before = self._position
        invalid         = False

        # ── 행동 실행 ──────────────────────────────────────────────────
        if action == ACTION_BUY:
            if self._position == 0 and self._cash > 0:
                self._shares      = self._cash / (price * (1.0 + self._tx_cost))
                self._entry_price = price
                self._cash        = 0.0
                self._position    = 1
            else:
                invalid = True

        elif action == ACTION_SELL:
            if self._position == 1:
                self._cash        = self._shares * price * (1.0 - self._tx_cost)
                self._shares      = 0.0
                self._entry_price = 0.0
                self._position    = 0
            else:
                invalid = True

        # ── 포트폴리오 가치 / 미실현 손익 ───────────────────────────────
        self._equity = self._cash + self._shares * price
        if self._position == 1:
            self._unrealized_pnl = (price - self._entry_price) / self._entry_price
        else:
            self._unrealized_pnl = 0.0

        # ── 보상 ────────────────────────────────────────────────────────
        base_reward = (self._equity - prev_equity) / prev_equity

        if self._reward_shaper is not None:
            rsi14       = float(row["rsi14"])       if "rsi14"       in row.index else 50.0
            golden_flag = int(row["golden_flag"])   if "golden_flag" in row.index else 0
            disparity20 = float(row["disparity20"]) if "disparity20" in row.index else 100.0
            shaped_reward, shaping_detail = self._reward_shaper.compute(
                base_reward     = base_reward,
                action          = action,
                position_before = position_before,
                position_after  = self._position,
                rsi14           = rsi14,
                golden_flag     = golden_flag,
                disparity20     = disparity20,
            )
        else:
            shaped_reward  = base_reward
            shaping_detail = {}

        reward = shaped_reward + (self._invalid_pen if invalid else 0.0)

        # ── 커서 전진 ───────────────────────────────────────────────────
        self._cursor += 1

        # ── 종료 조건 ───────────────────────────────────────────────────
        terminated = (
            self._cursor >= len(self._df)
            or self._equity < self._initial_eq * self._ruin_ratio
        )

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros((self._lookback, N_FEATURES), dtype=np.float32)
        )

        date = row["date"]
        info = {
            "code"           : self._code,
            "date"           : date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "price"          : price,
            "position"       : self._position,
            "unrealized_pnl" : self._unrealized_pnl,
            "action"         : action,
            "invalid_action" : invalid,
            "equity"         : self._equity,
            "reward_base"    : base_reward,
            "shaping"        : shaping_detail,
        }
        if terminated and self._reward_shaper is not None:
            info["episode_shaping_summary"] = self._reward_shaper.get_episode_summary()

        return obs, float(reward), terminated, False, info

    def render(self) -> None:
        if self._df is None:
            print("[TradingEnv] reset() 필요")
            return
        idx = max(self._cursor - 1, 0)
        row = self._df.iloc[idx]
        date    = row["date"]
        date_s  = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        pos_str = "보유" if self._position else "미보유"
        print(
            f"[{self._code}] 일자={date_s}  종가={row['close']:>10,.0f}"
            f"  포지션={pos_str:<4}  미실현={self._unrealized_pnl:+.2%}"
            f"  자산={self._equity:.5f}  step={self._cursor}/{len(self._df)}"
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        cursor 위치 기준 lookback 창의 9차원 특징 행렬을 반환.

        feature index:
          0: SMA5_disp            (%)
          1: SMA20_disp           (%)
          2: SMA60_disp           (%)
          3: golden_flag          {0, 1}
          4: dead_flag            {0, 1}
          5: disparity20_centered (%)
          6: rsi14_norm           [-1, 1]
          7: position             {0, 1}  — lookback 전 구간 브로드캐스트
          8: unrealized_pnl       — lookback 전 구간 브로드캐스트
        """
        lo = self._cursor - self._lookback + 1
        hi = self._cursor + 1
        w  = self._df.iloc[lo:hi]

        close = w["close"].to_numpy(dtype=np.float64)
        sma5  = w["sma5"].to_numpy(dtype=np.float64)
        sma20 = w["sma20"].to_numpy(dtype=np.float64)
        sma60 = w["sma60"].to_numpy(dtype=np.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            sma5_disp  = np.where(sma5  > 0, (close / sma5  - 1.0) * 100.0, 0.0)
            sma20_disp = np.where(sma20 > 0, (close / sma20 - 1.0) * 100.0, 0.0)
            sma60_disp = np.where(sma60 > 0, (close / sma60 - 1.0) * 100.0, 0.0)

        obs = np.column_stack([
            sma5_disp,
            sma20_disp,
            sma60_disp,
            w["golden_flag"].to_numpy(dtype=np.float64),
            w["dead_flag"].to_numpy(dtype=np.float64),
            w["disparity20_centered"].to_numpy(dtype=np.float64),
            w["rsi14_norm"].to_numpy(dtype=np.float64),
            np.full(self._lookback, float(self._position)),
            np.full(self._lookback, self._unrealized_pnl),
        ]).astype(np.float32)

        return obs
