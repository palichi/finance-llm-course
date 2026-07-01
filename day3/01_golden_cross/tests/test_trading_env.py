"""
env/trading_env.py 스모크 테스트.

실행: pytest tests/test_trading_env.py -v  (프로젝트 루트에서)

검증 항목:
  - reset / step 기본 동작
  - 매수 / 매도 / 유보 행동 처리
  - 무효 행동 페널티
  - 에피소드 종료 조건
  - 짧은 종목 __init__ 단계 필터링
  - 종목 전환 시 포지션 / 손익 / 자산 초기화 (핵심)
  - random policy 멀티 에피소드 안정성
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
import numpy as np
import pandas as pd
import pytest

from indicators.technical import compute_indicators
from env.trading_env import (
    TradingEnv, LOOKBACK, N_FEATURES,
    ACTION_SELL, ACTION_HOLD, ACTION_BUY,
)


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def make_stock_df(code: str, prices: list[float]) -> pd.DataFrame:
    """indicator 계산 완료, NaN 제거 후 종목 DataFrame 반환 (표준명 컬럼)."""
    n     = len(prices)
    dates = [int(d.strftime("%Y%m%d"))
             for d in pd.date_range("20240101", periods=n, freq="B")]
    df = pd.DataFrame({
        "srtnCd": code,
        "basDt" : dates,
        "mkp"   : prices,
        "clpr"  : prices,
        "hipr"  : prices,
        "lopr"  : prices,
    })
    return compute_indicators(df, nan_policy="drop").reset_index(drop=True)


def make_stock_data(n: int = 120) -> dict[str, pd.DataFrame]:
    """A(상승), B(하락) 두 종목 stock_data 반환."""
    return {
        "A": make_stock_df("A", [100.0 + i * 0.5 for i in range(n)]),
        "B": make_stock_df("B", [200.0 - i * 0.3 for i in range(n)]),
    }


@pytest.fixture
def env():
    return TradingEnv(make_stock_data())


@pytest.fixture
def env_a(env):
    """A 종목으로 reset된 환경."""
    env.reset(options={"code": "A"})
    return env


# ---------------------------------------------------------------------------
# 1. reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (LOOKBACK, N_FEATURES)

    def test_obs_dtype_float32(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_info_keys(self, env):
        _, info = env.reset(options={"code": "A"})
        assert {"code", "start_idx", "date"}.issubset(info)

    def test_forced_code(self, env):
        _, info = env.reset(options={"code": "B"})
        assert info["code"] == "B"

    def test_unknown_code_raises(self, env):
        with pytest.raises(KeyError):
            env.reset(options={"code": "UNKNOWN"})

    def test_equity_starts_at_1(self, env):
        env.reset()
        assert env._equity == pytest.approx(1.0)

    def test_position_starts_at_0(self, env):
        env.reset()
        assert env._position == 0

    def test_unrealized_pnl_starts_at_0(self, env):
        env.reset()
        assert env._unrealized_pnl == pytest.approx(0.0)

    def test_cursor_at_lookback_minus_1(self, env):
        env.reset()
        assert env._cursor == LOOKBACK - 1

    def test_obs_no_nan(self, env):
        obs, _ = env.reset(options={"code": "A"})
        assert not np.isnan(obs).any()

    def test_date_string_in_info(self, env):
        """date는 YYYY-MM-DD 형태 문자열."""
        _, info = env.reset(options={"code": "A"})
        assert "-" in info["date"]  # 날짜 구분자 포함


# ---------------------------------------------------------------------------
# 2. 짧은 종목 필터링 (__init__ 단계)
# ---------------------------------------------------------------------------

class TestShortStockFiltering:
    def test_short_only_raises_at_init(self):
        """lookback보다 짧은 종목만 있으면 __init__에서 ValueError."""
        tiny = make_stock_df("T", [100.0] * 10)  # < lookback 행
        with pytest.raises(ValueError, match="유효 종목 없음"):
            TradingEnv({"T": tiny})

    def test_mixed_short_and_valid(self):
        """짧은 종목은 경고 후 제외, 유효 종목은 정상 사용."""
        tiny  = make_stock_df("T", [100.0] * 10)
        valid = make_stock_df("A", [100.0 + i * 0.5 for i in range(120)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env = TradingEnv({"T": tiny, "A": valid})
        assert any("T" in str(warning.message) for warning in w)
        assert "T" not in env._stock_data
        assert "A" in env._stock_data

    def test_short_stock_not_selectable(self):
        """필터링된 종목은 reset options로도 선택 불가."""
        tiny  = make_stock_df("T", [100.0] * 10)
        valid = make_stock_df("A", [100.0 + i * 0.5 for i in range(120)])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            env = TradingEnv({"T": tiny, "A": valid})
        with pytest.raises(KeyError):
            env.reset(options={"code": "T"})


# ---------------------------------------------------------------------------
# 3. step() 기본
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_correct_types(self, env_a):
        obs, reward, terminated, truncated, info = env_a.step(ACTION_HOLD)
        assert obs.shape == (LOOKBACK, N_FEATURES)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_truncated_always_false(self, env_a):
        _, _, _, truncated, _ = env_a.step(ACTION_HOLD)
        assert truncated is False

    def test_info_keys(self, env_a):
        _, _, _, _, info = env_a.step(ACTION_HOLD)
        expected = {"code", "date", "price", "position", "unrealized_pnl",
                    "action", "invalid_action", "equity", "reward_base", "shaping"}
        assert expected.issubset(info)

    def test_hold_does_not_change_position(self, env_a):
        env_a.step(ACTION_HOLD)
        assert env_a._position == 0

    def test_cursor_advances_per_step(self, env_a):
        start = env_a._cursor
        env_a.step(ACTION_HOLD)
        assert env_a._cursor == start + 1

    def test_step_without_reset_raises(self):
        env = TradingEnv(make_stock_data())
        with pytest.raises(RuntimeError):
            env.step(ACTION_HOLD)


# ---------------------------------------------------------------------------
# 4. 매수 / 매도 동작
# ---------------------------------------------------------------------------

class TestBuySell:
    def test_buy_sets_position_1(self, env_a):
        env_a.step(ACTION_BUY)
        assert env_a._position == 1

    def test_buy_empties_cash(self, env_a):
        env_a.step(ACTION_BUY)
        assert env_a._cash == pytest.approx(0.0)

    def test_buy_sets_shares_positive(self, env_a):
        env_a.step(ACTION_BUY)
        assert env_a._shares > 0

    def test_buy_reduces_equity_by_tx_cost(self, env_a):
        """매수 직후 equity = 1/(1+tc) < 1."""
        env_a.step(ACTION_BUY)
        assert env_a._equity < 1.0
        assert env_a._equity == pytest.approx(1.0 / (1.0 + env_a._tx_cost), rel=1e-6)

    def test_sell_after_buy_clears_position(self, env_a):
        env_a.step(ACTION_BUY)
        env_a.step(ACTION_SELL)
        assert env_a._position == 0

    def test_sell_after_buy_clears_shares(self, env_a):
        env_a.step(ACTION_BUY)
        env_a.step(ACTION_SELL)
        assert env_a._shares == pytest.approx(0.0)

    def test_roundtrip_tx_cost_applied_twice(self, env_a):
        """매수/매도 가격이 다를 수 있으므로 두 가격을 모두 반영."""
        buy_price  = float(env_a._df.iloc[env_a._cursor]["close"])
        env_a.step(ACTION_BUY)
        sell_price = float(env_a._df.iloc[env_a._cursor]["close"])
        env_a.step(ACTION_SELL)
        tc       = env_a._tx_cost
        expected = (sell_price * (1.0 - tc)) / (buy_price * (1.0 + tc))
        assert env_a._equity == pytest.approx(expected, rel=1e-6)

    def test_unrealized_pnl_nonzero_while_holding(self, env_a):
        """A 종목은 상승 시계열이므로 보유 중 unrealized_pnl > 0."""
        env_a.step(ACTION_BUY)
        for _ in range(5):
            env_a.step(ACTION_HOLD)
        assert env_a._unrealized_pnl > 0

    def test_unrealized_pnl_zero_when_not_holding(self, env_a):
        env_a.step(ACTION_HOLD)
        assert env_a._unrealized_pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. 무효 행동 처리
# ---------------------------------------------------------------------------

class TestInvalidActions:
    def test_sell_when_not_holding_is_flagged(self, env_a):
        _, _, _, _, info = env_a.step(ACTION_SELL)
        assert info["invalid_action"] is True

    def test_sell_when_not_holding_applies_penalty(self):
        stock_data = make_stock_data()
        env1 = TradingEnv(stock_data)
        env2 = TradingEnv(stock_data)
        env1.reset(seed=0, options={"code": "A"})
        env2.reset(seed=0, options={"code": "A"})
        _, r_invalid, _, _, _ = env1.step(ACTION_SELL)
        _, r_hold,    _, _, _ = env2.step(ACTION_HOLD)
        assert r_invalid < r_hold

    def test_buy_when_holding_is_flagged(self, env_a):
        env_a.step(ACTION_BUY)
        _, _, _, _, info = env_a.step(ACTION_BUY)
        assert info["invalid_action"] is True

    def test_buy_when_holding_does_not_double_buy(self, env_a):
        env_a.step(ACTION_BUY)
        shares_before = env_a._shares
        env_a.step(ACTION_BUY)
        assert env_a._shares == pytest.approx(shares_before)

    def test_invalid_action_position_unchanged(self, env_a):
        env_a.step(ACTION_SELL)
        assert env_a._position == 0


# ---------------------------------------------------------------------------
# 6. 에피소드 종료
# ---------------------------------------------------------------------------

class TestTermination:
    def test_terminates_at_data_end(self, env):
        env.reset(options={"code": "A"})
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(ACTION_HOLD)
            steps += 1
            assert steps < 5000
        assert terminated

    def test_terminates_on_ruin(self):
        env = TradingEnv(make_stock_data(), cfg={"ruin_ratio": 0.999})
        env.reset(options={"code": "A"})
        _, _, terminated, _, _ = env.step(ACTION_BUY)
        assert terminated  # equity = 1/(1+tc) < 0.999 → ruin

    def test_terminal_obs_is_zeros(self, env):
        env.reset(options={"code": "A"})
        terminated = False
        obs = None
        while not terminated:
            obs, _, terminated, _, _ = env.step(ACTION_HOLD)
        assert (obs == 0).all()


# ---------------------------------------------------------------------------
# 7. 관측값 구조 검증
# ---------------------------------------------------------------------------

class TestObservation:
    def test_position_broadcast_all_1_when_holding(self, env_a):
        env_a.step(ACTION_BUY)
        obs = env_a._get_obs()
        assert np.allclose(obs[:, 7], 1.0)

    def test_position_broadcast_all_0_when_not_holding(self, env_a):
        obs = env_a._get_obs()
        assert np.allclose(obs[:, 7], 0.0)

    def test_unrealized_pnl_broadcast_consistent(self, env_a):
        env_a.step(ACTION_BUY)
        env_a.step(ACTION_HOLD)
        obs = env_a._get_obs()
        assert np.allclose(obs[:, 8], obs[0, 8])

    def test_obs_finite_values(self, env_a):
        for _ in range(10):
            env_a.step(ACTION_HOLD)
        assert np.isfinite(env_a._get_obs()).all()

    def test_golden_dead_flag_binary(self, env_a):
        for _ in range(30):
            env_a.step(ACTION_HOLD)
            o = env_a._get_obs()
            assert set(o[:, 3].astype(int)).issubset({0, 1})
            assert set(o[:, 4].astype(int)).issubset({0, 1})


# ---------------------------------------------------------------------------
# 8. 종목 경계 격리 (핵심 검증)
# ---------------------------------------------------------------------------

class TestEpisodeBoundaryIsolation:
    def test_position_resets_on_stock_switch(self, env):
        env.reset(options={"code": "A"})
        env.step(ACTION_BUY)
        assert env._position == 1

        env.reset(options={"code": "B"})
        assert env._position == 0

    def test_equity_resets_to_1_on_stock_switch(self, env):
        env.reset(options={"code": "A"})
        env.step(ACTION_BUY)
        env.step(ACTION_SELL)
        assert env._equity != pytest.approx(1.0)

        env.reset(options={"code": "B"})
        assert env._equity == pytest.approx(1.0)

    def test_entry_price_resets_on_stock_switch(self, env):
        env.reset(options={"code": "A"})
        env.step(ACTION_BUY)
        assert env._entry_price > 0

        env.reset(options={"code": "B"})
        assert env._entry_price == pytest.approx(0.0)

    def test_unrealized_pnl_resets_on_stock_switch(self, env):
        env.reset(options={"code": "A"})
        env.step(ACTION_BUY)
        for _ in range(3):
            env.step(ACTION_HOLD)
        assert env._unrealized_pnl > 0

        env.reset(options={"code": "B"})
        assert env._unrealized_pnl == pytest.approx(0.0)

    def test_cursor_resets_on_stock_switch(self, env):
        env.reset(options={"code": "A"})
        initial_cursor = env._cursor
        for _ in range(15):
            env.step(ACTION_HOLD)
        assert env._cursor == initial_cursor + 15

        env.reset(options={"code": "B"})
        assert env._cursor == initial_cursor

    def test_obs_reflects_new_stock_after_switch(self, env):
        obs_a, _ = env.reset(options={"code": "A"})
        obs_b, _ = env.reset(options={"code": "B"})
        assert not np.allclose(obs_a, obs_b)

    def test_shares_reset_on_stock_switch(self, env):
        env.reset(options={"code": "A"})
        env.step(ACTION_BUY)
        assert env._shares > 0

        env.reset(options={"code": "B"})
        assert env._shares == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. random policy 멀티 에피소드 스모크 테스트
# ---------------------------------------------------------------------------

class TestRandomPolicySmoke:
    def test_10_episodes_no_crash(self):
        stock_data = make_stock_data()
        env        = TradingEnv(stock_data)
        codes      = list(stock_data.keys())
        rng        = np.random.default_rng(99)

        for ep in range(10):
            code      = codes[ep % len(codes)]
            obs, info = env.reset(options={"code": code})

            assert obs.shape == (LOOKBACK, N_FEATURES)
            assert info["code"] == code
            assert env._position == 0
            assert env._equity   == pytest.approx(1.0)

            terminated = False
            step_count = 0
            while not terminated:
                action = int(rng.integers(3))
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(reward)
                assert info["equity"] > 0
                assert not truncated
                step_count += 1
            assert step_count > 0

    def test_position_never_exceeds_1(self):
        env = TradingEnv(make_stock_data())
        rng = np.random.default_rng(7)
        for _ in range(3):
            env.reset()
            terminated = False
            while not terminated:
                _, _, terminated, _, info = env.step(int(rng.integers(3)))
                assert info["position"] in (0, 1)

    def test_reward_finite_throughout(self):
        env = TradingEnv(make_stock_data())
        rng = np.random.default_rng(3)
        for _ in range(5):
            env.reset()
            terminated = False
            while not terminated:
                _, reward, terminated, _, _ = env.step(int(rng.integers(3)))
                assert np.isfinite(reward)
