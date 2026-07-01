"""
train/reward.py 단위 테스트.

실행: pytest tests/test_reward.py -v  (프로젝트 루트에서)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from train.reward import ACTION_SELL, ACTION_HOLD, ACTION_BUY, RewardConfig, RewardShaper
from indicators.technical import compute_indicators
from env.trading_env import TradingEnv


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def make_shaper(**kw) -> RewardShaper:
    """RewardConfig 오버라이드를 받아 RewardShaper 생성."""
    return RewardShaper(RewardConfig(**kw))


def compute(shaper: RewardShaper, **kw) -> tuple[float, dict]:
    defaults = dict(
        base_reward=0.0,
        action=ACTION_HOLD,
        position_before=0,
        position_after=0,
        rsi14=50.0,
        golden_flag=0,
        disparity20=100.0,
    )
    return shaper.compute(**{**defaults, **kw})


# ---------------------------------------------------------------------------
# 1. 셰이핑 off → 기본 보상만 반환
# ---------------------------------------------------------------------------

class TestNoShaping:
    def test_all_flags_off_returns_base(self):
        shaper = make_shaper(
            enable_rsi_penalty=False,
            enable_golden_bonus=False,
            enable_disparity_penalty=False,
        )
        final, detail = compute(shaper, base_reward=0.03)
        assert final == pytest.approx(0.03)
        assert detail["total_shaping"] == pytest.approx(0.0)

    def test_shaping_detail_zeros_when_all_off(self):
        shaper = make_shaper(
            enable_rsi_penalty=False,
            enable_golden_bonus=False,
            enable_disparity_penalty=False,
        )
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0, position_after=1,
                            rsi14=80.0, golden_flag=1, disparity20=115.0)
        assert detail["rsi_penalty"]       == pytest.approx(0.0)
        assert detail["golden_bonus"]      == pytest.approx(0.0)
        assert detail["disparity_penalty"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. RSI 페널티
# ---------------------------------------------------------------------------

class TestRsiPenalty:
    def test_fires_on_new_buy_with_high_rsi(self):
        shaper = make_shaper()
        final, detail = compute(shaper,
                                base_reward=0.0,
                                action=ACTION_BUY,
                                position_before=0,
                                position_after=1,
                                rsi14=75.0)
        assert detail["rsi_penalty"] == pytest.approx(-0.1)

    def test_not_fired_when_disabled(self):
        shaper = make_shaper(enable_rsi_penalty=False)
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, rsi14=80.0)
        assert detail["rsi_penalty"] == pytest.approx(0.0)

    def test_not_fired_on_hold_even_if_rsi_high(self):
        shaper = make_shaper()
        _, detail = compute(shaper, action=ACTION_HOLD, position_before=1,
                            position_after=1, rsi14=80.0)
        assert detail["rsi_penalty"] == pytest.approx(0.0)

    def test_not_fired_on_double_buy_attempt(self):
        """이미 포지션 보유 상태에서 BUY(무효) — 신규 매수 아님."""
        shaper = make_shaper()
        _, detail = compute(shaper, action=ACTION_BUY, position_before=1,
                            position_after=1, rsi14=80.0)
        assert detail["rsi_penalty"] == pytest.approx(0.0)

    def test_not_fired_exactly_at_threshold(self):
        """RSI == 70.0 은 초과가 아니므로 페널티 없음."""
        shaper = make_shaper()
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, rsi14=70.0)
        assert detail["rsi_penalty"] == pytest.approx(0.0)

    def test_custom_weight_applied(self):
        shaper = make_shaper(rsi_buy_penalty=0.2)
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, rsi14=75.0)
        assert detail["rsi_penalty"] == pytest.approx(-0.2)

    def test_custom_threshold(self):
        shaper = make_shaper(rsi_overbought_threshold=60.0)
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, rsi14=65.0)
        assert detail["rsi_penalty"] == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# 3. 골든크로스 보너스
# ---------------------------------------------------------------------------

class TestGoldenBonus:
    def test_fires_when_golden_and_holding(self):
        shaper = make_shaper()
        _, detail = compute(shaper, golden_flag=1,
                            position_before=1, position_after=1, action=ACTION_HOLD)
        assert detail["golden_bonus"] == pytest.approx(0.05)

    def test_fires_when_golden_and_just_bought(self):
        shaper = make_shaper()
        _, detail = compute(shaper, golden_flag=1,
                            position_before=0, position_after=1, action=ACTION_BUY)
        assert detail["golden_bonus"] == pytest.approx(0.05)

    def test_not_fired_when_not_holding(self):
        shaper = make_shaper()
        _, detail = compute(shaper, golden_flag=1,
                            position_before=0, position_after=0, action=ACTION_HOLD)
        assert detail["golden_bonus"] == pytest.approx(0.0)

    def test_not_fired_when_disabled(self):
        shaper = make_shaper(enable_golden_bonus=False)
        _, detail = compute(shaper, golden_flag=1,
                            position_before=1, position_after=1, action=ACTION_HOLD)
        assert detail["golden_bonus"] == pytest.approx(0.0)

    def test_not_fired_when_no_golden_flag(self):
        shaper = make_shaper()
        _, detail = compute(shaper, golden_flag=0,
                            position_before=1, position_after=1, action=ACTION_HOLD)
        assert detail["golden_bonus"] == pytest.approx(0.0)

    def test_custom_weight_applied(self):
        shaper = make_shaper(golden_hold_bonus=0.08)
        _, detail = compute(shaper, golden_flag=1,
                            position_before=1, position_after=1, action=ACTION_HOLD)
        assert detail["golden_bonus"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# 4. 이격도 페널티
# ---------------------------------------------------------------------------

class TestDisparityPenalty:
    def test_fires_on_new_buy_with_high_disparity(self):
        shaper = make_shaper()
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, disparity20=115.0)
        assert detail["disparity_penalty"] == pytest.approx(-0.1)

    def test_not_fired_when_disabled(self):
        shaper = make_shaper(enable_disparity_penalty=False)
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, disparity20=115.0)
        assert detail["disparity_penalty"] == pytest.approx(0.0)

    def test_not_fired_on_hold_even_if_disparity_high(self):
        shaper = make_shaper()
        _, detail = compute(shaper, action=ACTION_HOLD, position_before=1,
                            position_after=1, disparity20=120.0)
        assert detail["disparity_penalty"] == pytest.approx(0.0)

    def test_not_fired_exactly_at_threshold(self):
        """disparity == 110.0 은 초과가 아님."""
        shaper = make_shaper()
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, disparity20=110.0)
        assert detail["disparity_penalty"] == pytest.approx(0.0)

    def test_custom_weight_applied(self):
        shaper = make_shaper(disparity_buy_penalty=0.15)
        _, detail = compute(shaper, action=ACTION_BUY, position_before=0,
                            position_after=1, disparity20=115.0)
        assert detail["disparity_penalty"] == pytest.approx(-0.15)


# ---------------------------------------------------------------------------
# 5. 복합 조건 누적
# ---------------------------------------------------------------------------

class TestStackedShaping:
    def test_rsi_and_disparity_both_fire(self):
        """RSI 과매수 + 이격도 과열 신규 매수 → 두 페널티 모두 적용."""
        shaper = make_shaper()
        final, detail = compute(shaper, base_reward=0.0,
                                action=ACTION_BUY, position_before=0, position_after=1,
                                rsi14=75.0, disparity20=115.0)
        assert detail["rsi_penalty"]       == pytest.approx(-0.1)
        assert detail["disparity_penalty"] == pytest.approx(-0.1)
        assert detail["total_shaping"]     == pytest.approx(-0.2)
        assert final                       == pytest.approx(-0.2)

    def test_golden_bonus_offsets_rsi_penalty(self):
        """골든크로스 직후 RSI 과매수 매수: +0.05 - 0.1 = -0.05."""
        shaper = make_shaper()
        _, detail = compute(shaper, base_reward=0.0,
                            action=ACTION_BUY, position_before=0, position_after=1,
                            rsi14=75.0, golden_flag=1, disparity20=100.0)
        assert detail["total_shaping"] == pytest.approx(0.05 - 0.1)

    def test_base_reward_combined(self):
        shaper = make_shaper(
            enable_rsi_penalty=False,
            enable_disparity_penalty=False,
        )
        final, _ = compute(shaper, base_reward=0.02,
                           golden_flag=1, position_before=1, position_after=1,
                           action=ACTION_HOLD)
        assert final == pytest.approx(0.02 + 0.05)


# ---------------------------------------------------------------------------
# 6. 에피소드 카운터
# ---------------------------------------------------------------------------

class TestEpisodeCounter:
    def test_counters_start_at_zero(self):
        shaper = make_shaper()
        assert shaper.get_episode_summary() == {
            "rsi_penalty": 0, "golden_bonus": 0, "disparity_penalty": 0
        }

    def test_rsi_counter_increments(self):
        shaper = make_shaper()
        for _ in range(3):
            compute(shaper, action=ACTION_BUY, position_before=0,
                    position_after=1, rsi14=80.0)
        assert shaper.get_episode_summary()["rsi_penalty"] == 3

    def test_golden_counter_increments(self):
        shaper = make_shaper()
        for _ in range(5):
            compute(shaper, golden_flag=1, position_before=1,
                    position_after=1, action=ACTION_HOLD)
        assert shaper.get_episode_summary()["golden_bonus"] == 5

    def test_disparity_counter_increments(self):
        shaper = make_shaper()
        compute(shaper, action=ACTION_BUY, position_before=0,
                position_after=1, disparity20=112.0)
        compute(shaper, action=ACTION_BUY, position_before=0,
                position_after=1, disparity20=90.0)   # 미발동
        assert shaper.get_episode_summary()["disparity_penalty"] == 1

    def test_reset_episode_clears_counters(self):
        shaper = make_shaper()
        compute(shaper, action=ACTION_BUY, position_before=0,
                position_after=1, rsi14=80.0)
        shaper.reset_episode()
        assert shaper.get_episode_summary() == {
            "rsi_penalty": 0, "golden_bonus": 0, "disparity_penalty": 0
        }

    def test_counters_not_incremented_when_flag_off(self):
        shaper = make_shaper(
            enable_rsi_penalty=False,
            enable_golden_bonus=False,
            enable_disparity_penalty=False,
        )
        compute(shaper, action=ACTION_BUY, position_before=0, position_after=1,
                rsi14=80.0, golden_flag=1, disparity20=120.0)
        assert sum(shaper.get_episode_summary().values()) == 0


# ---------------------------------------------------------------------------
# 7. RewardConfig.from_dict
# ---------------------------------------------------------------------------

class TestRewardConfigFromDict:
    def test_flat_dict(self):
        cfg = RewardConfig.from_dict({"rsi_buy_penalty": 0.2, "golden_hold_bonus": 0.08})
        assert cfg.rsi_buy_penalty   == pytest.approx(0.2)
        assert cfg.golden_hold_bonus == pytest.approx(0.08)

    def test_reward_subkey(self):
        cfg = RewardConfig.from_dict({"reward": {"disparity_buy_penalty": 0.15}})
        assert cfg.disparity_buy_penalty == pytest.approx(0.15)

    def test_unknown_keys_ignored(self):
        cfg = RewardConfig.from_dict({"unknown_param": 999, "rsi_buy_penalty": 0.05})
        assert cfg.rsi_buy_penalty == pytest.approx(0.05)

    def test_defaults_preserved_for_missing_keys(self):
        cfg = RewardConfig.from_dict({"rsi_buy_penalty": 0.2})
        assert cfg.golden_hold_bonus    == pytest.approx(0.05)
        assert cfg.disparity_buy_penalty == pytest.approx(0.1)

    def test_flag_override(self):
        cfg = RewardConfig.from_dict({"enable_rsi_penalty": False})
        assert cfg.enable_rsi_penalty is False
        assert cfg.enable_golden_bonus is True   # 기본값 유지


# ---------------------------------------------------------------------------
# 8. env 통합 — step()에서 RewardShaper 연결 확인
# ---------------------------------------------------------------------------

def make_stock_df(code: str, prices: list[float]) -> pd.DataFrame:
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


class TestEnvRewardShaperIntegration:
    PRICES = [float(100 + i) for i in range(150)]

    def _env(self, **cfg_kw) -> tuple[TradingEnv, RewardShaper]:
        shaper = RewardShaper(RewardConfig(**cfg_kw))
        df     = make_stock_df("A", self.PRICES)
        env    = TradingEnv({"A": df}, reward_shaper=shaper)
        return env, shaper

    def test_shaping_detail_in_info(self):
        env, _ = self._env()
        env.reset(options={"code": "A"})
        _, _, _, _, info = env.step(ACTION_BUY)
        assert "shaping" in info
        assert "rsi_penalty" in info["shaping"]

    def test_reward_base_in_info(self):
        env, _ = self._env()
        env.reset(options={"code": "A"})
        _, _, _, _, info = env.step(ACTION_HOLD)
        assert "reward_base" in info

    def test_reset_clears_episode_counter(self):
        env, shaper = self._env()
        env.reset(options={"code": "A"})
        # 인위적으로 카운터 증가
        shaper._counts["golden_bonus"] = 5
        # reset() → reset_episode() 호출 → 카운터 초기화
        env.reset(options={"code": "A"})
        assert shaper.get_episode_summary()["golden_bonus"] == 0

    def test_episode_summary_in_terminal_info(self):
        df     = make_stock_df("T", [100.0] * 100)   # nan_policy="drop" 후 충분한 행 확보
        shaper = RewardShaper()
        env    = TradingEnv({"T": df}, reward_shaper=shaper)
        env.reset(options={"code": "T"})
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(ACTION_HOLD)
        assert "episode_shaping_summary" in info
        assert set(info["episode_shaping_summary"].keys()) == {
            "rsi_penalty", "golden_bonus", "disparity_penalty"
        }

    def test_no_shaper_still_works(self):
        """reward_shaper=None → 기존 동작 그대로."""
        df  = make_stock_df("A", self.PRICES)
        env = TradingEnv({"A": df})    # reward_shaper 없음
        env.reset(options={"code": "A"})
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
        assert np.isfinite(reward)
        assert "shaping" in info
        assert info["shaping"] == {}

    def test_golden_bonus_adds_reward(self):
        """골든크로스 플래그가 1인 행에서 HOLD하면 보너스만큼 reward 증가."""
        df     = make_stock_df("A", self.PRICES)
        shaper = RewardShaper()
        env    = TradingEnv({"A": df}, reward_shaper=shaper)
        env.reset(options={"code": "A"})

        # 포지션 진입 후 golden_flag가 1인 스텝을 찾아 reward 비교
        env.step(ACTION_BUY)   # 포지션 보유 상태로 만들기
        row = env._df.iloc[env._cursor]
        if int(row.get("golden_flag", 0) if hasattr(row, "get") else row["golden_flag"]) == 1:
            _, r_shaping, _, _, _    = env.step(ACTION_HOLD)
            env2 = TradingEnv({"A": df})
            env2.reset(options={"code": "A"})
            env2.step(ACTION_BUY)
            _, r_base, _, _, _ = env2.step(ACTION_HOLD)
            assert r_shaping > r_base - 1e-9   # 보너스로 더 높거나 동일

    def test_rsi_penalty_subtracts_reward(self):
        """RSI > 70 & 신규 매수 시 페널티만큼 reward 감소."""
        df  = make_stock_df("A", self.PRICES)

        shaper_on  = RewardShaper(RewardConfig(enable_rsi_penalty=True,  rsi_overbought_threshold=0.0))
        shaper_off = RewardShaper(RewardConfig(enable_rsi_penalty=False))

        env_on  = TradingEnv({"A": df}, reward_shaper=shaper_on)
        env_off = TradingEnv({"A": df}, reward_shaper=shaper_off)

        env_on.reset(options={"code": "A"})
        env_off.reset(options={"code": "A"})

        _, r_on,  _, _, _ = env_on.step(ACTION_BUY)
        _, r_off, _, _, _ = env_off.step(ACTION_BUY)

        # rsi_overbought_threshold=0.0 이면 어떤 RSI 값이든 페널티 발동
        assert r_on < r_off
