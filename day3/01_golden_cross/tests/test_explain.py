"""
explain/rule_based.py 단위 테스트.

가상의 PredictResult를 직접 생성해서 플래그 로직만 검증한다.
실제 모델·CSV 로드 없음.
"""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass

# ── 테스트 전용 PredictResult 스텁 ─────────────────────────────────────────

@dataclass
class _FakePredictResult:
    """explain()에 필요한 필드만 갖춘 최소 스텁."""
    action      : int
    action_name : str
    action_probs: dict
    date        : str
    ticker      : str
    name        : str
    indicators  : dict
    obs         : np.ndarray = None

    def __post_init__(self):
        if self.obs is None:
            self.obs = np.zeros((20, 9), dtype=np.float32)


def _make_result(
    action: int = 1,
    rsi14: float = 50.0,
    disparity20: float = 100.0,
    golden_flag: int = 0,
    dead_flag: int = 0,
    probs: dict | None = None,
) -> _FakePredictResult:
    if probs is None:
        probs = {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33}
    names = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return _FakePredictResult(
        action       = action,
        action_name  = names[action],
        action_probs = probs,
        date         = "2026-06-28",
        ticker       = "000000",
        name         = "테스트종목",
        indicators   = {
            "close": 10000.0,
            "open":  9900.0,
            "high":  10100.0,
            "low":   9800.0,
            "volume": 1000.0,
            "sma5":  9950.0,
            "sma20": 9800.0,
            "sma60": 9700.0,
            "ema20": 9820.0,
            "golden_flag":  float(golden_flag),
            "dead_flag":    float(dead_flag),
            "disparity20":  disparity20,
            "disparity20_centered": disparity20 - 100.0,
            "rsi14":        rsi14,
            "rsi14_norm":   (rsi14 - 50) / 50,
        },
    )


# ── import ──────────────────────────────────────────────────────────────────

from explain.rule_based import explain, ExplainResult


# ===========================================================================
# RSI 구간 판정
# ===========================================================================

class TestRsiZone:
    def test_overbought_exactly_70(self):
        r = explain(_make_result(rsi14=70.0))
        assert r.rsi_zone == "OVERBOUGHT"
        assert r.rsi_overbought is True
        assert r.rsi_oversold   is False

    def test_overbought_above_70(self):
        r = explain(_make_result(rsi14=75.0))
        assert r.rsi_zone == "OVERBOUGHT"
        assert r.rsi14 == pytest.approx(75.0)

    def test_oversold_exactly_30(self):
        r = explain(_make_result(rsi14=30.0))
        assert r.rsi_zone == "OVERSOLD"
        assert r.rsi_oversold   is True
        assert r.rsi_overbought is False

    def test_oversold_below_30(self):
        r = explain(_make_result(rsi14=25.0))
        assert r.rsi_zone == "OVERSOLD"

    def test_neutral_mid(self):
        r = explain(_make_result(rsi14=50.0))
        assert r.rsi_zone == "NEUTRAL"
        assert r.rsi_overbought is False
        assert r.rsi_oversold   is False

    def test_neutral_upper_boundary(self):
        r = explain(_make_result(rsi14=69.9))
        assert r.rsi_zone == "NEUTRAL"

    def test_neutral_lower_boundary(self):
        r = explain(_make_result(rsi14=30.1))
        assert r.rsi_zone == "NEUTRAL"


# ===========================================================================
# Disparity20 구간 판정
# ===========================================================================

class TestDisparityZone:
    def test_overheated_exactly_105(self):
        r = explain(_make_result(disparity20=105.0))
        assert r.disparity_zone == "OVERHEATED"
        assert r.disparity_overheated is True
        assert r.disparity_depressed  is False

    def test_overheated_above_105(self):
        r = explain(_make_result(disparity20=112.0))
        assert r.disparity_zone == "OVERHEATED"
        assert r.disparity20 == pytest.approx(112.0)

    def test_depressed_exactly_95(self):
        r = explain(_make_result(disparity20=95.0))
        assert r.disparity_zone == "DEPRESSED"
        assert r.disparity_depressed  is True
        assert r.disparity_overheated is False

    def test_depressed_below_95(self):
        r = explain(_make_result(disparity20=90.0))
        assert r.disparity_zone == "DEPRESSED"

    def test_neutral_exactly_100(self):
        r = explain(_make_result(disparity20=100.0))
        assert r.disparity_zone == "NEUTRAL"

    def test_neutral_upper_boundary(self):
        r = explain(_make_result(disparity20=104.9))
        assert r.disparity_zone == "NEUTRAL"

    def test_neutral_lower_boundary(self):
        r = explain(_make_result(disparity20=95.1))
        assert r.disparity_zone == "NEUTRAL"


# ===========================================================================
# 가상 케이스: RSI=75, disparity=112
# ===========================================================================

class TestKeyCase:
    """README 명시 케이스 — RSI=75, disparity=112로 플래그가 올바르게 켜지는지."""

    def setup_method(self):
        self.result_buy  = _make_result(
            action=2, rsi14=75.0, disparity20=112.0,
            probs={"BUY": 0.60, "HOLD": 0.30, "SELL": 0.10},
        )
        self.result_hold = _make_result(
            action=1, rsi14=75.0, disparity20=112.0,
            probs={"BUY": 0.40, "HOLD": 0.50, "SELL": 0.10},
        )

    def test_rsi_overbought_flag(self):
        r = explain(self.result_buy)
        assert r.rsi_zone      == "OVERBOUGHT"
        assert r.rsi_overbought is True

    def test_disparity_overheated_flag(self):
        r = explain(self.result_buy)
        assert r.disparity_zone      == "OVERHEATED"
        assert r.disparity_overheated is True

    def test_shaping_rsi_penalty_on_new_buy(self):
        # 미보유(position=0) + BUY → RSI 패널티 발동
        r = explain(self.result_buy, position=0)
        assert r.shaping_rsi_buy_penalty is True

    def test_shaping_rsi_penalty_off_on_hold(self):
        # BUY가 아니면 패널티 없음
        r = explain(self.result_hold, position=0)
        assert r.shaping_rsi_buy_penalty is False

    def test_shaping_disparity_penalty_on_new_buy(self):
        r = explain(self.result_buy, position=0)
        assert r.shaping_disparity_buy_penalty is True

    def test_shaping_disparity_penalty_off_already_holding(self):
        # 이미 보유 중(position=1) → "신규 매수"가 아니므로 패널티 없음
        r = explain(self.result_buy, position=1)
        assert r.shaping_disparity_buy_penalty is False

    def test_both_penalties_fire_together(self):
        r = explain(self.result_buy, position=0)
        assert r.shaping_rsi_buy_penalty       is True
        assert r.shaping_disparity_buy_penalty is True


# ===========================================================================
# Golden/Dead 크로스 플래그
# ===========================================================================

class TestCrossFlags:
    def test_golden_flag_true(self):
        r = explain(_make_result(golden_flag=1))
        assert r.golden_flag is True
        assert r.dead_flag   is False

    def test_dead_flag_true(self):
        r = explain(_make_result(dead_flag=1))
        assert r.dead_flag   is True
        assert r.golden_flag is False

    def test_both_flags_false(self):
        r = explain(_make_result(golden_flag=0, dead_flag=0))
        assert r.golden_flag is False
        assert r.dead_flag   is False

    def test_golden_hold_bonus_fires_when_holding(self):
        result = _make_result(golden_flag=1, action=1)
        r = explain(result, position=1)
        assert r.shaping_golden_hold_bonus is True

    def test_golden_hold_bonus_off_when_not_holding(self):
        result = _make_result(golden_flag=1, action=1)
        r = explain(result, position=0)
        assert r.shaping_golden_hold_bonus is False

    def test_golden_hold_bonus_off_without_golden_flag(self):
        result = _make_result(golden_flag=0, action=1)
        r = explain(result, position=1)
        assert r.shaping_golden_hold_bonus is False


# ===========================================================================
# 확신도 판정
# ===========================================================================

class TestConfidence:
    def test_low_confidence_exactly_10pct(self):
        # 차이 정확히 0.10 → 미만(< 0.10)이 아니므로 low_confidence=False
        probs = {"BUY": 0.45, "HOLD": 0.35, "SELL": 0.20}
        r = explain(_make_result(probs=probs))
        assert r.low_confidence is False

    def test_low_confidence_below_10pct(self):
        # 1위(HOLD=0.38) - 2위(BUY=0.35) = 0.03 < 0.10
        probs = {"BUY": 0.35, "HOLD": 0.38, "SELL": 0.27}
        r = explain(_make_result(probs=probs))
        assert r.low_confidence is True
        assert r.top1_action == "HOLD"
        assert r.top2_action == "BUY"

    def test_high_confidence(self):
        probs = {"BUY": 0.80, "HOLD": 0.15, "SELL": 0.05}
        r = explain(_make_result(probs=probs))
        assert r.low_confidence is False
        assert r.top1_action == "BUY"
        assert r.top1_prob == pytest.approx(0.80)

    def test_probs_passed_through(self):
        probs = {"BUY": 0.50, "HOLD": 0.30, "SELL": 0.20}
        r = explain(_make_result(probs=probs))
        assert r.action_probs == probs


# ===========================================================================
# summary() 문자열 기본 검증
# ===========================================================================

class TestSummary:
    def test_summary_contains_ticker(self):
        r = explain(_make_result())
        assert "000000" in r.summary()

    def test_summary_contains_rsi_value(self):
        r = explain(_make_result(rsi14=75.0))
        assert "75.0" in r.summary()

    def test_summary_low_confidence_warning(self):
        probs = {"BUY": 0.36, "HOLD": 0.35, "SELL": 0.29}
        r = explain(_make_result(probs=probs))
        assert "확신도 낮음" in r.summary()

    def test_summary_no_warning_when_confident(self):
        probs = {"BUY": 0.80, "HOLD": 0.15, "SELL": 0.05}
        r = explain(_make_result(probs=probs))
        assert "확신도 낮음" not in r.summary()
