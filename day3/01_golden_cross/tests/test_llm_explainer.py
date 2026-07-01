"""
explain/llm_explainer.py 단위 테스트.

LLM 호출 없이:
  - 폴백 템플릿 동작 검증
  - 숫자 검증(_validate_numbers) 로직 검증
  - API 오류 시 폴백 전환 검증
  - 숫자 불일치 시 폴백 전환 검증
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from explain.rule_based import ExplainResult
from explain.llm_explainer import (
    _fallback_template,
    _validate_numbers,
    _build_user_prompt,
    generate_explanation,
    ExplainOutput,
)


# ---------------------------------------------------------------------------
# 테스트용 ExplainResult 픽스처
# ---------------------------------------------------------------------------

def _make_explain(
    rsi14       : float = 65.6,
    disparity20 : float = 113.1,
    rsi_zone    : str   = "NEUTRAL",
    disp_zone   : str   = "OVERHEATED",
    golden_flag : bool  = False,
    dead_flag   : bool  = False,
    action      : int   = 1,
    action_name : str   = "HOLD",
    top1_action : str   = "HOLD",
    top1_prob   : float = 0.574,
    top2_action : str   = "BUY",
    top2_prob   : float = 0.284,
    low_conf    : bool  = False,
    shaping_rsi : bool  = False,
    shaping_gold: bool  = False,
    shaping_disp: bool  = False,
) -> ExplainResult:
    return ExplainResult(
        rsi14                        = rsi14,
        rsi_zone                     = rsi_zone,
        rsi_overbought               = rsi14 >= 70.0,
        rsi_oversold                 = rsi14 <= 30.0,
        disparity20                  = disparity20,
        disparity_zone               = disp_zone,
        disparity_overheated         = disparity20 >= 105.0,
        disparity_depressed          = disparity20 <= 95.0,
        golden_flag                  = golden_flag,
        dead_flag                    = dead_flag,
        shaping_rsi_buy_penalty      = shaping_rsi,
        shaping_golden_hold_bonus    = shaping_gold,
        shaping_disparity_buy_penalty= shaping_disp,
        action_probs                 = {"BUY": 0.284, "HOLD": 0.574, "SELL": 0.142},
        top1_action                  = top1_action,
        top1_prob                    = top1_prob,
        top2_action                  = top2_action,
        top2_prob                    = top2_prob,
        low_confidence               = low_conf,
        ticker                       = "005930",
        name                         = "삼성전자",
        date                         = "2026-06-18",
    )


# ===========================================================================
# 폴백 템플릿
# ===========================================================================

class TestFallbackTemplate:
    def test_contains_ticker(self):
        text = _fallback_template(_make_explain())
        assert "005930" in text

    def test_contains_action_korean(self):
        text = _fallback_template(_make_explain(top1_action="HOLD"))
        assert "유보" in text

    def test_contains_rsi_value(self):
        text = _fallback_template(_make_explain(rsi14=65.6))
        assert "65.6" in text

    def test_contains_disparity_value(self):
        text = _fallback_template(_make_explain(disparity20=113.1))
        assert "113.1" in text

    def test_rsi_overbought_mention(self):
        text = _fallback_template(_make_explain(rsi14=75.0, rsi_zone="OVERBOUGHT"))
        assert "과매수" in text

    def test_rsi_oversold_mention(self):
        text = _fallback_template(_make_explain(rsi14=25.0, rsi_zone="OVERSOLD"))
        assert "과매도" in text

    def test_rsi_neutral_mention(self):
        text = _fallback_template(_make_explain(rsi14=50.0, rsi_zone="NEUTRAL"))
        assert "중립" in text

    def test_disparity_overheated_mention(self):
        text = _fallback_template(_make_explain(disparity20=110.0, disp_zone="OVERHEATED"))
        assert "과열" in text

    def test_disparity_depressed_mention(self):
        text = _fallback_template(_make_explain(disparity20=90.0, disp_zone="DEPRESSED"))
        assert "침체" in text

    def test_golden_cross_mention(self):
        text = _fallback_template(_make_explain(golden_flag=True))
        assert "골든크로스" in text

    def test_dead_cross_mention(self):
        text = _fallback_template(_make_explain(dead_flag=True))
        assert "데드크로스" in text

    def test_low_confidence_mention(self):
        text = _fallback_template(_make_explain(
            top1_prob=0.36, top2_prob=0.34, low_conf=True
        ))
        assert "확신도" in text

    def test_shaping_rsi_mention(self):
        text = _fallback_template(_make_explain(shaping_rsi=True))
        assert "RSI" in text and ("패널티" in text or "셰이핑" in text)

    def test_shaping_golden_mention(self):
        text = _fallback_template(_make_explain(shaping_gold=True))
        assert "골든크로스 보유 보너스" in text

    def test_shaping_disp_mention(self):
        text = _fallback_template(_make_explain(shaping_disp=True))
        assert "이격도 과열 매수 패널티" in text

    def test_returns_string(self):
        result = _fallback_template(_make_explain())
        assert isinstance(result, str)
        assert len(result) > 30

    def test_buy_action_korean(self):
        text = _fallback_template(_make_explain(
            action=2, action_name="BUY", top1_action="BUY"
        ))
        assert "매수" in text

    def test_sell_action_korean(self):
        text = _fallback_template(_make_explain(
            action=0, action_name="SELL", top1_action="SELL"
        ))
        assert "매도" in text


# ===========================================================================
# 숫자 검증 (_validate_numbers)
# ===========================================================================

class TestValidateNumbers:
    BASE = {"BUY": 0.284, "HOLD": 0.574, "SELL": 0.142}

    def test_valid_rsi_exact(self):
        text = "RSI14는 65.6입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_valid_disparity_exact(self):
        text = "이격도는 113.1입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_valid_both_exact(self):
        text = "RSI14는 65.6이고 이격도는 113.1입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_invalid_rsi_hallucinated(self):
        # 55.0 이라는 잘못된 RSI 값
        text = "RSI14는 55.0으로 중립입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is False

    def test_invalid_disparity_hallucinated(self):
        # 120.0 이라는 잘못된 disparity 값
        text = "이격도는 120.0입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is False

    def test_valid_rsi_rounded(self):
        # 반올림 값 (66) 도 허용 (±1.0)
        text = "RSI는 약 66입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_valid_disparity_rounded(self):
        text = "이격도는 113 수준입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_valid_action_probs(self):
        # 확률 백분율 (57.4%) 도 허용
        text = "HOLD 확률은 57.4%입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_valid_small_numbers_ignored(self):
        # 1, 2, 3 등 작은 수사는 무시
        text = "3가지 행동 중 1위는 유보입니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True

    def test_valid_no_numbers(self):
        text = "모델이 유보를 선택했습니다."
        assert _validate_numbers(text, 65.6, 113.1, self.BASE) is True


# ===========================================================================
# generate_explanation: API 없는 경우 폴백
# ===========================================================================

class TestGenerateExplanationNoKey:
    def test_no_key_returns_fallback(self):
        r = _make_explain()
        out = generate_explanation(r, api_key="")
        assert isinstance(out, ExplainOutput)
        assert out.used_llm is False
        assert out.reason == "api_error"
        assert len(out.text) > 30

    def test_fallback_text_contains_ticker(self):
        r = _make_explain()
        out = generate_explanation(r, api_key="")
        assert "005930" in out.text


# ===========================================================================
# generate_explanation: API 오류 시 폴백
# ===========================================================================

class TestGenerateExplanationApiError:
    def test_api_exception_returns_fallback(self):
        r = _make_explain()
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = Exception("connection error")
            out = generate_explanation(r, api_key="sk-fake-key")
        assert out.used_llm is False
        assert out.reason == "api_error"

    def test_api_error_text_is_template(self):
        r = _make_explain()
        expected = _fallback_template(r)
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = RuntimeError("timeout")
            out = generate_explanation(r, api_key="sk-fake-key")
        assert out.text == expected


# ===========================================================================
# generate_explanation: 숫자 불일치 시 폴백
# ===========================================================================

class TestGenerateExplanationNumberMismatch:
    def _mock_llm(self, text: str):
        mock_msg    = MagicMock()
        mock_msg.content = [MagicMock(text=text)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        return mock_client

    def test_mismatch_rsi_triggers_fallback(self):
        r   = _make_explain(rsi14=65.6)
        bad = "RSI14는 55.0으로 중립 구간에 있습니다. 이격도는 113.1입니다. 모델은 유보를 선택했습니다."
        with patch("anthropic.Anthropic", return_value=self._mock_llm(bad)):
            out = generate_explanation(r, api_key="sk-fake-key")
        assert out.used_llm is False
        assert out.reason == "number_mismatch"

    def test_mismatch_disparity_triggers_fallback(self):
        r   = _make_explain(disparity20=113.1)
        bad = "RSI14는 65.6입니다. 이격도는 120.0으로 과열 상태입니다. 유보를 선택했습니다."
        with patch("anthropic.Anthropic", return_value=self._mock_llm(bad)):
            out = generate_explanation(r, api_key="sk-fake-key")
        assert out.used_llm is False
        assert out.reason == "number_mismatch"

    def test_correct_numbers_returns_llm(self):
        r    = _make_explain(rsi14=65.6, disparity20=113.1)
        good = "RSI14는 65.6으로 중립 구간입니다. 이격도는 113.1로 과열 상태입니다. 모델은 유보(57.4%)를 선택했습니다."
        with patch("anthropic.Anthropic", return_value=self._mock_llm(good)):
            out = generate_explanation(r, api_key="sk-fake-key")
        assert out.used_llm is True
        assert out.reason == "llm_ok"
        assert out.text == good

    def test_no_numbers_in_response_passes(self):
        r    = _make_explain()
        good = "모델은 유보를 선택했습니다. RSI는 중립 구간에 있고 이격도는 과열 상태입니다. 확신도는 충분합니다."
        with patch("anthropic.Anthropic", return_value=self._mock_llm(good)):
            out = generate_explanation(r, api_key="sk-fake-key")
        assert out.used_llm is True
        assert out.reason == "llm_ok"


# ===========================================================================
# 프롬프트 조립 (_build_user_prompt) 기본 검증
# ===========================================================================

class TestBuildUserPrompt:
    def test_contains_ticker(self):
        prompt = _build_user_prompt(_make_explain())
        assert "005930" in prompt

    def test_contains_rsi_value(self):
        prompt = _build_user_prompt(_make_explain(rsi14=65.6))
        assert "65.6" in prompt

    def test_contains_disparity_value(self):
        prompt = _build_user_prompt(_make_explain(disparity20=113.1))
        assert "113.1" in prompt

    def test_low_confidence_flagged_in_prompt(self):
        prompt = _build_user_prompt(_make_explain(low_conf=True))
        assert "예" in prompt

    def test_shaping_listed_in_prompt(self):
        prompt = _build_user_prompt(_make_explain(shaping_rsi=True))
        assert "RSI" in prompt and "패널티" in prompt

    def test_no_shaping_shows_none(self):
        prompt = _build_user_prompt(_make_explain())
        assert "없음" in prompt
