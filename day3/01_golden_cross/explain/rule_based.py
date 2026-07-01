"""
규칙 기반 설명 모듈 — LLM 미사용, 100% 결정론적.

predict.py의 PredictResult를 받아서 기술지표 상태와
reward shaping 발동 조건, 확신도를 플래그로 변환한다.

사용법:
    from inference.predict import predict
    from explain.rule_based import explain

    result  = predict("005930")
    explain_result = explain(result)
    print(explain_result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference.predict import PredictResult

# ---------------------------------------------------------------------------
# 임계값 상수 (reward shaping과 동일 기준 사용)
# ---------------------------------------------------------------------------

RSI_OVERBOUGHT  = 70.0
RSI_OVERSOLD    = 30.0
DISP_OVERHEATED = 105.0   # explain 기준 (reward shaping은 110)
DISP_DEPRESSED  =  95.0
DISP_RSI_HOT    = 110.0   # reward shaping 발동 기준 (disparity_buy_penalty)
LOW_CONFIDENCE_THRESHOLD = 0.10   # 1위·2위 확률 차이 10%p 이하


# ---------------------------------------------------------------------------
# 반환 타입
# ---------------------------------------------------------------------------

@dataclass
class ExplainResult:
    # ── RSI 상태 ─────────────────────────────────────────────────
    rsi14            : float
    rsi_zone         : str     # "OVERBOUGHT" | "OVERSOLD" | "NEUTRAL"
    rsi_overbought   : bool    # rsi14 >= 70
    rsi_oversold     : bool    # rsi14 <= 30

    # ── 이격도(Disparity20) 상태 ──────────────────────────────────
    disparity20      : float
    disparity_zone   : str     # "OVERHEATED" | "DEPRESSED" | "NEUTRAL"
    disparity_overheated: bool # disparity20 >= 105
    disparity_depressed : bool # disparity20 <= 95

    # ── 크로스 플래그 ─────────────────────────────────────────────
    golden_flag      : bool    # 최근 window 내 골든크로스 발생
    dead_flag        : bool    # 최근 window 내 데드크로스 발생

    # ── Reward shaping 발동 조건 매칭 ────────────────────────────
    shaping_rsi_buy_penalty    : bool  # 신규 매수 AND rsi14 > 70
    shaping_golden_hold_bonus  : bool  # golden_flag AND 보유(position=1 가정)
    shaping_disparity_buy_penalty: bool  # 신규 매수 AND disparity20 > 110

    # ── 모델 확신도 ───────────────────────────────────────────────
    action_probs     : dict[str, float]
    top1_action      : str
    top1_prob        : float
    top2_action      : str
    top2_prob        : float
    low_confidence   : bool    # top1-top2 차이 < 10%p

    # ── 메타 ──────────────────────────────────────────────────────
    ticker           : str
    name             : str
    date             : str

    def summary(self) -> str:
        """사람이 읽기 편한 요약 문자열 반환."""
        lines = [
            f"[{self.ticker} | {self.name} | {self.date}]",
            f"  모델 결정  : {self.top1_action} ({self.top1_prob:.1%})"
            + (" ⚠️ 확신도 낮음" if self.low_confidence else ""),
            f"  확률 분포  : BUY={self.action_probs['BUY']:.1%}  "
            f"HOLD={self.action_probs['HOLD']:.1%}  "
            f"SELL={self.action_probs['SELL']:.1%}",
            "",
            f"  RSI14      : {self.rsi14:.1f}  [{self.rsi_zone}]",
            f"  Disparity20: {self.disparity20:.1f}  [{self.disparity_zone}]",
            f"  골든크로스  : {'발생' if self.golden_flag else '없음'}",
            f"  데드크로스  : {'발생' if self.dead_flag   else '없음'}",
            "",
            "  Reward 셰이핑 조건:",
            f"    RSI>70 매수 패널티   : {'발동' if self.shaping_rsi_buy_penalty     else '미발동'}",
            f"    골든크로스 보유 보너스: {'발동' if self.shaping_golden_hold_bonus   else '미발동'}",
            f"    이격도>110 매수 패널티: {'발동' if self.shaping_disparity_buy_penalty else '미발동'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def explain(
    result  : "PredictResult",
    position: int = 0,
) -> ExplainResult:
    """
    PredictResult를 받아 규칙 기반 ExplainResult를 반환.

    Parameters
    ----------
    result : PredictResult
        inference.predict.predict() 의 반환값.
    position : int
        현재 포지션 (0=미보유, 1=보유).
        reward shaping golden_hold_bonus 조건 판단에 사용.
        기본값 0 (predict() 기본값과 동일).
    """
    ind = result.indicators

    rsi14       = float(ind.get("rsi14",       50.0))
    disparity20 = float(ind.get("disparity20", 100.0))
    golden_flag = bool(ind.get("golden_flag",  0))
    dead_flag   = bool(ind.get("dead_flag",    0))

    # ── RSI 구간 판정 ──────────────────────────────────────────────
    if rsi14 >= RSI_OVERBOUGHT:
        rsi_zone = "OVERBOUGHT"
    elif rsi14 <= RSI_OVERSOLD:
        rsi_zone = "OVERSOLD"
    else:
        rsi_zone = "NEUTRAL"

    # ── 이격도 구간 판정 ───────────────────────────────────────────
    if disparity20 >= DISP_OVERHEATED:
        disparity_zone = "OVERHEATED"
    elif disparity20 <= DISP_DEPRESSED:
        disparity_zone = "DEPRESSED"
    else:
        disparity_zone = "NEUTRAL"

    # ── Reward shaping 발동 조건 ───────────────────────────────────
    # "신규 매수" = action이 BUY이고 position이 0인 경우
    is_new_buy = (result.action == 2) and (position == 0)

    shaping_rsi      = is_new_buy and (rsi14 > RSI_OVERBOUGHT)
    shaping_golden   = golden_flag and (position == 1)
    shaping_disp     = is_new_buy and (disparity20 > DISP_RSI_HOT)

    # ── 확신도 ─────────────────────────────────────────────────────
    probs = result.action_probs   # {"BUY": p, "HOLD": p, "SELL": p}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top1_action, top1_prob = sorted_probs[0]
    top2_action, top2_prob = sorted_probs[1]
    low_confidence = (top1_prob - top2_prob) < LOW_CONFIDENCE_THRESHOLD

    return ExplainResult(
        rsi14              = rsi14,
        rsi_zone           = rsi_zone,
        rsi_overbought     = rsi14 >= RSI_OVERBOUGHT,
        rsi_oversold       = rsi14 <= RSI_OVERSOLD,

        disparity20        = disparity20,
        disparity_zone     = disparity_zone,
        disparity_overheated = disparity20 >= DISP_OVERHEATED,
        disparity_depressed  = disparity20 <= DISP_DEPRESSED,

        golden_flag        = golden_flag,
        dead_flag          = dead_flag,

        shaping_rsi_buy_penalty      = shaping_rsi,
        shaping_golden_hold_bonus    = shaping_golden,
        shaping_disparity_buy_penalty= shaping_disp,

        action_probs   = probs,
        top1_action    = top1_action,
        top1_prob      = top1_prob,
        top2_action    = top2_action,
        top2_prob      = top2_prob,
        low_confidence = low_confidence,

        ticker = result.ticker,
        name   = result.name,
        date   = result.date,
    )


# ---------------------------------------------------------------------------
# 편의 함수: predict + explain 한 번에
# ---------------------------------------------------------------------------

def explain_ticker(
    query     : str,
    model_path: str | None = None,
    data_path : str | None = None,
    position  : int = 0,
) -> tuple["PredictResult", ExplainResult]:
    """predict() + explain() 를 한 번에 실행."""
    from inference.predict import predict  # noqa: PLC0415
    result = predict(query, model_path=model_path,
                     data_path=data_path, position=position)
    return result, explain(result, position=position)
