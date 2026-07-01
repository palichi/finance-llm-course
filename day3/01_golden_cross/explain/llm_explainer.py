"""
LLM 기반 설명 모듈 — Claude API로 rule_based + RAG 출력을 자연어 한국어로 변환.

흐름:
  1. ExplainResult + RAGResult 목록 → 구조화 프롬프트 조립
  2. Claude API 호출 (claude-haiku-4-5)
  3. 응답에 포함된 숫자 및 사례 인용이 실제 데이터와 일치하는지 검증
  4. API 오류 또는 검증 실패 → 폴백 템플릿으로 자동 전환

중요 — PPO(주) / RAG(부) 순서:
  PPO가 먼저 결정을 내리고 수치 근거가 확정된 뒤에만 RAG를 조회한다.
  RAG 검색 결과는 설명만 보강하며, PPO의 결정 자체를 바꾸지 않는다.

사용법:
    from explain.rule_based  import explain
    from explain.rag_retriever import retrieve
    from explain.llm_explainer import generate_explanation
    from inference.predict   import predict

    result       = predict("005930")
    explain_r    = explain(result)
    rag_results  = retrieve(explain_r, ticker="005930")
    explanation  = generate_explanation(explain_r, rag_results)
    print(explanation.text)
    print("LLM 사용:", explanation.used_llm)
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from explain.rule_based import ExplainResult
    from explain.rag_retriever import RAGResult

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "../../.env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL     = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "너는 주어진 수치 데이터와 검색된 과거 사례만 보고 설명을 작성한다. "
    "데이터에 없는 값을 추측하거나 새로운 판단을 만들어내지 않는다. "
    "검색된 과거 사례를 인용할 때는 반드시 그 사례의 종목코드와 날짜를 함께 언급해서 출처를 명확히 한다. "
    "검색된 사례가 없으면 '참고할 과거 사례를 찾지 못했습니다'라고 명시하고 지어내지 않는다. "
    "확신도가 낮게 표시된 경우 그 사실을 분명히 언급한다. "
    "출력은 4~6문장 한국어로 작성하며, 마크다운·불릿 없이 평문으로만 작성한다."
)


# ---------------------------------------------------------------------------
# 반환 타입
# ---------------------------------------------------------------------------

@dataclass
class ExplainOutput:
    text    : str    # 최종 설명 (LLM 또는 템플릿)
    used_llm: bool   # True = Claude 응답 사용, False = 폴백 템플릿
    reason  : str    # "llm_ok" | "api_error" | "number_mismatch"


# ---------------------------------------------------------------------------
# 폴백 템플릿 (LLM 없이도 수치 기반 완전한 설명 생성)
# ---------------------------------------------------------------------------

def _fallback_template(r: "ExplainResult") -> str:
    """ExplainResult 수치만으로 3~4문장 한국어 설명 구성."""
    action_ko = {"BUY": "매수", "HOLD": "유보", "SELL": "매도"}
    top_ko    = action_ko.get(r.top1_action, r.top1_action)

    # 문장 1: 모델 결정 + 확률
    s1 = (
        f"PPO 모델은 {r.ticker}({r.name})에 대해 {r.date} 기준 "
        f"{top_ko}({r.top1_prob:.1%})를 선택했습니다."
    )

    # 문장 2: RSI 상태
    rsi_desc = {
        "OVERBOUGHT": f"RSI14는 {r.rsi14:.1f}로 과매수 구간(≥70)에 진입해 있습니다.",
        "OVERSOLD"  : f"RSI14는 {r.rsi14:.1f}로 과매도 구간(≤30)에 위치합니다.",
        "NEUTRAL"   : f"RSI14는 {r.rsi14:.1f}로 중립 구간(30~70)에 있습니다.",
    }
    s2 = rsi_desc[r.rsi_zone]

    # 문장 3: 이격도 + 크로스
    disp_desc = {
        "OVERHEATED": f"이격도(disparity20)는 {r.disparity20:.1f}로 과열 구간(≥105)입니다",
        "DEPRESSED" : f"이격도(disparity20)는 {r.disparity20:.1f}로 침체 구간(≤95)입니다",
        "NEUTRAL"   : f"이격도(disparity20)는 {r.disparity20:.1f}로 중립 수준입니다",
    }
    cross = ""
    if r.golden_flag:
        cross = " 최근 골든크로스(단기 이평 상향 돌파)가 발생한 상태입니다."
    elif r.dead_flag:
        cross = " 최근 데드크로스(단기 이평 하향 이탈)가 발생한 상태입니다."
    s3 = disp_desc[r.disparity_zone] + ("." if not cross else ",") + cross

    # 문장 4: 확신도 또는 shaping 발동 정보
    extra_parts = []
    if r.low_confidence:
        extra_parts.append(
            f"모델의 1·2위 행동 확률 차이가 "
            f"{abs(r.top1_prob - r.top2_prob):.1%}에 불과해 확신도가 낮습니다"
        )
    active_shaping = []
    if r.shaping_rsi_buy_penalty:
        active_shaping.append("RSI 과매수 매수 패널티")
    if r.shaping_golden_hold_bonus:
        active_shaping.append("골든크로스 보유 보너스")
    if r.shaping_disparity_buy_penalty:
        active_shaping.append("이격도 과열 매수 패널티")
    if active_shaping:
        extra_parts.append(f"학습 시 적용된 보상 셰이핑({', '.join(active_shaping)})이 이 상황과 일치합니다")

    if extra_parts:
        s4 = "; ".join(extra_parts) + "."
    else:
        s4 = (
            f"BUY {r.action_probs['BUY']:.1%} / "
            f"HOLD {r.action_probs['HOLD']:.1%} / "
            f"SELL {r.action_probs['SELL']:.1%} 분포를 기반으로 위 결정이 내려졌습니다."
        )

    return " ".join([s1, s2, s3, s4])


# ---------------------------------------------------------------------------
# 숫자 검증
# ---------------------------------------------------------------------------

def _validate_numbers(
    response: str,
    rsi14   : float,
    disparity20: float,
    action_probs: dict[str, float],
) -> bool:
    """
    LLM 응답에 포함된 숫자가 실제 RSI14·disparity20 값과 일치하는지 검증.

    - 응답에서 소수·정수 추출
    - RSI 범위(5~100) 내 숫자가 있으면 rsi14와 ±1.0 이내여야 함
    - disparity 범위(80~200) 내 숫자가 있으면 disparity20과 ±1.0 이내여야 함
    - action_probs 백분율(×100)은 허용

    Returns True if valid, False if mismatch detected.
    """
    numbers = [float(m) for m in re.findall(r'\d+(?:\.\d+)?', response)]

    # 허용 숫자 집합 구성
    allowed: set[float] = {
        rsi14, round(rsi14, 1), round(rsi14),
        disparity20, round(disparity20, 1), round(disparity20),
    }
    for p in action_probs.values():
        pct = round(p * 100, 1)
        allowed.update({pct, round(pct), pct - 0.1, pct + 0.1})

    TOLERANCE = 1.0

    for num in numbers:
        if num < 5:   # 단순 수사(1, 2, 3, 4) 제외
            continue

        # RSI 범위 내 숫자 검증
        if 5 <= num <= 100:
            if not any(abs(num - a) <= TOLERANCE for a in allowed):
                return False

        # disparity 범위 내 숫자 검증 (RSI 범위와 겹치지 않는 구간)
        if 100 < num <= 200:
            if not any(abs(num - a) <= TOLERANCE for a in allowed):
                return False

    return True


# ---------------------------------------------------------------------------
# 프롬프트 조립
# ---------------------------------------------------------------------------

def _format_rag_section(rag_results: "list[RAGResult] | None") -> str:
    """RAG 검색 결과를 프롬프트 섹션 문자열로 변환."""
    if not rag_results:
        return "[참고 과거 사례]\n  - 검색된 사례 없음"
    lines = ["[참고 과거 사례]"]
    for i, r in enumerate(rag_results, 1):
        meta = r.metadata
        ticker  = meta.get("ticker",  "?")
        date    = meta.get("date",    "?")
        ret5    = meta.get("ret5",    -999)
        ret10   = meta.get("ret10",   -999)
        act_num = meta.get("ppo_action", -1)
        act_ko  = {0: "매도", 1: "유보", 2: "매수"}.get(act_num, "?")
        r5_str  = f"{ret5:.1f}%" if ret5 != -999 else "N/A"
        r10_str = f"{ret10:.1f}%" if ret10 != -999 else "N/A"
        lines.append(
            f"  {i}. ({ticker}, {date}) — 당시 PPO: {act_ko}, "
            f"5일후 {r5_str}, 10일후 {r10_str}"
        )
        # 카드 텍스트 앞 100자 (뉴스/공시 포함)
        excerpt = r.card_text[:120].replace("\n", " ")
        lines.append(f"     요약: {excerpt}…")
    return "\n".join(lines)


def _build_user_prompt(
    r: "ExplainResult",
    rag_results: "list[RAGResult] | None" = None,
) -> str:
    action_ko = {"BUY": "매수", "HOLD": "유보", "SELL": "매도"}

    shaping_lines = []
    if r.shaping_rsi_buy_penalty:
        shaping_lines.append("  - RSI>70 신규 매수 패널티 발동")
    if r.shaping_golden_hold_bonus:
        shaping_lines.append("  - 골든크로스 보유 보너스 발동")
    if r.shaping_disparity_buy_penalty:
        shaping_lines.append("  - 이격도>110 신규 매수 패널티 발동")
    shaping_str = "\n".join(shaping_lines) if shaping_lines else "  - 없음"
    rag_str = _format_rag_section(rag_results)

    return f"""다음 수치 데이터와 과거 사례를 바탕으로 4~6문장 한국어 설명을 작성하세요.
현재 판단 근거를 먼저 설명하고, "과거 비슷한 사례에서는 ~" 형식으로 RAG 근거를 붙이세요.
과거 사례를 인용할 때는 반드시 (종목코드, 날짜) 형태로 출처를 표기하세요.

[종목 정보]
- 종목: {r.ticker} ({r.name})
- 기준일: {r.date}

[모델 결정]
- 행동: {r.top1_action} ({action_ko.get(r.top1_action, r.top1_action)}) — {r.top1_prob:.1%}
- 확률 분포: BUY={r.action_probs['BUY']:.1%} / HOLD={r.action_probs['HOLD']:.1%} / SELL={r.action_probs['SELL']:.1%}
- 확신도 낮음: {"예 (1·2위 차이 " + f"{abs(r.top1_prob - r.top2_prob):.1%})" if r.low_confidence else "아니오"}

[기술 지표]
- RSI14: {r.rsi14:.1f} → {r.rsi_zone}
- disparity20: {r.disparity20:.1f} → {r.disparity_zone}
- 골든크로스: {"발생" if r.golden_flag else "없음"}
- 데드크로스: {"발생" if r.dead_flag else "없음"}

[Reward 셰이핑 발동 조건]
{shaping_str}

{rag_str}"""


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def _validate_rag_citations(
    response    : str,
    rag_results : "list[RAGResult] | None",
) -> bool:
    """
    LLM 응답에 등장하는 (종목코드, 날짜) 인용이 실제 RAG 검색결과에 존재하는지 검증.
    존재하지 않는 사례를 지어내 인용하는 경우 False 반환.
    RAG 결과가 없으면 항상 True (인용 자체가 없어야 함).
    """
    if not rag_results:
        return True

    valid_pairs: set[str] = set()
    for r in rag_results:
        ticker = r.metadata.get("ticker", "")
        date   = r.metadata.get("date",   "")
        if ticker and date:
            # "(005930, 2025-06-05)" 형태 또는 "005930, 2025-06-05" 형태 허용
            valid_pairs.add(f"{ticker},{date}")
            valid_pairs.add(f"{ticker}, {date}")

    # 응답에서 (종목코드, 날짜) 패턴 추출
    patterns = re.findall(r'\(([0-9A-Za-z]+),\s*(\d{4}-\d{2}-\d{2})\)', response)
    for ticker, date in patterns:
        key1 = f"{ticker},{date}"
        key2 = f"{ticker}, {date}"
        if key1 not in valid_pairs and key2 not in valid_pairs:
            return False
    return True


def generate_explanation(
    explain_result: "ExplainResult",
    rag_results   : "list[RAGResult] | None" = None,
    model         : str = DEFAULT_MODEL,
    api_key       : str | None = None,
) -> ExplainOutput:
    """
    ExplainResult + RAG 검색결과를 Claude API로 자연어 한국어 설명으로 변환.

    Parameters
    ----------
    explain_result : ExplainResult
        explain.rule_based.explain() 반환값.
    rag_results : list[RAGResult] | None
        explain.rag_retriever.retrieve() 반환값. None이면 RAG 없이 동작.
    model : str
        사용할 Claude 모델 ID.
    api_key : str | None
        ANTHROPIC_API_KEY. None이면 환경변수/env 파일에서 자동 로드.

    Returns
    -------
    ExplainOutput
        .text     : 최종 설명 문자열
        .used_llm : True=LLM 사용, False=폴백 템플릿
        .reason   : "llm_ok" | "api_error" | "number_mismatch" | "citation_error"
    """
    key = api_key or ANTHROPIC_API_KEY
    if not key:
        return ExplainOutput(
            text     = _fallback_template(explain_result),
            used_llm = False,
            reason   = "api_error",
        )

    try:
        import anthropic  # noqa: PLC0415
        client = anthropic.Anthropic(api_key=key)

        message = client.messages.create(
            model      = model,
            max_tokens = 768,
            system     = SYSTEM_PROMPT,
            messages   = [{
                "role"   : "user",
                "content": _build_user_prompt(explain_result, rag_results),
            }],
        )
        llm_text = message.content[0].text.strip()

        # 숫자 검증
        if not _validate_numbers(
            response     = llm_text,
            rsi14        = explain_result.rsi14,
            disparity20  = explain_result.disparity20,
            action_probs = explain_result.action_probs,
        ):
            return ExplainOutput(
                text     = _fallback_template(explain_result),
                used_llm = False,
                reason   = "number_mismatch",
            )

        # 사례 인용 검증
        if not _validate_rag_citations(llm_text, rag_results):
            return ExplainOutput(
                text     = _fallback_template(explain_result),
                used_llm = False,
                reason   = "citation_error",
            )

        return ExplainOutput(
            text     = llm_text,
            used_llm = True,
            reason   = "llm_ok",
        )

    except Exception:
        return ExplainOutput(
            text     = _fallback_template(explain_result),
            used_llm = False,
            reason   = "api_error",
        )
