"""
Reward 셰이핑 모듈.

기본 보상:
    R_base = (V_t - V_{t-1}) / V_{t-1}
    거래비용 효과는 V_t 계산(env)에 이미 반영되어 있음.

셰이핑 항 (각각 config로 가중치 조정, on/off 가능):
    penalty_rsi      = -w_rsi   if (신규 매수 AND rsi14 > rsi_thresh)
    bonus_golden     = +w_gold  if (golden_flag==1 AND 보유 중)
    penalty_disp     = -w_disp  if (신규 매수 AND disparity20 > disp_thresh)

최종 보상:
    R_final = R_base + penalty_rsi + bonus_golden + penalty_disp

무효 행동 페널티(invalid_penalty)는 env가 별도로 처리.

카운트 로깅:
    각 셰이핑 항의 episode 내 발동 횟수를 get_episode_summary()로 조회.
    에이전트가 지표 규칙만 흉내내고 있는지 점검하는 데 사용.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY  = 2


@dataclass
class RewardConfig:
    """reward 셰이핑 전체 설정. 모든 가중치·임계값·on/off 플래그."""

    # ── on/off 플래그 (False = 해당 셰이핑 비활성, 순수 수익 기반 비교용) ──
    enable_rsi_penalty      : bool  = True
    enable_golden_bonus     : bool  = True
    enable_disparity_penalty: bool  = True

    # ── 조건 임계값 ───────────────────────────────────────────────────
    rsi_overbought_threshold: float = 70.0    # RSI 과매수 판단 기준
    disparity_hot_threshold : float = 110.0   # 이격도 과열 판단 기준

    # ── 가중치 ────────────────────────────────────────────────────────
    rsi_buy_penalty     : float = 0.1
    golden_hold_bonus   : float = 0.05
    disparity_buy_penalty: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        """
        cfg 딕셔너리로부터 RewardConfig 생성.
        최상위 key 또는 "reward" 서브섹션을 모두 지원.
        없는 key는 기본값 유지.
        """
        src    = d.get("reward", d)          # "reward" 서브키 있으면 우선
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: src[k] for k in fields if k in src})


class RewardShaper:
    """
    episode 단위로 셰이핑 발동 횟수를 추적하는 reward 계산기.

    Parameters
    ----------
    cfg : dict | RewardConfig | None
        설정 딕셔너리, RewardConfig, 또는 None(전부 기본값).
    """

    def __init__(self, cfg: dict | RewardConfig | None = None) -> None:
        if isinstance(cfg, RewardConfig):
            self.config = cfg
        elif isinstance(cfg, dict):
            self.config = RewardConfig.from_dict(cfg)
        else:
            self.config = RewardConfig()

        self._counts: dict[str, int] = {}
        self.reset_episode()

    # ------------------------------------------------------------------
    # episode 관리
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """에피소드 시작 시 발동 카운터 초기화. env.reset() 직후 호출."""
        self._counts = {
            "rsi_penalty"      : 0,
            "golden_bonus"     : 0,
            "disparity_penalty": 0,
        }

    def get_episode_summary(self) -> dict[str, int]:
        """
        에피소드 동안 각 셰이핑 항이 몇 번 발동했는지 반환.

        활용 예:
          - 학습 중 tensorboard/wandb 로깅
          - 에이전트가 지표 규칙만 흉내내는지 점검
          - RAG corpus 카드에 메타데이터로 첨부
        """
        return dict(self._counts)

    # ------------------------------------------------------------------
    # 보상 계산
    # ------------------------------------------------------------------

    def compute(
        self,
        base_reward    : float,
        action         : int,
        position_before: int,
        position_after : int,
        rsi14          : float,
        golden_flag    : int,
        disparity20    : float,
    ) -> tuple[float, dict]:
        """
        기본 보상에 셰이핑을 더해 최종 보상과 상세 내역을 반환한다.

        Parameters
        ----------
        base_reward     : (V_t - V_{t-1}) / V_{t-1}  (거래비용 포함)
        action          : 0=매도, 1=유보, 2=매수 (실제 실행된 action)
        position_before : action 실행 전 포지션 (0 or 1)
        position_after  : action 실행 후 포지션 (0 or 1)
        rsi14           : 현재 RSI 값 (0 ~ 100 스케일, not normalized)
        golden_flag     : 골든크로스 플래그 (0 or 1)
        disparity20     : (close / SMA20) * 100  (100 기준, 110 초과 시 과열)

        Returns
        -------
        (final_reward, shaping_detail)
            shaping_detail : {
                "rsi_penalty", "golden_bonus", "disparity_penalty",
                "total_shaping", "final_reward"
            }
        """
        cfg        = self.config
        rsi_pen    = 0.0
        gold_bon   = 0.0
        disp_pen   = 0.0
        is_new_buy = (action == ACTION_BUY) and (position_before == 0)

        # ── RSI 과매수 신규 매수 페널티 ──────────────────────────────
        if (cfg.enable_rsi_penalty
                and is_new_buy
                and rsi14 > cfg.rsi_overbought_threshold):
            rsi_pen = -cfg.rsi_buy_penalty
            self._counts["rsi_penalty"] += 1

        # ── 골든크로스 직후 보유 포지션 유지 보너스 ──────────────────
        if (cfg.enable_golden_bonus
                and golden_flag == 1
                and position_after == 1):
            gold_bon = cfg.golden_hold_bonus
            self._counts["golden_bonus"] += 1

        # ── 이격도 과열 신규 매수 페널티 ─────────────────────────────
        if (cfg.enable_disparity_penalty
                and is_new_buy
                and disparity20 > cfg.disparity_hot_threshold):
            disp_pen = -cfg.disparity_buy_penalty
            self._counts["disparity_penalty"] += 1

        total_shaping = rsi_pen + gold_bon + disp_pen
        final_reward  = base_reward + total_shaping

        detail = {
            "rsi_penalty"      : rsi_pen,
            "golden_bonus"     : gold_bon,
            "disparity_penalty": disp_pen,
            "total_shaping"    : total_shaping,
            "final_reward"     : final_reward,
        }
        return final_reward, detail
