"""
PPO 모델 추론 모듈.

사용법:
    from inference.predict import predict

    result = predict("005930")          # 종목코드
    result = predict("삼성전자")         # 종목명도 허용
    result = predict("005930", model_path="models/run/best_model.zip")

반환 PredictResult:
    action          : int   — 0=매도, 1=유보, 2=매수
    action_name     : str   — "SELL" | "HOLD" | "BUY"
    action_probs    : dict  — {"BUY": float, "HOLD": float, "SELL": float}
    date            : str   — 기준일자 (YYYY-MM-DD)
    ticker          : str   — 종목코드
    name            : str   — 종목명
    indicators      : dict  — 최신 raw 지표값 전부
    obs             : ndarray — 모델에 입력된 (20, 9) 상태 행렬
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators.technical import compute_indicators

LOOKBACK   = 20
N_FEATURES = 9

ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

DEFAULT_DATA_PATH  = ROOT / "../../day2/03_fss_api/data/stock_prices.csv"
DEFAULT_MODEL_PATH = ROOT / "models/20260628_231252/best_model.zip"


# ---------------------------------------------------------------------------
# 반환 타입
# ---------------------------------------------------------------------------

@dataclass
class PredictResult:
    action      : int
    action_name : str
    action_probs: dict[str, float]   # {"BUY": p, "HOLD": p, "SELL": p}
    date        : str
    ticker      : str
    name        : str
    indicators  : dict[str, float]   # 최신 행 raw 지표 전부
    obs         : np.ndarray = field(repr=False)  # (20, 9) 모델 입력


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _load_stock_df(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="utf-8-sig", dtype={"srtnCd": str})
    df["srtnCd"] = df["srtnCd"].astype(str).str.zfill(6)
    return df


def _resolve_ticker(query: str, raw_df: pd.DataFrame) -> tuple[str, str]:
    """
    종목코드 또는 종목명으로 ticker, name 해석.
    ambiguous하면 첫 번째 매칭 반환.
    """
    # 6자리 코드 직접 매칭
    if query.isdigit():
        query = query.zfill(6)

    # srtnCd 완전 일치
    if query in raw_df["srtnCd"].values:
        name = raw_df[raw_df["srtnCd"] == query]["itmsNm"].iloc[0]
        return query, str(name)

    # itmsNm 부분 일치
    mask = raw_df["itmsNm"].str.contains(query, na=False)
    if mask.any():
        row = raw_df[mask].iloc[0]
        return str(row["srtnCd"]), str(row["itmsNm"])

    raise ValueError(
        f"종목 '{query}'를 찾을 수 없습니다. "
        f"종목코드(6자리) 또는 종목명을 확인하세요."
    )


def _build_obs(df_ind: pd.DataFrame, position: int = 0,
               unrealized_pnl: float = 0.0) -> np.ndarray:
    """
    compute_indicators() 결과 중 마지막 LOOKBACK 행으로 (LOOKBACK, 9) obs 생성.
    TradingEnv._get_obs() 와 동일한 feature 순서.
    """
    if len(df_ind) < LOOKBACK:
        raise ValueError(
            f"지표 계산 후 행이 {len(df_ind)}개 — lookback({LOOKBACK}) 이상 필요."
        )

    w = df_ind.iloc[-LOOKBACK:].copy()

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
        np.full(LOOKBACK, float(position)),
        np.full(LOOKBACK, unrealized_pnl),
    ]).astype(np.float32)

    return obs


def _extract_indicators(row: pd.Series) -> dict[str, float]:
    """최신 행에서 raw 지표값 전부 추출."""
    cols = [
        "close", "open", "high", "low", "volume",
        "sma5", "sma20", "sma60", "ema20",
        "golden_flag", "dead_flag",
        "disparity20", "disparity20_centered",
        "rsi14", "rsi14_norm",
    ]
    return {c: float(row[c]) for c in cols if c in row.index}


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def predict(
    query     : str,
    model_path: str | Path | None = None,
    data_path : str | Path | None = None,
    position  : int   = 0,
    unrealized_pnl: float = 0.0,
) -> PredictResult:
    """
    단일 종목에 대한 PPO 모델 추론.

    Parameters
    ----------
    query : str
        종목코드("005930") 또는 종목명("삼성전자").
    model_path : str | Path | None
        .zip 모델 경로. None이면 DEFAULT_MODEL_PATH 사용.
    data_path : str | Path | None
        stock_prices.csv 경로. None이면 DEFAULT_DATA_PATH 사용.
    position : int
        현재 포지션 가정 (0=미보유, 1=보유). 기본 0.
    unrealized_pnl : float
        현재 미실현 손익 가정. 기본 0.0.

    Returns
    -------
    PredictResult
    """
    # stable_baselines3 는 무거운 import → 함수 내에서 지연 로드
    from stable_baselines3 import PPO  # noqa: PLC0415

    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    data_path  = Path(data_path)  if data_path  else DEFAULT_DATA_PATH

    # ── 데이터 로드 ────────────────────────────────────────────────────
    raw_df = _load_stock_df(data_path)
    ticker, name = _resolve_ticker(query, raw_df)

    stock_rows = raw_df[raw_df["srtnCd"] == ticker].copy()
    if stock_rows.empty:
        raise ValueError(f"'{ticker}' 데이터가 CSV에 없습니다.")

    # ── 지표 계산 ──────────────────────────────────────────────────────
    # nan_policy="drop": sma60 warm-up(~59행) NaN 제거
    df_ind = compute_indicators(stock_rows, nan_policy="drop")
    if df_ind.empty:
        raise ValueError(f"'{ticker}' 지표 계산 후 유효 데이터 없음 (데이터 부족).")

    # ── obs 구성 ───────────────────────────────────────────────────────
    obs = _build_obs(df_ind, position=position, unrealized_pnl=unrealized_pnl)

    # ── 모델 로드 & 추론 ───────────────────────────────────────────────
    model = PPO.load(str(model_path))

    # action_probs: policy 분포에서 직접 추출
    import torch  # noqa: PLC0415
    device = model.policy.device
    obs_tensor = torch.tensor(obs[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()  # (3,)

    action = int(probs.argmax())
    action_probs = {
        "SELL": float(probs[0]),
        "HOLD": float(probs[1]),
        "BUY" : float(probs[2]),
    }

    # ── 기준일자 & 지표 ────────────────────────────────────────────────
    latest_row = df_ind.iloc[-1]
    date = latest_row["date"]
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

    indicators = _extract_indicators(latest_row)

    return PredictResult(
        action       = action,
        action_name  = ACTION_NAMES[action],
        action_probs = action_probs,
        date         = date_str,
        ticker       = ticker,
        name         = name,
        indicators   = indicators,
        obs          = obs,
    )
