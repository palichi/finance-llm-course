#!/usr/bin/env python
"""
PPO 학습 스크립트 — 200종목 에피소드를 섞어 단일 모델 학습.

══ 에피소드 혼합 전략 (선택 근거) ════════════════════════════════════════
  채택: DummyVecEnv(n_envs) + 각 env에 전체 train 종목 pool 탑재
  ──────────────────────────────────────────────────────────────────────
  ① SubprocVecEnv 미채택 이유
     - pandas DataFrame 직렬화(pickle) 오버헤드: 200종목 × ~500행 DF를
       subprocess 경계마다 직렬화하면 속도 이득보다 비용이 큼.
     - 디버깅 난이도 증가.

  ② "종목별 별도 env" 미채택 이유
     - n_envs(4~8) << 종목수(200)이므로 대부분의 종목이 학습에서 제외됨.
     - 특정 종목 고착 문제(curriculum 없이).

  ③ DummyVecEnv + 전체 pool 방식의 장점
     - reset() 마다 200종목 중 무작위 1종목 선택 → 하나의 PPO 롤아웃에
       여러 종목 에피소드가 자연스럽게 섞임.
     - 구현 단순, pickle 문제 없음, 디버깅 용이.
════════════════════════════════════════════════════════════════════════

실행:
  # 스모크 테스트 (3종목, 50,000 스텝)
  python -m train.run_training --smoke

  # 본 학습
  python -m train.run_training --total_steps 2000000 --n_envs 8

TensorBoard:
  tensorboard --logdir logs/
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators.technical import compute_indicators
from env.trading_env import TradingEnv
from train.reward import RewardShaper, RewardConfig


# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    # ── 데이터 ────────────────────────────────────────────────────────
    "data_path"           : "../../day2/03_fss_api/data/stock_prices.csv",
    "val_date"            : "20260101",  # 미만=train / 이상=val
    # ── 환경 ──────────────────────────────────────────────────────────
    "tx_cost"             : 0.003,
    "ruin_ratio"          : 0.7,
    "invalid_penalty"     : -0.01,
    # ── Reward 셰이핑 ──────────────────────────────────────────────────
    "enable_shaping"      : True,
    # ── 정책망 ────────────────────────────────────────────────────────
    # LSTM 교체 시: policy="MlpLstmPolicy", net_arch 제거하고
    # policy_kwargs={"n_lstm_layers":1,"lstm_hidden_size":64} 사용.
    "policy"              : "MlpPolicy",
    "net_arch"            : [64, 64],
    # ── PPO 하이퍼파라미터 ─────────────────────────────────────────────
    "learning_rate"       : 3e-4,
    "n_steps"             : 2048,   # 롤아웃 길이(환경 당)
    "batch_size"          : 64,
    "gamma"               : 0.99,
    "gae_lambda"          : 0.95,
    "clip_range"          : 0.2,
    "ent_coef"            : 0.01,
    "n_epochs"            : 10,
    "n_envs"              : 4,
    "total_steps"         : 1_000_000,
    # ── 평가 ──────────────────────────────────────────────────────────
    "eval_freq"           : 20_000,  # 몇 global step마다 평가
    "n_eval_stocks"       : 20,      # val 평가 종목 수
    "n_train_eval_stocks" : 10,      # train 평가 종목 수 (곡선용)
    # ── 경로 ──────────────────────────────────────────────────────────
    "model_dir"           : "models",
    "log_dir"             : "logs",
    "run_name"            : None,    # None → 타임스탬프 자동 생성
    # ── 스모크 테스트 ──────────────────────────────────────────────────
    "smoke"               : False,
}


# ---------------------------------------------------------------------------
# 데이터 로드 & train/val 분리
# ---------------------------------------------------------------------------

def load_and_split(cfg: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    CSV 로드 → 종목별 지표 계산 → 날짜 기준 train/val 분리.

    val_date 미만 = train, 이상 = val (per-stock 분리).
    NaN 제거(nan_policy='drop') 후 lookback(20) 이상 행만 유효.
    """
    data_path = ROOT / cfg["data_path"]
    print(f"[data] 로딩: {data_path}")
    raw = pd.read_csv(data_path, dtype={"srtnCd": str})

    val_cutoff = int(cfg["val_date"])
    train_data: dict[str, pd.DataFrame] = {}
    val_data  : dict[str, pd.DataFrame] = {}

    codes = sorted(raw["srtnCd"].unique())
    for code in codes:
        group = raw[raw["srtnCd"] == code].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = compute_indicators(group, nan_policy="drop")
        if df.empty:
            continue

        date_int = df["date"].dt.strftime("%Y%m%d").astype(int)
        tr = df[date_int  < val_cutoff].reset_index(drop=True)
        va = df[date_int >= val_cutoff].reset_index(drop=True)

        if len(tr) >= 20:
            train_data[code] = tr
        if len(va) >= 20:
            val_data[code] = va

    print(
        f"[data] train {len(train_data)}종목 / val {len(val_data)}종목  "
        f"(split: {cfg['val_date']})"
    )
    return train_data, val_data


# ---------------------------------------------------------------------------
# 성과 지표 계산
# ---------------------------------------------------------------------------

def compute_episode_metrics(equity_curve: np.ndarray) -> dict[str, float]:
    """
    단일 에피소드 equity curve → 누적수익률, MDD, 연환산 Sharpe, 승률.
    """
    if len(equity_curve) < 2:
        return {"cum_return": 0.0, "mdd": 0.0, "sharpe": 0.0, "win_rate": 0.0}

    returns    = np.diff(equity_curve) / equity_curve[:-1]
    cum_return = float(equity_curve[-1] / equity_curve[0] - 1)

    peak     = np.maximum.accumulate(equity_curve)
    drawdown = np.where(peak > 0, (equity_curve - peak) / peak, 0.0)
    mdd      = float(drawdown.min())

    std = returns.std()
    sharpe   = float(returns.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
    win_rate = float((returns > 0).mean())

    return {"cum_return": cum_return, "mdd": mdd, "sharpe": sharpe, "win_rate": win_rate}


# ---------------------------------------------------------------------------
# 종목 집합에 대한 평가
# ---------------------------------------------------------------------------

def evaluate_portfolio(
    model    : PPO,
    data     : dict[str, pd.DataFrame],
    cfg      : dict,
    n_stocks : int = 20,
    seed     : int = 42,
) -> dict[str, float]:
    """
    n_stocks개 종목을 deterministic 정책으로 롤아웃 → 지표 집계.

    Returns
    -------
    {"mean_cum_return", "mean_mdd", "mean_sharpe", "mean_win_rate",
     "std_cum_return",  ...}
    """
    codes = sorted(data.keys())[:n_stocks]
    if not codes:
        return {}

    env_cfg = {k: cfg[k] for k in ("tx_cost", "ruin_ratio", "invalid_penalty")}
    all_metrics: list[dict] = []

    for code in codes:
        env = TradingEnv({code: data[code]}, cfg=env_cfg)
        obs, _ = env.reset(seed=seed, options={"code": code})
        equity_curve = [1.0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            equity_curve.append(info["equity"])
            done = terminated or truncated
        all_metrics.append(compute_episode_metrics(np.array(equity_curve)))

    result: dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"std_{key}"]  = float(np.std(vals))
    return result


# ---------------------------------------------------------------------------
# Validation Callback
# ---------------------------------------------------------------------------

class ValidationCallback(BaseCallback):
    """
    주기적으로 train/val 집합을 평가하고,
    best model(val Sharpe 기준)을 별도 저장하며 TensorBoard에 기록한다.
    """

    def __init__(
        self,
        train_data    : dict[str, pd.DataFrame],
        val_data      : dict[str, pd.DataFrame],
        cfg           : dict,
        best_model_dir: Path,
        verbose       : int = 1,
    ) -> None:
        super().__init__(verbose)
        self.train_data     = train_data
        self.val_data       = val_data
        self.cfg            = cfg
        self.best_model_dir = best_model_dir

        self.eval_freq      = cfg["eval_freq"]
        self.best_sharpe    = -np.inf
        self.history        : list[dict] = []   # {step, val/*, train/*}
        self._last_eval_step: int = -1

    def _on_step(self) -> bool:
        # eval_freq global step마다 평가
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        # ── Validation 평가 ────────────────────────────────────────
        val_m = evaluate_portfolio(
            self.model, self.val_data, self.cfg,
            n_stocks=self.cfg["n_eval_stocks"],
        )
        for k, v in val_m.items():
            self.logger.record(f"val/{k}", v)

        # ── Train 평가 (곡선 기록용 subset) ───────────────────────
        train_m = evaluate_portfolio(
            self.model, self.train_data, self.cfg,
            n_stocks=self.cfg["n_train_eval_stocks"],
        )
        for k, v in train_m.items():
            self.logger.record(f"train_eval/{k}", v)

        self.logger.dump(self.num_timesteps)

        rec = {"step": self.num_timesteps}
        rec.update({f"val/{k}": v   for k, v in val_m.items()})
        rec.update({f"train/{k}": v for k, v in train_m.items()})
        self.history.append(rec)

        # ── Best model 저장 (val Sharpe 최대) ─────────────────────
        sharpe = val_m.get("mean_sharpe", -np.inf)
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            path = self.best_model_dir / "best_model"
            self.model.save(str(path))
            if self.verbose:
                print(
                    f"\n[best] step={self.num_timesteps:,}  "
                    f"val_sharpe={sharpe:.4f}  → {path}.zip"
                )

        if self.verbose:
            cr  = val_m.get("mean_cum_return", 0)
            mdd = val_m.get("mean_mdd", 0)
            wr  = val_m.get("mean_win_rate", 0)
            print(
                f"  [eval] step={self.num_timesteps:,}  "
                f"val cum_ret={cr:+.2%}  MDD={mdd:.2%}  "
                f"sharpe={sharpe:.3f}  win_rate={wr:.1%}"
            )
        return True


# ---------------------------------------------------------------------------
# 환경 팩토리
# ---------------------------------------------------------------------------

def make_env_fn(stock_data: dict, cfg: dict):
    """
    DummyVecEnv용 env 생성 클로저.
    각 env는 동일한 stock_data pool을 보유 → reset()마다 무작위 종목 선택.
    """
    env_cfg = {k: cfg[k] for k in ("tx_cost", "ruin_ratio", "invalid_penalty")}

    def _init():
        shaper = None
        if cfg.get("enable_shaping", True):
            shaper = RewardShaper(RewardConfig.from_dict(cfg))
        env = TradingEnv(stock_data, cfg=env_cfg, reward_shaper=shaper)
        # Monitor: 에피소드 보상·길이를 SB3가 TensorBoard에 자동 기록
        return Monitor(env)

    return _init


# ---------------------------------------------------------------------------
# 결과 차트 저장
# ---------------------------------------------------------------------------

def save_curves(history: list[dict], run_dir: Path) -> None:
    """Train/Val 지표 이력을 2×2 subplot 차트로 저장."""
    if not history:
        return

    steps = [h["step"] for h in history]

    metrics = [
        ("cum_return", "Cumulative Return"),
        ("mdd",        "Max Drawdown"),
        ("sharpe",     "Sharpe Ratio"),
        ("win_rate",   "Win Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for (key, title), ax in zip(metrics, axes.flat):
        val_vals   = [h.get(f"val/mean_{key}",   np.nan) for h in history]
        train_vals = [h.get(f"train/mean_{key}", np.nan) for h in history]
        ax.plot(steps, val_vals,   label="val",   marker="o", markersize=3)
        ax.plot(steps, train_vals, label="train", marker="s", markersize=3,
                linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Train / Validation Performance", fontsize=14)
    fig.tight_layout()
    path = run_dir / "curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[chart] {path}")


# ---------------------------------------------------------------------------
# PPO 생성
# ---------------------------------------------------------------------------

def build_ppo(vec_env: DummyVecEnv, cfg: dict, log_dir: str) -> PPO:
    """
    PPO 생성.

    LSTM 교체 방법:
      cfg["policy"] = "MlpLstmPolicy"
      cfg["policy_kwargs"] = {"n_lstm_layers": 1, "lstm_hidden_size": 64}
    """
    policy_kwargs = cfg.get("policy_kwargs", {"net_arch": cfg["net_arch"]})
    return PPO(
        policy         = cfg["policy"],
        env            = vec_env,
        learning_rate  = cfg["learning_rate"],
        n_steps        = cfg["n_steps"],
        batch_size     = cfg["batch_size"],
        gamma          = cfg["gamma"],
        gae_lambda     = cfg["gae_lambda"],
        clip_range     = cfg["clip_range"],
        ent_coef       = cfg["ent_coef"],
        n_epochs       = cfg["n_epochs"],
        policy_kwargs  = policy_kwargs,
        tensorboard_log= log_dir,
        verbose        = 1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    run_name = cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = ROOT / cfg["log_dir"]   / run_name
    model_dir= ROOT / cfg["model_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Run : {run_name}")
    print(f"  Strategy : DummyVecEnv({cfg['n_envs']}) + 전체 train pool")
    print(f"  Policy   : {cfg['policy']}  net_arch={cfg['net_arch']}")
    print(f"  Steps    : {cfg['total_steps']:,}  (eval every {cfg['eval_freq']:,})")
    print("=" * 65)

    # ── 데이터 ──────────────────────────────────────────────────────────
    train_data, val_data = load_and_split(cfg)

    if cfg.get("smoke"):
        train_codes = sorted(train_data.keys())[:3]
        val_codes   = sorted(val_data.keys())[:3]
        train_data  = {c: train_data[c] for c in train_codes}
        val_data    = {c: val_data[c]   for c in val_codes}
        cfg.update({
            "total_steps"         : 50_000,
            "eval_freq"           : 10_000,
            "n_eval_stocks"       : 3,
            "n_train_eval_stocks" : 3,
            "n_steps"             : 512,
            "n_envs"              : 2,
        })
        print(f"[smoke] 종목 train={train_codes} / val={val_codes}")
        print(f"[smoke] steps={cfg['total_steps']:,}  n_envs={cfg['n_envs']}")

    if not train_data:
        raise RuntimeError("train 데이터 없음 — val_date 설정 확인")

    # ── VecEnv 생성 ──────────────────────────────────────────────────────
    env_fns = [make_env_fn(train_data, cfg) for _ in range(cfg["n_envs"])]
    vec_env = DummyVecEnv(env_fns)
    print(
        f"[env] DummyVecEnv({cfg['n_envs']})  "
        f"obs={vec_env.observation_space.shape}  "
        f"act={vec_env.action_space.n}"
    )

    # ── PPO ─────────────────────────────────────────────────────────────
    model = build_ppo(vec_env, cfg, str(run_dir))
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[model] 파라미터 수: {total_params:,}")

    # ── Callback ─────────────────────────────────────────────────────────
    val_cb = ValidationCallback(
        train_data    = train_data,
        val_data      = val_data,
        cfg           = cfg,
        best_model_dir= model_dir,
        verbose       = 1,
    )

    # ── 학습 ────────────────────────────────────────────────────────────
    print(f"\n[train] 시작  →  tensorboard --logdir {run_dir.parent}\n")
    try:
        model.learn(
            total_timesteps     = cfg["total_steps"],
            callback            = val_cb,
            progress_bar        = True,
            tb_log_name         = run_name,
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        print("\n[train] 중단됨 — 현재 모델 저장 중...")

    # ── 최종 모델 저장 ────────────────────────────────────────────────────
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"[done] final model: {final_path}.zip")

    # ── 차트 저장 ─────────────────────────────────────────────────────────
    save_curves(val_cb.history, run_dir)

    # ── Best model 최종 평가 ──────────────────────────────────────────────
    best_path = model_dir / "best_model.zip"
    if best_path.exists():
        print("\n[final eval] best model → 전체 val 종목")
        best_model = PPO.load(str(model_dir / "best_model"), env=vec_env)
        final_m = evaluate_portfolio(best_model, val_data, cfg, n_stocks=len(val_data))
        for k, v in sorted(final_m.items()):
            print(f"  {k:<28} {v:+.4f}")

    vec_env.close()
    print(f"\n[done] run_dir: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="PPO 주식 매매 학습 (stable-baselines3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    D = DEFAULT_CFG

    p.add_argument("--smoke",         action="store_true", help="3종목 50k스텝 파이프라인 검증")
    p.add_argument("--data_path",     default=D["data_path"])
    p.add_argument("--val_date",      default=D["val_date"],         help="YYYYMMDD: 이 날짜 이상은 val")
    p.add_argument("--total_steps",   type=int,   default=D["total_steps"])
    p.add_argument("--n_envs",        type=int,   default=D["n_envs"])
    p.add_argument("--learning_rate", type=float, default=D["learning_rate"])
    p.add_argument("--n_steps",       type=int,   default=D["n_steps"],    help="env당 롤아웃 길이")
    p.add_argument("--batch_size",    type=int,   default=D["batch_size"])
    p.add_argument("--gamma",         type=float, default=D["gamma"])
    p.add_argument("--gae_lambda",    type=float, default=D["gae_lambda"])
    p.add_argument("--clip_range",    type=float, default=D["clip_range"])
    p.add_argument("--ent_coef",      type=float, default=D["ent_coef"])
    p.add_argument("--n_epochs",      type=int,   default=D["n_epochs"])
    p.add_argument("--policy",        default=D["policy"],           help="MlpPolicy / MlpLstmPolicy")
    p.add_argument("--net_arch", nargs="+", type=int, default=D["net_arch"], metavar="N",
                   help="예: --net_arch 128 128")
    p.add_argument("--no_shaping",    action="store_true",           help="reward shaping 비활성")
    p.add_argument("--eval_freq",     type=int,   default=D["eval_freq"])
    p.add_argument("--n_eval_stocks", type=int,   default=D["n_eval_stocks"])
    p.add_argument("--model_dir",     default=D["model_dir"])
    p.add_argument("--log_dir",       default=D["log_dir"])
    p.add_argument("--run_name",      default=D["run_name"])

    args = p.parse_args()
    cfg  = {**D, **vars(args)}
    cfg["enable_shaping"] = not args.no_shaping
    return cfg


if __name__ == "__main__":
    main(parse_args())
