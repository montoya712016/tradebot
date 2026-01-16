# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Motor de backtest Sniper (ciclo stateful):
- Entrada apenas se EntryScore >= tau_entry e DangerScore < tau_danger
- Adds respeitando max_adds, add_spacing, risco máximo, e thresholds mais rígidos
- Saídas por TP/SL/Timeout (contrato fixo)
"""

from dataclasses import dataclass
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
from config.thresholds import DEFAULT_THRESHOLD_OVERRIDES


@dataclass
class SniperModels:
    entry_model: "xgb.Booster"
    danger_model: "xgb.Booster"
    exit_model: "xgb.Booster | None"
    entry_feature_cols: List[str]
    danger_feature_cols: List[str]
    exit_feature_cols: List[str]
    entry_calib: dict
    danger_calib: dict
    exit_calib: dict
    tau_entry: float
    tau_danger: float
    tau_exit: float
    tau_add: float
    tau_danger_add: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _apply_calibration(p: np.ndarray, calib: dict) -> np.ndarray:
    if not isinstance(calib, dict):
        return p
    if calib.get("type") != "platt":
        return p
    a = float(calib.get("coef", 1.0))
    b = float(calib.get("intercept", 0.0))
    z = a * p + b
    return _sigmoid(z)


def _load_booster(path_json: Path) -> "xgb.Booster":
    bst = xgb.Booster()
    p_ubj = path_json.with_suffix(".ubj")
    if p_ubj.exists() and p_ubj.stat().st_size > 0:
        bst.load_model(str(p_ubj))
        return bst
    bst.load_model(str(path_json))
    try:
        bst.save_model(str(p_ubj))
    except Exception:
        pass
    return bst


def load_sniper_models(
    run_dir: Path,
    period_days: int,
    *,
    tau_add_multiplier: float = 1.10,
    tau_danger_add_multiplier: float = 0.90,
) -> SniperModels:
    """
    Carrega EntryScore e DangerScore + meta (features, calibração).

    tau_add e tau_danger_add são derivados do threshold principal por multiplicadores
    (thresholds definidos manualmente em config/thresholds.py).
    """
    pd_dir = Path(run_dir) / f"period_{int(period_days)}d"
    meta_path = pd_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    entry_path = pd_dir / "entry_model" / "model_entry.json"
    danger_path = pd_dir / "danger_model" / "model_danger.json"
    exit_path = pd_dir / "exit_model" / "model_exit.json"

    entry_model = _load_booster(entry_path)
    danger_model = _load_booster(danger_path)
    exit_model = _load_booster(exit_path) if exit_path.exists() else None

    entry_cols = list(meta["entry"]["feature_cols"])
    danger_cols = list(meta["danger"]["feature_cols"])
    exit_cols = list((meta.get("exit") or {}).get("feature_cols") or [])
    entry_calib = dict(meta["entry"].get("calibration") or {"type": "identity"})
    danger_calib = dict(meta["danger"].get("calibration") or {"type": "identity"})
    exit_calib = dict((meta.get("exit") or {}).get("calibration") or {"type": "identity"})
    tau_entry = DEFAULT_THRESHOLD_OVERRIDES.tau_entry
    tau_danger = DEFAULT_THRESHOLD_OVERRIDES.tau_danger
    tau_exit = DEFAULT_THRESHOLD_OVERRIDES.tau_exit
    if tau_entry is None:
        tau_entry = float(meta["entry"].get("threshold", 0.5))
    if tau_danger is None:
        tau_danger = float(meta["danger"].get("threshold", 0.5))
    if tau_exit is None:
        tau_exit = float((meta.get("exit") or {}).get("threshold", 1.0))

    tau_add = float(min(0.99, max(0.01, tau_entry * float(tau_add_multiplier))))
    tau_danger_add = float(min(0.99, max(0.01, tau_danger * float(tau_danger_add_multiplier))))

    return SniperModels(
        entry_model=entry_model,
        danger_model=danger_model,
        exit_model=exit_model,
        entry_feature_cols=entry_cols,
        danger_feature_cols=danger_cols,
        exit_feature_cols=exit_cols,
        entry_calib=entry_calib,
        danger_calib=danger_calib,
        exit_calib=exit_calib,
        tau_entry=tau_entry,
        tau_danger=tau_danger,
        tau_exit=tau_exit,
        tau_add=tau_add,
        tau_danger_add=tau_danger_add,
    )


def _predict_1row(bst: "xgb.Booster", row: np.ndarray) -> float:
    # IMPORTANT (XGBoost CUDA):
    # `inplace_predict` com booster em CUDA e input numpy (CPU) gera warning
    # "mismatched devices" e cai em DMatrix internamente. Evitamos isso aqui.
    dm = xgb.DMatrix(row.reshape(1, -1))
    p = bst.predict(dm, validate_features=False)
    return float(p[0])


def _build_row_vector(
    df: pd.DataFrame,
    i: int,
    feat_cols: List[str],
    *,
    cycle_state: dict,
) -> np.ndarray:
    out = np.zeros(len(feat_cols), dtype=np.float32)
    for k, col in enumerate(feat_cols):
        if col.startswith("cycle_"):
            out[k] = float(cycle_state.get(col, 0.0))
            continue
        try:
            v = df[col].iat[i]
            out[k] = float(v) if np.isfinite(v) else 0.0
        except Exception:
            out[k] = 0.0
    return out


@dataclass
class SniperTrade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    num_adds: int
    reason: str
    r_net: float
    # Extras para diagnóstico/plots (opcionais para compat)
    avg_entry_price: float | None = None
    entries: int | None = None
    sides: int | None = None
    costs: float | None = None
    r_gross: float | None = None


@dataclass
class SniperBacktestResult:
    trades: List[SniperTrade]
    equity_curve: np.ndarray
    timestamps: pd.DatetimeIndex
    monthly_returns: List[float]
    max_dd: float
    win_rate: float
    profit_factor: float
    # métricas adicionais (baratas; calculadas junto com dd)
    ulcer_index: float = 0.0
    dd_duration_ratio: float = 0.0


def _max_true_run_np(flags: np.ndarray) -> int:
    """
    Maior sequência consecutiva de True em um array booleano.
    """
    if flags.size == 0:
        return 0
    x = flags.astype(np.int8, copy=False)
    # detecta inícios/fins de runs com diff
    dx = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(dx == 1)
    ends = np.flatnonzero(dx == -1)
    if starts.size == 0 or ends.size == 0:
        return 0
    lens = ends - starts
    return int(lens.max()) if lens.size else 0


def _finalize_monthly_returns(idx: pd.DatetimeIndex, eq: np.ndarray) -> List[float]:
    if len(idx) == 0:
        return []
    month_first: Dict[Tuple[int, int], float] = {}
    month_last: Dict[Tuple[int, int], float] = {}
    for t, e in zip(idx, eq):
        key = (int(t.year), int(t.month))
        month_first.setdefault(key, float(e))
        month_last[key] = float(e)
    out: List[float] = []
    for k in sorted(month_first.keys()):
        s = month_first[k]
        e = month_last.get(k, s)
        if s > 0:
            out.append((e / s) - 1.0)
    return out


def simulate_sniper_cycle(
    df: pd.DataFrame,
    *,
    models: SniperModels,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
    capital_per_cycle: float = 1.0,
    # Config explícita (sem depender de env vars)
    tp_hard_when_exit: bool | None = None,
    exit_min_hold_bars: int = 3,
    exit_confirm_bars: int = 1,
) -> SniperBacktestResult:
    """
    Simula um único símbolo usando ciclo stateful e os modelos treinados.
    Espera que `df` já contenha as features usadas no treino (mesmos nomes).
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    candle_sec = int(candle_sec or 60)
    timeout_bars = contract.timeout_bars(candle_sec)

    close = df["close"].to_numpy(np.float64, copy=False)
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)
    idx = df.index
    n = len(df)

    eq = 1.0
    eq_curve = np.ones(n, dtype=np.float64)
    trades: List[SniperTrade] = []

    in_pos = False
    entry_i = 0
    entry_price = 0.0
    avg_price = 0.0
    last_fill = 0.0
    total_size = 0.0
    num_adds = 0
    size_sched = tuple(float(x) for x in contract.add_sizing) if contract.add_sizing else (1.0,)
    if len(size_sched) < contract.max_adds + 1:
        size_sched = size_sched + (size_sched[-1],) * (contract.max_adds + 1 - len(size_sched))

    # Exit score: confirmation to avoid 1-bar spikes
    exit_min_hold = int(exit_min_hold_bars)
    exit_confirm = int(exit_confirm_bars)
    if exit_confirm <= 0:
        exit_confirm = 1
    exit_streak = 0
    exit_threshold = float(getattr(contract, "exit_score_threshold", 0.0))
    time_per_bar_hours = float(candle_sec) / 3600.0

    for i in range(n):
        px = close[i]
        if not np.isfinite(px) or px <= 0.0:
            eq_curve[i] = eq
            continue

        # estado do ciclo observável
        time_in_trade = (i - entry_i) if in_pos else 0
        dd_pct = (px / avg_price - 1.0) if (in_pos and avg_price > 0) else 0.0
        tp_price = (avg_price * (1.0 + contract.tp_min_pct)) if in_pos else (px * (1.0 + contract.tp_min_pct))
        sl_price = (avg_price * (1.0 - contract.sl_pct)) if in_pos else (px * (1.0 - contract.sl_pct))
        dist_to_tp = ((tp_price - px) / tp_price) if tp_price > 0 else 0.0
        dist_to_sl = ((px - sl_price) / sl_price) if sl_price > 0 else 0.0
        risk_used = (total_size * contract.sl_pct) if in_pos else contract.sl_pct
        next_size = size_sched[min(num_adds + 1, len(size_sched) - 1)] if in_pos else size_sched[0]
        risk_if_add = ((total_size + next_size) * contract.sl_pct) if in_pos else (contract.sl_pct + contract.add_spacing_pct)

        cycle_state = {
            # compatível com os datasets: 1.0 apenas quando estamos avaliando um add
            "cycle_is_add": 0.0,
            "cycle_num_adds": float(num_adds),
            "cycle_time_in_trade": float(time_in_trade),
            "cycle_dd_pct": float(dd_pct),
            "cycle_dist_to_tp": float(dist_to_tp),
            "cycle_dist_to_sl": float(dist_to_sl),
            "cycle_avg_entry_price": float(avg_price if in_pos else px),
            "cycle_last_fill_price": float(last_fill if in_pos else px),
            "cycle_risk_used_pct": float(risk_used),
            "cycle_risk_if_add_pct": float(risk_if_add),
        }

        # danger sempre é calculado
        x_d = _build_row_vector(df, i, models.danger_feature_cols, cycle_state=cycle_state)
        p_d = _predict_1row(models.danger_model, x_d)
        p_d = float(_apply_calibration(np.array([p_d], dtype=np.float64), models.danger_calib)[0])

        if not in_pos:
            exit_streak = 0
            x_e = _build_row_vector(df, i, models.entry_feature_cols, cycle_state=cycle_state)
            p_e = _predict_1row(models.entry_model, x_e)
            p_e = float(_apply_calibration(np.array([p_e], dtype=np.float64), models.entry_calib)[0])
            if (p_e >= models.tau_entry) and (p_d < models.tau_danger):
                in_pos = True
                entry_i = i
                entry_price = px
                avg_price = px
                last_fill = px
                total_size = size_sched[0]
                num_adds = 0
            eq_curve[i] = eq
            continue

        # in position: checa exits (ordem conservadora SL -> TP -> EXIT -> timeout)
        hi = high[i] if np.isfinite(high[i]) else px
        lo = low[i] if np.isfinite(low[i]) else px
        reason = None
        exit_px = None
        # exit score (pnl/dd/tempo/danger)
        exit_score = 0.0
        if (exit_threshold > 0.0) and (time_in_trade >= exit_min_hold):
            pnl_pct = ((px / avg_price) - 1.0) * 100.0 if avg_price > 0.0 else 0.0
            if pnl_pct < 0.0:
                pnl_pct = 0.0
            dd_pct_score = ((avg_price - lo) / avg_price) * 100.0 if (avg_price > 0.0 and lo < avg_price) else 0.0
            time_hours = float(time_in_trade) * time_per_bar_hours
            danger_hit = bool(p_d >= models.tau_danger)
            exit_score = contract.exit_score(
                pnl_pct=float(pnl_pct),
                dd_pct=float(dd_pct_score),
                time_hours=float(time_hours),
                danger_hit=bool(danger_hit),
            )

        if (exit_threshold > 0.0) and (time_in_trade >= exit_min_hold) and (exit_score >= exit_threshold):
            exit_streak += 1
        else:
            exit_streak = 0

        # TP hard:
        # - se existe ExitScore, por padrão desligamos TP hard (deixa o Exit decidir)
        # - se não existe ExitScore, TP hard fica ligado como antes
        has_exit = bool(exit_threshold > 0.0)
        if tp_hard_when_exit is None:
            tp_hard = (not has_exit)
        else:
            tp_hard = bool(tp_hard_when_exit) if has_exit else True

        if lo <= (avg_price * (1.0 - contract.sl_pct)):
            reason = "SL"
            exit_px = avg_price * (1.0 - contract.sl_pct)
        elif tp_hard and (hi >= (avg_price * (1.0 + contract.tp_min_pct))):
            reason = "TP"
            exit_px = avg_price * (1.0 + contract.tp_min_pct)
        elif (has_exit) and (time_in_trade >= exit_min_hold) and (exit_streak >= exit_confirm):
            reason = "EXIT"
            exit_px = px
        elif time_in_trade >= timeout_bars:
            reason = "TO"
            exit_px = px

        if reason is not None and exit_px is not None:
            entries = 1 + num_adds
            sides = entries + 1
            costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
            r = (exit_px / avg_price) - 1.0
            r_net = r - costs
            eq = eq * (1.0 + capital_per_cycle * r_net)
            trades.append(
                SniperTrade(
                    entry_ts=pd.to_datetime(idx[entry_i]),
                    exit_ts=pd.to_datetime(idx[i]),
                    entry_price=float(entry_price),
                    exit_price=float(exit_px),
                    num_adds=int(num_adds),
                    reason=str(reason),
                    r_net=float(r_net),
                    avg_entry_price=float(avg_price),
                    entries=int(entries),
                    sides=int(sides),
                    costs=float(costs),
                    r_gross=float(r),
                )
            )
            in_pos = False
            exit_streak = 0
            entry_i = 0
            entry_price = 0.0
            avg_price = 0.0
            last_fill = 0.0
            total_size = 0.0
            num_adds = 0
            eq_curve[i] = eq
            continue

        # add logic (somente se ainda no ciclo)
        if num_adds < int(contract.max_adds):
            trigger = last_fill * (1.0 - float(contract.add_spacing_pct))
            if trigger > 0 and lo <= trigger:
                # thresholds mais rígidos para add
                cycle_state["cycle_is_add"] = 1.0
                x_e = _build_row_vector(df, i, models.entry_feature_cols, cycle_state=cycle_state)
                p_e = _predict_1row(models.entry_model, x_e)
                p_e = float(_apply_calibration(np.array([p_e], dtype=np.float64), models.entry_calib)[0])
                next_size = size_sched[num_adds + 1]
                risk_after = (total_size + next_size) * contract.sl_pct
                if (
                    (p_e >= models.tau_add)
                    and (p_d < models.tau_danger_add)
                    and (risk_after <= contract.risk_max_cycle_pct + 1e-9)
                ):
                    # executa add no preço trigger
                    new_total = total_size + next_size
                    avg_price = (avg_price * total_size + trigger * next_size) / new_total
                    total_size = new_total
                    last_fill = trigger
                    num_adds += 1

        eq_curve[i] = eq

    # métricas
    if len(eq_curve):
        eq_max = np.maximum.accumulate(eq_curve)
        dd = (eq_max - eq_curve) / np.where(eq_max > 0, eq_max, 1.0)
        max_dd = float(np.nanmax(dd))
        dd2 = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
        ulcer = float(np.sqrt(float(np.mean(dd2 * dd2))))
        dd_dur = float(_max_true_run_np(dd2 > 1e-12)) / float(len(dd2))
    else:
        max_dd = 0.0
        ulcer = 0.0
        dd_dur = 0.0
    if trades:
        wins = [t.r_net for t in trades if t.r_net > 0]
        losses = [-t.r_net for t in trades if t.r_net <= 0]
        win_rate = float(len(wins) / len(trades))
        pf = float(sum(wins) / max(1e-12, sum(losses))) if losses else float("inf")
    else:
        win_rate = 0.0
        pf = 0.0

    monthly = _finalize_monthly_returns(idx, eq_curve)
    return SniperBacktestResult(
        trades=trades,
        equity_curve=eq_curve.astype(np.float64, copy=False),
        timestamps=idx,
        monthly_returns=monthly,
        max_dd=max_dd,
        ulcer_index=float(ulcer),
        dd_duration_ratio=float(dd_dur),
        win_rate=win_rate,
        profit_factor=pf,
    )


__all__ = [
    "SniperModels",
    "SniperTrade",
    "SniperBacktestResult",
    "load_sniper_models",
    "simulate_sniper_cycle",
]


