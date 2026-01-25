# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Walk-forward backtest Sniper:
- Carrega todos os period_* de um wf_XXX
- Para cada timestamp, escolhe o modelo cujo train_end_utc está antes do timestamp
  (preferindo o menor tail válido, i.e., o mais "recente" que não vazou)
- Roda o ciclo usando p_entry/p_danger pré-computados em batch
"""

from dataclasses import dataclass, replace
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise

from .sniper_simulator import (
    SniperBacktestResult,
    SniperTrade,
    _apply_calibration,
    _finalize_monthly_returns,
)
from config.thresholds import DEFAULT_THRESHOLD_OVERRIDES


try:
    # quando importado como pacote (ex.: backtest.*)
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
except Exception:
    try:
        # fallback legado (quando o repo_root está no sys.path com nome de pacote)
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window  # type: ignore[import]
    except Exception:
        # modo comum: rodar a partir do repo_root (sys.path inclui modules/)
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window


@dataclass
class PeriodModel:
    period_days: int
    train_end_utc: pd.Timestamp
    entry_model: xgb.Booster
    entry_models: dict[str, xgb.Booster]
    danger_model: xgb.Booster | None  # legado (não usado)
    exit_model: xgb.Booster | None  # legado (não usado)
    entry_cols: List[str]
    entry_cols_map: dict[str, List[str]]
    danger_cols: List[str]
    exit_cols: List[str]
    entry_calib: dict
    entry_calib_map: dict[str, dict]
    danger_calib: dict
    exit_calib: dict
    tau_entry: float
    tau_entry_map: dict[str, float]
    tau_danger: float
    tau_exit: float
    tau_add: float
    tau_danger_add: float


def _load_booster(path_json: Path) -> xgb.Booster:
    bst = xgb.Booster()
    ubj = path_json.with_suffix(".ubj")
    if ubj.exists() and ubj.stat().st_size > 0:
        bst.load_model(str(ubj))
        return bst
    bst.load_model(str(path_json))
    return bst


def _entry_specs() -> list[tuple[str, int]]:
    windows = list(getattr(DEFAULT_TRADE_CONTRACT, "entry_label_windows_minutes", []) or [])
    if len(windows) < 1:
        raise ValueError("entry_label_windows_minutes deve ter ao menos 1 valor")
    return [("mid", int(windows[0]))]


def select_entry_mid(p_entry_map: dict[str, np.ndarray]) -> np.ndarray:
    if "mid" in p_entry_map:
        return p_entry_map["mid"]
    if "short" in p_entry_map:
        return p_entry_map["short"]
    if "long" in p_entry_map:
        return p_entry_map["long"]
    # fallback: primeiro disponível
    for v in p_entry_map.values():
        return v
    return np.array([], dtype=np.float32)


def load_period_models(
    run_dir: Path,
    *,
    tau_add_multiplier: float = 1.10,
    tau_danger_add_multiplier: float = 0.90,
) -> List[PeriodModel]:
    periods: List[PeriodModel] = []
    for pd_dir in sorted([p for p in Path(run_dir).iterdir() if p.is_dir() and p.name.startswith("period_") and p.name.endswith("d")]):
        period_days = int(pd_dir.name.replace("period_", "").replace("d", ""))
        meta = json.loads((pd_dir / "meta.json").read_text(encoding="utf-8"))
        train_end = meta.get("train_end_utc")
        if not train_end:
            continue
        train_end_ts = pd.to_datetime(train_end)

        entry_models: dict[str, xgb.Booster] = {}
        entry_cols_map: dict[str, list[str]] = {}
        entry_calib_map: dict[str, dict] = {}
        # tenta carregar modelos multi-janela
        for name, w in _entry_specs():
            mdir = pd_dir / f"entry_model_{int(w)}m" / "model_entry.json"
            if mdir.exists():
                entry_models[name] = _load_booster(mdir)
                meta_key = f"entry_{int(w)}m"
                entry_cols_map[name] = list((meta.get(meta_key) or {}).get("feature_cols") or meta["entry"]["feature_cols"])
                entry_calib_map[name] = dict((meta.get(meta_key) or {}).get("calibration") or meta["entry"].get("calibration") or {"type": "identity"})
        # fallback legado (single)
        if not entry_models:
            entry_models["mid"] = _load_booster(pd_dir / "entry_model" / "model_entry.json")
            entry_cols_map["mid"] = list(meta["entry"]["feature_cols"])
            entry_calib_map["mid"] = dict(meta["entry"].get("calibration") or {"type": "identity"})

        entry_model = entry_models.get("mid") or list(entry_models.values())[0]
        # modelos de danger/exit removidos (pipeline entry-only)
        danger_model = None
        exit_model = None
        entry_cols = list(meta["entry"]["feature_cols"])
        danger_cols: list[str] = []
        exit_cols: list[str] = []
        entry_calib = dict(meta["entry"].get("calibration") or {"type": "identity"})
        danger_calib: dict = {}
        exit_calib: dict = {}
        tau_entry = DEFAULT_THRESHOLD_OVERRIDES.tau_entry
        tau_danger = 1.0
        tau_exit = 1.0
        if tau_entry is None:
            tau_entry = float(meta["entry"].get("threshold", 0.5))
        tau_add = float(min(0.99, max(0.01, tau_entry * float(tau_add_multiplier))))
        tau_danger_add = 1.0
        tau_entry_map = {"mid": float(tau_entry)}

        periods.append(
            PeriodModel(
                period_days=period_days,
                train_end_utc=train_end_ts,
                entry_model=entry_model,
                entry_models=entry_models,
                danger_model=danger_model,
                exit_model=exit_model,
                entry_cols=entry_cols,
                entry_cols_map=entry_cols_map,
                danger_cols=danger_cols,
                exit_cols=exit_cols,
                entry_calib=entry_calib,
                entry_calib_map=entry_calib_map,
                danger_calib=danger_calib,
                exit_calib=exit_calib,
                tau_entry=tau_entry,
                tau_entry_map=tau_entry_map,
                tau_danger=tau_danger,
                tau_exit=tau_exit,
                tau_add=tau_add,
                tau_danger_add=tau_danger_add,
            )
        )
    # mais recente primeiro (menor tail) => train_end maior
    periods.sort(key=lambda p: p.train_end_utc, reverse=True)
    return periods


def apply_threshold_overrides(
    periods: List[PeriodModel],
    *,
    tau_entry: float | None = None,
    tau_add_multiplier: float = 1.10,
) -> List[PeriodModel]:
    """
    Aplica overrides (simulação) em todos os períodos, mantendo consistência de thresholds derivados.
    """
    out: List[PeriodModel] = []
    for pm in periods:
        te = pm.tau_entry if tau_entry is None else float(tau_entry)
        ta = float(min(0.99, max(0.01, te * float(tau_add_multiplier))))
        tau_entry_map = dict(pm.tau_entry_map or {})
        if tau_entry is not None:
            tau_entry_map = {k: float(te) for k in (tau_entry_map.keys() or ["mid"])}
        out.append(
            replace(
                pm,
                tau_entry=float(te),
                tau_entry_map=tau_entry_map,
                tau_add=float(ta),
            )
        )
    return out


def _build_matrix_rows(df: pd.DataFrame, cols: List[str], rows: np.ndarray) -> np.ndarray:
    """
    Monta matriz float32 apenas para as linhas `rows`, evitando `df.loc[mask]` (muito caro).
    """
    rows = np.asarray(rows, dtype=np.int64)
    mat = np.zeros((int(rows.size), len(cols)), dtype=np.float32)
    if rows.size == 0 or (len(cols) == 0):
        return mat
    df_cols = set(df.columns)
    for j, c in enumerate(cols):
        if c not in df_cols:
            continue
        v = df[c].to_numpy()
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        mat[:, j] = v[rows]
    return mat


def predict_scores_walkforward(
    df: pd.DataFrame,
    *,
    periods: List[PeriodModel],
    return_period_id: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, PeriodModel] | tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, PeriodModel, np.ndarray]:
    """
    Prediz p_entry/p_danger para cada timestamp, escolhendo o modelo mais recente com train_end_utc < t.
    Retorna p_entry por janela, p_danger e o primeiro período efetivamente usado.
    """
    import xgboost as xgb
    idx = pd.to_datetime(df.index)
    n = len(idx)
    p_entry_map = {name: np.full(n, np.nan, dtype=np.float32) for name, _w in _entry_specs()}
    p_danger = np.zeros(n, dtype=np.float32)  # danger removido
    # p_exit removido (mantido nulo para compatibilidade)
    p_exit = np.zeros(n, dtype=np.float32)
    period_id = np.full(n, -1, dtype=np.int16)

    used_any: PeriodModel | None = None

    for pid, pm in enumerate(periods):
        mask = idx > pm.train_end_utc
        # só preenche onde ainda não foi preenchido (usa mid como proxy)
        mask &= ~np.isfinite(p_entry_map.get("mid", np.full(n, np.nan, dtype=np.float32)))
        if not mask.any():
            continue
        mask_np = np.asarray(mask, dtype=bool)
        rows = np.flatnonzero(mask_np)
        pdg = np.zeros(rows.size, dtype=np.float32)
        # entry por janela
        for name, _w in _entry_specs():
            model = pm.entry_models.get(name, pm.entry_model)
            cols = pm.entry_cols_map.get(name, pm.entry_cols)
            X_e = _build_matrix_rows(df, cols, rows)
            pe = model.predict(xgb.DMatrix(X_e), validate_features=False).astype(np.float32, copy=False)
            calib = pm.entry_calib_map.get(name, pm.entry_calib)
            pe = _apply_calibration(pe.astype(np.float64), calib).astype(np.float32, copy=False)
            p_entry_map[name][rows] = pe
        p_danger[rows] = pdg
        period_id[rows] = np.int16(pid)
        if used_any is None:
            used_any = pm

    if used_any is None:
        if len(periods) == 0:
            raise RuntimeError("Nenhum período encontrado (periods vazio).")
        t0 = pd.to_datetime(idx.min())
        t1 = pd.to_datetime(idx.max())
        te_max = max(p.train_end_utc for p in periods)
        te_min = min(p.train_end_utc for p in periods)
        raise RuntimeError(
            "Nenhum período do run_dir é válido para o range de timestamps fornecido. "
            f"df=[{t0}..{t1}] train_end_range=[{te_min}..{te_max}]"
        )
    if return_period_id:
        return p_entry_map, p_danger, p_exit, used_any, period_id
    return p_entry_map, p_danger, p_exit, used_any



def _build_row_vector_for_exit(
    df: pd.DataFrame,
    i: int,
    cols: List[str],
    *,
    cycle_state: dict,
    col_arrays: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    out = np.zeros(len(cols), dtype=np.float32)
    for k, col in enumerate(cols):
        if col.startswith("cycle_"):
            out[k] = float(cycle_state.get(col, 0.0))
            continue
        if col_arrays is not None:
            arr = col_arrays.get(col)
            if arr is not None:
                try:
                    v = float(arr[i])
                    out[k] = v if np.isfinite(v) else 0.0
                    continue
                except Exception:
                    pass
        try:
            v = df[col].iat[i]
            out[k] = float(v) if np.isfinite(v) else 0.0
        except Exception:
            out[k] = 0.0
    return out


def _make_exit_row_cache(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    """
    Prepara estruturas para preencher row de ExitScore sem alocar/sem pandas no hot-loop.

    Retorna: (kind[int8], idx[int16], data_cols(typed.List|list), row_buf[float32])
    """
    # mapeamento fixo de cycle_* (ordem estável)
    cycle_names = [
        "cycle_is_add",
        "cycle_num_adds",
        "cycle_time_in_trade",
        "cycle_dd_pct",
        "cycle_dist_to_tp",
        "cycle_dist_to_sl",
        "cycle_avg_entry_price",
        "cycle_last_fill_price",
        "cycle_risk_used_pct",
        "cycle_risk_if_add_pct",
    ]
    cycle_map = {name: j for j, name in enumerate(cycle_names)}

    kind = np.zeros(len(cols), dtype=np.int8)
    idx = np.zeros(len(cols), dtype=np.int16)

    # data colunas: lista de arrays float32 (numba typed.List se disponível)
    data_py: list[np.ndarray] = []
    data_pos: dict[str, int] = {}

    for k, col in enumerate(cols):
        sc = str(col)
        if sc.startswith("cycle_"):
            kind[k] = 0
            idx[k] = np.int16(cycle_map.get(sc, 0))
            continue
        # non-cycle: coluna do df
        if sc in df.columns:
            j = data_pos.get(sc)
            if j is None:
                try:
                    arr = df[sc].to_numpy(np.float32, copy=False)
                except Exception:
                    arr = None
                if arr is None:
                    kind[k] = 2
                    idx[k] = np.int16(0)
                    continue
                j = len(data_py)
                data_pos[sc] = j
                data_py.append(arr)
            kind[k] = 1
            idx[k] = np.int16(j)
        else:
            kind[k] = 2
            idx[k] = np.int16(0)

    if _NUMBA_OK:
        data_cols = NList()  # type: ignore[misc]
        for a in data_py:
            data_cols.append(a)
    else:
        data_cols = data_py

    row_buf = np.empty(len(cols), dtype=np.float32)
    return kind, idx, data_cols, row_buf


def simulate_sniper_from_scores(
    df: pd.DataFrame,
    *,
    p_entry: np.ndarray,
    p_danger: np.ndarray,
    p_exit: np.ndarray | None = None,
    thresholds: PeriodModel,
    periods: List[PeriodModel] | None = None,
    period_id: np.ndarray | None = None,
    contract: TradeContract | None = None,
    candle_sec: int = 60,
    # Config explícita (sem depender de env vars)
    exit_min_hold_bars: int = 0,
    exit_confirm_bars: int = 1,
) -> SniperBacktestResult:
    """
    Simula ciclo Sniper usando scores pré-computados (mais rápido) e thresholds do período selecionado.
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
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
    # confirmação simples do exit
    exit_min_hold = int(max(0, exit_min_hold_bars))
    exit_confirm = int(exit_confirm_bars)
    if exit_confirm <= 0:
        exit_confirm = 1
    exit_streak = 0
    ema_span = exit_ema_span_from_window(contract, int(candle_sec))
    use_ema_exit = ema_span > 0
    ema_alpha = 2.0 / float(ema_span + 1) if use_ema_exit else 0.0
    ema_offset = float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0)
    ema = 0.0

    size_sched = tuple(float(x) for x in contract.add_sizing) if contract.add_sizing else (1.0,)
    if len(size_sched) < contract.max_adds + 1:
        size_sched = size_sched + (size_sched[-1],) * (contract.max_adds + 1 - len(size_sched))


    for i in range(n):
        pm = thresholds
        pid_i = 0
        if periods is not None and period_id is not None:
            try:
                pid_i = int(period_id[i])
                if 0 <= pid_i < len(periods):
                    pm = periods[pid_i]
            except Exception:
                pm = thresholds
                pid_i = 0

        px = close[i]
        if not np.isfinite(px) or px <= 0.0:
            eq_curve[i] = eq
            continue
        pe = float(p_entry[i]) if np.isfinite(p_entry[i]) else 0.0

        if not in_pos:
            exit_streak = 0
            if i + int(exit_min_hold) >= (n - 1):
                eq_curve[i] = eq
                continue
            if (pe >= pm.tau_entry):
                in_pos = True
                entry_i = i
                entry_price = px
                avg_price = px
                last_fill = px
                total_size = size_sched[0]
                num_adds = 0
                if use_ema_exit:
                    ema = float(entry_price) * (1.0 - ema_offset)
            eq_curve[i] = eq
            continue

        time_in_trade = i - entry_i
        lo = low[i] if np.isfinite(low[i]) else px
        reason = None
        exit_px = None
        if use_ema_exit:
            ema = ema + (ema_alpha * (px - ema))
            if px < ema:
                exit_streak += 1
            else:
                exit_streak = 0

        if use_ema_exit and (px < ema) and (exit_streak >= exit_confirm):
            reason = "EMA"
            exit_px = px

        if reason is not None and exit_px is not None:
            entries = 1 + num_adds
            sides = entries + 1
            costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
            r_gross = (exit_px / avg_price) - 1.0
            r_net = float(r_gross) - float(costs)
            eq = eq * (1.0 + r_net)
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
                    r_gross=float(r_gross),
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

        # add logic com scores précomputados
        if num_adds < int(contract.max_adds):
            trigger = last_fill * (1.0 - float(contract.add_spacing_pct))
            if trigger > 0 and lo <= trigger:
                next_size = size_sched[num_adds + 1]
                risk_after = 0.0
                if (pe >= pm.tau_add) and (risk_after <= contract.risk_max_cycle_pct + 1e-9):
                    new_total = total_size + next_size
                    avg_price = (avg_price * total_size + trigger * next_size) / new_total
                    total_size = new_total
                    last_fill = trigger
                    num_adds += 1

        eq_curve[i] = eq

    # métricas
    if in_pos and n > 0:
        exit_px = float(close[-1])
        entries = 1 + num_adds
        sides = entries + 1
        costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
        r_gross = (exit_px / avg_price) - 1.0 if avg_price > 0 else 0.0
        r_net = float(r_gross) - float(costs)
        eq = eq * (1.0 + r_net)
        trades.append(
            SniperTrade(
                entry_ts=pd.to_datetime(idx[entry_i]),
                exit_ts=pd.to_datetime(idx[-1]),
                entry_price=float(entry_price),
                exit_price=float(exit_px),
                num_adds=int(num_adds),
                reason="EOD",
                r_net=float(r_net),
                avg_entry_price=float(avg_price),
                entries=int(entries),
                sides=int(sides),
                costs=float(costs),
                r_gross=float(r_gross),
            )
        )
        eq_curve[-1] = eq

    eq_max = np.maximum.accumulate(eq_curve) if len(eq_curve) else eq_curve
    dd = (eq_max - eq_curve) / np.where(eq_max > 0, eq_max, 1.0) if len(eq_curve) else np.array([0.0])
    max_dd = float(np.nanmax(dd)) if len(dd) else 0.0
    dd2 = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
    ulcer = float(np.sqrt(float(np.mean(dd2 * dd2)))) if len(dd2) else 0.0
    # maior sequência de drawdown (dd > 0)
    if len(dd2):
        x = (dd2 > 1e-12).astype(np.int8, copy=False)
        dx = np.diff(np.concatenate(([0], x, [0])))
        starts = np.flatnonzero(dx == 1)
        ends = np.flatnonzero(dx == -1)
        dd_dur = float((ends - starts).max()) / float(len(dd2)) if starts.size and ends.size else 0.0
    else:
        dd_dur = 0.0
    if trades:
        wins = [t.r_net for t in trades if t.r_net > 0]
        losses = [-t.r_net for t in trades if t.r_net <= 0]
        win_rate = float(len(wins) / len(trades))
        pf = float(sum(wins) / max(1e-12, sum(losses))) if losses else float("inf")
    else:
        win_rate = 0.0
        pf = 0.0
    monthly = _finalize_monthly_returns(df.index, eq_curve)
    return SniperBacktestResult(
        trades=trades,
        equity_curve=eq_curve,
        timestamps=df.index,
        monthly_returns=monthly,
        max_dd=max_dd,
        ulcer_index=float(ulcer),
        dd_duration_ratio=float(dd_dur),
        win_rate=win_rate,
        profit_factor=pf,
    )


__all__ = [
    "PeriodModel",
    "load_period_models",
    "predict_scores_walkforward",
    "select_entry_mid",
    "simulate_sniper_from_scores",
]


