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
import math

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

# Numba (opcional): acelera montagem do vetor do ExitScore (hot-loop).
_NUMBA_OK = False
try:  # pragma: no cover
    from numba import njit  # type: ignore
    from numba.typed import List as NList  # type: ignore

    _NUMBA_OK = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    NList = None  # type: ignore
    _NUMBA_OK = False


def _apply_calibration_scalar(x: float, calib: dict) -> float:
    """
    Versão escalar da calibração (evita alocar array 1x).
    """
    if not isinstance(calib, dict):
        return float(x)
    if calib.get("type") != "platt":
        return float(x)
    a = float(calib.get("coef", 1.0))
    b = float(calib.get("intercept", 0.0))
    z = a * float(x) + b
    return float(1.0 / (1.0 + math.exp(-z)))


if _NUMBA_OK:  # pragma: no cover
    @njit(cache=True)
    def _fill_exit_row_nb(out, cycle_vals, data_cols, i, kind, idx):  # type: ignore[no-redef]
        """
        kind: 0 -> cycle, 1 -> data, 2 -> zero
        idx: índice no vetor cycle_vals ou na lista data_cols
        """
        for k in range(out.shape[0]):
            kk = kind[k]
            if kk == 0:
                out[k] = cycle_vals[idx[k]]
            elif kk == 1:
                out[k] = data_cols[idx[k]][i]
            else:
                out[k] = 0.0


try:
    # quando importado como pacote (ex.: backtest.*)
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    try:
        # fallback legado (quando o repo_root está no sys.path com nome de pacote)
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    except Exception:
        # modo comum: rodar a partir do repo_root (sys.path inclui modules/)
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT


@dataclass
class PeriodModel:
    period_days: int
    train_end_utc: pd.Timestamp
    entry_model: xgb.Booster
    danger_model: xgb.Booster
    exit_model: xgb.Booster | None
    entry_cols: List[str]
    danger_cols: List[str]
    exit_cols: List[str]
    entry_calib: dict
    danger_calib: dict
    exit_calib: dict
    tau_entry: float
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

        entry_model = _load_booster(pd_dir / "entry_model" / "model_entry.json")
        danger_model = _load_booster(pd_dir / "danger_model" / "model_danger.json")
        exit_path = pd_dir / "exit_model" / "model_exit.json"
        exit_model = _load_booster(exit_path) if exit_path.exists() else None
        entry_cols = list(meta["entry"]["feature_cols"])
        danger_cols = list(meta["danger"]["feature_cols"])
        exit_cols = list((meta.get("exit") or {}).get("feature_cols") or [])
        entry_calib = dict(meta["entry"].get("calibration") or {"type": "identity"})
        danger_calib = dict(meta["danger"].get("calibration") or {"type": "identity"})
        exit_calib = dict((meta.get("exit") or {}).get("calibration") or {"type": "identity"})
        tau_entry = float(meta["entry"].get("threshold", 0.5))
        tau_danger = float(meta["danger"].get("threshold", 0.5))
        tau_exit = float((meta.get("exit") or {}).get("threshold", 1.0))
        tau_add = float(min(0.99, max(0.01, tau_entry * float(tau_add_multiplier))))
        tau_danger_add = float(min(0.99, max(0.01, tau_danger * float(tau_danger_add_multiplier))))

        periods.append(
            PeriodModel(
                period_days=period_days,
                train_end_utc=train_end_ts,
                entry_model=entry_model,
                danger_model=danger_model,
                exit_model=exit_model,
                entry_cols=entry_cols,
                danger_cols=danger_cols,
                exit_cols=exit_cols,
                entry_calib=entry_calib,
                danger_calib=danger_calib,
                exit_calib=exit_calib,
                tau_entry=tau_entry,
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
    tau_danger: float | None = None,
    tau_exit: float | None = None,
    tau_add_multiplier: float = 1.10,
    tau_danger_add_multiplier: float = 0.90,
) -> List[PeriodModel]:
    """
    Aplica overrides (simulação) em todos os períodos, mantendo consistência de thresholds derivados.
    """
    out: List[PeriodModel] = []
    for pm in periods:
        te = pm.tau_entry if tau_entry is None else float(tau_entry)
        td = pm.tau_danger if tau_danger is None else float(tau_danger)
        tx = pm.tau_exit if tau_exit is None else float(tau_exit)
        ta = float(min(0.99, max(0.01, te * float(tau_add_multiplier))))
        tda = float(min(0.99, max(0.01, td * float(tau_danger_add_multiplier))))
        out.append(
            replace(
                pm,
                tau_entry=float(te),
                tau_danger=float(td),
                tau_exit=float(tx),
                tau_add=float(ta),
                tau_danger_add=float(tda),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PeriodModel] | tuple[np.ndarray, np.ndarray, np.ndarray, PeriodModel, np.ndarray]:
    """
    Prediz p_entry/p_danger para cada timestamp, escolhendo o modelo mais recente com train_end_utc < t.
    Retorna arrays e o primeiro período efetivamente usado (para thresholds).
    """
    idx = pd.to_datetime(df.index)
    n = len(idx)
    p_entry = np.full(n, np.nan, dtype=np.float32)
    p_danger = np.full(n, np.nan, dtype=np.float32)
    # IMPORTANT: ExitScore depende de features "cycle_*" (estado da posição),
    # que NÃO existem no cache (df) e precisam ser calculadas durante a simulação.
    # Então aqui retornamos NaN e o simulador preenche p_exit on-the-fly apenas quando em posição.
    p_exit = np.full(n, np.nan, dtype=np.float32)
    period_id = np.full(n, -1, dtype=np.int16)

    used_any: PeriodModel | None = None

    for pid, pm in enumerate(periods):
        mask = idx > pm.train_end_utc
        mask &= ~np.isfinite(p_entry)  # só preenche onde ainda não foi preenchido
        if not mask.any():
            continue
        mask_np = np.asarray(mask, dtype=bool)
        rows = np.flatnonzero(mask_np)
        X_e = _build_matrix_rows(df, pm.entry_cols, rows)
        X_d = _build_matrix_rows(df, pm.danger_cols, rows)
        # IMPORTANT (XGBoost CUDA):
        # `inplace_predict` com booster em CUDA e input numpy (CPU) gera warning
        # "mismatched devices" e faz fallback interno. Usamos DMatrix direto.
        import xgboost as xgb  # local import para reduzir custo em ambientes sem xgb
        pe = pm.entry_model.predict(xgb.DMatrix(X_e), validate_features=False).astype(np.float32, copy=False)
        pdg = pm.danger_model.predict(xgb.DMatrix(X_d), validate_features=False).astype(np.float32, copy=False)
        pe = _apply_calibration(pe.astype(np.float64), pm.entry_calib).astype(np.float32, copy=False)
        pdg = _apply_calibration(pdg.astype(np.float64), pm.danger_calib).astype(np.float32, copy=False)
        p_entry[rows] = pe
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
        return p_entry, p_danger, p_exit, used_any, period_id
    return p_entry, p_danger, p_exit, used_any



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
    tp_hard_when_exit: bool | None = None,
    exit_min_hold_bars: int = 3,
    exit_confirm_bars: int = 1,
    exit_on_danger: bool = True,
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

    # Cache de montagem de row do ExitScore (por pid): evita alocações e pandas no hot-loop
    exit_cols_by_pid: dict[int, list[str]] = {}
    exit_cache_by_pid: dict[int, tuple[np.ndarray, np.ndarray, object, np.ndarray]] = {}
    if periods is not None:
        for pid, pmx in enumerate(periods):
            exit_cols_by_pid[int(pid)] = list(getattr(pmx, "exit_cols", None) or [])
    else:
        exit_cols_by_pid[0] = list(getattr(thresholds, "exit_cols", None) or [])

    timeout_bars = contract.timeout_bars(int(candle_sec))
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
    exit_min_hold = int(exit_min_hold_bars)
    exit_confirm = int(exit_confirm_bars)
    if exit_confirm <= 0:
        exit_confirm = 1
    exit_streak = 0
    use_numba = bool(_NUMBA_OK)

    size_sched = tuple(float(x) for x in contract.add_sizing) if contract.add_sizing else (1.0,)
    if len(size_sched) < contract.max_adds + 1:
        size_sched = size_sched + (size_sched[-1],) * (contract.max_adds + 1 - len(size_sched))

    # Reusa buffer de cycle_* (evita alocação por barra)
    cycle_vals = np.empty(10, dtype=np.float32)

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
        pdg = float(p_danger[i]) if np.isfinite(p_danger[i]) else 1.0
        pe = float(p_entry[i]) if np.isfinite(p_entry[i]) else 0.0

        if not in_pos:
            exit_streak = 0
            if (pe >= pm.tau_entry) and (pdg < pm.tau_danger):
                in_pos = True
                entry_i = i
                entry_price = px
                avg_price = px
                last_fill = px
                total_size = size_sched[0]
                num_adds = 0
            eq_curve[i] = eq
            continue

        time_in_trade = i - entry_i
        hi = high[i] if np.isfinite(high[i]) else px
        lo = low[i] if np.isfinite(low[i]) else px
        reason = None
        exit_px = None
        cur_profit = (px / avg_price - 1.0) if (avg_price > 0.0) else 0.0
        if (reason is None) and exit_on_danger and (pdg >= float(pm.tau_danger)) and (time_in_trade >= exit_min_hold):
            reason = "DANGER"
            exit_px = px

        # calcula ExitScore on-the-fly (precisa de cycle_* => depende do estado da posição)
        has_exit = (getattr(pm, "exit_model", None) is not None) and bool(getattr(pm, "exit_cols", []))
        pxit = 0.0
        # OTIMIZAÇÃO: não faz inferência de exit antes do min_hold (não pode sair)
        if has_exit and in_pos and (time_in_trade >= exit_min_hold):
            dd_pct = (px / avg_price - 1.0) if (avg_price > 0.0) else 0.0
            tp_price = avg_price * (1.0 + contract.tp_min_pct)
            sl_price = avg_price * (1.0 - contract.sl_pct)
            dist_to_tp = ((tp_price - px) / tp_price) if tp_price > 0 else 0.0
            dist_to_sl = ((px - sl_price) / sl_price) if sl_price > 0 else 0.0
            risk_used = total_size * contract.sl_pct
            next_size = size_sched[min(num_adds + 1, len(size_sched) - 1)]
            risk_if_add = (total_size + next_size) * contract.sl_pct

            pid2 = int(pid_i) if (periods is not None and period_id is not None) else 0

            cols = exit_cols_by_pid.get(pid2) or list(getattr(pm, "exit_cols", []) or [])
            cache = exit_cache_by_pid.get(pid2)
            if cache is None:
                cache = _make_exit_row_cache(df, cols)
                exit_cache_by_pid[pid2] = cache
            kind, fidx, data_cols, row_buf = cache

            cycle_vals[0] = np.float32(0.0)  # is_add
            cycle_vals[1] = np.float32(num_adds)
            cycle_vals[2] = np.float32(time_in_trade)
            cycle_vals[3] = np.float32(dd_pct)
            cycle_vals[4] = np.float32(dist_to_tp)
            cycle_vals[5] = np.float32(dist_to_sl)
            cycle_vals[6] = np.float32(avg_price)
            cycle_vals[7] = np.float32(last_fill)
            cycle_vals[8] = np.float32(risk_used)
            cycle_vals[9] = np.float32(risk_if_add)

            if _NUMBA_OK:
                if use_numba:
                    try:
                        _fill_exit_row_nb(row_buf, cycle_vals, data_cols, i, kind, fidx)  # type: ignore[misc]
                    except Exception:
                        use_numba = False
            else:
                # fallback Python (sem alocar row novo)
                for kk in range(row_buf.shape[0]):
                    t = int(kind[kk])
                    if t == 0:
                        row_buf[kk] = float(cycle_vals[int(fidx[kk])])
                    elif t == 1:
                        row_buf[kk] = float(data_cols[int(fidx[kk])][i])  # type: ignore[index]
                    else:
                        row_buf[kk] = 0.0
            if (not _NUMBA_OK) or (not use_numba):
                # fallback Python (sem alocar row novo)
                for kk in range(row_buf.shape[0]):
                    t = int(kind[kk])
                    if t == 0:
                        row_buf[kk] = float(cycle_vals[int(fidx[kk])])
                    elif t == 1:
                        row_buf[kk] = float(data_cols[int(fidx[kk])][i])  # type: ignore[index]
                    else:
                        row_buf[kk] = 0.0

            # IMPORTANT (XGBoost CUDA): evitar warning de mismatched devices
            dm = xgb.DMatrix(row_buf.reshape(1, -1))
            pxit = float(pm.exit_model.predict(dm, validate_features=False)[0])
            pxit = float(_apply_calibration_scalar(float(pxit), pm.exit_calib))
            # salva para plot/diagnóstico (opcional)
            if p_exit is not None:
                try:
                    p_exit[i] = np.float32(pxit)
                except Exception:
                    pass

        # atualiza confirmação do EXIT
        if has_exit and (time_in_trade >= exit_min_hold) and (pxit >= float(getattr(pm, "tau_exit", 1.0))):
            exit_streak += 1
        else:
            exit_streak = 0
        exit_ok = has_exit and (exit_streak >= exit_confirm) and (time_in_trade >= exit_min_hold)

        # TP hard:
        # - se existe ExitScore, por padrão desligamos TP hard (deixa o Exit decidir)
        # - se não existe ExitScore, TP hard fica ligado como antes
        if tp_hard_when_exit is None:
            tp_hard = (not has_exit)
        else:
            tp_hard = bool(tp_hard_when_exit) if has_exit else True

        if reason is None and lo <= (avg_price * (1.0 - contract.sl_pct)):
            reason = "SL"
            exit_px = avg_price * (1.0 - contract.sl_pct)
        elif reason is None and tp_hard and (hi >= (avg_price * (1.0 + contract.tp_min_pct))):
            reason = "TP"
            exit_px = avg_price * (1.0 + contract.tp_min_pct)
        elif reason is None and exit_ok:
            reason = "EXIT"
            exit_px = px
        elif reason is None and time_in_trade >= timeout_bars:
            reason = "TO"
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
                risk_after = (total_size + next_size) * contract.sl_pct
                if (
                    (pe >= pm.tau_add)
                    and (pdg < pm.tau_danger_add)
                    and (risk_after <= contract.risk_max_cycle_pct + 1e-9)
                ):
                    new_total = total_size + next_size
                    avg_price = (avg_price * total_size + trigger * next_size) / new_total
                    total_size = new_total
                    last_fill = trigger
                    num_adds += 1

        eq_curve[i] = eq

    # métricas
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
    "simulate_sniper_from_scores",
]


