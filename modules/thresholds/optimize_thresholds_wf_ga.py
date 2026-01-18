# -*- coding: utf-8 -*-
from __future__ import annotations

"""
GA (Genetic Algorithm) para calibrar thresholds (tau_entry/tau_danger/tau_exit) sem vazamento,
em walk-forward por janelas fixas.

Ideia (sem leakage):
- Dividimos o histÃ³rico em janelas consecutivas de tamanho `step_days` (ex.: 90d).
- Para cada passo i:
    - Otimiza thresholds SOMENTE na janela i (train/calibraÃ§Ã£o).
    - Avalia (out-of-sample) esses thresholds na janela i+1 (teste).
    - AvanÃ§a (i <- i+1).

ObservaÃ§Ãµes importantes:
- `ExitScore` Ã© calculado on-the-fly dentro do simulador (depende de cycle_*),
  entÃ£o o custo de simulaÃ§Ã£o domina. Para viabilizar, este script suporta:
    - `--bar_stride` para downsample (ex.: 5 => usa 1 a cada 5 candles).
    - `--symbols_per_eval` para amostrar sÃ­mbolos por avaliaÃ§Ã£o (fitness estocÃ¡stico).
    - Paralelismo em nÃ­vel de indivÃ­duos (`--jobs`).

SaÃ­das:
- CSV com thresholds e mÃ©tricas in-sample e out-of-sample por passo.
- JSON por passo em `out_dir` (opcional), Ãºtil para reaplicar thresholds na simulaÃ§Ã£o.

Executar:
  python modules/thresholds/optimize_thresholds_wf_ga.py --run-dir D:/astra/models_sniper/wf_020 --max-symbols 150
"""

import argparse
from pathlib import Path
import sys
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np

# permitir rodar como script direto (sem PYTHONPATH)
if __package__ in (None, ""):
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break
import pandas as pd

try:
    # pacote
    from backtest.sniper_walkforward import (
        load_period_models,
        predict_scores_walkforward,
        apply_threshold_overrides,
    )
    from backtest.sniper_portfolio import PortfolioConfig, SymbolData, simulate_portfolio
    from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL
    from trade_contract import DEFAULT_TRADE_CONTRACT
    from config.symbols import load_top_market_cap_symbols
    from utils.paths import resolve_generated_path
except Exception:
    # fallback para execuÃ§Ã£o direta
    import sys
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break
    from backtest.sniper_walkforward import (  # type: ignore[import]
        load_period_models,
        predict_scores_walkforward,
        apply_threshold_overrides,
    )
    from backtest.sniper_portfolio import PortfolioConfig, SymbolData, simulate_portfolio  # type: ignore[import]
    from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore[import]
    from trade_contract import DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    from config.symbols import load_top_market_cap_symbols  # type: ignore[import]
    from utils.paths import resolve_generated_path  # type: ignore[import]

try:
    # opcional: notificaÃ§Ãµes (nÃ£o pode quebrar o GA)
    from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send
except Exception:
    try:
        from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore[import]
    except Exception:
        _pushover_load_default = None
        _pushover_send = None


# ------------------------------
# Progress helper (tqdm optional)
# ------------------------------
def _progress(it: Iterable, *, total: int | None = None, desc: str = "") -> Iterable:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, total=total, desc=desc)
    except Exception:
        # fallback minimalista
        if desc:
            print(f"[prog] {desc} ...", flush=True)
        return it


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, float(x))))


def _set_thread_limits(n: int) -> int:
    """
    Limita threads internas (OpenMP/BLAS/numexpr/pyarrow) e XGBoost.

    IMPORTANT:
    - `--jobs` controla QUANTOS processos; mesmo com jobs=1 vocÃª pode ver CPU=100%
      porque XGBoost (OpenMP) e/ou BLAS podem usar vÃ¡rios cores dentro de um Ãºnico processo.
    """
    try:
        n = int(max(1, int(n)))
    except Exception:
        n = 1

    # Stacks comuns (Windows/Linux)
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        # Arrow/Parquet
        "PYARROW_NUM_THREADS",
    ):
        try:
            os.environ[k] = str(n)
        except Exception:
            pass

    # XGBoost global config (3.x): nthread=0 => usa "mÃ¡ximo"
    try:
        import xgboost as xgb  # type: ignore

        try:
            xgb.set_config(nthread=int(n))
        except Exception:
            pass
    except Exception:
        pass

    return int(n)


def _xgb_has_cuda() -> bool:
    try:
        import xgboost as xgb  # type: ignore

        bi = getattr(xgb, "build_info", None)
        if bi is None:
            return False
        info = bi()
        return bool((info or {}).get("USE_CUDA", False))
    except Exception:
        return False


# Threshold grids especÃ­ficos para cada variÃ¡vel
_ENTRY_MIN, _ENTRY_MAX = 0.70, 0.85
_DANGER_MIN, _DANGER_MAX = 0.40, 0.65
_EXIT_MIN, _EXIT_MAX = 0.60, 0.85
_TAU_STEP = 0.05


def _quantize_tau(x: float, lo: float, hi: float) -> float:
    """
    Quantiza tau para a grade [lo..hi] com passo _TAU_STEP.
    """
    step = float(_TAU_STEP)
    if not np.isfinite(x):
        return lo
    q = round(float(x) / step) * step
    q = _clamp(q, lo, hi)
    return float(round(q + 1e-12, 2))


def _read_parquet_best_effort(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame | None:
    """
    LÃª parquet tentando selecionar colunas, mas tolerando colunas ausentes (evita crash).
    """
    p = Path(path)
    cols = list(columns) if columns else None
    if cols:
        try:
            return pd.read_parquet(p, columns=cols)
        except Exception:
            # tenta interseÃ§Ã£o via schema (pyarrow)
            try:
                import pyarrow.parquet as pq  # type: ignore

                schema_cols = set(pq.ParquetFile(str(p)).schema.names)
                use = [c for c in cols if c in schema_cols]
                if use:
                    return pd.read_parquet(p, columns=use)
            except Exception:
                pass
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _read_cache_meta_times(data_path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    LÃª start/end do cache via *.meta.json (rÃ¡pido).
    Fallback para None se nÃ£o conseguir.
    """
    try:
        mp = Path(data_path).with_suffix(".meta.json")
        if not mp.exists():
            return None, None
        meta = json.loads(mp.read_text(encoding="utf-8"))
        st = meta.get("start_ts_utc")
        en = meta.get("end_ts_utc")
        if not st or not en:
            return None, None
        return pd.to_datetime(st), pd.to_datetime(en)
    except Exception:
        return None, None


@dataclass(frozen=True)
class Individual:
    tau_entry: float
    tau_danger: float
    tau_exit: float


@dataclass(frozen=True)
class FitnessWeights:
    """
    Fitness simples (como vocÃª pediu): otimiza SOMENTE 4 mÃ©tricas via mÃ©dia geomÃ©trica de margens:
    - retorno final (%)
    - profit factor (PF)
    - win rate (hit)
    - max drawdown (DD)

    Sem mÃ­nimo/mÃ¡ximo de trades, sem consistÃªncia mensal.
    """

    pf_cap: float = 8.0

    # Targets (margens >= 1.0)
    # Obs: definimos em % para vocÃª nÃ£o ter que "converter na cabeÃ§a".
    # Ex.: 30.0 = +30% no perÃ­odo.
    target_return_pct: float = 30.0  # +30% / 180d
    target_pf: float = 1.50
    target_win: float = 0.50
    target_dd: float = 0.30

    # Penaliza mais forte nesses limites "ruins"
    hard_return_pct: float = 0.0  # nÃ£o aceita retorno negativo (padrÃ£o)
    hard_pf: float = 1.00
    hard_win: float = 0.30
    hard_dd: float = 0.50

    # RegularizaÃ§Ã£o numÃ©rica e clipping (evita 0/inf)
    eps: float = 1e-9
    margin_min: float = 0.20
    # caps por mÃ©trica (evita que UMA mÃ©trica domine a GM)
    # - PF muito alto (>~2.3) costuma indicar "quase nÃ£o opera" (suspeito)
    # - win muito alto (>~0.6) idem
    # - DD muito baixo nÃ£o deve "comprar" fitness
    eq_margin_max: float = 3.00
    pf_margin_max: float = 1.55  # ~ 2.3 / 1.5
    win_margin_max: float = 1.20  # 0.60 / 0.50
    dd_margin_max: float = 1.20  # recompensa limitada por DD abaixo do alvo

    pf_suspicious_hi: float = 2.30
    win_suspicious_hi: float = 0.60
    pen_pf_hi: float = 1.25
    pen_win_hi: float = 1.00

    # Pesos das penalidades hard (multiplicativas; nÃ£o vai a -inf)
    pen_eq: float = 8.0
    pen_pf: float = 6.0
    pen_win: float = 2.0
    pen_dd: float = 6.0

    # Se True e o sÃ­mbolo nÃ£o operou, tratamos como neutro (nÃ£o derruba a mÃ©dia)
    no_trade_neutral: bool = True

    # --- NOVAS TRAVAS DE ESTABILIDADE ---
    # Alvos de quantidade de trades (total do portfÃ³lio na janela de 180 dias)
    # Para 24 sÃ­mbolos em 180 dias, ~100-800 trades Ã© um range saudÃ¡vel (0.5 a 4 trades/dia).
    target_trades_min: int = 100
    target_trades_max: int = 800
    pen_trade_count: float = 5.0

    # O tamanho real da janela (setado em runtime pelo main via worker init)
    step_days: int = 90


@dataclass
class EvalResult:
    fitness: float
    eq_end: float
    max_dd: float
    profit_factor: float
    win_rate: float
    trades: int  # trades do sÃ­mbolo (quando per-symbol) ou mÃ©dia (quando agregado)
    neg_months: int  # idem
    # Extras p/ leitura humana (agregado):
    symbols: int = 1
    trades_total: int = 0
    trades_avg: float = 0.0
    neg_month_ratio: float = 0.0


@dataclass(frozen=True)
class UniverseSelectorConfig:
    """
    Config para seleÃ§Ã£o dinÃ¢mica de universo por step, baseado em OOS acumulado por sÃ­mbolo.

    Regras (sem vazamento):
    - O universo ATIVO do step k Ã© decidido usando apenas o histÃ³rico OOS atÃ© k-1.
    - No final do step k, rodamos um "shadow OOS" com TODO o universo disponÃ­vel do step
      para atualizar mÃ©tricas por sÃ­mbolo (incluindo sÃ­mbolos inativos).
    """

    enabled: bool = True
    warmup_steps: int = 2
    ewm_decay: float = 0.85

    min_steps: int = 2
    min_trades_total: int = 10

    # Histerese (ban/unban) usando retorno normalizado e PF (OOS, por sÃ­mbolo)
    ban_ret180: float = -1.0
    ban_pf: float = 1.00
    unban_ret180: float = 0.50
    unban_pf: float = 1.05

    # Remove "nÃ£o-operÃ¡veis" (ex.: stablecoins que nunca geram trade) apÃ³s algum histÃ³rico
    ban_if_avg_trades_per_step_below: float = 0.25

    # MantÃ©m diversificaÃ§Ã£o mÃ­nima, mesmo que o critÃ©rio baniria muita coisa
    min_active: int = 20

    # Limita churn por step (estabilidade de universo)
    max_changes_per_step: int = 12


def _is_stable_symbol(sym: str) -> bool:
    s = str(sym or "").strip().upper()
    return bool(s.startswith(("USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD")))


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return
    try:
        if p.exists():
            df.to_csv(p, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(p, mode="w", header=True, index=False, encoding="utf-8")
    except Exception:
        # nunca quebrar o GA por I/O de relatÃ³rios
        return


def _load_universe_state(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_universe_state(path: Path, state: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _per_symbol_eval_from_portfolio_trades(
    trades: list[Any],
    *,
    symbols: list[str],
    weights: FitnessWeights,
) -> dict[str, EvalResult]:
    """
    ConstrÃ³i mÃ©tricas por sÃ­mbolo a partir de trades de um backtest de PORTFÃ“LIO.

    Nota: retornos por trade sÃ£o tratados como multiplicativos via (1 + weight*r_net)
    em ordem cronolÃ³gica de saÃ­da (exit_ts), para obter eq_end/max_dd por sÃ­mbolo.
    """
    by_sym: dict[str, list[Any]] = {str(s).upper(): [] for s in (symbols or [])}
    for tr in list(trades or []):
        try:
            s = str(getattr(tr, "symbol", "") or "").upper()
        except Exception:
            s = ""
        if not s:
            continue
        if s not in by_sym:
            by_sym[s] = []
        by_sym[s].append(tr)

    out: dict[str, EvalResult] = {}
    for sym in list(symbols or []):
        s = str(sym).upper()
        trs = list(by_sym.get(s, []) or [])
        try:
            trs.sort(key=lambda t: pd.to_datetime(getattr(t, "exit_ts", None)))
        except Exception:
            pass

        pnl: list[float] = []
        for tr in trs:
            w = float(getattr(tr, "weight", 0.0) or 0.0)
            r = float(getattr(tr, "r_net", 0.0) or 0.0)
            pnl.append(float(w * r))

        eq = 1.0
        peak = 1.0
        max_dd = 0.0
        eq_curve = [1.0]
        for x in pnl:
            eq *= float(max(1e-12, 1.0 + float(x)))
            peak = float(max(peak, eq))
            dd = float(1.0 - (eq / float(max(1e-12, peak))))
            if dd > max_dd:
                max_dd = dd
            eq_curve.append(eq)

        wins = float(sum(x for x in pnl if x > 0.0))
        losses = float(sum(-x for x in pnl if x < 0.0))
        pf = float(wins / max(1e-12, losses)) if losses > 0.0 else float("inf")
        win = float(sum(1 for x in pnl if x > 0.0) / max(1, len(pnl)))

        class _R:
            pass

        r0 = _R()
        r0.equity_curve = np.asarray(eq_curve, dtype=np.float64)
        r0.max_dd = float(max_dd)
        r0.profit_factor = float(pf)
        r0.win_rate = float(win)
        r0.trades = trs
        r0.monthly_returns = []
        ev = _score_result(r0, weights, symbol=s)
        out[s] = ev
    return out


def _compute_symbol_ewm(history: pd.DataFrame, *, decay: float) -> dict[str, dict[str, float]]:
    """
    Agrega histÃ³rico por sÃ­mbolo usando EWM simples (decay).

    Espera colunas (quando disponÃ­veis):
    - fitness
    - ret180_pct
    - profit_factor
    - win_rate
    - max_dd
    - trades
    - step_idx
    """
    if history is None or history.empty:
        return {}

    cols = set(str(c) for c in history.columns)
    need = {"symbol", "step_idx"}
    if not need.issubset(cols):
        return {}

    d = float(_clamp(float(decay), 0.0, 0.999))
    a = float(1.0 - d)

    out: dict[str, dict[str, float]] = {}
    try:
        hist = history.copy()
        hist["symbol"] = [str(x).upper() for x in hist["symbol"].tolist()]
        hist = hist.sort_values(["symbol", "step_idx"], ascending=[True, True])
    except Exception:
        hist = history

    for sym, g in hist.groupby("symbol", sort=False):
        symu = str(sym).upper()
        g2 = g
        steps = int(len(g2))
        trades_total = 0.0
        # EWM de mÃ©tricas-chave (fallback = NaN)
        ew_f = None
        ew_r = None
        ew_pf = None
        ew_wr = None
        ew_dd = None
        for _, row in g2.iterrows():
            try:
                tr = float(row.get("trades", 0.0) or 0.0)
            except Exception:
                tr = 0.0
            trades_total += float(max(0.0, tr))

            def _upd(prev, val):
                try:
                    v = float(val)
                except Exception:
                    return prev
                if not np.isfinite(v):
                    return prev
                if prev is None or (not np.isfinite(float(prev))):
                    return float(v)
                return float(d * float(prev) + a * float(v))

            ew_f = _upd(ew_f, row.get("fitness", np.nan))
            ew_r = _upd(ew_r, row.get("ret180_pct", np.nan))
            ew_pf = _upd(ew_pf, row.get("profit_factor", np.nan))
            ew_wr = _upd(ew_wr, row.get("win_rate", np.nan))
            ew_dd = _upd(ew_dd, row.get("max_dd", np.nan))

        avg_trades = float(trades_total) / float(max(1, steps))
        out[symu] = {
            "steps": float(steps),
            "trades_total": float(trades_total),
            "avg_trades": float(avg_trades),
            "ewm_fitness": float(ew_f) if ew_f is not None else float("nan"),
            "ewm_ret180": float(ew_r) if ew_r is not None else float("nan"),
            "ewm_pf": float(ew_pf) if ew_pf is not None else float("nan"),
            "ewm_win": float(ew_wr) if ew_wr is not None else float("nan"),
            "ewm_dd": float(ew_dd) if ew_dd is not None else float("nan"),
        }
    return out


def _select_active_symbols_for_step(
    *,
    step_idx: int,
    syms_available: list[str],
    history: pd.DataFrame,
    state_prev: dict[str, Any],
    cfg: UniverseSelectorConfig,
) -> tuple[list[str], dict[str, str]]:
    """
    Decide universo ativo do step (somente com base no histÃ³rico OOS anterior).

    Retorna (active_syms, reason_by_symbol) para logging/debug.
    """
    av = [str(s).upper() for s in list(syms_available or []) if str(s).strip()]
    av_set = set(av)
    if not cfg.enabled:
        return sorted(av_set), {}

    if int(step_idx) < int(max(0, cfg.warmup_steps)):
        return sorted(av_set), {s: "warmup" for s in av_set}

    feats = _compute_symbol_ewm(history, decay=float(cfg.ewm_decay))

    prev_active = [str(x).upper() for x in (state_prev.get("active_symbols") or [])]
    prev_banned = {str(x).upper() for x in (state_prev.get("banned_symbols") or [])}

    # MantÃ©m coerÃªncia com o universo disponÃ­vel no step
    prev_active = [s for s in prev_active if s in av_set]
    prev_banned = {s for s in prev_banned if s in av_set}

    # Se nÃ£o havia estado (primeiro step pÃ³s-warmup), comeÃ§a com tudo disponÃ­vel
    if not prev_active:
        prev_active = sorted(av_set)

    reasons: dict[str, str] = {}
    desired_active = set(prev_active)
    desired_banned = set(prev_banned)

    def _ok_min_hist(symu: str) -> bool:
        f = feats.get(symu) or {}
        steps = int(f.get("steps", 0.0) or 0.0)
        trades_total = float(f.get("trades_total", 0.0) or 0.0)
        return (steps >= int(cfg.min_steps)) and (trades_total >= float(cfg.min_trades_total))

    # Aplica ban/unban com histerese (ret180 + pf) + regra de "nÃ£o opera"
    for s in sorted(av_set):
        f = feats.get(s) or {}
        steps = int(f.get("steps", 0.0) or 0.0)
        avg_tr = float(f.get("avg_trades", 0.0) or 0.0)
        r180 = float(f.get("ewm_ret180", float("nan")))
        pf = float(f.get("ewm_pf", float("nan")))

        # Ainda sem histÃ³rico suficiente: mantÃ©m como estava (e se nÃ£o estava, ativa por padrÃ£o)
        if steps < int(cfg.min_steps):
            if s in desired_banned:
                reasons[s] = "ban_keep_insufficient_history"
            else:
                desired_active.add(s)
                reasons[s] = "active_insufficient_history"
            continue

        # Ban por "nÃ£o opera"
        if avg_tr < float(cfg.ban_if_avg_trades_per_step_below):
            # apÃ³s min_steps, se nÃ£o gera trade, Ã© candidato a ban (stable coins etc.)
            desired_banned.add(s)
            desired_active.discard(s)
            reasons[s] = f"ban_low_activity(avg_trades={avg_tr:.2f})"
            continue

        # Ban (exige histÃ³rico mÃ­nimo e trades mÃ­nimos)
        if _ok_min_hist(s) and np.isfinite(r180) and np.isfinite(pf):
            if (r180 <= float(cfg.ban_ret180)) and (pf <= float(cfg.ban_pf)):
                desired_banned.add(s)
                desired_active.discard(s)
                reasons[s] = f"ban(ret180={r180:+.2f} pf={pf:.2f})"
                continue

        # Unban
        if s in desired_banned and _ok_min_hist(s) and np.isfinite(r180) and np.isfinite(pf):
            if (r180 >= float(cfg.unban_ret180)) and (pf >= float(cfg.unban_pf)):
                desired_banned.discard(s)
                desired_active.add(s)
                reasons[s] = f"unban(ret180={r180:+.2f} pf={pf:.2f})"
                continue

        # default: mantÃ©m estado
        if s in desired_banned:
            reasons[s] = "ban_keep"
        else:
            desired_active.add(s)
            reasons[s] = "active_keep"

    # ImpÃµe churn mÃ¡ximo por step (remove piores / adiciona melhores)
    prev_active_set = set(prev_active)
    removed = sorted((prev_active_set - desired_active))
    added = sorted((desired_active - prev_active_set))
    maxchg = int(max(0, cfg.max_changes_per_step))
    if maxchg > 0:
        # ranking por ewm_ret180 (fallback = -inf)
        def _rank(symu: str) -> float:
            try:
                v = float((feats.get(symu) or {}).get("ewm_ret180", float("-inf")))
            except Exception:
                v = float("-inf")
            if not np.isfinite(v):
                v = float("-inf")
            return v

        if len(removed) > maxchg:
            removed_sorted = sorted(removed, key=_rank)  # remove os piores primeiro
            keep_removed = set(removed_sorted[:maxchg])
            # desfaz remoÃ§Ãµes alÃ©m do cap
            for s in removed:
                if s not in keep_removed:
                    desired_active.add(s)
                    desired_banned.discard(s)
                    reasons[s] = "keep_due_to_churn_cap"
        if len(added) > maxchg:
            added_sorted = sorted(added, key=_rank, reverse=True)  # adiciona os melhores primeiro
            keep_added = set(added_sorted[:maxchg])
            for s in added:
                if s not in keep_added:
                    desired_active.discard(s)
                    reasons[s] = "skip_add_due_to_churn_cap"

    # DiversificaÃ§Ã£o mÃ­nima
    active = sorted(set(desired_active) & av_set)
    if int(cfg.min_active) > 0 and len(active) < int(cfg.min_active):
        feats = feats or {}
        def _rank2(symu: str) -> float:
            try:
                r = float((feats.get(symu) or {}).get("ewm_ret180", float("-inf")))
            except Exception:
                r = float("-inf")
            if not np.isfinite(r):
                r = float("-inf")
            # stable/no-trade tende a ficar com -inf / NaN e ir pro fim naturalmente
            return r

        pool = sorted(av_set, key=_rank2, reverse=True)
        active_set = set(active)
        for s in pool:
            if len(active_set) >= int(cfg.min_active):
                break
            active_set.add(s)
            reasons[s] = "added_to_meet_min_active"
        active = sorted(active_set)

    return active, reasons


def _max_true_streak(flags: list[bool]) -> int:
    best = 0
    cur = 0
    for f in flags:
        if f:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _ret_pct(eq_end: float) -> float:
    """
    Converte equity (ex.: 1.07) -> retorno em % (ex.: 7.0).
    """
    try:
        return (float(eq_end) - 1.0) * 100.0
    except Exception:
        return 0.0


def _ret_pct_per_180d(eq_end: float, window_days: int) -> float:
    """
    Retorno normalizado para uma janela equivalente de 180d.

    Ãštil porque no expanding window o IN (treino) cresce (360d, 540d, ...),
    enquanto o OOS continua 180d.
    """
    try:
        d = float(max(1, int(window_days)))
        eq = float(eq_end)
        # (eq^(180/d) - 1) * 100
        return (float(np.power(max(1e-12, eq), 180.0 / d)) - 1.0) * 100.0
    except Exception:
        return 0.0


def _normalize_weights_dict(weights: dict[str, Any]) -> dict[str, Any]:
    """
    Compatibilidade entre versÃµes:
    - versÃµes antigas usavam `target_eq`/`hard_eq` (equity multiplicativa)
    - versÃµes novas usam `target_return_pct`/`hard_return_pct`
    """
    w: dict[str, Any] = dict(weights or {})

    # target_eq -> target_return_pct
    if ("target_eq" in w) and ("target_return_pct" not in w):
        try:
            w["target_return_pct"] = (float(w.pop("target_eq")) - 1.0) * 100.0
        except Exception:
            w.pop("target_eq", None)

    # hard_eq -> hard_return_pct
    if ("hard_eq" in w) and ("hard_return_pct" not in w):
        try:
            w["hard_return_pct"] = (float(w.pop("hard_eq")) - 1.0) * 100.0
        except Exception:
            w.pop("hard_eq", None)

    # Remove chaves desconhecidas (evita crash quando rodar com caches/versÃµes misturadas)
    try:
        from dataclasses import fields

        allowed = {f.name for f in fields(FitnessWeights)}
        w = {k: v for (k, v) in w.items() if k in allowed}
    except Exception:
        pass

    return w


def _score_result(res, weights: FitnessWeights, *, symbol: str | None = None) -> EvalResult:
    eq_curve = getattr(res, "equity_curve", None)
    if eq_curve is None or len(eq_curve) == 0:
        eq_end = 1.0
    else:
        eq_end = float(eq_curve[-1])
    max_dd = float(getattr(res, "max_dd", 0.0) or 0.0)
    pf = float(getattr(res, "profit_factor", 0.0) or 0.0)
    if not np.isfinite(pf):
        pf = float(getattr(weights, "pf_cap", 10.0) or 10.0)
    pf_cap = float(getattr(weights, "pf_cap", 10.0) or 10.0)
    pf = float(min(pf_cap, max(0.0, pf)))
    win_rate = float(getattr(res, "win_rate", 0.0) or 0.0)
    trades = int(len(getattr(res, "trades", []) or []))
    # -------- fitness geomÃ©trico (margens vs targets) --------
    eps = float(getattr(weights, "eps", 1e-9) or 1e-9)
    # alvo de retorno em %, mas a conta continua multiplicativa (equity) para ser bem-comportada.
    # Como agora o treino pode ser "expanding", escalamos o alvo por duraÃ§Ã£o (base=180d).
    t_ret_pct_180 = float(getattr(weights, "target_return_pct", 30.0) or 30.0)
    w_days = float(getattr(weights, "step_days", 180) or 180)
    # compounding: target_eq = (1 + r_180)^(days/180)
    t_eq = float(np.power(1.0 + (t_ret_pct_180 / 100.0), max(0.25, w_days / 180.0)))
    t_pf = float(getattr(weights, "target_pf", 1.5) or 1.5)
    t_win = float(getattr(weights, "target_win", 0.5) or 0.5)
    t_dd = float(getattr(weights, "target_dd", 0.30) or 0.30)

    # neutro para stablecoins sem trades (nÃ£o derruba mÃ©dia), mas NÃƒO vira Ã³timo.
    if int(trades) <= 0 and bool(getattr(weights, "no_trade_neutral", True)):
        symu = str(symbol or "").strip().upper()
        is_stable = symu.startswith(("USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD")) if symu else False
        if is_stable:
            return EvalResult(
                fitness=0.70,
                eq_end=1.0,
                max_dd=0.0,
                profit_factor=0.0,
                win_rate=0.0,
                trades=0,
                neg_months=0,
                symbols=1,
                trades_total=0,
                trades_avg=0.0,
                neg_month_ratio=0.0,
            )

    # Margens vs targets:
    # - reward acima do target, mas com caps por mÃ©trica
    # - DD recompensa abaixo do target, mas tambÃ©m com cap (evita dominar)
    mmin = float(getattr(weights, "margin_min", 0.2) or 0.2)
    eq_mmax = float(getattr(weights, "eq_margin_max", 3.0) or 3.0)
    pf_mmax = float(getattr(weights, "pf_margin_max", 1.55) or 1.55)
    win_mmax = float(getattr(weights, "win_margin_max", 1.20) or 1.20)
    dd_mmax = float(getattr(weights, "dd_margin_max", 1.20) or 1.20)

    meq = float(eq_end) / float(max(eps, float(t_eq)))
    mpf = float(pf) / float(max(eps, float(t_pf)))
    mwin = float(win_rate) / float(max(eps, float(t_win)))
    mdd = float(t_dd) / float(max(eps, float(max_dd)))

    meq = float(_clamp(meq, mmin, eq_mmax))
    mpf = float(_clamp(mpf, mmin, pf_mmax))
    mwin = float(_clamp(mwin, mmin, win_mmax))
    mdd = float(_clamp(mdd, mmin, dd_mmax))

    gm = float(np.exp((np.log(meq) + np.log(mpf) + np.log(mwin) + np.log(mdd)) / 4.0))

    # penalidades hard (multiplicativas), sem ir a -inf
    pen = 1.0
    h_ret_pct_180 = float(getattr(weights, "hard_return_pct", 0.0) or 0.0)
    h_eq = float(np.power(1.0 + (h_ret_pct_180 / 100.0), max(0.25, w_days / 180.0)))
    h_pf = float(getattr(weights, "hard_pf", 1.0) or 1.0)
    h_win = float(getattr(weights, "hard_win", 0.30) or 0.30)
    h_dd = float(getattr(weights, "hard_dd", 0.50) or 0.50)

    if float(eq_end) < h_eq:
        k = float(getattr(weights, "pen_eq", 8.0) or 8.0)
        pen *= float(np.exp(-k * (h_eq - float(eq_end))))
    if float(pf) < h_pf:
        k = float(getattr(weights, "pen_pf", 6.0) or 6.0)
        pen *= float(np.exp(-k * (h_pf - float(pf))))
    if float(win_rate) < h_win:
        k = float(getattr(weights, "pen_win", 2.0) or 2.0)
        pen *= float(np.exp(-k * (h_win - float(win_rate))))
    if float(max_dd) > h_dd:
        k = float(getattr(weights, "pen_dd", 6.0) or 6.0)
        pen *= float(np.exp(-k * (float(max_dd) - h_dd)))

    # penalidades "suspeitas": PF/win bons demais normalmente = poucas operaÃ§Ãµes / regime muito especÃ­fico
    try:
        pf_hi = float(getattr(weights, "pf_suspicious_hi", 2.30) or 2.30)
        if float(pf) > pf_hi:
            k = float(getattr(weights, "pen_pf_hi", 1.25) or 1.25)
            pen *= float(np.exp(-k * (float(pf) - pf_hi)))
    except Exception:
        pass
    try:
        win_hi = float(getattr(weights, "win_suspicious_hi", 0.60) or 0.60)
        if float(win_rate) > win_hi:
            k = float(getattr(weights, "pen_win_hi", 1.00) or 1.00)
            pen *= float(np.exp(-k * (float(win_rate) - win_hi)))
    except Exception:
        pass

    fitness = float(gm * pen)

    # --- PENALIDADE DE QUANTIDADE DE TRADES ---
    # Se operar demais ou de menos, derruba o fitness para evitar extremos e overfitting ao ruÃ­do.
    # IMPORTANTE: com janela "expanding", trades crescem ~linear com os dias.
    # Escalamos os targets (definidos por ~180d) proporcionalmente ao tamanho da janela avaliada.
    base_min = float(getattr(weights, "target_trades_min", 100))
    base_max = float(getattr(weights, "target_trades_max", 800))
    step_days = float(getattr(weights, "step_days", 180) or 180)
    scale = float(max(0.25, step_days / 180.0))
    t_min = float(base_min * scale)
    t_max = float(base_max * scale)
    k_pen = float(getattr(weights, "pen_trade_count", 5.0))

    if trades > 0:
        if trades < t_min:
            # Penaliza falta de amostragem estatÃ­stica
            fitness *= float(np.exp(-k_pen * (t_min - trades) / t_min))
        elif trades > t_max:
            # Penaliza overtrading (provÃ¡vel overfitting ao ruÃ­do de baixa confianÃ§a)
            fitness *= float(np.exp(-k_pen * (trades - t_max) / t_max))

    # mÃ©tricas mensais continuam sendo reportadas, mas nÃ£o entram no fitness
    monthly_raw = getattr(res, "monthly_returns", None) or []
    monthly = [float(x) for x in list(monthly_raw)]
    neg_months = int(sum(1 for r in monthly if float(r) < 0.0))
    neg_ratio = float(neg_months) / float(max(1, len(monthly)))
    return EvalResult(
        fitness=float(fitness),
        eq_end=float(eq_end),
        max_dd=max_dd,
        profit_factor=float(pf),
        win_rate=win_rate,
        trades=trades,
        neg_months=neg_months,
        symbols=1,
        trades_total=int(trades),
        trades_avg=float(trades),
        neg_month_ratio=float(neg_ratio),
    )


def _aggregate_eval(results: list[EvalResult]) -> EvalResult:
    if not results:
        return EvalResult(
            fitness=-1e9,
            eq_end=1.0,
            max_dd=1.0,
            profit_factor=0.0,
            win_rate=0.0,
            trades=0,
            neg_months=999,
            symbols=0,
            trades_total=0,
            trades_avg=0.0,
            neg_month_ratio=1.0,
        )
    # mÃ©dia simples (o objetivo Ã© a MÃ‰DIA do universo)
    n = int(len(results))
    fitness = float(np.mean([r.fitness for r in results]))
    eq_end = float(np.mean([r.eq_end for r in results]))
    max_dd = float(np.mean([r.max_dd for r in results]))
    pf = float(np.mean([r.profit_factor for r in results]))
    wr = float(np.mean([r.win_rate for r in results]))
    trades_avg = float(np.mean([r.trades for r in results]))
    trades_total = int(sum(int(r.trades) for r in results))
    neg_months = int(np.mean([r.neg_months for r in results]))
    neg_ratio = float(np.mean([float(getattr(r, "neg_month_ratio", 0.0) or 0.0) for r in results]))
    return EvalResult(
        fitness=fitness,
        eq_end=eq_end,
        max_dd=max_dd,
        profit_factor=pf,
        win_rate=wr,
        trades=int(trades_avg),
        neg_months=neg_months,
        symbols=n,
        trades_total=trades_total,
        trades_avg=trades_avg,
        neg_month_ratio=neg_ratio,
    )


# ------------------------------
# Dataset preparation per step
# ------------------------------
def _downsample_df(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    s = int(stride)
    if s <= 1:
        return df
    return df.iloc[::s].copy()


def _needed_feature_columns(periods) -> list[str]:
    cols: set[str] = {"open", "high", "low", "close", "volume"}
    for pm in periods:
        cols.update([str(c) for c in (pm.entry_cols or [])])
        cols.update([str(c) for c in (pm.danger_cols or [])])
        cols.update([str(c) for c in (getattr(pm, "exit_cols", None) or [])])
    # o simulador lÃª df[col].iat[i]; se a coluna nÃ£o existe, vira 0.0 (mas piora o exit)
    return sorted(cols)


def _prepare_symbol_frame(
    df_full: pd.DataFrame,
    *,
    periods,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    bar_stride: int,
) -> pd.DataFrame:
    idx = pd.to_datetime(df_full.index)
    m = (idx >= t_start) & (idx <= t_end)
    df = df_full.loc[m]
    if df.empty:
        return df
    df = _downsample_df(df, int(bar_stride))
    if df.empty:
        return df
    try:
        pe_map, pdg, pex, used, pid = predict_scores_walkforward(df, periods=periods, return_period_id=True)
        pe = pe_map.get('mid') if isinstance(pe_map, dict) else pe_map
    except RuntimeError:
        # range de timestamps pode estar antes do primeiro perÃ­odo do wf_* (sem modelo "vÃ¡lido").
        # NÃ£o deve derrubar o GA; apenas ignora este sÃ­mbolo neste step.
        return df.iloc[0:0].copy()
    # anexar como colunas (facilita slicing em worker)
    df = df.copy()
    df["__p_entry"] = np.asarray(pe, dtype=np.float32)
    df["__p_danger"] = np.asarray(pdg, dtype=np.float32)
    df["__period_id"] = np.asarray(pid, dtype=np.int16)
    # OtimizaÃ§Ã£o: NÃƒO guardar __p_exit (GA nÃ£o usa); reduz tamanho de parquet e I/O.

    # OtimizaÃ§Ã£o: apÃ³s prediÃ§Ã£o, podemos descartar colunas de entry/danger (sÃ³ eram necessÃ¡rias pra predizer).
    # Mantemos OHLCV + colunas do Exit (nÃ£o-cycle) + colunas internas.
    keep: set[str] = {"open", "high", "low", "close", "volume", "__p_entry", "__p_danger", "__period_id"}
    for pm in periods:
        for c in list(getattr(pm, "exit_cols", None) or []):
            cc = str(c)
            if cc.startswith("cycle_"):
                continue
            keep.add(cc)
    df = df[[c for c in df.columns if c in keep]].copy()
    return df


# ------------------------------
# GA core (simples e eficiente)
# ------------------------------
def _init_population(rng: np.random.Generator, pop_size: int, seed_best: Individual | None) -> list[Individual]:
    out: list[Individual] = []
    for i in range(int(pop_size)):
        if seed_best is not None and i < max(1, int(pop_size) // 5):
            # perturbaÃ§Ã£o pequena ao redor do melhor anterior (warm start)
            te = _quantize_tau(seed_best.tau_entry + float(rng.normal(0, 0.05)), _ENTRY_MIN, _ENTRY_MAX)
            td = _quantize_tau(seed_best.tau_danger + float(rng.normal(0, 0.05)), _DANGER_MIN, _DANGER_MAX)
            tx = _quantize_tau(seed_best.tau_exit + float(rng.normal(0, 0.05)), _EXIT_MIN, _EXIT_MAX)
        else:
            # amostra aleatÃ³ria dentro dos limites especÃ­ficos
            te = rng.uniform(_ENTRY_MIN, _ENTRY_MAX)
            td = rng.uniform(_DANGER_MIN, _DANGER_MAX)
            tx = rng.uniform(_EXIT_MIN, _EXIT_MAX)
        
        out.append(Individual(
            tau_entry=_quantize_tau(te, _ENTRY_MIN, _ENTRY_MAX), 
            tau_danger=_quantize_tau(td, _DANGER_MIN, _DANGER_MAX), 
            tau_exit=_quantize_tau(tx, _EXIT_MIN, _EXIT_MAX)
        ))
    return out


def _tournament_select(rng: np.random.Generator, scored: list[tuple[Individual, EvalResult]], k: int = 3) -> Individual:
    best: tuple[Individual, EvalResult] | None = None
    for _ in range(int(k)):
        cand = scored[int(rng.integers(0, len(scored)))]
        if best is None or cand[1].fitness > best[1].fitness:
            best = cand
    assert best is not None
    return best[0]


def _crossover(rng: np.random.Generator, a: Individual, b: Individual, p: float = 0.7) -> tuple[Individual, Individual]:
    if float(rng.random()) > float(p):
        return a, b
    # blend crossover
    alpha = float(rng.uniform(0.2, 0.8))
    c1 = Individual(
        tau_entry=_quantize_tau(alpha * a.tau_entry + (1 - alpha) * b.tau_entry, _ENTRY_MIN, _ENTRY_MAX),
        tau_danger=_quantize_tau(alpha * a.tau_danger + (1 - alpha) * b.tau_danger, _DANGER_MIN, _DANGER_MAX),
        tau_exit=_quantize_tau(alpha * a.tau_exit + (1 - alpha) * b.tau_exit, _EXIT_MIN, _EXIT_MAX),
    )
    c2 = Individual(
        tau_entry=_quantize_tau(alpha * b.tau_entry + (1 - alpha) * a.tau_entry, _ENTRY_MIN, _ENTRY_MAX),
        tau_danger=_quantize_tau(alpha * b.tau_danger + (1 - alpha) * a.tau_danger, _DANGER_MIN, _DANGER_MAX),
        tau_exit=_quantize_tau(alpha * b.tau_exit + (1 - alpha) * a.tau_exit, _EXIT_MIN, _EXIT_MAX),
    )
    return c1, c2


def _mutate(rng: np.random.Generator, ind: Individual, p: float = 0.25) -> Individual:
    if float(rng.random()) > float(p):
        return ind
    te = _quantize_tau(ind.tau_entry + float(rng.normal(0, 0.06)), _ENTRY_MIN, _ENTRY_MAX)
    td = _quantize_tau(ind.tau_danger + float(rng.normal(0, 0.06)), _DANGER_MIN, _DANGER_MAX)
    tx = _quantize_tau(ind.tau_exit + float(rng.normal(0, 0.06)), _EXIT_MIN, _EXIT_MAX)
    return Individual(tau_entry=te, tau_danger=td, tau_exit=tx)


# ------------------------------
# Worker pool (Windows-friendly)
# ------------------------------
_W_RUN_DIR: Path | None = None
_W_PERIODS_SIM = None
_W_WEIGHTS: FitnessWeights | None = None
_W_SYMBOL_PATHS: dict[str, Path] | None = None
_W_DF_CACHE: dict[str, pd.DataFrame] | None = None
_W_DF_CACHE_LRU: list[str] | None = None
_W_DF_CACHE_SIZE: int = 0
_W_EVAL_SYMS: list[str] | None = None
_W_TRAIN_START: pd.Timestamp | None = None
_W_TRAIN_END: pd.Timestamp | None = None
_W_SYMBOLS_PER_EVAL: int = 0
_W_SEED: int = 42
_W_PORTF_CFG: PortfolioConfig | None = None


@dataclass(frozen=True)
class SimPeriod:
    """
    PerÃ­odo mÃ­nimo para simulaÃ§Ã£o.

    IMPORTANT: NÃƒO carrega entry_model/danger_model no worker (economiza RAM).
    O simulador sÃ³ precisa de taus + exit_model (para pxit on-the-fly) e calib/cols.
    """

    period_days: int
    train_end_utc: pd.Timestamp
    exit_model: Any
    exit_cols: list[str]
    exit_calib: dict
    tau_entry: float
    tau_danger: float
    tau_exit: float
    tau_add: float
    tau_danger_add: float


def _load_xgb_booster_best_effort(path_json: Path) -> Any:
    """
    Carrega Booster preferindo .ubj (menor/mais rÃ¡pido), com fallback para .json.
    """
    import xgboost as xgb  # type: ignore

    bst = xgb.Booster()
    ubj = Path(path_json).with_suffix(".ubj")
    if ubj.exists() and ubj.stat().st_size > 0:
        bst.load_model(str(ubj))
        return bst
    bst.load_model(str(path_json))
    return bst


def _load_periods_for_sim(run_dir: Path) -> list[SimPeriod]:
    """
    VersÃ£o leve do load_period_models: carrega APENAS o necessÃ¡rio para simular.

    - lÃª meta.json
    - carrega exit_model (se existir)
    - NÃƒO carrega entry/danger models (jÃ¡ usamos __p_entry/__p_danger prÃ©-computados no dataset)
    """
    periods: list[SimPeriod] = []
    run_dir = Path(run_dir)
    for pd_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("period_") and p.name.endswith("d")]):
        try:
            period_days = int(pd_dir.name.replace("period_", "").replace("d", ""))
        except Exception:
            continue
        try:
            meta = json.loads((pd_dir / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        train_end = meta.get("train_end_utc")
        if not train_end:
            continue
        train_end_ts = pd.to_datetime(train_end)

        # thresholds
        try:
            tau_entry = float(meta["entry"].get("threshold", 0.5))
            tau_danger = float(meta["danger"].get("threshold", 0.5))
        except Exception:
            tau_entry = float(meta.get("tau_entry", 0.5) or 0.5)
            tau_danger = float(meta.get("tau_danger", 0.5) or 0.5)
        tau_exit = float((meta.get("exit") or {}).get("threshold", 1.0))

        # exit model (opcional)
        exit_cols = list((meta.get("exit") or {}).get("feature_cols") or [])
        exit_calib = dict((meta.get("exit") or {}).get("calibration") or {"type": "identity"})
        exit_path = pd_dir / "exit_model" / "model_exit.json"
        exit_model = None
        if exit_path.exists() and exit_cols:
            try:
                exit_model = _load_xgb_booster_best_effort(exit_path)
            except Exception:
                exit_model = None

        tau_add = float(min(0.99, max(0.01, tau_entry * 1.10)))
        tau_danger_add = float(min(0.99, max(0.01, tau_danger * 0.90)))

        periods.append(
            SimPeriod(
                period_days=int(period_days),
                train_end_utc=train_end_ts,
                exit_model=exit_model,
                exit_cols=list(exit_cols),
                exit_calib=dict(exit_calib),
                tau_entry=float(tau_entry),
                tau_danger=float(tau_danger),
                tau_exit=float(tau_exit),
                tau_add=float(tau_add),
                tau_danger_add=float(tau_danger_add),
            )
        )

    # mesma ordenaÃ§Ã£o do sniper_walkforward.load_period_models (mais recente primeiro)
    periods.sort(key=lambda p: p.train_end_utc, reverse=True)
    return periods


def _apply_threshold_overrides_sim(
    periods: list[SimPeriod],
    *,
    tau_entry: float,
    tau_danger: float,
    tau_exit: float,
    tau_add_multiplier: float = 1.10,
    tau_danger_add_multiplier: float = 0.90,
) -> list[SimPeriod]:
    out: list[SimPeriod] = []
    te = float(tau_entry)
    td = float(tau_danger)
    tx = float(tau_exit)
    ta = float(min(0.99, max(0.01, te * float(tau_add_multiplier))))
    tda = float(min(0.99, max(0.01, td * float(tau_danger_add_multiplier))))
    for pm in periods:
        out.append(
            SimPeriod(
                period_days=int(pm.period_days),
                train_end_utc=pd.to_datetime(pm.train_end_utc),
                exit_model=pm.exit_model,
                exit_cols=list(pm.exit_cols),
                exit_calib=dict(pm.exit_calib),
                tau_entry=float(te),
                tau_danger=float(td),
                tau_exit=float(tx),
                tau_add=float(ta),
                tau_danger_add=float(tda),
            )
        )
    return out


def _df_cache_get(sym: str) -> pd.DataFrame | None:
    global _W_SYMBOL_PATHS, _W_DF_CACHE, _W_DF_CACHE_LRU, _W_DF_CACHE_SIZE
    assert _W_SYMBOL_PATHS is not None and _W_DF_CACHE is not None and _W_DF_CACHE_LRU is not None
    p = _W_SYMBOL_PATHS.get(sym)
    if p is None:
        return None
    if sym in _W_DF_CACHE:
        # move para o fim (LRU)
        try:
            _W_DF_CACHE_LRU.remove(sym)
        except ValueError:
            pass
        _W_DF_CACHE_LRU.append(sym)
        return _W_DF_CACHE[sym]
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    # garante DatetimeIndex uma Ãºnica vez (evita pd.to_datetime em cada avaliaÃ§Ã£o)
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    if int(_W_DF_CACHE_SIZE) > 0:
        _W_DF_CACHE[sym] = df
        _W_DF_CACHE_LRU.append(sym)
        while len(_W_DF_CACHE_LRU) > int(_W_DF_CACHE_SIZE):
            old = _W_DF_CACHE_LRU.pop(0)
            _W_DF_CACHE.pop(old, None)
    return df


def _worker_init(
    run_dir: str,
    dataset_dir: str,
    train_start_ns: int,
    train_end_ns: int,
    symbols_per_eval: int,
    weights: dict[str, Any],
    seed: int,
    df_cache_size: int,
    xgb_threads: int,
    xgb_device: str,
    allowed_symbols_csv: str,
) -> None:
    global _W_RUN_DIR, _W_PERIODS_SIM, _W_WEIGHTS, _W_SYMBOL_PATHS, _W_DF_CACHE, _W_DF_CACHE_LRU, _W_DF_CACHE_SIZE, _W_EVAL_SYMS
    global _W_TRAIN_START, _W_TRAIN_END, _W_SYMBOLS_PER_EVAL, _W_SEED
    global _W_PORTF_CFG
    _W_RUN_DIR = Path(run_dir).expanduser().resolve()
    # Limita threads internas dentro do worker para evitar oversubscription.
    xgb_threads = _set_thread_limits(int(xgb_threads))

    # Carrega perÃ­odos leves (sem entry/danger models) para evitar "bad allocation".
    _W_PERIODS_SIM = _load_periods_for_sim(_W_RUN_DIR)
    # reforÃ§a nthread no Booster do exit (alguns modelos vÃªm com nthread=0)
    try:
        import xgboost as xgb  # type: ignore

        dev = str(xgb_device or "cpu").strip().lower()
        if dev not in {"cpu", "cuda"}:
            dev = "cpu"
        # safety: nÃ£o tentar cuda se nÃ£o tiver build com cuda
        if dev == "cuda" and (not _xgb_has_cuda()):
            dev = "cpu"

        for pm in (_W_PERIODS_SIM or []):
            bst = getattr(pm, "exit_model", None)
            if bst is not None:
                try:
                    bst.set_param({"nthread": int(xgb_threads), "device": dev})
                except Exception:
                    pass
    except Exception:
        pass
    # Compatibilidade: pode receber dicionÃ¡rio de uma versÃ£o anterior do GA
    # (por exemplo se o processo principal ficou carregado com cÃ³digo antigo e os workers spawnaram com cÃ³digo novo).
    _W_WEIGHTS = FitnessWeights(**_normalize_weights_dict(weights))
    _W_TRAIN_START = pd.to_datetime(int(train_start_ns))
    _W_TRAIN_END = pd.to_datetime(int(train_end_ns))
    _W_SYMBOLS_PER_EVAL = int(symbols_per_eval)
    _W_SEED = int(seed)
    # Config do portfÃ³lio (fixo, estilo demo)
    try:
        _W_PORTF_CFG = PortfolioConfig(
            max_positions=20,
            total_exposure=0.75,
            max_trade_exposure=0.10,
            min_trade_exposure=0.03,
            exit_min_hold_bars=3,
            exit_confirm_bars=2,
        )
    except Exception:
        _W_PORTF_CFG = PortfolioConfig()

    ddir = Path(dataset_dir).expanduser().resolve()
    allowed = {s.strip().upper() for s in str(allowed_symbols_csv or "").split(",") if s.strip()}
    paths: dict[str, Path] = {}
    for p in ddir.glob("*.parquet"):
        sym = p.stem.upper()
        if allowed and sym not in allowed:
            continue
        paths[sym] = p
    _W_SYMBOL_PATHS = paths
    _W_DF_CACHE = {}
    _W_DF_CACHE_LRU = []
    _W_DF_CACHE_SIZE = int(df_cache_size)

    # IMPORTANT: escolhe subconjunto de sÃ­mbolos UMA VEZ por step (nÃ£o por indivÃ­duo).
    # Isso reduz MUITO o thrash de disco/cache e deixa o fitness estÃ¡vel/determinÃ­stico.
    syms_all = sorted(paths.keys())
    if _W_SYMBOLS_PER_EVAL > 0 and _W_SYMBOLS_PER_EVAL < len(syms_all):
        rr = np.random.default_rng(int(_W_SEED) ^ 0xA5A5_1234)
        pick = rr.choice(np.arange(len(syms_all)), size=int(_W_SYMBOLS_PER_EVAL), replace=False)
        _W_EVAL_SYMS = [syms_all[int(i)] for i in pick]
    else:
        _W_EVAL_SYMS = syms_all

    # IMPORTANT: evitar thrash de disco:
    # se o cache LRU for menor que o conjunto avaliado, vamos reler parquets o tempo todo.
    # Ajusta automaticamente para caber todos os sÃ­mbolos avaliados neste step.
    try:
        need = int(len(_W_EVAL_SYMS or []))
        if need > 0 and _W_DF_CACHE_SIZE < need:
            _W_DF_CACHE_SIZE = need
    except Exception:
        pass


def _eval_one(ind: Individual) -> tuple[Individual, EvalResult]:
    assert _W_PERIODS_SIM is not None and _W_SYMBOL_PATHS is not None and _W_WEIGHTS is not None and _W_PORTF_CFG is not None
    syms = list(_W_EVAL_SYMS or sorted(_W_SYMBOL_PATHS.keys()))

    periods = _apply_threshold_overrides_sim(
        _W_PERIODS_SIM,
        tau_entry=float(ind.tau_entry),
        tau_danger=float(ind.tau_danger),
        tau_exit=float(ind.tau_exit),
    )
    # --- SimulaÃ§Ã£o de portfÃ³lio (carteira Ãºnica) ---
    t0 = _W_TRAIN_START
    t1 = _W_TRAIN_END
    sym_data: dict[str, SymbolData] = {}
    for sym in syms:
        df0 = _df_cache_get(sym)
        if df0 is None or df0.empty:
            continue
        try:
            df = df0.loc[t0:t1]
        except Exception:
            idx = pd.to_datetime(df0.index)
            m = (idx >= t0) & (idx <= t1)
            df = df0.loc[m]
        if df is None or df.empty or len(df) < 400:
            continue
        try:
            pe = df["__p_entry"].to_numpy(np.float32, copy=False)
            pdg = df["__p_danger"].to_numpy(np.float32, copy=False)
            pid = df["__period_id"].to_numpy(np.int16, copy=False)
        except Exception:
            continue
        n = int(len(df))
        tau_entry = float(ind.tau_entry)
        tau_danger = float(ind.tau_danger)
        tau_exit = float(ind.tau_exit)
        tau_add = float(min(0.99, max(0.01, tau_entry * 1.10)))
        tau_danger_add = float(min(0.99, max(0.01, tau_danger * 0.90)))
        sym_data[sym] = SymbolData(
            df=df,
            p_entry=pe,
            p_danger=pdg,
            p_exit=np.full(n, np.nan, dtype=np.float32),
            tau_entry=tau_entry,
            tau_danger=tau_danger,
            tau_add=tau_add,
            tau_danger_add=tau_danger_add,
            tau_exit=tau_exit,
            period_id=pid,
            periods=periods,
        )

    if not sym_data:
        return ind, EvalResult(
            fitness=-1e9,
            eq_end=1.0,
            max_dd=1.0,
            profit_factor=0.0,
            win_rate=0.0,
            trades=0,
            neg_months=999,
            symbols=0,
            trades_total=0,
            trades_avg=0.0,
            neg_month_ratio=1.0,
        )

    pres = simulate_portfolio(sym_data, cfg=_W_PORTF_CFG, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60)
    eq_ser = getattr(pres, "equity_curve", None)
    if eq_ser is None or len(eq_ser) == 0:
        eq_ser = pd.Series([1.0], index=[pd.to_datetime(t1)], name="equity")
    else:
        try:
            last = float(eq_ser.iloc[-1])
        except Exception:
            last = 1.0
        if pd.to_datetime(eq_ser.index.max()) < pd.to_datetime(t1):
            eq_ser = pd.concat([eq_ser, pd.Series([last], index=[pd.to_datetime(t1)], name="equity")]).sort_index()

    trades_list = list(getattr(pres, "trades", []) or [])
    pnl = []
    for tr in trades_list:
        w = float(getattr(tr, "weight", 0.0) or 0.0)
        r = float(getattr(tr, "r_net", 0.0) or 0.0)
        pnl.append(w * r)
    wins = sum(x for x in pnl if x > 0)
    losses = sum(-x for x in pnl if x < 0)
    pf = float(wins / max(1e-12, losses)) if losses > 0 else float("inf")
    win = float(sum(1 for x in pnl if x > 0) / max(1, len(pnl)))

    class _R:
        pass

    r0 = _R()
    r0.equity_curve = np.asarray(eq_ser.to_numpy(np.float64, copy=False))
    r0.max_dd = float(getattr(pres, "max_dd", 0.0) or 0.0)
    r0.profit_factor = float(pf)
    r0.win_rate = float(win)
    r0.trades = trades_list
    r0.monthly_returns = []

    ev = _score_result(r0, _W_WEIGHTS, symbol="PORTFOLIO")
    # Ajusta metadados agregados
    return ind, EvalResult(
        fitness=float(ev.fitness),
        eq_end=float(ev.eq_end),
        max_dd=float(ev.max_dd),
        profit_factor=float(ev.profit_factor),
        win_rate=float(ev.win_rate),
        trades=int(len(trades_list)),
        neg_months=int(ev.neg_months),
        symbols=int(len(sym_data)),
        trades_total=int(len(trades_list)),
        trades_avg=float(len(trades_list)),
        neg_month_ratio=float(ev.neg_month_ratio),
    )


def _save_step_json(
    out_dir: Path,
    *,
    step_idx: int,
    train: dict,
    test: dict,
    best: Individual,
    in_metrics: EvalResult,
    oos_metrics: EvalResult,
    universe: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step_idx": int(step_idx),
        "train": train,
        "test": test,
        "universe": dict(universe or {}),
        "best_thresholds": asdict(best),
        "in_sample": asdict(in_metrics),
        "out_of_sample": asdict(oos_metrics),
    }
    # Windows: ":" e espaÃ§os em filenames causam OSError 22. Sanitiza timestamps para o nome do arquivo.
    def _ts_slug(x: Any) -> str:
        try:
            ts = pd.to_datetime(x)
            s = ts.strftime("%Y%m%d_%H%M%S")
        except Exception:
            s = str(x)
        # nÃ£o depende de regex (Windows-friendly)
        out = []
        for ch in str(s):
            if ch.isalnum() or ch in "._-":
                out.append(ch)
            else:
                out.append("_")
        s2 = "".join(out).strip("_")
        return s2 or "ts"

    train_end = _ts_slug(train.get("end"))
    test_end = _ts_slug(test.get("end"))
    p = out_dir / f"step_{int(step_idx):03d}_{train_end}_to_{test_end}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # Silencia warning ruidoso do XGBoost quando hÃ¡ device mismatch (CUDA booster + numpy CPU),
    # caso algum caminho legado ainda chame inplace_predict internamente.
    try:
        warnings.filterwarnings(
            "ignore",
            message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
            category=UserWarning,
        )
    except Exception:
        pass
    # ======== CONFIG PADRÃƒO (Sem precisar de flags) ========
    # Se quiser forÃ§ar um diretÃ³rio especÃ­fico, mude aqui. 
    # None = busca o wf_* mais recente em models_sniper/
    run_dir_override: str | None = None 
    
    # Universo fixo (sem aleatoriedade)
    symbols_default = [
        "ADAUSDT",
        "XLMUSDT",
        "XRPUSDT",
        "FILUSDT",
        "APEUSDT",
        "STXUSDT",
        "SHIBUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "SUIUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "SOLUSDT",
        "ETCUSDT",
        "ATOMUSDT",
        "LTCUSDT",
        "TRXUSDT",
        "NEARUSDT",
        "AAVEUSDT",
        "HBARUSDT",
        "UNIUSDT",
        "APTUSDT",
        "POLUSDT",
        "ALGOUSDT"
    ]

    # Default: usar TODO o top_market_cap.txt (sem limite), a menos que o usuÃ¡rio especifique.
    max_symbols_default = 0
    # 180d reduz o ruÃ­do e melhora a estabilidade do GA (menos overfit de curto prazo).
    step_days_default = 180
    years_default = 6
    bar_stride_default = 5
    # Para usar bem CPUs com muitos threads:
    # - paralelizamos por indivÃ­duo (cada indivÃ­duo = 1 processo)
    # - entÃ£o pop_size precisa ser >= jobs para ocupar tudo.
    pop_size_default = 10
    gens_default = 10
    # Default conservador no Windows: muitos processos duplicam RAM (datasets + modelos exit).
    # VocÃª pode subir via --jobs quando tiver RAM sobrando.
    jobs_default = 4
    # =======================================================

    # Se existir CUDA no xgboost, deixa cuda como default (mas ainda dÃ¡ para forÃ§ar cpu via flag).
    xgb_device_default = "cuda" if _xgb_has_cuda() else "cpu"

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=run_dir_override)
    # Deixe vazio para usar top_market_cap.txt (recomendado)
    ap.add_argument("--symbols", type=str, default="", help="Lista fixa de sÃ­mbolos (CSV). Se vazio, usa top_market_cap.txt.")
    ap.add_argument("--max-symbols", type=int, default=max_symbols_default)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step-days", type=int, default=step_days_default)
    ap.add_argument("--years", type=int, default=years_default)
    ap.add_argument("--bar-stride", type=int, default=bar_stride_default)
    ap.add_argument("--refresh-cache", action="store_true", help="Recria caches de features (forÃ§a atualizar dados).")
    # 0 = usa todos os sÃ­mbolos do dataset/step (recomendado com lista fixa pequena)
    ap.add_argument("--symbols-per-eval", type=int, default=0)
    ap.add_argument("--pop-size", type=int, default=pop_size_default)
    ap.add_argument("--gens", type=int, default=gens_default)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--log-top-k", type=int, default=5)
    ap.add_argument("--log-all", action="store_true")
    ap.add_argument("--jobs", type=int, default=jobs_default)
    ap.add_argument("--xgb-threads", type=int, default=1)
    ap.add_argument("--xgb-device", type=str, default=xgb_device_default, choices=["cpu", "cuda"], help="Device do XGBoost (cpu/cuda).")
    ap.add_argument("--df-cache-size", type=int, default=24)
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.01)
    # Pushover notifications:
    # Por padrÃ£o, fica LIGADO automaticamente se houver credenciais (ENV ou config/secrets.py).
    # Use --no-pushover para desativar.
    ap.add_argument("--no-pushover", action="store_true")
    ap.add_argument("--pushover-user-env", type=str, default="PUSHOVER_USER_KEY")
    ap.add_argument("--pushover-token-env", type=str, default="PUSHOVER_TOKEN_TRADE")
    ap.add_argument("--pushover-title", type=str, default="tradebot GA")
    ap.add_argument("--pushover-every", type=int, default=1, help="Enviar a cada N geraÃ§Ãµes (1 = toda geraÃ§Ã£o).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=str, default="thresholds_ga_wf.csv")
    ap.add_argument("--out-dir", type=str, default="thresholds_ga_steps")
    # Universo dinÃ¢mico (por sÃ­mbolo, usando OOS acumulado)
    ap.add_argument("--no-dynamic-universe", action="store_true", help="Desativa seleÃ§Ã£o dinÃ¢mica de universo por sÃ­mbolo.")
    ap.add_argument("--universe-warmup-steps", type=int, default=2)
    ap.add_argument("--universe-ewm-decay", type=float, default=0.85)
    ap.add_argument("--universe-min-steps", type=int, default=2)
    ap.add_argument("--universe-min-trades", type=int, default=10)
    ap.add_argument("--universe-ban-ret180", type=float, default=-1.0)
    ap.add_argument("--universe-ban-pf", type=float, default=1.0)
    ap.add_argument("--universe-unban-ret180", type=float, default=0.5)
    ap.add_argument("--universe-unban-pf", type=float, default=1.05)
    ap.add_argument("--universe-min-active", type=int, default=20)
    ap.add_argument("--universe-max-changes", type=int, default=12)
    ap.add_argument("--symbol-perf-csv", type=str, default="thresholds_ga_wf_symbols.csv")
    ap.add_argument("--universe-state", type=str, default="thresholds_ga_universe_state.json")
    ap.add_argument("--no-shadow-eval", action="store_true", help="NÃ£o rodar shadow OOS (por sÃ­mbolo) a cada step.")
    args = ap.parse_args()
    # Limita threads tambÃ©m no processo principal (dataset build + OOS por geraÃ§Ã£o)
    _set_thread_limits(int(getattr(args, "xgb_threads", 1) or 1))

    pushover_on = not bool(getattr(args, "no_pushover", False))

    pushover_cfg = None
    if pushover_on and (_pushover_load_default is not None) and (_pushover_send is not None):
        try:
            pushover_cfg = _pushover_load_default(
                user_env=str(getattr(args, "pushover_user_env", "PUSHOVER_USER_KEY")),
                token_env=str(getattr(args, "pushover_token_env", "PUSHOVER_TOKEN_TRADE")),
                token_name_fallback=str(getattr(args, "pushover_token_env", "PUSHOVER_TOKEN_TRADE")),
                title=str(getattr(args, "pushover_title", "tradebot GA")),
            )
        except Exception:
            pushover_cfg = None

    _pushover_warned = False

    def _notify(msg: str) -> None:
        nonlocal _pushover_warned
        if pushover_cfg is None or _pushover_send is None:
            return
        try:
            res = _pushover_send(msg, cfg=pushover_cfg)
            # nossa implementaÃ§Ã£o retorna (ok, payload, err). Se falhar, avisar 1x.
            if isinstance(res, tuple) and len(res) >= 1:
                ok = bool(res[0])
                err = None
                if (not ok) and len(res) >= 3:
                    err = res[2]
                if (not ok) and (not _pushover_warned):
                    _pushover_warned = True
                    print(f"[pushover][warn] falhou ao enviar (ok=False) err={err}", flush=True)
        except Exception as e:
            # nunca quebrar o GA por notificaÃ§Ã£o
            if not _pushover_warned:
                _pushover_warned = True
                print(f"[pushover][warn] exceÃ§Ã£o ao enviar: {type(e).__name__}: {e}", flush=True)
            return

    if pushover_on:
        if pushover_cfg is None:
            print(
                f"[pushover] NÃƒO ativado (credenciais nÃ£o encontradas). "
                f"ENV esperadas: {getattr(args,'pushover_user_env','PUSHOVER_USER_KEY')} + {getattr(args,'pushover_token_env','PUSHOVER_TOKEN_TRADE')} "
                f"(ou config/secrets.py).",
                flush=True,
            )
        else:
            print("[pushover] ativado automaticamente.", flush=True)

    # Auto-detect run_dir se for None
    if args.run_dir is None:
        try:
            # Tenta caminhos provÃ¡veis no seu ambiente (absolutos primeiro)
            paths_to_check = [
                Path("D:/astra/models_sniper"),
                Path(__file__).resolve().parents[2].parent / "models_sniper",
                Path.cwd().parent / "models_sniper",
                Path.cwd() / "models_sniper"
            ]
            
            for models_root in paths_to_check:
                if models_root.is_dir():
                    wf_list = sorted([p for p in models_root.glob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
                    if wf_list:
                        args.run_dir = str(wf_list[-1])
                        break
        except Exception:
            pass

    if args.run_dir is None:
        raise RuntimeError("NÃ£o foi possÃ­vel encontrar nenhum wf_* automaticamente. Por favor, defina run_dir_override no cÃ³digo.")

    run_dir = Path(args.run_dir).expanduser().resolve()
    print(f"[ga] Usando run_dir: {run_dir}")
    if not run_dir.is_dir():
        raise RuntimeError(f"run_dir invÃ¡lido: {run_dir}")

    xgb_device = str(getattr(args, "xgb_device", "cpu") or "cpu").strip().lower()
    if xgb_device == "cuda" and (not _xgb_has_cuda()):
        print("[ga][warn] --xgb-device=cuda solicitado, mas este xgboost nÃ£o tem CUDA. Revertendo para cpu.", flush=True)
        xgb_device = "cpu"
    print(f"[ga] xgb_device={xgb_device} xgb_threads={int(getattr(args,'xgb_threads',1) or 1)}", flush=True)

    # teste de notificaÃ§Ã£o (apÃ³s saber qual wf_* estÃ¡ rodando)
    if pushover_on and (pushover_cfg is not None) and (_pushover_send is not None):
        _notify(f"GA iniciado: run={run_dir.name} | pop={int(args.pop_size)} gens={int(args.gens)} step_days={int(args.step_days)}")

    rng = np.random.default_rng(int(args.seed))
    periods_base = load_period_models(run_dir)
    # Base leve de perÃ­odos (mesma estrutura usada pelos workers).
    # Usaremos isso tambÃ©m no OOS no processo principal para garantir que IN/OOS usem
    # exatamente a mesma lÃ³gica de simulaÃ§Ã£o (diferenÃ§a apenas no recorte temporal).
    periods_sim_base = _load_periods_for_sim(run_dir)
    # reforÃ§a nthread nos boosters do processo principal (dataset build usa entry/danger)
    try:
        for pm in (periods_base or []):
            try:
                pm.entry_model.set_param({"nthread": int(getattr(args, "xgb_threads", 1) or 1), "device": xgb_device})
            except Exception:
                pass
            try:
                pm.danger_model.set_param({"nthread": int(getattr(args, "xgb_threads", 1) or 1), "device": xgb_device})
            except Exception:
                pass
            try:
                em = getattr(pm, "exit_model", None)
                if em is not None:
                    em.set_param({"nthread": int(getattr(args, "xgb_threads", 1) or 1), "device": xgb_device})
            except Exception:
                pass
    except Exception:
        pass

    # SeleÃ§Ã£o de sÃ­mbolos (lista fixa do usuÃ¡rio; sem aleatoriedade)
    syms = [s.strip().upper() for s in str(getattr(args, "symbols", "")).split(",") if s.strip()]
    if not syms:
        # default: top_market_cap.txt (tudo), com opcionais --limit/--max-symbols
        syms = load_top_market_cap_symbols(limit=int(args.limit) if int(args.limit) > 0 else None)
        syms = [s.strip().upper() for s in syms if str(s).strip()]
    if int(args.max_symbols) > 0:
        syms = syms[: int(args.max_symbols)]
    if not syms:
        raise RuntimeError("Sem sÃ­mbolos (top_market_cap.txt vazio?)")

    try:
        if int(len(syms)) >= 120 and int(getattr(args, "symbols_per_eval", 0) or 0) <= 0:
            print(
                f"[ga][warn] universo grande (symbols={len(syms)}) e --symbols-per-eval=0 => cada avaliaÃ§Ã£o vai usar TODOS os sÃ­mbolos. "
                f"Se ficar lento demais, use --symbols-per-eval (ex.: 60-120) para amostrar por step.",
                flush=True,
            )
    except Exception:
        pass

    # Cache: precisamos de (years + 1 step) para cobrir train + test
    total_days_cache = int(args.years) * 365 + int(args.step_days) * 2 + 60
    cache_map = ensure_feature_cache(
        syms,
        total_days=int(total_days_cache),
        contract=DEFAULT_TRADE_CONTRACT,
        flags=dict(GLOBAL_FLAGS_FULL, **{"_quiet": True}),
        refresh=bool(getattr(args, "refresh_cache", False)),
        # IMPORTANT: evita "hit" com cache antigo/curto (que reduz end_global e corta steps).
        strict_total_days=True,
        parallel=True,
        max_workers=32,
    )
    syms = [s for s in syms if s in cache_map]
    if not syms:
        raise RuntimeError("Nenhum sÃ­mbolo restou apÃ³s cache")

    # Janela global:
    # - Antes: usava o menor end_ts (todos alinhados) -> corta steps quando 1 sÃ­mbolo Ã© curto.
    # - Agora: usamos o MAIOR end_ts e deixamos sÃ­mbolos entrarem/sair ao longo do tempo.
    ends_by_sym: dict[str, pd.Timestamp] = {}
    starts_by_sym: dict[str, pd.Timestamp] = {}

    # IMPORTANTE (performance): NÃƒO ler 408 parquets gigantes sÃ³ pra descobrir start/end.
    # Usamos os meta.json do cache (bem rÃ¡pido). SÃ³ faz fallback para parquet se faltar meta.
    t_scan0 = time.perf_counter()
    miss_meta = 0
    for i, s in enumerate(syms, start=1):
        try:
            p = Path(cache_map[s])
        except Exception:
            continue
        st, en = _read_cache_meta_times(p)
        if st is not None and en is not None:
            starts_by_sym[s] = pd.to_datetime(st)
            ends_by_sym[s] = pd.to_datetime(en)
        else:
            miss_meta += 1
            try:
                df0 = _read_parquet_best_effort(p, columns=["close"])
                if df0 is None or df0.empty:
                    continue
                starts_by_sym[s] = pd.to_datetime(df0.index.min())
                ends_by_sym[s] = pd.to_datetime(df0.index.max())
            except Exception:
                continue
        if i in {1, 5, 10, 25, 50, 100, 200, 300, 400} or i == len(syms):
            try:
                dt = time.perf_counter() - t_scan0
                print(f"[ga] scan caches: {i}/{len(syms)} (miss_meta={miss_meta}) elapsed={dt:.1f}s", flush=True)
            except Exception:
                pass
    if not ends_by_sym:
        raise RuntimeError("Sem dados vÃ¡lidos nos caches para end_global")
    end_global = max(ends_by_sym.values())
    try:
        worst = sorted(ends_by_sym.items(), key=lambda kv: kv[1])[:5]
        best = sorted(ends_by_sym.items(), key=lambda kv: kv[1], reverse=True)[:5]
        worst_s = ", ".join([f"{k}={v}" for k, v in worst])
        best_s = ", ".join([f"{k}={v}" for k, v in best])
        print(f"[ga] end_global(max)={end_global} | melhores_ends: {best_s}", flush=True)
        print(f"[ga] ends_mais_curto: {worst_s}", flush=True)
        if starts_by_sym:
            starts_worst = sorted(starts_by_sym.items(), key=lambda kv: kv[1], reverse=True)[:5]
            starts_best = sorted(starts_by_sym.items(), key=lambda kv: kv[1])[:5]
            sw_s = ", ".join([f"{k}={v}" for k, v in starts_worst])
            sb_s = ", ".join([f"{k}={v}" for k, v in starts_best])
            print(f"[ga] starts_mais_tardio: {sw_s}", flush=True)
            print(f"[ga] starts_mais_antigo: {sb_s}", flush=True)
    except Exception:
        pass
    start_global = end_global - pd.Timedelta(days=int(args.years) * 365)

    # Garante que as janelas nÃ£o comecem ANTES do primeiro modelo disponÃ­vel no wf_*
    # (caso contrÃ¡rio, `predict_scores_walkforward` nÃ£o consegue preencher nenhum ponto e lanÃ§a erro)
    try:
        min_train_end = min(pd.to_datetime(pm.train_end_utc) for pm in (periods_base or []))
        if pd.to_datetime(end_global) <= min_train_end:
            raise RuntimeError(
                f"end_global={end_global} estÃ¡ <= min_train_end(model)={min_train_end}. "
                "Seu cache/dados terminam antes do primeiro modelo do wf_*. "
                "Use um wf_* compatÃ­vel ou aumente o range de dados."
            )
        if pd.to_datetime(start_global) < min_train_end:
            print(
                f"[ga][warn] start_global={start_global} < min_train_end(model)={min_train_end}; ajustando start_global para {min_train_end}",
                flush=True,
            )
            start_global = pd.to_datetime(min_train_end)
    except Exception:
        pass

    step = pd.Timedelta(days=int(args.step_days))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    t0 = pd.to_datetime(start_global)
    while t0 + step <= end_global:
        windows.append((t0, t0 + step))
        t0 = t0 + step
    if len(windows) < 3:
        raise RuntimeError(f"Poucas janelas geradas (windows={len(windows)}). Ajuste years/step_days.")

    # colunas necessÃ¡rias para o Exit (e Entry/Danger)
    need_cols = _needed_feature_columns(periods_base)

    out_rows: list[dict[str, Any]] = []
    best_prev: Individual | None = None
    out_csv = resolve_generated_path(str(args.out_csv))
    out_dir = resolve_generated_path(str(args.out_dir))
    dataset_root = (out_dir / "ga_cache" / run_dir.name).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Universo dinÃ¢mico (por sÃ­mbolo)
    universe_cfg = UniverseSelectorConfig(
        enabled=not bool(getattr(args, "no_dynamic_universe", False)),
        warmup_steps=int(getattr(args, "universe_warmup_steps", 2) or 2),
        ewm_decay=float(getattr(args, "universe_ewm_decay", 0.85) or 0.85),
        min_steps=int(getattr(args, "universe_min_steps", 2) or 2),
        min_trades_total=int(getattr(args, "universe_min_trades", 10) or 10),
        ban_ret180=float(getattr(args, "universe_ban_ret180", -1.0) or -1.0),
        ban_pf=float(getattr(args, "universe_ban_pf", 1.0) or 1.0),
        unban_ret180=float(getattr(args, "universe_unban_ret180", 0.5) or 0.5),
        unban_pf=float(getattr(args, "universe_unban_pf", 1.05) or 1.05),
        min_active=int(getattr(args, "universe_min_active", 20) or 20),
        max_changes_per_step=int(getattr(args, "universe_max_changes", 12) or 12),
    )
    symbol_perf_csv = resolve_generated_path(str(getattr(args, "symbol_perf_csv", "thresholds_ga_wf_symbols.csv")))
    universe_state_path = resolve_generated_path(str(getattr(args, "universe_state", "thresholds_ga_universe_state.json")))
    shadow_eval_on = not bool(getattr(args, "no_shadow_eval", False))

    # Pesos para score POR SÃMBOLO (menos rÃ­gidos que o portfÃ³lio; sem penalidade de trade count)
    weights_symbol_oos = FitnessWeights(
        step_days=int(getattr(args, "step_days", 180) or 180),
        target_return_pct=5.0,
        target_pf=1.20,
        target_win=0.45,
        target_dd=0.35,
        hard_return_pct=0.0,
        hard_pf=0.90,
        hard_win=0.25,
        hard_dd=0.60,
        target_trades_min=0,
        target_trades_max=10_000_000,
        pen_trade_count=0.0,
        no_trade_neutral=True,
    )

    # Walk-forward OOS consolidado (portfÃ³lio): compÃµe equity ao longo dos steps
    wf_oos_logeq_portf: float = 0.0
    wf_oos_steps: int = 0

    # Expanding window:
    # - step 0: train = 180d (windows[0])
    # - step 1: train = 360d (windows[0] + windows[1])
    # - step k: train = (k+1)*step_days (sempre comeÃ§ando em windows[0][0])
    train_start0 = pd.to_datetime(windows[0][0])

    for step_idx in range(len(windows) - 1):
        # janela de treino cresce a cada step
        train_start = train_start0
        train_end = pd.to_datetime(windows[step_idx][1])
        # IMPORTANT: evita "vazamento" de 1 candle por slicing inclusivo (train_end entra no IN e no OOS).
        # Como estamos em 1m, 1 minuto Ã© suficiente; com bar_stride > 1, ainda Ã© seguro.
        test_start = pd.to_datetime(train_end) + pd.Timedelta(minutes=1)
        test_end = pd.to_datetime(windows[step_idx + 1][1])

        try:
            train_len_days = int(round(float((train_end - train_start) / pd.Timedelta(days=1))))
        except Exception:
            train_len_days = int((step_idx + 1) * int(args.step_days))
        train_len_days = max(1, int(train_len_days))

        # FitnessWeights Ã© frozen (imutÃ¡vel) -> setar via construtor
        # - treino: usa o tamanho real da janela (expanding) para escalar targets/penalidades
        # - OOS: mantÃ©m "por step" (180d) para comparabilidade entre steps
        weights_train = FitnessWeights(step_days=int(train_len_days))
        weights_train_dict = asdict(weights_train)
        weights_oos = FitnessWeights(step_days=int(args.step_days))

        print(
            f"\n[ga] step={step_idx}/{len(windows)-2} "
            f"train={train_start}..{train_end} ({train_len_days}d) "
            f"test={test_start}..{test_end} ({int(args.step_days)}d)",
            flush=True,
        )

        # PrÃ©-filtro: evita preparar dataset para sÃ­mbolos sem barras suficientes neste step.
        # Isso Ã© CRÃTICO quando `syms` Ã© grande (ex.: 400+).
        min_bars_step = 400
        stride = int(getattr(args, "bar_stride", 1) or 1)
        stride = max(1, stride)
        min_span = pd.Timedelta(minutes=int(min_bars_step) * int(stride))
        syms_build: list[str] = []
        for s in syms:
            st = starts_by_sym.get(s)
            en = ends_by_sym.get(s)
            if st is None or en is None:
                # fallback: nÃ£o sabemos; tenta
                syms_build.append(s)
                continue
            try:
                # precisa ter pelo menos `min_span` dentro do treino e do teste
                ok_train = (pd.to_datetime(st) <= (pd.to_datetime(train_end) - min_span)) and (pd.to_datetime(en) >= (pd.to_datetime(train_start) + min_span))
                ok_test = (pd.to_datetime(st) <= (pd.to_datetime(test_end) - min_span)) and (pd.to_datetime(en) >= (pd.to_datetime(test_start) + min_span))
                if ok_train and ok_test:
                    syms_build.append(s)
            except Exception:
                syms_build.append(s)

        if len(syms_build) != len(syms):
            try:
                print(
                    f"[ga] symbols: build_candidates={len(syms_build)}/{len(syms)} (min_bars={min_bars_step} stride={stride})",
                    flush=True,
                )
            except Exception:
                pass
        if not syms_build:
            raise RuntimeError("Nenhum sÃ­mbolo tem barras suficientes neste step (syms_build=0).")

        # prepara dataset do passo (train+test) em parquet por sÃ­mbolo (Windows-friendly)
        ds_dir = dataset_root / f"step_{int(step_idx):03d}"
        # Rebuild automÃ¡tico se o dataset for de uma versÃ£o antiga (com colunas a mais, ex.: __p_exit)
        need_rebuild = False
        if ds_dir.is_dir():
            meta_p = ds_dir / "_ga_dataset_meta.json"
            if meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    if str(meta.get("format", "")) != "ga_v2_mincols":
                        need_rebuild = True
                    # rebuild se o conjunto de sÃ­mbolos mudou (evita misturar runs antigos)
                    meta_syms = [str(x).upper() for x in (meta.get("symbols") or [])]
                    # se meta nÃ£o tem lista, assume legado -> rebuild
                    if (not meta_syms) or meta_syms != [str(x).upper() for x in syms_build]:
                        need_rebuild = True
                    # rebuild se o range mudou (expanding window muda t_start em relaÃ§Ã£o ao legado)
                    if str(meta.get("t_start", "")) != str(train_start):
                        need_rebuild = True
                    if str(meta.get("t_end", "")) != str(test_end):
                        need_rebuild = True
                except Exception:
                    need_rebuild = True
            else:
                # sem meta => pode ser legado
                need_rebuild = True
        if (not ds_dir.is_dir()) or (not any(ds_dir.glob("*.parquet"))) or need_rebuild:
            ds_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ga] preparando dataset (cols={len(need_cols)}) em {ds_dir}", flush=True)
            # limpa parquets antigos (se existirem) para nÃ£o misturar formatos
            try:
                for p in ds_dir.glob("*.parquet"):
                    p.unlink(missing_ok=True)
            except Exception:
                pass
            for sym in _progress(syms_build, total=len(syms_build), desc=f"prep step {step_idx}"):
                p = cache_map[sym]
                df_full = _read_parquet_best_effort(Path(p), columns=need_cols)
                if df_full is None:
                    continue
                if df_full is None or df_full.empty:
                    continue
                # cobre train+test
                df_step = _prepare_symbol_frame(
                    df_full,
                    periods=periods_base,
                    t_start=train_start,
                    t_end=test_end,
                    bar_stride=int(args.bar_stride),
                )
                if df_step is None or df_step.empty or len(df_step) < 400:
                    continue
                # grava
                out_p = ds_dir / f"{sym}.parquet"
                df_step.to_parquet(out_p, index=True)
            # marca versÃ£o do dataset (para permitir upgrades futuros sem confusÃ£o)
            try:
                (ds_dir / "_ga_dataset_meta.json").write_text(
                    json.dumps(
                        {
                            "format": "ga_v2_mincols",
                            "symbols": list(syms_build),
                            "t_start": str(train_start),
                            "t_end": str(test_end),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # Carrega parquets do step uma vez (reuso para OOS por geraÃ§Ã£o)
        frames_step: dict[str, pd.DataFrame] = {}
        try:
            syms_set = {s.upper() for s in syms_build}
            for p in ds_dir.glob("*.parquet"):
                s = p.stem.upper()
                if s not in syms_set:
                    continue
                frames_step[s] = pd.read_parquet(p)
        except Exception:
            frames_step = {}

        # IMPORTANT: trave o universo por step para comparabilidade IN vs OOS.
        # Sem isso, pode acontecer do treino usar N sÃ­mbolos e o OOS usar N+1 (ou N-1),
        # sÃ³ porque algum sÃ­mbolo nÃ£o tinha barras suficientes em um dos intervalos.
        syms_step: list[str] = []
        for sym, df0 in frames_step.items():
            try:
                n_tr = int(len(df0.loc[train_start:train_end]))
                n_te = int(len(df0.loc[test_start:test_end]))
            except Exception:
                idx0 = pd.to_datetime(df0.index)
                n_tr = int(np.sum((idx0 >= train_start) & (idx0 <= train_end)))
                n_te = int(np.sum((idx0 >= test_start) & (idx0 <= test_end)))
            if n_tr >= 400 and n_te >= 400:
                syms_step.append(sym)
        syms_step = sorted(set(syms_step))
        if syms_step:
            frames_step = {s: frames_step[s] for s in syms_step if s in frames_step}
        else:
            # fallback: mantÃ©m tudo (nÃ£o deve acontecer com seus 24 sÃ­mbolos)
            syms_step = sorted(frames_step.keys())

        # ------------------------------
        # Universo dinÃ¢mico (ativo vs shadow) por step
        # - disponÃ­vel: intersecao trainâˆ©test (syms_step)
        # - ativo: subset usado pelo GA (e pelo OOS "real")
        # - shadow: avaliado no final do step (OOS) para pontuar TODOS os sÃ­mbolos
        # ------------------------------
        state_prev = _load_universe_state(universe_state_path)
        hist_prev = pd.DataFrame()
        try:
            if symbol_perf_csv.exists():
                hist_all = pd.read_csv(symbol_perf_csv)
                if "step_idx" in hist_all.columns:
                    hist_prev = hist_all.loc[hist_all["step_idx"] < int(step_idx)].copy()
                else:
                    hist_prev = hist_all.copy()
        except Exception:
            hist_prev = pd.DataFrame()

        syms_active_step, _reasons = _select_active_symbols_for_step(
            step_idx=int(step_idx),
            syms_available=syms_step,
            history=hist_prev,
            state_prev=state_prev,
            cfg=universe_cfg,
        )
        if not syms_active_step:
            syms_active_step = list(syms_step)
        syms_active_step = sorted({s.upper() for s in syms_active_step if s and s.upper() in set(syms_step)})
        frames_step_active = {s: frames_step[s] for s in syms_active_step if s in frames_step}

        if universe_cfg.enabled:
            try:
                banned_now = sorted({s.upper() for s in syms_step} - {s.upper() for s in syms_active_step})
                print(
                    f"[ga][universe] available={len(syms_step)} active={len(syms_active_step)} banned={len(banned_now)} "
                    f"(warmup={int(universe_cfg.warmup_steps)} min_active={int(universe_cfg.min_active)} shadow={'on' if shadow_eval_on else 'off'})",
                    flush=True,
                )
            except Exception:
                pass

        # GA: avalia indivÃ­duos no TRAIN somente
        pop = _init_population(rng, int(args.pop_size), best_prev)
        elite_n = max(0, int(args.elite))

        # Paralelismo efetivo aqui Ã© "por indivÃ­duo".
        # Se jobs > pop_size, processos vÃ£o ficar ociosos (sÃ³ hÃ¡ pop_size tarefas por geraÃ§Ã£o).
        eff_workers = max(1, min(int(args.jobs), int(args.pop_size)))
        print(
            f"[ga] workers: eff={eff_workers} (jobs={int(args.jobs)} pop={int(args.pop_size)} xgb_threads={int(args.xgb_threads)} df_cache={int(args.df_cache_size)})",
            flush=True,
        )
        print(
            f"[ga] symbols: total_step={len(syms)} symbols_per_eval={int(args.symbols_per_eval)} (dataset_build={len(syms)})",
            flush=True,
        )
        if len(syms_build) != len(syms):
            print(f"[ga] symbols: dataset_build={len(syms_build)} (prefilter)", flush=True)
        if syms_step and (len(syms_step) != len(syms)):
            print(f"[ga] symbols: usando intersecao trainâˆ©test = {len(syms_step)}", flush=True)
        if universe_cfg.enabled and syms_active_step and (len(syms_active_step) != len(syms_step)):
            print(f"[ga] symbols: universo ATIVO (para GA) = {len(syms_active_step)}", flush=True)
        with ProcessPoolExecutor(
            max_workers=eff_workers,
            initializer=_worker_init,
            initargs=(
                str(run_dir),
                str(ds_dir),
                int(pd.to_datetime(train_start).value),
                int(pd.to_datetime(train_end).value),
                int(args.symbols_per_eval),
                dict(weights_train_dict),
                int(args.seed) + int(step_idx) * 1337,
                int(args.df_cache_size),
                int(args.xgb_threads),
                str(xgb_device),
                ",".join(syms_active_step),
            ),
        ) as ex:
            best_ind: Individual | None = None
            best_fit: EvalResult | None = None

            memo: dict[Individual, EvalResult] = {}
            best_gen_fit: float | None = None
            no_improve = 0

            for g in range(int(args.gens)):
                tgen0 = time.perf_counter()
                # Evita reavaliar indivÃ­duos idÃªnticos (acontece muito com elitismo/crossover).
                futs = []
                missing: list[Individual] = []
                for ind in pop:
                    ev0 = memo.get(ind)
                    if ev0 is None:
                        missing.append(ind)
                        futs.append(ex.submit(_eval_one, ind))
                scored: list[tuple[Individual, EvalResult]] = []
                # jÃ¡ coloca os que estavam no memo
                for ind in pop:
                    ev0 = memo.get(ind)
                    if ev0 is not None:
                        scored.append((ind, ev0))
                # e adiciona os recÃ©m-calculados
                for f in _progress(as_completed(futs), total=len(futs), desc=f"gen {g+1}/{int(args.gens)}"):
                    ind, ev = f.result()
                    memo[ind] = ev
                    scored.append((ind, ev))

                scored.sort(key=lambda t: t[1].fitness, reverse=True)
                top_k = int(args.log_top_k) if int(args.log_top_k) > 0 else 0
                top = scored[: min(top_k, len(scored))] if top_k > 0 else []
                if top:
                    b0, e0 = top[0]
                    tgen = time.perf_counter() - tgen0
                    print(
                        (
                            f"[ga] gen={g+1} best fitness={e0.fitness:.4f} "
                            f"tau=(E{b0.tau_entry:.2f},D{b0.tau_danger:.2f},X{b0.tau_exit:.2f}) "
                            f"ret={_ret_pct(e0.eq_end):+.1f}% (ret180={_ret_pct_per_180d(e0.eq_end, int(train_len_days)):+.1f}%) "
                            f"dd={e0.max_dd:.2%} pf={e0.profit_factor:.2f} "
                            f"hit={e0.win_rate:.2f} syms={e0.symbols} "
                            f"trades(avg)~{e0.trades_avg:.1f} trades(total)~{e0.trades_total} negR~{e0.neg_month_ratio:.2f}"
                        ),
                        flush=True,
                    )
                    # notificaÃ§Ã£o a cada N geraÃ§Ãµes (se habilitada) com IN + OOS
                    every = max(1, int(getattr(args, "pushover_every", 1) or 1))
                    if pushover_on and ((g + 1) % every == 0):
                        oos_msg = ""
                        try:
                            # avalia o melhor da geraÃ§Ã£o no OOS (TEST) sÃ³ para monitoramento (nÃ£o afeta seleÃ§Ã£o)
                            if frames_step_active:
                                periods_gen = _apply_threshold_overrides_sim(
                                    periods_sim_base,
                                    tau_entry=float(b0.tau_entry),
                                    tau_danger=float(b0.tau_danger),
                                    tau_exit=float(b0.tau_exit),
                                )
                                sym_data_g: dict[str, SymbolData] = {}
                                for sym_g, df0_g in frames_step_active.items():
                                    try:
                                        df_g = df0_g.loc[test_start:test_end]
                                    except Exception:
                                        idx_g = pd.to_datetime(df0_g.index)
                                        m_g = (idx_g >= test_start) & (idx_g <= test_end)
                                        df_g = df0_g.loc[m_g]
                                    if df_g is None or df_g.empty or len(df_g) < 400:
                                        continue
                                    try:
                                        pe_g = df_g["__p_entry"].to_numpy(np.float32, copy=False)
                                        pdg_g = df_g["__p_danger"].to_numpy(np.float32, copy=False)
                                        pid_g = df_g["__period_id"].to_numpy(np.int16, copy=False)
                                    except Exception:
                                        continue
                                    n_g = int(len(df_g))
                                    te = float(b0.tau_entry)
                                    td = float(b0.tau_danger)
                                    tx = float(b0.tau_exit)
                                    ta = float(min(0.99, max(0.01, te * 1.10)))
                                    tda = float(min(0.99, max(0.01, td * 0.90)))
                                    sym_data_g[sym_g] = SymbolData(
                                        df=df_g,
                                        p_entry=pe_g,
                                        p_danger=pdg_g,
                                        p_exit=np.full(n_g, np.nan, dtype=np.float32),
                                        tau_entry=te,
                                        tau_danger=td,
                                        tau_add=ta,
                                        tau_danger_add=tda,
                                        tau_exit=tx,
                                        period_id=pid_g,
                                        periods=periods_gen,
                                    )

                                if sym_data_g:
                                    cfg_g = PortfolioConfig(
                                        max_positions=20,
                                        total_exposure=0.75,
                                        max_trade_exposure=0.10,
                                        min_trade_exposure=0.03,
                                        exit_min_hold_bars=3,
                                        exit_confirm_bars=2,
                                    )
                                    pres_g = simulate_portfolio(sym_data_g, cfg=cfg_g, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60)
                                    eq_ser_g = getattr(pres_g, "equity_curve", None)
                                    if eq_ser_g is None or len(eq_ser_g) == 0:
                                        eq_ser_g = pd.Series([1.0], index=[pd.to_datetime(test_end)], name="equity")
                                    else:
                                        last = float(eq_ser_g.iloc[-1]) if len(eq_ser_g) else 1.0
                                        if pd.to_datetime(eq_ser_g.index.max()) < pd.to_datetime(test_end):
                                            eq_ser_g = pd.concat(
                                                [eq_ser_g, pd.Series([last], index=[pd.to_datetime(test_end)], name="equity")]
                                            ).sort_index()
                                    trades_g = list(getattr(pres_g, "trades", []) or [])
                                    pnl_g = []
                                    for tr in trades_g:
                                        w = float(getattr(tr, "weight", 0.0) or 0.0)
                                        r = float(getattr(tr, "r_net", 0.0) or 0.0)
                                        pnl_g.append(w * r)
                                    wins = sum(x for x in pnl_g if x > 0)
                                    losses = sum(-x for x in pnl_g if x < 0)
                                    pf_g = float(wins / max(1e-12, losses)) if losses > 0 else float("inf")
                                    win_g = float(sum(1 for x in pnl_g if x > 0) / max(1, len(pnl_g)))

                                    class _R:
                                        pass

                                    r1 = _R()
                                    r1.equity_curve = np.asarray(eq_ser_g.to_numpy(np.float64, copy=False))
                                    r1.max_dd = float(getattr(pres_g, "max_dd", 0.0) or 0.0)
                                    r1.profit_factor = float(pf_g)
                                    r1.win_rate = float(win_g)
                                    r1.trades = trades_g
                                    r1.monthly_returns = []
                                    evg = _score_result(r1, weights_oos, symbol="PORTFOLIO")
                                    oos_g = EvalResult(
                                        fitness=float(evg.fitness),
                                        eq_end=float(evg.eq_end),
                                        max_dd=float(evg.max_dd),
                                        profit_factor=float(evg.profit_factor),
                                        win_rate=float(evg.win_rate),
                                        trades=int(len(trades_g)),
                                        neg_months=int(evg.neg_months),
                                        symbols=int(len(sym_data_g)),
                                        trades_total=int(len(trades_g)),
                                        trades_avg=float(len(trades_g)),
                                        neg_month_ratio=float(evg.neg_month_ratio),
                                    )
                                else:
                                    oos_g = EvalResult(
                                        fitness=-1e9,
                                        eq_end=1.0,
                                        max_dd=1.0,
                                        profit_factor=0.0,
                                        win_rate=0.0,
                                        trades=0,
                                        neg_months=999,
                                        symbols=0,
                                        trades_total=0,
                                        trades_avg=0.0,
                                        neg_month_ratio=1.0,
                                    )
                                oos_msg = (
                                    f" | OOS fit={oos_g.fitness:.4f} ret={_ret_pct(oos_g.eq_end):+.1f}% dd={oos_g.max_dd:.1%} "
                                    f"pf={oos_g.profit_factor:.2f} win={oos_g.win_rate:.2f} tr~{oos_g.trades_total}"
                                )
                        except Exception:
                            oos_msg = ""

                        _notify(
                            (
                                f"GA step {step_idx}/{len(windows)-2} gen {g+1}/{int(args.gens)} "
                                f"IN fit={e0.fitness:.4f} ret={_ret_pct(e0.eq_end):+.1f}% "
                                f"(ret180={_ret_pct_per_180d(e0.eq_end, int(train_len_days)):+.1f}%) "
                                f"dd={e0.max_dd:.1%} pf={e0.profit_factor:.2f} "
                                f"win={e0.win_rate:.2f} tr~{e0.trades_total} "
                                f"tau=({b0.tau_entry:.2f},{b0.tau_danger:.2f},{b0.tau_exit:.2f})"
                                f"{oos_msg} gen_sec={tgen:.1f}"
                            )
                        )
                    # early stopping simples por estagnaÃ§Ã£o
                    if best_gen_fit is None:
                        best_gen_fit = float(e0.fitness)
                        no_improve = 0
                    else:
                        if float(e0.fitness) >= float(best_gen_fit) + float(args.early_stop_min_delta):
                            best_gen_fit = float(e0.fitness)
                            no_improve = 0
                        else:
                            no_improve += 1
                    if int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
                        print(
                            f"[ga] early-stop: sem melhora por {no_improve} geraÃ§Ãµes (min_delta={float(args.early_stop_min_delta):.4f})",
                            flush=True,
                        )
                        break
                if bool(getattr(args, "log_all", False)):
                    for ind, ev in scored:
                        print(
                            (
                                f"[ga] gen={g+1} ind tau=(E{ind.tau_entry:.3f},D{ind.tau_danger:.3f},X{ind.tau_exit:.3f}) "
                                f"fit={ev.fitness:.4f} ret={_ret_pct(ev.eq_end):+.1f}% "
                                f"(ret180={_ret_pct_per_180d(ev.eq_end, int(train_len_days)):+.1f}%) "
                                f"dd={ev.max_dd:.2%} pf={ev.profit_factor:.2f} "
                                f"hit={ev.win_rate:.2f} trades(avg)~{ev.trades_avg:.1f} trades(total)~{ev.trades_total}"
                            ),
                            flush=True,
                        )

                # update best global do passo
                if scored:
                    cand_ind, cand_ev = scored[0]
                    if best_fit is None or cand_ev.fitness > best_fit.fitness:
                        best_ind, best_fit = cand_ind, cand_ev

                # elitismo + reproduÃ§Ã£o
                next_pop: list[Individual] = [x[0] for x in scored[:elite_n]]
                while len(next_pop) < int(args.pop_size):
                    p1 = _tournament_select(rng, scored, k=3)
                    p2 = _tournament_select(rng, scored, k=3)
                    c1, c2 = _crossover(rng, p1, p2, p=0.7)
                    c1 = _mutate(rng, c1, p=0.30)
                    c2 = _mutate(rng, c2, p=0.30)
                    next_pop.append(c1)
                    if len(next_pop) < int(args.pop_size):
                        next_pop.append(c2)
                pop = next_pop

        assert best_ind is not None and best_fit is not None

        # Avalia OOS no TEST usando o melhor do passo (sequencial, determinÃ­stico)
        frames: dict[str, pd.DataFrame] = {}
        # `syms_step` Ã© o universo disponÃ­vel do step (trainâˆ©test). `syms_active_step` Ã© o subset operado.
        syms_set = {s.upper() for s in syms_step} if syms_step else {s.upper() for s in syms}
        for p in ds_dir.glob("*.parquet"):
            s = p.stem.upper()
            if s not in syms_set:
                continue
            frames[s] = pd.read_parquet(p)
        periods_best = _apply_threshold_overrides_sim(
            periods_sim_base,
            tau_entry=float(best_ind.tau_entry),
            tau_danger=float(best_ind.tau_danger),
            tau_exit=float(best_ind.tau_exit),
        )
        sym_data_oos_shadow: dict[str, SymbolData] = {}
        for sym, df0 in frames.items():
            try:
                df = df0.loc[test_start:test_end]
            except Exception:
                idx = pd.to_datetime(df0.index)
                m = (idx >= test_start) & (idx <= test_end)
                df = df0.loc[m]
            if df is None or df.empty or len(df) < 400:
                continue
            try:
                pe = df["__p_entry"].to_numpy(np.float32, copy=False)
                pdg = df["__p_danger"].to_numpy(np.float32, copy=False)
                pid = df["__period_id"].to_numpy(np.int16, copy=False)
            except Exception:
                continue
            n = int(len(df))
            te = float(best_ind.tau_entry)
            td = float(best_ind.tau_danger)
            tx = float(best_ind.tau_exit)
            ta = float(min(0.99, max(0.01, te * 1.10)))
            tda = float(min(0.99, max(0.01, td * 0.90)))
            sym_data_oos_shadow[sym] = SymbolData(
                df=df,
                p_entry=pe,
                p_danger=pdg,
                p_exit=np.full(n, np.nan, dtype=np.float32),
                tau_entry=te,
                tau_danger=td,
                tau_add=ta,
                tau_danger_add=tda,
                tau_exit=tx,
                period_id=pid,
                periods=periods_best,
            )

        cfg_oos = PortfolioConfig(
            max_positions=20,
            total_exposure=0.75,
            max_trade_exposure=0.10,
            min_trade_exposure=0.03,
            exit_min_hold_bars=3,
            exit_confirm_bars=2,
        )
        syms_active_set = {str(s).upper() for s in (syms_active_step or [])}
        sym_data_oos = {s: sd for (s, sd) in sym_data_oos_shadow.items() if s in syms_active_set}

        if not sym_data_oos:
            oos = EvalResult(
                fitness=-1e9,
                eq_end=1.0,
                max_dd=1.0,
                profit_factor=0.0,
                win_rate=0.0,
                trades=0,
                neg_months=999,
                symbols=0,
                trades_total=0,
                trades_avg=0.0,
                neg_month_ratio=1.0,
            )
        else:
            pres = simulate_portfolio(sym_data_oos, cfg=cfg_oos, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60)
            eq_ser = getattr(pres, "equity_curve", None)
            if eq_ser is None or len(eq_ser) == 0:
                eq_ser = pd.Series([1.0], index=[pd.to_datetime(test_end)], name="equity")
            else:
                last = float(eq_ser.iloc[-1]) if len(eq_ser) else 1.0
                if pd.to_datetime(eq_ser.index.max()) < pd.to_datetime(test_end):
                    eq_ser = pd.concat([eq_ser, pd.Series([last], index=[pd.to_datetime(test_end)], name="equity")]).sort_index()

            trades_list = list(getattr(pres, "trades", []) or [])
            pnl = []
            for tr in trades_list:
                w = float(getattr(tr, "weight", 0.0) or 0.0)
                r = float(getattr(tr, "r_net", 0.0) or 0.0)
                pnl.append(w * r)
            wins = sum(x for x in pnl if x > 0)
            losses = sum(-x for x in pnl if x < 0)
            pf = float(wins / max(1e-12, losses)) if losses > 0 else float("inf")
            win = float(sum(1 for x in pnl if x > 0) / max(1, len(pnl)))

            class _R:
                pass

            r0 = _R()
            r0.equity_curve = np.asarray(eq_ser.to_numpy(np.float64, copy=False))
            r0.max_dd = float(getattr(pres, "max_dd", 0.0) or 0.0)
            r0.profit_factor = float(pf)
            r0.win_rate = float(win)
            r0.trades = trades_list
            r0.monthly_returns = []
            ev = _score_result(r0, weights_oos, symbol="PORTFOLIO")
            oos = EvalResult(
                fitness=float(ev.fitness),
                eq_end=float(ev.eq_end),
                max_dd=float(ev.max_dd),
                profit_factor=float(ev.profit_factor),
                win_rate=float(ev.win_rate),
                trades=int(len(trades_list)),
                neg_months=int(ev.neg_months),
                symbols=int(len(sym_data_oos)),
                trades_total=int(len(trades_list)),
                trades_avg=float(len(trades_list)),
                neg_month_ratio=float(ev.neg_month_ratio),
            )

        # Shadow OOS: simula portfÃ³lio com TODO o universo disponÃ­vel do step
        if universe_cfg.enabled and shadow_eval_on and sym_data_oos_shadow:
            try:
                pres_sh = simulate_portfolio(
                    sym_data_oos_shadow, cfg=cfg_oos, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60
                )
                trades_sh = list(getattr(pres_sh, "trades", []) or [])
                per_sym = _per_symbol_eval_from_portfolio_trades(
                    trades_sh,
                    symbols=sorted(set(syms_step)),
                    weights=weights_symbol_oos,
                )

                rows_sym: list[dict[str, Any]] = []
                for s in sorted(set(syms_step)):
                    evs = per_sym.get(str(s).upper())
                    if evs is None:
                        continue
                    rows_sym.append(
                        {
                            "step_idx": int(step_idx),
                            "train_start": str(train_start),
                            "train_end": str(train_end),
                            "test_start": str(test_start),
                            "test_end": str(test_end),
                            "symbol": str(s).upper(),
                            "is_active": bool(str(s).upper() in syms_active_set),
                            "fitness": float(evs.fitness),
                            "eq_end": float(evs.eq_end),
                            "ret_pct": float(_ret_pct(float(evs.eq_end))),
                            "ret180_pct": float(_ret_pct_per_180d(float(evs.eq_end), int(args.step_days))),
                            "max_dd": float(evs.max_dd),
                            "profit_factor": float(evs.profit_factor),
                            "win_rate": float(evs.win_rate),
                            "trades": int(evs.trades_total),
                        }
                    )

                _append_csv(symbol_perf_csv, pd.DataFrame(rows_sym))

                # Atualiza estado do universo (para limitar churn e registrar banidos)
                banned_now = sorted(set(syms_step) - set(syms_active_step))
                _save_universe_state(
                    universe_state_path,
                    {
                        "step_idx": int(step_idx),
                        "active_symbols": sorted(list(set(syms_active_step))),
                        "banned_symbols": banned_now,
                        "available_symbols": sorted(list(set(syms_step))),
                        "updated_at": time.time(),
                    },
                )
            except Exception as e:
                print(f"[ga][universe][warn] shadow eval falhou: {type(e).__name__}: {e}", flush=True)

        print(
            (
                f"[ga] step={step_idx} BEST tau=(E{best_ind.tau_entry:.2f},D{best_ind.tau_danger:.2f},X{best_ind.tau_exit:.2f}) "
                f"IN fitness={best_fit.fitness:.4f} ret={_ret_pct(best_fit.eq_end):+.1f}% "
                f"(ret180={_ret_pct_per_180d(best_fit.eq_end, int(train_len_days)):+.1f}%) "
                f"dd={best_fit.max_dd:.2%} pf={best_fit.profit_factor:.2f} hit={best_fit.win_rate:.2f} "
                f"tr(avg)~{best_fit.trades_avg:.1f} tr(total)~{best_fit.trades_total} syms={best_fit.symbols} | "
                f"OOS fitness={oos.fitness:.4f} ret={_ret_pct(oos.eq_end):+.1f}% dd={oos.max_dd:.2%} pf={oos.profit_factor:.2f} hit={oos.win_rate:.2f} "
                f"tr(avg)~{oos.trades_avg:.1f} tr(total)~{oos.trades_total} syms={oos.symbols}"
            ),
            flush=True,
        )
        # consolida walk-forward OOS (portfÃ³lio)
        try:
            wf_oos_logeq_portf = float(wf_oos_logeq_portf) + float(np.log(max(1e-12, float(oos.eq_end))))
            wf_oos_steps = int(wf_oos_steps) + 1
        except Exception:
            pass
        if pushover_on:
            _notify(
                f"step {step_idx}/{len(windows)-2} BEST E{best_ind.tau_entry:.2f} D{best_ind.tau_danger:.2f} X{best_ind.tau_exit:.2f} | "
                f"IN ret={_ret_pct(best_fit.eq_end):+.1f}% dd={best_fit.max_dd:.0%} pf={best_fit.profit_factor:.2f} tr={best_fit.trades_total} | "
                f"OOS ret={_ret_pct(oos.eq_end):+.1f}% dd={oos.max_dd:.0%} pf={oos.profit_factor:.2f} tr={oos.trades_total}"
            )

        out_rows.append(
            {
                "step_idx": int(step_idx),
                "train_start": str(train_start),
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "universe_available": int(len(syms_step or [])),
                "universe_active": int(len(syms_active_step or [])),
                "universe_banned": int(max(0, int(len(syms_step or [])) - int(len(syms_active_step or [])))),
                "universe_dynamic": bool(universe_cfg.enabled),
                "tau_entry": float(best_ind.tau_entry),
                "tau_danger": float(best_ind.tau_danger),
                "tau_exit": float(best_ind.tau_exit),
                "in_fitness": float(best_fit.fitness),
                "in_eq_end": float(best_fit.eq_end),
                "in_return_pct": float(_ret_pct(best_fit.eq_end)),
                "in_max_dd": float(best_fit.max_dd),
                "in_pf": float(best_fit.profit_factor),
                "in_win_rate": float(best_fit.win_rate),
                "in_trades": int(best_fit.trades),
                "in_neg_months": int(best_fit.neg_months),
                "in_symbols": int(best_fit.symbols),
                "in_trades_total": int(best_fit.trades_total),
                "in_trades_avg": float(best_fit.trades_avg),
                "in_neg_month_ratio": float(best_fit.neg_month_ratio),
                "oos_fitness": float(oos.fitness),
                "oos_eq_end": float(oos.eq_end),
                "oos_return_pct": float(_ret_pct(oos.eq_end)),
                "oos_max_dd": float(oos.max_dd),
                "oos_pf": float(oos.profit_factor),
                "oos_win_rate": float(oos.win_rate),
                "oos_trades": int(oos.trades),
                "oos_neg_months": int(oos.neg_months),
                "oos_symbols": int(oos.symbols),
                "oos_trades_total": int(oos.trades_total),
                "oos_trades_avg": float(oos.trades_avg),
                "oos_neg_month_ratio": float(oos.neg_month_ratio),
            }
        )

        _save_step_json(
            out_dir,
            step_idx=step_idx,
            train={"start": str(train_start), "end": str(train_end)},
            test={"start": str(test_start), "end": str(test_end)},
            universe={
                "available": int(len(syms_step or [])),
                "active": int(len(syms_active_step or [])),
                "banned": int(max(0, int(len(syms_step or [])) - int(len(syms_active_step or [])))),
                "dynamic": bool(universe_cfg.enabled),
                "shadow_eval": bool(shadow_eval_on),
            },
            best=best_ind,
            in_metrics=best_fit,
            oos_metrics=oos,
        )

        # salva CSV incremental (para nÃ£o perder progresso)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(out_csv, index=False, encoding="utf-8")

        best_prev = best_ind

    print(f"\n[ga] concluÃ­do. CSV: {out_csv} | steps: {len(out_rows)}", flush=True)
    # resumo WF OOS consolidado (portfÃ³lio)
    try:
        if int(wf_oos_steps) > 0:
            eq_total = float(np.exp(float(wf_oos_logeq_portf)))
            eq_gmean_step = float(np.exp(float(wf_oos_logeq_portf) / float(max(1, int(wf_oos_steps)))))
            print(
                f"[ga] WF-OOS (portfÃ³lio): steps={int(wf_oos_steps)} ret_total={_ret_pct(eq_total):+.1f}% ret_gmean_step={_ret_pct(eq_gmean_step):+.1f}%",
                flush=True,
            )
    except Exception:
        pass

    # resumo de variaÃ§Ã£o dos taus entre steps (ajuda a ver instabilidade)
    try:
        if out_rows:
            te = np.asarray([float(r.get("tau_entry", np.nan)) for r in out_rows], dtype=np.float64)
            td = np.asarray([float(r.get("tau_danger", np.nan)) for r in out_rows], dtype=np.float64)
            tx = np.asarray([float(r.get("tau_exit", np.nan)) for r in out_rows], dtype=np.float64)
            te = te[np.isfinite(te)]
            td = td[np.isfinite(td)]
            tx = tx[np.isfinite(tx)]
            if te.size and td.size and tx.size:
                print(
                    (
                        f"[ga] taus steps: "
                        f"E[min={float(te.min()):.2f} max={float(te.max()):.2f} std={float(te.std()):.2f}] "
                        f"D[min={float(td.min()):.2f} max={float(td.max()):.2f} std={float(td.std()):.2f}] "
                        f"X[min={float(tx.min()):.2f} max={float(tx.max()):.2f} std={float(tx.std()):.2f}]"
                    ),
                    flush=True,
                )
    except Exception:
        pass


if __name__ == "__main__":
    main()


