# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest WALK-FORWARD (portfolio multi-cripto) com taus base e opcional otimizacao.

Objetivo:
- Rodar um backtest por passos (step_days) ao longo de um range de anos
- Usar os modelos wf_* para Entry/Danger/ExitScore (Exit on-the-fly)
- Operar portfÃ³lio (capital compartilhado) com taus base: tau_entry/tau_danger/tau_exit
- Manter universo baseado em top_market_cap.txt, mas filtrar por "tem barras suficientes no step"
- Mostrar progresso por passo (percentual do tempo do step processado)
- Enviar Pushover a cada passo

Executar:
  python modules/backtest/wf_portfolio.py --run-dir D:/astra/models_sniper/wf_022 --tau-entry 0.80 --tau-danger 0.40 --tau-exit 0.85
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward
    from backtest.sniper_portfolio import PortfolioConfig, SymbolData, simulate_portfolio
    from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL
    from trade_contract import DEFAULT_TRADE_CONTRACT
    from config.symbols import load_top_market_cap_symbols
    from utils.paths import resolve_generated_path
except Exception:
    import sys

    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break
    from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward  # type: ignore[import]
    from backtest.sniper_portfolio import PortfolioConfig, SymbolData, simulate_portfolio  # type: ignore[import]
    from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore[import]
    from trade_contract import DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    from config.symbols import load_top_market_cap_symbols  # type: ignore[import]
    from utils.paths import resolve_generated_path  # type: ignore[import]

try:
    from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send
except Exception:
    try:
        from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore[import]
    except Exception:
        _pushover_load_default = None
        _pushover_send = None


def _set_thread_limits(n: int) -> int:
    try:
        n = int(max(1, int(n)))
    except Exception:
        n = 1
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "PYARROW_NUM_THREADS",
    ):
        try:
            os.environ[k] = str(n)
        except Exception:
            pass
    try:
        import xgboost as xgb  # type: ignore

        try:
            xgb.set_config(nthread=int(n))
        except Exception:
            pass
    except Exception:
        pass
    return int(n)


def _progress(it, *, total: int | None = None, desc: str = ""):
    """
    Barra de progresso (tqdm opcional). Fallback com ETA simples.
    """
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, total=total, desc=desc)
    except Exception:
        # fallback: imprime progresso bÃ¡sico (sem spammar)
        t0 = time.perf_counter()
        last = 0.0

        def _gen():
            nonlocal last
            i = 0
            n = int(total) if total is not None else None
            for x in it:
                i += 1
                now = time.perf_counter()
                if now - last >= 1.0:
                    last = now
                    if n:
                        pct = 100.0 * i / max(1, n)
                        avg = (now - t0) / max(1, i)
                        eta = avg * max(0, n - i)
                        print(f"[wf] {desc}: {i}/{n} ({pct:5.1f}%) ETA {eta/60.0:5.1f}m", flush=True)
                    else:
                        print(f"[wf] {desc}: {i}", flush=True)
                yield x

        return _gen()


def _read_parquet_best_effort(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame | None:
    p = Path(path)
    cols = list(columns) if columns else None
    if cols:
        try:
            return pd.read_parquet(p, columns=cols)
        except Exception:
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


def _needed_feature_columns(periods) -> list[str]:
    cols: set[str] = {"open", "high", "low", "close", "volume"}
    for pm in periods:
        cols.update([str(c) for c in (pm.entry_cols or [])])
        cols.update([str(c) for c in (pm.danger_cols or [])])
        cols.update([str(c) for c in (getattr(pm, "exit_cols", None) or [])])
    return sorted(cols)


def _downsample_df(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    s = int(stride)
    if s <= 1:
        return df
    return df.iloc[::s].copy()


def _prepare_symbol_frame_for_window(
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
        pe, pdg, _pex, _used, pid = predict_scores_walkforward(df, periods=periods, return_period_id=True)
    except RuntimeError:
        return df.iloc[0:0].copy()
    df = df.copy()
    df["__p_entry"] = np.asarray(pe, dtype=np.float32)
    df["__p_danger"] = np.asarray(pdg, dtype=np.float32)
    df["__period_id"] = np.asarray(pid, dtype=np.int16)
    # MantÃ©m somente OHLCV + colunas nÃ£o-cycle do Exit + colunas internas
    keep: set[str] = {"open", "high", "low", "close", "volume", "__p_entry", "__p_danger", "__period_id"}
    for pm in periods:
        for c in list(getattr(pm, "exit_cols", None) or []):
            cc = str(c)
            if cc.startswith("cycle_"):
                continue
            keep.add(cc)
    df = df[[c for c in df.columns if c in keep]].copy()
    return df


def _ret_pct(eq_end: float) -> float:
    try:
        return (float(eq_end) - 1.0) * 100.0
    except Exception:
        return 0.0


def _max_drawdown_from_curve(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    x = eq.to_numpy(np.float64, copy=False)
    peak = np.maximum.accumulate(x)
    dd = 1.0 - (x / np.maximum(1e-12, peak))
    return float(np.max(dd)) if len(dd) else 0.0


def _fmt_hms(seconds: float) -> str:
    try:
        s = int(max(0.0, float(seconds)))
    except Exception:
        s = 0
    if s < 60:
        return f"{s:d}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m:d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


def _mk_progress_cb(step_start: pd.Timestamp, step_end: pd.Timestamp) -> Callable[[pd.Timestamp], None]:
    t0 = pd.to_datetime(step_start)
    t1 = pd.to_datetime(step_end)
    span = float(max(1.0, (t1 - t0) / pd.Timedelta(seconds=1)))
    last_print = 0.0
    last_pct = -1.0
    wall_start = time.perf_counter()

    def _cb(t: pd.Timestamp) -> None:
        nonlocal last_print, last_pct
        now_wall = time.perf_counter()
        if now_wall - last_print < 0.75:
            return
        last_print = now_wall
        try:
            tt = pd.to_datetime(t)
            pct = float(((tt - t0) / pd.Timedelta(seconds=1)) / span) * 100.0
            pct = float(max(0.0, min(100.0, pct)))
        except Exception:
            pct = 0.0
        if int(pct) == int(last_pct):
            return
        last_pct = pct
        elapsed = float(max(0.0, now_wall - wall_start))
        # ETA simples por percentual (pode oscilar no comeÃ§o)
        if pct >= 1e-6:
            eta = elapsed * (100.0 / float(max(1e-6, pct)) - 1.0)
        else:
            eta = 0.0
        print(
            f"\r[wf] progresso step: {pct:5.1f}% | elapsed={_fmt_hms(elapsed)} ETA={_fmt_hms(eta)} | t={pd.to_datetime(t)}",
            end="",
            flush=True,
        )

    return _cb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=False, default=None)
    ap.add_argument("--symbols", type=str, default="", help="CSV. Se vazio, usa top_market_cap.txt")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--step-days", type=int, default=90)
    # IMPORTANTE: para backtest "verÃ­dico" em 1m, use 1.
    # Valores >1 fazem downsample (ex.: 5 => ~5m) e alteram a escala temporal efetiva do sistema.
    ap.add_argument("--bar-stride", type=int, default=1)
    ap.add_argument("--refresh-cache", action="store_true")

    ap.add_argument("--tau-entry", type=float, default=0.85)
    ap.add_argument("--tau-danger", type=float, default=0.40)
    ap.add_argument("--tau-exit", type=float, default=0.85)

    # Universo: usar histÃ³rico ao invÃ©s de sÃ³ o step atual (mais robusto)
    ap.add_argument(
        "--universe-history-mode",
        type=str,
        default="rolling",
        choices=["step", "expanding", "rolling"],
        help="Como calcular mÃ©tricas por sÃ­mbolo para seleÃ§Ã£o do prÃ³ximo step: "
        "'step' usa somente o step atual; 'expanding' usa todo histÃ³rico atÃ© o fim do step; "
        "'rolling' usa apenas os Ãºltimos N dias (ver --universe-history-days).",
    )
    ap.add_argument(
        "--universe-history-days",
        type=int,
        default=730,
        help="Janela (dias) quando universe-history-mode=rolling. Ex.: 730 ~ 2 anos.",
    )

    # OtimizaÃ§Ã£o simples de tau_entry por step (histÃ³rico -> aplica no step seguinte)
    ap.add_argument(
        "--tau-opt-mode",
        type=str,
        default="entry_grid",
        choices=["none", "entry_grid"],
        help="Otimiza thresholds por step para aplicar no step seguinte. "
        "'entry_grid' faz grid search apenas em tau_entry (tau_danger e tau_exit ficam fixos).",
    )
    ap.add_argument("--tau-opt-lookback-days", type=int, default=730, help="Janela histÃ³rica (dias) para otimizar tau (ex.: 730 ~ 2 anos).")
    ap.add_argument("--tau-opt-bar-stride", type=int, default=10, help="Downsample sÃ³ para otimizaÃ§Ã£o de tau (reduz custo).")
    ap.add_argument("--tau-opt-symbols", type=int, default=40, help="Qtd mÃ¡x de sÃ­mbolos usados para otimizar tau (subset dos selecionados).")
    ap.add_argument("--tau-opt-entry-min", type=float, default=0.60, help="MÃ­nimo do grid de tau_entry.")
    ap.add_argument("--tau-opt-entry-max", type=float, default=0.85, help="MÃ¡ximo do grid de tau_entry.")
    ap.add_argument("--tau-opt-entry-step", type=float, default=0.05)
    ap.add_argument("--tau-opt-min-trades", type=int, default=20, help="Exige mÃ­nimo de trades no histÃ³rico para aceitar o tau Ã³timo.")
    ap.add_argument("--tau-opt-ewm-decay", type=float, default=0.80, help="Suaviza mudanÃ§as de tau entre steps (0..1).")

    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--total-exposure", type=float, default=0.75)
    ap.add_argument("--max-trade-exposure", type=float, default=0.10)
    ap.add_argument("--min-trade-exposure", type=float, default=0.03)
    ap.add_argument("--exit-min-hold-bars", type=int, default=3)
    ap.add_argument("--exit-confirm-bars", type=int, default=2)

    # Universo dinÃ¢mico (walk-forward):
    # - step 0: opera TODOS os sÃ­mbolos disponÃ­veis no step
    # - step i: opera o universo "selecionado" no step i-1
    # - em todos os steps: ainda assim simula TODOS os sÃ­mbolos do step para selecionar o prÃ³ximo
    ap.add_argument("--no-dynamic-universe", action="store_true", help="Desativa seleÃ§Ã£o walk-forward de universo.")
    ap.add_argument(
        "--active-target",
        type=int,
        default=0,
        help="(LEGADO) Se >0, vira teto (max) de sÃ­mbolos ativos por step.",
    )
    ap.add_argument("--universe-ewm-decay", type=float, default=0.75, help="SuavizaÃ§Ã£o (0..1). Maior = mais estÃ¡vel.")
    ap.add_argument(
        "--universe-min-trades",
        type=int,
        default=0,
        help="(Opcional) MÃ­nimo de trades por sÃ­mbolo para aplicar filtros. 0 = nÃ£o filtra por trades.",
    )
    ap.add_argument(
        "--universe-use-effective-metrics",
        action="store_true",
        help="Usa mÃ©tricas 'shrinkadas' por nÃºmero de trades (recomendado).",
    )
    ap.add_argument(
        "--universe-use-raw-metrics",
        dest="universe_use_effective_metrics",
        action="store_false",
        help="Usa mÃ©tricas cruas (PF pode ficar inf com poucas trades).",
    )
    ap.set_defaults(universe_use_effective_metrics=True)
    ap.add_argument("--universe-trade-weight-k", type=float, default=20.0, help="Controle do shrinkage (maior = mais conservador).")
    ap.add_argument("--universe-pf-prior", type=float, default=0.25, help="Pseudo-loss/win para regularizar PF (evita inf).")
    ap.add_argument("--universe-pf-cap", type=float, default=10.0, help="Cap de PF para score (evita dominÃ¢ncia).")
    ap.add_argument(
        "--universe-score-mode",
        type=str,
        default="log_pf_x_trades",
        choices=["log_pf_x_trades", "pf_x_trades", "pfm1_x_trades"],
        help="Como combinar PF e nÂº de trades para ranquear/selecionar.",
    )
    ap.add_argument(
        "--universe-min-score",
        type=float,
        default=0.0,
        help="Filtro opcional: score mÃ­nimo para entrar no universo do prÃ³ximo step (penaliza baixa atividade).",
    )
    ap.add_argument("--universe-min-pf", type=float, default=1.0, help="Filtro principal: remove sÃ­mbolos com PF abaixo disso.")
    ap.add_argument("--universe-min-win", type=float, default=0.30, help="Filtro opcional de win_rate (0..1).")
    ap.add_argument("--universe-max-dd", type=float, default=1.0, help="Filtro opcional de drawdown (0..1).")
    ap.add_argument(
        "--universe-min-active",
        type=int,
        default=20,
        help="Se poucos passam no filtro, completa com os melhores scores (evita colapsar trades). Use 0 para desativar.",
    )
    ap.add_argument("--universe-max-active", type=int, default=0, help="Teto de sÃ­mbolos ativos (0 => sem teto).")
    ap.add_argument("--universe-max-changes", type=int, default=0, help="Se houver teto, limita novas entradas por step (0 => ilimitado).")

    ap.add_argument("--jobs", type=int, default=1, help="Apenas para parallelismo de dataset (I/O). SimulaÃ§Ã£o Ã© sequencial.")
    ap.add_argument("--xgb-threads", type=int, default=8)
    ap.add_argument("--pushover-user-env", type=str, default="PUSHOVER_USER_KEY")
    ap.add_argument("--pushover-token-env", type=str, default="PUSHOVER_TOKEN_TRADE")
    ap.add_argument("--pushover-title", type=str, default="tradebot WF backtest")
    ap.add_argument("--no-pushover", action="store_true")
    ap.add_argument("--out-dir", type=str, default="wf_backtest_fixed_tau")
    ap.add_argument("--no-plot", action="store_true", help="NÃ£o exibir/salvar grÃ¡fico ao final.")
    args = ap.parse_args()

    _set_thread_limits(int(getattr(args, "xgb_threads", 8) or 8))

    pushover_on = not bool(getattr(args, "no_pushover", False))
    pushover_cfg = None
    if pushover_on and (_pushover_load_default is not None) and (_pushover_send is not None):
        try:
            pushover_cfg = _pushover_load_default(
                user_env=str(getattr(args, "pushover_user_env", "PUSHOVER_USER_KEY")),
                token_env=str(getattr(args, "pushover_token_env", "PUSHOVER_TOKEN_TRADE")),
                token_name_fallback=str(getattr(args, "pushover_token_env", "PUSHOVER_TOKEN_TRADE")),
                title=str(getattr(args, "pushover_title", "tradebot WF backtest")),
            )
        except Exception:
            pushover_cfg = None

    def _notify(msg: str) -> None:
        if pushover_cfg is None or _pushover_send is None:
            return
        try:
            _pushover_send(msg, cfg=pushover_cfg)
        except Exception:
            return

    # Auto-detect run_dir
    if args.run_dir is None:
        try:
            paths_to_check = [
                Path("D:/astra/models_sniper"),
                Path(__file__).resolve().parents[2].parent / "models_sniper",
                Path.cwd().parent / "models_sniper",
                Path.cwd() / "models_sniper",
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
        raise RuntimeError("NÃ£o foi possÃ­vel encontrar wf_* automaticamente; passe --run-dir.")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"run_dir invÃ¡lido: {run_dir}")
    print(f"[wf] run_dir={run_dir}", flush=True)

    # SÃ­mbolos
    syms = [s.strip().upper() for s in str(getattr(args, "symbols", "")).split(",") if s.strip()]
    if not syms:
        syms = load_top_market_cap_symbols(limit=int(args.limit) if int(args.limit) > 0 else None)
        syms = [s.strip().upper() for s in syms if str(s).strip()]
    if int(args.max_symbols) > 0:
        syms = syms[: int(args.max_symbols)]
    if not syms:
        raise RuntimeError("Sem sÃ­mbolos (top_market_cap.txt vazio?)")

    # Cache de features (histÃ³rico suficiente para os steps)
    total_days_cache = int(args.years) * 365 + int(args.step_days) * 2 + 60
    cache_map = ensure_feature_cache(
        syms,
        total_days=int(total_days_cache),
        contract=DEFAULT_TRADE_CONTRACT,
        flags=dict(GLOBAL_FLAGS_FULL, **{"_quiet": True}),
        refresh=bool(getattr(args, "refresh_cache", False)),
        strict_total_days=True,
        parallel=True,
        max_workers=32,
    )
    syms = [s for s in syms if s in cache_map]
    if not syms:
        raise RuntimeError("Nenhum sÃ­mbolo restou apÃ³s cache")

    # Carrega perÃ­odos/modelos (CPU)
    periods_base = load_period_models(run_dir)
    # forÃ§a device=cpu
    try:
        for pm in (periods_base or []):
            try:
                pm.entry_model.set_param({"nthread": int(getattr(args, "xgb_threads", 8) or 8), "device": "cpu"})
            except Exception:
                pass
            try:
                pm.danger_model.set_param({"nthread": int(getattr(args, "xgb_threads", 8) or 8), "device": "cpu"})
            except Exception:
                pass
            try:
                em = getattr(pm, "exit_model", None)
                if em is not None:
                    em.set_param({"nthread": int(getattr(args, "xgb_threads", 8) or 8), "device": "cpu"})
            except Exception:
                pass
    except Exception:
        pass

    need_cols = _needed_feature_columns(periods_base)

    # start/end global via meta.json
    starts_by_sym: dict[str, pd.Timestamp] = {}
    ends_by_sym: dict[str, pd.Timestamp] = {}
    for s in syms:
        st, en = _read_cache_meta_times(Path(cache_map[s]))
        if st is None or en is None:
            continue
        starts_by_sym[s] = pd.to_datetime(st)
        ends_by_sym[s] = pd.to_datetime(en)
    if not ends_by_sym:
        raise RuntimeError("Sem end_ts nos caches")
    end_global = max(ends_by_sym.values())
    start_global = end_global - pd.Timedelta(days=int(args.years) * 365)

    # Ajuste para nÃ£o comeÃ§ar antes do primeiro modelo disponÃ­vel
    try:
        min_train_end = min(pd.to_datetime(pm.train_end_utc) for pm in (periods_base or []))
        if pd.to_datetime(start_global) < pd.to_datetime(min_train_end):
            print(f"[wf][warn] start_global={start_global} < min_train_end={min_train_end}; ajustando", flush=True)
            start_global = pd.to_datetime(min_train_end) + pd.Timedelta(minutes=1)
    except Exception:
        pass

    step = pd.Timedelta(days=int(args.step_days))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    t0 = pd.to_datetime(start_global)
    while t0 + step <= pd.to_datetime(end_global):
        windows.append((t0, t0 + step))
        t0 = t0 + step
    if not windows:
        raise RuntimeError("Sem janelas (windows=0).")

    tau_e = float(args.tau_entry)
    tau_d = float(args.tau_danger)
    tau_x = float(args.tau_exit)
    tau_add = float(min(0.99, max(0.01, tau_e * 1.10)))
    tau_dadd = float(min(0.99, max(0.01, tau_d * 0.90)))

    out_dir = resolve_generated_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "wf_cache" / run_dir.name
    cache_root.mkdir(parents=True, exist_ok=True)

    cfg = PortfolioConfig(
        max_positions=int(args.max_positions),
        total_exposure=float(args.total_exposure),
        max_trade_exposure=float(args.max_trade_exposure),
        min_trade_exposure=float(args.min_trade_exposure),
        exit_min_hold_bars=int(args.exit_min_hold_bars),
        exit_confirm_bars=int(args.exit_confirm_bars),
    )

    if pushover_on:
        _notify(
            f"WF backtest iniciado: run={run_dir.name} steps={len(windows)} "
            f"tau=(E{tau_e:.2f},D{tau_d:.2f},X{tau_x:.2f}) step={int(args.step_days)}d years={int(args.years)} syms={len(syms)}"
        )

    # Equity contÃ­nua (stitch por multiplicaÃ§Ã£o)
    eq_scale = 1.0
    equity_all: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    trades_csv = out_dir / "wf_trades.csv"
    symbol_stats_csv = out_dir / "wf_symbol_stats.csv"
    eval_trades_csv = out_dir / "wf_eval_trades.csv"
    eval_symbol_metrics_csv = out_dir / "wf_eval_symbol_metrics.csv"
    eval_symbol_metrics_full_csv = out_dir / "wf_eval_symbol_metrics_full.csv"
    universe_csv = out_dir / "wf_universe.csv"
    # evita append duplicado quando vocÃª re-roda o script
    for p in (trades_csv, symbol_stats_csv, eval_trades_csv, eval_symbol_metrics_csv, eval_symbol_metrics_full_csv, universe_csv):
        try:
            if p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            pass

    min_bars = 400
    stride = max(1, int(args.bar_stride))
    min_span = pd.Timedelta(minutes=int(min_bars) * int(stride))

    dynamic_universe_on = not bool(getattr(args, "no_dynamic_universe", False))
    prev_selected: list[str] = []  # universo selecionado no step anterior (para operar step atual)
    prev_active: list[str] = []  # universo realmente operado no step anterior (para churn quando hÃ¡ teto)
    ewm_score: dict[str, float] = {}  # score suavizado por sÃ­mbolo (para ranking do prÃ³ximo step)
    # histÃ³rico (eval) para mÃ©tricas robustas por sÃ­mbolo
    hist_by_sym: dict[str, list[tuple[pd.Timestamp, float]]] = {}

    def _score_symbol(*, pf: float, win: float, dd: float, trades: int, min_trades: int) -> float:
        # score focado em PF + win; DD entra como penalidade (menos peso); retorno bruto nÃ£o entra.
        # Nota: PF pode dar infinito quando nÃ£o hÃ¡ perdas; cap para nÃ£o dominar com poucas trades.
        pf_c = float(min(10.0, max(1e-6, float(pf))))
        win_c = float(min(1.0, max(0.0, float(win))))
        dd_c = float(min(0.999, max(0.0, float(dd))))
        s = float(np.log(pf_c)) + 3.0 * (win_c - 0.50) - 1.5 * dd_c
        if int(min_trades) > 0 and int(trades) < int(min_trades):
            # penaliza poucos trades (evita escolher por "sorte" / amostra pequena)
            short = float(int(min_trades) - int(trades)) / float(max(1, int(min_trades)))
            s -= 1.0 * short
        return float(s)

    for step_idx, (ws, we) in enumerate(windows):
        ws = pd.to_datetime(ws)
        we = pd.to_datetime(we)
        print(f"\n[wf] step={step_idx+1}/{len(windows)} window={ws}..{we}", flush=True)

        # taus "planejados" para o prÃ³ximo step (podem ser atualizados via tau-opt)
        tau_e_next = float(tau_e)
        tau_d_next = float(tau_d)
        tau_x_next = float(tau_x)
        tau_opt_mode_step = str(getattr(args, "tau_opt_mode", "none") or "none").strip().lower()
        uni_hist_mode_step = str(getattr(args, "universe_history_mode", "step") or "step").strip().lower()
        uni_hist_days_step = int(getattr(args, "universe_history_days", 730) or 730)

        # PrÃ©-filtro por meta (rÃ¡pido)
        syms_build: list[str] = []
        for s in syms:
            st = starts_by_sym.get(s)
            en = ends_by_sym.get(s)
            if st is None or en is None:
                continue
            ok = (pd.to_datetime(st) <= (we - min_span)) and (pd.to_datetime(en) >= (ws + min_span))
            if ok:
                syms_build.append(s)
        syms_build = sorted(set(syms_build))
        print(f"[wf] symbols: candidates={len(syms_build)}/{len(syms)} (min_bars={min_bars} stride={stride})", flush=True)
        if not syms_build:
            rows.append({"step_idx": step_idx, "start": str(ws), "end": str(we), "symbols": 0, "eq_end": 1.0})
            continue

        ds_dir = cache_root / f"step_{step_idx:03d}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        meta_p = ds_dir / "_wf_dataset_meta.json"
        need_rebuild = True
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                if str(meta.get("format")) == "wf_v1" and str(meta.get("t_start")) == str(ws) and str(meta.get("t_end")) == str(we):
                    meta_syms = [str(x).upper() for x in (meta.get("symbols") or [])]
                    if meta_syms == [str(x).upper() for x in syms_build] and any(ds_dir.glob("*.parquet")):
                        need_rebuild = False
            except Exception:
                need_rebuild = True

        if need_rebuild:
            # limpa
            try:
                for p in ds_dir.glob("*.parquet"):
                    p.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[wf] preparando dataset step (cols={len(need_cols)}) ...", flush=True)
            for sym in _progress(syms_build, total=len(syms_build), desc=f"prep step {step_idx}"):
                df_full = _read_parquet_best_effort(Path(cache_map[sym]), columns=need_cols)
                if df_full is None or df_full.empty:
                    continue
                df_step = _prepare_symbol_frame_for_window(
                    df_full,
                    periods=periods_base,
                    t_start=ws,
                    t_end=we,
                    bar_stride=int(args.bar_stride),
                )
                if df_step is None or df_step.empty or len(df_step) < min_bars:
                    continue
                df_step.to_parquet(ds_dir / f"{sym}.parquet", index=True)
            meta_p.write_text(
                json.dumps({"format": "wf_v1", "t_start": str(ws), "t_end": str(we), "symbols": list(syms_build)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # carrega frames
        frames: dict[str, pd.DataFrame] = {}
        for p in ds_dir.glob("*.parquet"):
            sym = p.stem.upper()
            if sym.startswith("_"):
                continue
            try:
                df = pd.read_parquet(p)
                # Normaliza Ã­ndice para tz-naive (evita crash tz-aware vs tz-naive na equity).
                try:
                    idx = pd.to_datetime(df.index, errors="coerce")
                    if isinstance(idx, pd.DatetimeIndex) and (idx.tz is not None):
                        idx = idx.tz_convert(None)
                    df.index = idx
                    df = df.loc[df.index.notna()].copy()
                except Exception:
                    pass
                if df is None or df.empty or len(df) < min_bars:
                    continue
                frames[sym] = df
            except Exception:
                continue
        syms_step = sorted(frames.keys())
        print(f"[wf] symbols_step={len(syms_step)}", flush=True)
        if not syms_step:
            rows.append({"step_idx": step_idx, "start": str(ws), "end": str(we), "symbols": 0, "eq_end": 1.0})
            continue

        # monta SymbolData
        sym_data: dict[str, SymbolData] = {}
        for sym, df in frames.items():
            try:
                pe = df["__p_entry"].to_numpy(np.float32, copy=False)
                pdg = df["__p_danger"].to_numpy(np.float32, copy=False)
                pid = df["__period_id"].to_numpy(np.int16, copy=False)
            except Exception:
                continue
            n = int(len(df))
            sym_data[sym] = SymbolData(
                df=df,
                p_entry=pe,
                p_danger=pdg,
                p_exit=np.full(n, np.nan, dtype=np.float32),
                tau_entry=float(tau_e),
                tau_danger=float(tau_d),
                tau_add=float(tau_add),
                tau_danger_add=float(tau_dadd),
                tau_exit=float(tau_x),
                period_id=pid,
                periods=list(periods_base),
            )

        # define universo ATIVO para operar este step (sempre baseado no step anterior)
        max_active = int(getattr(args, "universe_max_active", 0) or 0)
        # compat: se usuÃ¡rio passar --active-target, tratamos como teto (max_active)
        if max_active <= 0:
            at = int(getattr(args, "active_target", 0) or 0)
            if at > 0:
                max_active = at

        active_syms: list[str]
        if (not dynamic_universe_on) or (step_idx == 0) or (not prev_selected):
            # step 0 (ou sem histÃ³rico): opera tudo
            active_syms = list(syms_step)
        else:
            sel_avail = [s for s in prev_selected if s in sym_data]
            if not sel_avail:
                active_syms = list(syms_step)
            else:
                active_syms = sel_avail
                if max_active > 0:
                    active_syms = active_syms[: int(max_active)]

        # export universo operado (para auditoria)
        try:
            (out_dir / f"wf_active_symbols_step_{step_idx:03d}.txt").write_text("\n".join(active_syms) + "\n", encoding="utf-8")
        except Exception:
            pass
        print(
            f"[wf] active_universe={len(active_syms)}/{len(syms_step)} "
            f"(dynamic={dynamic_universe_on} max_active={max_active if max_active>0 else 'off'})",
            flush=True,
        )

        # 1) SIMULAÃ‡ÃƒO COMPLETA (todas as criptos do step) â€” usada SOMENTE para selecionar o universo do PRÃ“XIMO step
        #    (Isso resolve exatamente o seu problema: evitar operar "lixo" que apareceu anos depois)
        try:
            cfg_eval = PortfolioConfig(
                max_positions=0,  # ilimitado
                total_exposure=1e9,  # orÃ§amento "infinito" => aceita tudo
                max_trade_exposure=1.0,  # cada trade com peso ~1 (equity por sÃ­mbolo vira produto(1+r_net))
                min_trade_exposure=0.0,
                exit_min_hold_bars=int(cfg.exit_min_hold_bars),
                exit_confirm_bars=int(cfg.exit_confirm_bars),
                rank_mode=str(getattr(cfg, "rank_mode", "p_entry_minus_p_danger")),
            )
            pres_eval = simulate_portfolio(sym_data, cfg=cfg_eval, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60)  # type: ignore[arg-type]
            eval_trades = list(getattr(pres_eval, "trades", []) or [])

            # acumula trades (eval) para histÃ³rico robusto (seleÃ§Ã£o por PF/WR em janela maior)
            for tr in eval_trades:
                sym = str(getattr(tr, "symbol", "") or "").upper()
                if not sym:
                    continue
                exit_ts = getattr(tr, "exit_ts", None)
                if exit_ts is None:
                    continue
                try:
                    xts = pd.to_datetime(exit_ts)
                except Exception:
                    continue
                r = float(getattr(tr, "r_net", 0.0) or 0.0)
                hist_by_sym.setdefault(sym, []).append((xts, float(r)))

            # salva trades completos (eval) para debug
            if eval_trades:
                etrows: list[dict[str, Any]] = []
                for tr in eval_trades:
                    sym = str(getattr(tr, "symbol", "") or "").upper()
                    entry_ts = pd.to_datetime(getattr(tr, "entry_ts", None)) if getattr(tr, "entry_ts", None) is not None else None
                    exit_ts = pd.to_datetime(getattr(tr, "exit_ts", None)) if getattr(tr, "exit_ts", None) is not None else None
                    r = float(getattr(tr, "r_net", 0.0) or 0.0)
                    etrows.append(
                        {
                            "step_idx": int(step_idx),
                            "window_start": str(ws),
                            "window_end": str(we),
                            "symbol": sym,
                            "entry_ts": str(entry_ts) if entry_ts is not None else "",
                            "exit_ts": str(exit_ts) if exit_ts is not None else "",
                            "r_net": float(r),
                            "reason": str(getattr(tr, "reason", "") or ""),
                            "num_adds": int(getattr(tr, "num_adds", 0) or 0),
                        }
                    )
                pd.DataFrame(etrows).to_csv(
                    eval_trades_csv,
                    mode="a",
                    header=not eval_trades_csv.exists(),
                    index=False,
                    encoding="utf-8",
                )

            # mÃ©tricas por sÃ­mbolo (eval)
            min_tr = int(getattr(args, "universe_min_trades", 0) or 0)
            use_eff = bool(getattr(args, "universe_use_effective_metrics", True))
            k_tr = float(getattr(args, "universe_trade_weight_k", 20.0) or 20.0)
            k_tr = float(max(0.0, k_tr))
            pf_prior = float(getattr(args, "universe_pf_prior", 0.25) or 0.25)
            pf_prior = float(max(0.0, pf_prior))
            pf_cap = float(getattr(args, "universe_pf_cap", 10.0) or 10.0)
            pf_cap = float(max(1.0, pf_cap))
            # monta base de trades por sÃ­mbolo conforme o modo de histÃ³rico
            hist_mode = str(getattr(args, "universe_history_mode", "step") or "step").strip().lower()
            hist_days = int(getattr(args, "universe_history_days", 730) or 730)
            by_sym: dict[str, list[tuple[pd.Timestamp, float]]] = {}
            if hist_mode == "step":
                for tr in eval_trades:
                    sym = str(getattr(tr, "symbol", "") or "").upper()
                    r = float(getattr(tr, "r_net", 0.0) or 0.0)
                    xt = getattr(tr, "exit_ts", None)
                    try:
                        xts = pd.to_datetime(xt) if xt is not None else pd.Timestamp.min
                    except Exception:
                        xts = pd.Timestamp.min
                    if not sym:
                        continue
                    by_sym.setdefault(sym, []).append((xts, float(r)))
            else:
                # expanding/rolling: usa hist_by_sym acumulado e filtra atÃ© o fim do step
                t_hi = pd.to_datetime(we)
                if hist_mode == "rolling":
                    hist_days = max(1, int(hist_days))
                    try:
                        t_lo = t_hi - pd.Timedelta(days=int(hist_days))
                    except Exception:
                        t_lo = pd.Timestamp.min
                else:
                    t_lo = pd.Timestamp.min
                for sym in syms_step:
                    symu = str(sym).upper()
                    rs0 = list(hist_by_sym.get(symu, []))
                    if not rs0:
                        continue
                    rs = [(t, rr) for (t, rr) in rs0 if (t_lo <= t <= t_hi)]
                    if rs:
                        by_sym[symu] = rs

            mrows: list[dict[str, Any]] = []
            for sym in syms_step:
                rs = by_sym.get(sym, [])
                if not rs:
                    mrows.append(
                        {
                            "step_idx": int(step_idx),
                            "window_start": str(ws),
                            "window_end": str(we),
                            "symbol": sym,
                            "trades": 0,
                            "win_rate": 0.0,
                            "profit_factor": 0.0,
                            "ret_total": 0.0,
                            "max_dd": 0.0,
                            "score_raw": _score_symbol(pf=0.0, win=0.0, dd=0.0, trades=0, min_trades=min_tr),
                        }
                    )
                    continue

                rs.sort(key=lambda t: t[0])
                arr = np.asarray([x[1] for x in rs], dtype=np.float64)
                wins = float(arr[arr > 0].sum())
                losses = float((-arr[arr < 0]).sum())
                pf = float(wins / max(1e-12, losses)) if losses > 0 else float("inf")
                win = float(np.mean(arr > 0))
                # equity por sÃ­mbolo (somente em exits, mas suficiente para ranking)
                eq = 1.0
                peak = 1.0
                max_dd = 0.0
                for r in arr:
                    eq *= (1.0 + float(r))
                    if eq > peak:
                        peak = eq
                    dd = 1.0 - (eq / max(1e-12, peak))
                    if dd > max_dd:
                        max_dd = dd
                ret_total = float(eq - 1.0)

                # ---- mÃ©tricas "effective" (shrinkadas por nÂº de trades) ----
                # objetivo: evitar PF=inf e evitar que 1 trade domine seleÃ§Ã£o/ranking.
                ntr = int(arr.size)
                if k_tr > 0.0:
                    tw = float(ntr) / float(ntr + k_tr)
                else:
                    tw = 1.0
                # PF regularizado: adiciona pseudo-loss e pseudo-win iguais (prior neutro pf~1)
                pf_reg = float((wins + pf_prior) / max(1e-12, (losses + pf_prior)))
                pf_reg = float(min(pf_cap, max(1e-6, pf_reg)))
                pf_eff = float(1.0 + (pf_reg - 1.0) * float(tw))
                win_eff = float(0.5 + (float(win) - 0.5) * float(tw))
                # score principal (PF x atividade) com cap e sem infinito:
                pf_for_score = float(min(pf_cap, max(1e-6, pf_reg)))
                score_mode = str(getattr(args, "universe_score_mode", "log_pf_x_trades"))
                if score_mode == "pf_x_trades":
                    score_raw = float(ntr) * float(pf_for_score)
                elif score_mode == "pfm1_x_trades":
                    score_raw = float(ntr) * float(pf_for_score - 1.0)
                else:
                    # default: log(PF_cap) * trades (recomendado)
                    score_raw = float(ntr) * float(np.log(float(pf_for_score)))

                mrows.append(
                    {
                        "step_idx": int(step_idx),
                        "window_start": str(ws),
                        "window_end": str(we),
                        "symbol": sym,
                        "trades": int(arr.size),
                        "win_rate": float(win),
                        "win_rate_eff": float(win_eff),
                        "profit_factor": float(pf),
                        "profit_factor_reg": float(pf_reg),
                        "profit_factor_eff": float(pf_eff),
                        "ret_total": float(ret_total),
                        "max_dd": float(max_dd),
                        "trade_weight": float(tw),
                        "score_raw": float(score_raw),
                    }
                )

            dfm = pd.DataFrame(mrows)
            decay = float(getattr(args, "universe_ewm_decay", 0.75) or 0.75)
            decay = float(min(0.99, max(0.0, decay)))
            alpha = 1.0 - decay
            dfm["score_ewm"] = 0.0
            for i, r in dfm.iterrows():
                sym = str(r["symbol"])
                sraw = float(r["score_raw"])
                prev = float(ewm_score.get(sym, sraw))
                sewm = float(prev * decay + sraw * alpha)
                ewm_score[sym] = sewm
                dfm.at[i, "score_ewm"] = sewm

            # ranking para o PRÃ“XIMO step
            dfm_rank = dfm.copy()
            # prioridade: score_ewm desc; depois trades desc (desempate)
            dfm_rank.sort_values(["score_ewm", "trades"], ascending=[False, False], inplace=True, na_position="last")
            ranked_all = [str(s).upper() for s in dfm_rank["symbol"].astype(str).tolist() if str(s).strip()]

            # salva mÃ©tricas + ranking
            dfm.to_csv(
                eval_symbol_metrics_csv,
                mode="a",
                header=not eval_symbol_metrics_csv.exists(),
                index=False,
                encoding="utf-8",
            )

            # SeleÃ§Ã£o para o PRÃ“XIMO step (filtro por PF>=1 + confiabilidade por nÂº de trades)
            min_pf = float(getattr(args, "universe_min_pf", 1.0) or 1.0)
            min_win = float(getattr(args, "universe_min_win", 0.30) or 0.30)
            max_dd_f = float(getattr(args, "universe_max_dd", 1.0) or 1.0)
            min_tr = int(getattr(args, "universe_min_trades", 0) or 0)
            min_score = float(getattr(args, "universe_min_score", 0.0) or 0.0)
            max_changes = int(getattr(args, "universe_max_changes", 0) or 0)
            min_active = int(getattr(args, "universe_min_active", 0) or 0)
            use_eff = bool(getattr(args, "universe_use_effective_metrics", True))

            pf_col = "profit_factor_eff" if (use_eff and ("profit_factor_eff" in dfm_rank.columns)) else "profit_factor"
            win_col = "win_rate_eff" if (use_eff and ("win_rate_eff" in dfm_rank.columns)) else "win_rate"

            sel_mask = (dfm_rank[pf_col].astype(float) >= float(min_pf))
            if int(min_tr) > 0:
                sel_mask &= (dfm_rank["trades"].astype(int) >= int(min_tr))
            sel_mask &= (dfm_rank[win_col].astype(float) >= float(min_win))
            sel_mask &= (dfm_rank["max_dd"].astype(float) <= float(max_dd_f))
            # filtro por score (penaliza baixa atividade + "PF bom por acaso")
            if "score_raw" in dfm_rank.columns and float(min_score) > 0:
                sel_mask &= (dfm_rank["score_raw"].astype(float) >= float(min_score))
            selected_next = [str(s).upper() for s in dfm_rank.loc[sel_mask, "symbol"].astype(str).tolist() if str(s).strip()]

            # aplica teto (se houver) com churn limitado (se configurado)
            if max_active > 0:
                max_active = int(max(1, max_active))
                if max_changes > 0 and prev_active:
                    keep_n = max(0, int(max_active) - int(max_changes))
                    keep = [s for s in prev_active if s in selected_next][:keep_n]
                    keep_set = set(keep)
                    fill = [s for s in selected_next if s not in keep_set]
                    selected_next = keep + fill[: max(0, int(max_active) - len(keep))]
                else:
                    selected_next = selected_next[: int(max_active)]

            # garante mÃ­nimo (se pedido): completa com melhores scores mesmo que falhem no PF
            if min_active > 0 and len(selected_next) < int(min_active):
                need = int(min_active) - int(len(selected_next))
                sel_set = set(selected_next)
                # Prefere sÃ­mbolos que tiveram trades no step (evita completar com "no-signal").
                try:
                    trades_map = {str(r["symbol"]).upper(): int(r["trades"]) for _, r in dfm_rank[["symbol", "trades"]].iterrows()}
                except Exception:
                    trades_map = {}
                ranked_rest = [s for s in ranked_all if s not in sel_set]
                ranked_with_tr = [s for s in ranked_rest if int(trades_map.get(s, 0) or 0) > 0]
                ranked_zero_tr = [s for s in ranked_rest if int(trades_map.get(s, 0) or 0) <= 0]
                add = ranked_with_tr + ranked_zero_tr
                selected_next = list(selected_next) + add[: max(0, need)]

            # fallback de seguranÃ§a: se o filtro eliminar tudo, NÃƒO zera o universo (evita colapsar trades).
            if not selected_next:
                if prev_selected:
                    print("[wf][warn] seleÃ§Ã£o vazia; mantendo universo do step anterior.", flush=True)
                    selected_next = list(prev_selected)
                else:
                    print("[wf][warn] seleÃ§Ã£o vazia no step 0; usando ranking completo (sem filtros).", flush=True)
                    selected_next = list(ranked_all)
            prev_selected = list(selected_next)

            # salva seleÃ§Ã£o do prÃ³ximo step
            try:
                (out_dir / f"wf_selected_symbols_next_step_{step_idx:03d}.txt").write_text("\n".join(prev_selected) + "\n", encoding="utf-8")
            except Exception:
                pass

            # ------------------------------
            # OtimizaÃ§Ã£o de tau_entry (para o PRÃ“XIMO step)
            # ------------------------------
            tau_opt_mode = str(getattr(args, "tau_opt_mode", "none") or "none").strip().lower()
            if tau_opt_mode != "none" and int(step_idx) < int(len(windows) - 1):
                try:
                    # parÃ¢metros
                    lb_days = int(getattr(args, "tau_opt_lookback_days", 730) or 730)
                    lb_days = max(1, int(lb_days))
                    stride_opt = int(getattr(args, "tau_opt_bar_stride", 10) or 10)
                    stride_opt = max(1, int(stride_opt))
                    n_syms_opt = int(getattr(args, "tau_opt_symbols", 40) or 40)
                    n_syms_opt = max(1, int(n_syms_opt))
                    te_min = float(getattr(args, "tau_opt_entry_min", 0.60) or 0.60)
                    te_max = float(getattr(args, "tau_opt_entry_max", 0.85) or 0.85)
                    te_step = float(getattr(args, "tau_opt_entry_step", 0.05) or 0.05)
                    te_step = float(max(1e-6, te_step))
                    min_trades_opt = int(getattr(args, "tau_opt_min_trades", 20) or 20)
                    min_trades_opt = max(0, int(min_trades_opt))
                    tau_decay = float(getattr(args, "tau_opt_ewm_decay", 0.80) or 0.80)
                    tau_decay = float(min(0.99, max(0.0, tau_decay)))

                    # janela histÃ³rica (atÃ© o fim do step atual)
                    t_hi = pd.to_datetime(we)
                    t_lo = t_hi - pd.Timedelta(days=int(lb_days))

                    # escolhe subset de sÃ­mbolos (melhores do ranking e que estÃ£o no universo selecionado)
                    sel_set = set(str(s).upper() for s in (prev_selected or []))
                    df_sel = dfm_rank[dfm_rank["symbol"].astype(str).str.upper().isin(sel_set)].copy()
                    df_sel.sort_values(["score_ewm", "trades"], ascending=[False, False], inplace=True, na_position="last")
                    syms_opt = [str(s).upper() for s in df_sel.head(int(n_syms_opt))["symbol"].astype(str).tolist() if str(s).strip()]
                    if not syms_opt:
                        syms_opt = list(prev_selected)[: int(n_syms_opt)]

                    # carrega histÃ³rico (usa wf_cache/step_XXX jÃ¡ gerado; sem recomputar scores)
                    base_frames: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]] = {}
                    for sym in syms_opt:
                        parts: list[pd.DataFrame] = []
                        for j in range(int(step_idx) + 1):
                            wsj, wej = windows[int(j)]
                            wsj = pd.to_datetime(wsj)
                            wej = pd.to_datetime(wej)
                            if wej < t_lo or wsj > t_hi:
                                continue
                            p = cache_root / f"step_{int(j):03d}" / f"{str(sym).upper()}.parquet"
                            if not p.exists():
                                continue
                            try:
                                df0 = pd.read_parquet(p)
                            except Exception:
                                continue
                            try:
                                idx0 = pd.to_datetime(df0.index, errors="coerce")
                                if isinstance(idx0, pd.DatetimeIndex) and (idx0.tz is not None):
                                    idx0 = idx0.tz_convert(None)
                                df0.index = idx0
                                df0 = df0.loc[df0.index.notna()].copy()
                            except Exception:
                                pass
                            try:
                                df0 = df0.loc[(df0.index >= t_lo) & (df0.index <= t_hi)]
                            except Exception:
                                pass
                            if df0 is None or df0.empty:
                                continue
                            if stride_opt > 1:
                                df0 = df0.iloc[:: int(stride_opt)].copy()
                            # com stride alto (ou cache jÃ¡ downsampled), nÃ£o dÃ¡ pra exigir muitos pontos
                            if df0 is None or df0.empty or len(df0) < 50:
                                continue
                            parts.append(df0)
                        if not parts:
                            continue
                        try:
                            dfh = pd.concat(parts).sort_index()
                            dfh = dfh[~dfh.index.duplicated(keep="last")]
                        except Exception:
                            continue
                        need_cols = {"close", "__p_entry", "__p_danger", "__period_id"}
                        if not need_cols.issubset(set(dfh.columns)):
                            continue
                        try:
                            pe = dfh["__p_entry"].to_numpy(np.float32, copy=False)
                            pdg = dfh["__p_danger"].to_numpy(np.float32, copy=False)
                            pid = dfh["__period_id"].to_numpy(np.int16, copy=False)
                        except Exception:
                            continue
                        base_frames[str(sym).upper()] = (dfh, pe, pdg, pid)

                    if base_frames:
                        # grid de tau_entry
                        grid = []
                        x = float(te_min)
                        # evita problemas de float acumulado
                        while x <= float(te_max) + 1e-9:
                            grid.append(float(round(x + 1e-12, 2)))
                            x += float(te_step)

                        # (score, tau_entry, eq_end, trades)
                        # IMPORTANTE: nÃ£o otimizamos somente por equity total do lookback,
                        # senÃ£o quase sempre empurra tau_entry para o mÃ­nimo (overtrading e piora OOS).
                        best = None
                        for te in grid:
                            te = float(min(float(te_max), max(float(te_min), float(te))))
                            ta = float(min(0.99, max(0.01, te * 1.10)))
                            tda = float(min(0.99, max(0.01, float(tau_d) * 0.90)))

                            sym_data_opt: dict[str, SymbolData] = {}
                            for sym, (dfh, pe, pdg, pid) in base_frames.items():
                                n = int(len(dfh))
                                sym_data_opt[sym] = SymbolData(
                                    df=dfh,
                                    p_entry=pe,
                                    p_danger=pdg,
                                    p_exit=np.full(n, np.nan, dtype=np.float32),
                                    tau_entry=float(te),
                                    tau_danger=float(tau_d),
                                    tau_add=float(ta),
                                    tau_danger_add=float(tda),
                                    tau_exit=float(tau_x),
                                    period_id=pid,
                                    periods=list(periods_base),
                                )
                            pres_opt = simulate_portfolio(sym_data_opt, cfg=cfg, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60)  # type: ignore[arg-type]
                            eq_ser_opt = getattr(pres_opt, "equity_curve", None)
                            eq_end_opt = float(eq_ser_opt.iloc[-1]) if (eq_ser_opt is not None and len(eq_ser_opt) > 0) else 1.0
                            ntr_opt = int(len(getattr(pres_opt, "trades", []) or []))
                            # score: prioriza retorno RECENTE (Ãºltimo step_days) + qualidade (PF/win) + penaliza DD e extremos de trades.
                            max_dd_opt = float(getattr(pres_opt, "max_dd", 0.0) or 0.0)
                            # retorno recente (mesmo horizonte do prÃ³ximo step)
                            try:
                                eval_days = int(getattr(args, "step_days", 90) or 90)
                                eval_days = max(30, min(int(lb_days), int(eval_days)))
                            except Exception:
                                eval_days = 90
                            try:
                                if eq_ser_opt is not None and len(eq_ser_opt) > 0:
                                    t_eval0 = pd.to_datetime(eq_ser_opt.index.max()) - pd.Timedelta(days=int(eval_days))
                                    # valor mais recente <= t_eval0
                                    m0 = eq_ser_opt.index <= pd.to_datetime(t_eval0)
                                    if bool(m0.any()):
                                        eq0 = float(eq_ser_opt.loc[m0].iloc[-1])
                                    else:
                                        eq0 = float(eq_ser_opt.iloc[0])
                                else:
                                    eq0 = 1.0
                            except Exception:
                                eq0 = 1.0
                            ret_recent = float(np.log(max(1e-12, float(eq_end_opt) / max(1e-12, float(eq0)))))

                            # PF/win do portfÃ³lio (ponderado por weight)
                            trades_opt = list(getattr(pres_opt, "trades", []) or [])
                            pnl = []
                            for tr in trades_opt:
                                w = float(getattr(tr, "weight", 0.0) or 0.0)
                                r = float(getattr(tr, "r_net", 0.0) or 0.0)
                                pnl.append(float(w) * float(r))
                            if pnl:
                                wins = float(sum(x for x in pnl if x > 0.0))
                                losses = float(sum(-x for x in pnl if x < 0.0))
                                pf_opt = float(wins / max(1e-12, losses)) if losses > 0.0 else 10.0
                                win_opt = float(sum(1 for x in pnl if x > 0.0) / max(1, len(pnl)))
                            else:
                                pf_opt = 0.0
                                win_opt = 0.0
                            pf_opt = float(min(10.0, max(0.0, float(pf_opt))))

                            score = float(ret_recent)
                            # qualidade / robustez
                            score += 0.60 * float(np.log(max(1e-12, min(10.0, float(pf_opt)))))
                            score += 1.20 * float(float(win_opt) - 0.50)
                            score -= 2.00 * float(max(0.0, min(1.0, float(max_dd_opt))))
                            # penaliza PF<1 explicitamente
                            if float(pf_opt) < 1.0:
                                score -= 2.5 * float(1.0 - float(pf_opt))

                            # penalidade por trades muito baixos (amostra fraca)
                            if min_trades_opt > 0 and ntr_opt < int(min_trades_opt):
                                score -= 4.0 * float(int(min_trades_opt) - int(ntr_opt)) / float(max(1, int(min_trades_opt)))
                            # penalidade por overtrading (tende a destruir OOS em regimes ruins)
                            try:
                                base_min = 100.0
                                base_max = 800.0
                                scale = float(max(0.25, float(lb_days) / 180.0))
                                t_min = float(base_min * scale)
                                t_max = float(base_max * scale)
                                if float(ntr_opt) > float(t_max):
                                    score -= 1.75 * float(float(ntr_opt) - float(t_max)) / float(max(1.0, float(t_max)))
                            except Exception:
                                pass

                            cur = (float(score), float(te), float(eq_end_opt), int(ntr_opt))
                            if best is None or cur[0] > best[0]:
                                best = cur

                        if best is not None:
                            best_te = float(best[1])
                            # suaviza mudanÃ§a e quantiza na grade
                            te_sm = float(float(tau_e) * float(tau_decay) + float(best_te) * float(1.0 - float(tau_decay)))
                            te_q = float(round(round(te_sm / float(te_step)) * float(te_step) + 1e-12, 2))
                            te_q = float(min(float(te_max), max(float(te_min), float(te_q))))
                            tau_e_next = float(te_q)
                            print(
                                f"[wf] tau_opt(entry_grid): lookback={lb_days}d stride={stride_opt} syms={len(base_frames)}/{len(syms_opt)} "
                                f"best_te={best_te:.2f} best_eq={best[2]:.4f} best_trades={best[3]} -> tau_entry_next={tau_e_next:.2f}",
                                flush=True,
                            )
                    else:
                        print("[wf][warn] tau_opt: sem histÃ³rico suficiente (base_frames=0). Mantendo tau_entry.", flush=True)
                except Exception as e:
                    print(f"[wf][warn] tau_opt falhou: {type(e).__name__}: {e}", flush=True)

            # CSV "longo": mÃ©tricas de TODAS as criptos + flags de uso/seleÃ§Ã£o (para anÃ¡lise de consistÃªncia)
            try:
                used_set = {str(s).upper() for s in active_syms}
                selected_set = {str(s).upper() for s in prev_selected}
                df_full = dfm_rank.copy()
                df_full["used_this_step"] = df_full["symbol"].astype(str).str.upper().isin(used_set)
                df_full["selected_next_step"] = df_full["symbol"].astype(str).str.upper().isin(selected_set)
                df_full["passed_filters"] = sel_mask.to_numpy(dtype=bool, copy=False)
                # ranks (1 = melhor)
                df_full["rank_score_ewm"] = np.arange(1, len(df_full) + 1, dtype=np.int32)
                # metadados do step
                df_full["symbols_step"] = int(len(syms_step))
                # `sym_data_active` ainda nÃ£o existe aqui no step 0 (Ã© criado depois), entÃ£o usamos `active_syms`.
                df_full["symbols_active"] = int(len(active_syms))
                df_full["dynamic_universe"] = bool(dynamic_universe_on)
                df_full["max_active"] = int(max_active) if int(max_active) > 0 else 0
                df_full["filter_min_pf"] = float(min_pf)
                df_full["filter_min_win"] = float(min_win)
                df_full["filter_max_dd"] = float(max_dd_f)
                df_full["filter_min_trades"] = int(min_tr)
                df_full["filter_min_score"] = float(min_score)
                df_full["score_mode"] = str(getattr(args, "universe_score_mode", "log_pf_x_trades"))
                # taus e modos (para auditoria / consistÃªncia)
                df_full["tau_entry"] = float(tau_e)
                df_full["tau_danger"] = float(tau_d)
                df_full["tau_exit"] = float(tau_x)
                df_full["tau_entry_next"] = float(tau_e_next)
                df_full["tau_danger_next"] = float(tau_d_next)
                df_full["tau_exit_next"] = float(tau_x_next)
                df_full["tau_opt_mode"] = str(tau_opt_mode_step)
                df_full["universe_history_mode"] = str(uni_hist_mode_step)
                df_full["universe_history_days"] = int(uni_hist_days_step) if str(uni_hist_mode_step) == "rolling" else 0
                # IMPORTANTE: fixa a ordem de colunas para nÃ£o corromper o CSV quando o schema muda.
                schema = [
                    "step_idx",
                    "window_start",
                    "window_end",
                    "symbol",
                    "used_this_step",
                    "selected_next_step",
                    "passed_filters",
                    "rank_score_ewm",
                    "trades",
                    "win_rate",
                    "win_rate_eff",
                    "profit_factor",
                    "profit_factor_reg",
                    "profit_factor_eff",
                    "ret_total",
                    "max_dd",
                    "trade_weight",
                    "score_raw",
                    "score_ewm",
                    "symbols_step",
                    "symbols_active",
                    "dynamic_universe",
                    "max_active",
                    "filter_min_pf",
                    "filter_min_win",
                    "filter_max_dd",
                    "filter_min_trades",
                    "filter_min_score",
                    "score_mode",
                    "tau_entry",
                    "tau_danger",
                    "tau_exit",
                    "tau_entry_next",
                    "tau_danger_next",
                    "tau_exit_next",
                    "tau_opt_mode",
                    "universe_history_mode",
                    "universe_history_days",
                ]
                for c in schema:
                    if c not in df_full.columns:
                        df_full[c] = np.nan
                extra = [c for c in df_full.columns if c not in set(schema)]
                df_full = df_full[schema + extra]
                df_full.to_csv(
                    eval_symbol_metrics_full_csv,
                    mode="a",
                    header=not eval_symbol_metrics_full_csv.exists(),
                    index=False,
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[wf][warn] falhou ao salvar wf_eval_symbol_metrics_full.csv: {type(e).__name__}: {e}", flush=True)

            # log curto (top 5)
            try:
                top5 = dfm_rank.head(5)[["symbol", "trades", "win_rate", "profit_factor", "max_dd", "score_ewm"]]
                print("[wf] top5 (ranking):")
                print(top5.to_string(index=False))
                try:
                    top5_sel = dfm_rank[dfm_rank["symbol"].astype(str).str.upper().isin(set(prev_selected))].head(5)[
                        ["symbol", "trades", "win_rate", "profit_factor", "max_dd", "score_ewm"]
                    ]
                    print("[wf] top5 (selecionadas p/ prÃ³ximo step):")
                    print(top5_sel.to_string(index=False))
                except Exception:
                    pass
                print(
                    f"[wf] selected_next_step: n={len(prev_selected)} "
                    f"(min_trades={min_tr} min_pf={min_pf:.2f} min_win={min_win:.2f} max_dd={max_dd_f:.2f} min_score={min_score:.2f})",
                    flush=True,
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[wf][warn] avaliaÃ§Ã£o de universo falhou: {type(e).__name__}: {e}", flush=True)

        # 2) SIMULAÃ‡ÃƒO DO PORTFÃ“LIO REAL (apenas universo ativo do step atual)
        sym_data_active = {s: sym_data[s] for s in active_syms if s in sym_data}
        prev_active = list(active_syms)

        # export universo (csv agregador)
        try:
            pd.DataFrame(
                [
                    {
                        "step_idx": int(step_idx),
                        "window_start": str(ws),
                        "window_end": str(we),
                        "symbols_step": int(len(syms_step)),
                        "active_symbols": int(len(sym_data_active)),
                        "dynamic": bool(dynamic_universe_on),
                        "max_active": int(max_active),
                    }
                ]
            ).to_csv(universe_csv, mode="a", header=not universe_csv.exists(), index=False, encoding="utf-8")
        except Exception:
            pass

        # simula com callback de progresso (por tempo do step)
        cb = _mk_progress_cb(ws, we)
        pres = simulate_portfolio(sym_data_active, cfg=cfg, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60, progress_cb=cb)  # type: ignore[arg-type]
        print("")  # quebra a linha depois do \r

        eq_ser = getattr(pres, "equity_curve", None)
        if eq_ser is None or len(eq_ser) == 0:
            eq_ser = pd.Series([1.0], index=[we], name="equity")
        # Normaliza Ã­ndice para tz-naive (evita crash em concat/sort)
        try:
            eidx = pd.to_datetime(eq_ser.index, errors="coerce")
            if isinstance(eidx, pd.DatetimeIndex) and (eidx.tz is not None):
                eidx = eidx.tz_convert(None)
            eq_ser = pd.Series(eq_ser.to_numpy(np.float64, copy=False), index=eidx, name=str(getattr(eq_ser, "name", "equity")))
            eq_ser = eq_ser.loc[eq_ser.index.notna()].sort_index()
        except Exception:
            pass

        # stitch contÃ­nuo
        eq_scaled = (eq_ser / float(eq_ser.iloc[0])) * float(eq_scale)
        equity_all.append(eq_scaled.rename("equity"))
        eq_end = float(eq_ser.iloc[-1]) if len(eq_ser) else 1.0
        eq_scale *= float(eq_end)

        trades = list(getattr(pres, "trades", []) or [])
        pnl = []
        for tr in trades:
            w = float(getattr(tr, "weight", 0.0) or 0.0)
            r = float(getattr(tr, "r_net", 0.0) or 0.0)
            pnl.append(float(w) * float(r))
        # Export: trades + stats por sÃ­mbolo (ajuda a investigar "quais moedas puxaram pra cima/baixo")
        if trades:
            try:
                trows: list[dict[str, Any]] = []
                for tr in trades:
                    sym = str(getattr(tr, "symbol", "") or "").upper()
                    entry_ts = pd.to_datetime(getattr(tr, "entry_ts", None)) if getattr(tr, "entry_ts", None) is not None else None
                    exit_ts = pd.to_datetime(getattr(tr, "exit_ts", None)) if getattr(tr, "exit_ts", None) is not None else None
                    w = float(getattr(tr, "weight", 0.0) or 0.0)
                    r = float(getattr(tr, "r_net", 0.0) or 0.0)
                    trows.append(
                        {
                            "step_idx": int(step_idx),
                            "window_start": str(ws),
                            "window_end": str(we),
                            "symbol": sym,
                            "entry_ts": str(entry_ts) if entry_ts is not None else "",
                            "exit_ts": str(exit_ts) if exit_ts is not None else "",
                            "weight": float(w),
                            "r_net": float(r),
                            "pnl_w": float(w) * float(r),
                            "reason": str(getattr(tr, "reason", "") or ""),
                            "num_adds": int(getattr(tr, "num_adds", 0) or 0),
                        }
                    )
                df_tr = pd.DataFrame(trows)
                df_tr.to_csv(
                    trades_csv,
                    mode="a",
                    header=not trades_csv.exists(),
                    index=False,
                    encoding="utf-8",
                )

                # stats por sÃ­mbolo (aproxima contribuiÃ§Ãµes por "pnl_w" e por log-ret)
                df_tr["log1p_pnl_w"] = np.log1p(df_tr["pnl_w"].astype(np.float64))
                g = df_tr.groupby("symbol", dropna=False)
                df_sym = g.agg(
                    trades=("pnl_w", "size"),
                    pnl_w_sum=("pnl_w", "sum"),
                    pnl_w_pos=("pnl_w", lambda x: float(x[x > 0].sum())),
                    pnl_w_neg=("pnl_w", lambda x: float(x[x < 0].sum())),
                    win_rate=("pnl_w", lambda x: float((x > 0).mean()) if len(x) else 0.0),
                    log_ret_sum=("log1p_pnl_w", "sum"),
                ).reset_index()
                df_sym["step_idx"] = int(step_idx)
                df_sym["window_start"] = str(ws)
                df_sym["window_end"] = str(we)
                # retorno aproximado por sÃ­mbolo (composto pelas trades dele)
                df_sym["ret_approx"] = np.expm1(df_sym["log_ret_sum"].astype(np.float64))
                # profit factor simples
                df_sym["profit_factor"] = df_sym.apply(
                    lambda r: (float(r["pnl_w_pos"]) / max(1e-12, float(-r["pnl_w_neg"]))) if float(r["pnl_w_neg"]) < 0 else float("inf"),
                    axis=1,
                )
                # ordena para inspeÃ§Ã£o (maior contribuiÃ§Ã£o primeiro)
                df_sym.sort_values(["ret_approx", "pnl_w_sum"], ascending=[False, False], inplace=True, na_position="last")
                df_sym.to_csv(
                    symbol_stats_csv,
                    mode="a",
                    header=not symbol_stats_csv.exists(),
                    index=False,
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[wf][warn] falhou ao exportar trades/stats: {type(e).__name__}: {e}", flush=True)
        if len(pnl) == 0:
            wins = 0.0
            losses = 0.0
            pf = 0.0
            win_rate = 0.0
        else:
            wins = float(sum(x for x in pnl if x > 0.0))
            losses = float(sum(-x for x in pnl if x < 0.0))
            pf = float(wins / max(1e-12, losses)) if losses > 0.0 else float("inf")
            win_rate = float(sum(1 for x in pnl if x > 0.0) / max(1, len(pnl)))
        max_dd = float(getattr(pres, "max_dd", 0.0) or 0.0)
        if max_dd <= 0.0:
            max_dd = _max_drawdown_from_curve(eq_ser)

        row = {
            "step_idx": int(step_idx),
            "start": str(ws),
            "end": str(we),
            "symbols_step": int(len(syms_step)),
            "symbols_active": int(len(sym_data_active)),
            "tau_entry": float(tau_e),
            "tau_danger": float(tau_d),
            "tau_exit": float(tau_x),
            "tau_entry_next": float(tau_e_next),
            "tau_danger_next": float(tau_d_next),
            "tau_exit_next": float(tau_x_next),
            "tau_opt_mode": str(tau_opt_mode_step),
            "universe_history_mode": str(uni_hist_mode_step),
            "universe_history_days": int(uni_hist_days_step) if str(uni_hist_mode_step) == "rolling" else 0,
            "eq_end": float(eq_end),
            "ret_pct": float(_ret_pct(eq_end)),
            "max_dd": float(max_dd),
            "trades": int(len(trades)),
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
        }
        rows.append(row)

        print(
            f"[wf] step={step_idx+1}/{len(windows)} done: ret={row['ret_pct']:+.1f}% dd={row['max_dd']:.1%} "
            f"pf={row['profit_factor']:.2f} win={row['win_rate']:.2f} trades={row['trades']} "
            f"syms_active={row['symbols_active']}/{row['symbols_step']}",
            flush=True,
        )
        if pushover_on:
            _notify(
                f"WF step {step_idx+1}/{len(windows)}: ret={row['ret_pct']:+.1f}% dd={row['max_dd']:.0%} "
                f"pf={row['profit_factor']:.2f} win={row['win_rate']:.2f} "
                f"tr={row['trades']} syms_active={row['symbols_active']}/{row['symbols_step']} "
                f"tau=(E{tau_e:.2f},D{tau_d:.2f},X{tau_x:.2f})"
            )

        # salva incremental
        pd.DataFrame(rows).to_csv(out_dir / "wf_steps.csv", index=False, encoding="utf-8")
        if equity_all:
            eq_join = pd.concat(equity_all).sort_index()
            eq_join = eq_join[~eq_join.index.duplicated(keep="last")]
            eq_join.to_csv(out_dir / "wf_equity_curve.csv", index=True, encoding="utf-8")

        # aplica taus planejados (para o prÃ³ximo step)
        if float(tau_e_next) != float(tau_e) or float(tau_d_next) != float(tau_d) or float(tau_x_next) != float(tau_x):
            print(
                f"[wf] taus_next aplicados: "
                f"E {float(tau_e):.2f}->{float(tau_e_next):.2f} | "
                f"D {float(tau_d):.2f}->{float(tau_d_next):.2f} | "
                f"X {float(tau_x):.2f}->{float(tau_x_next):.2f}",
                flush=True,
            )
        tau_e = float(tau_e_next)
        tau_d = float(tau_d_next)
        tau_x = float(tau_x_next)

    print(f"\n[wf] concluÃ­do. out_dir={out_dir}", flush=True)
    if pushover_on:
        try:
            df_steps = pd.DataFrame(rows)
            total_eq = float(np.prod(df_steps["eq_end"].to_numpy(np.float64, copy=False))) if len(df_steps) else 1.0
            _notify(f"WF backtest concluÃ­do: steps={len(windows)} ret_total={_ret_pct(total_eq):+.1f}%")
        except Exception:
            _notify(f"WF backtest concluÃ­do: steps={len(windows)}")

    # Plot final: evoluÃ§Ã£o do capital / profit ao longo do tempo
    if not bool(getattr(args, "no_plot", False)):
        try:
            eq_path = out_dir / "wf_equity_curve.csv"
            if eq_path.exists():
                eq_df = pd.read_csv(eq_path)
                if "equity" in eq_df.columns and len(eq_df) > 0:
                    # primeira coluna Ã© o index salvo (timestamp)
                    ts_col = str(eq_df.columns[0])
                    t = pd.to_datetime(eq_df[ts_col])
                    eq = pd.Series(eq_df["equity"].to_numpy(np.float64, copy=False), index=t, name="equity").sort_index()
                else:
                    eq = None
            else:
                eq = None

            if eq is not None and len(eq) > 1:
                profit_pct = (eq - 1.0) * 100.0
                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
                    ax[0].plot(eq.index, eq.values, color="#2E86AB", linewidth=1.6)
                    ax[0].set_title("EvoluÃ§Ã£o do capital (equity)")
                    ax[0].set_ylabel("Equity")
                    ax[0].grid(True, alpha=0.25)

                    ax[1].plot(profit_pct.index, profit_pct.values, color="#18A558", linewidth=1.6)
                    ax[1].set_title("Profit acumulado (%)")
                    ax[1].set_ylabel("%")
                    ax[1].grid(True, alpha=0.25)

                    fig.tight_layout()
                    out_png = out_dir / "wf_profit_equity.png"
                    try:
                        fig.savefig(out_png, dpi=140)
                        print(f"[wf] grÃ¡fico salvo: {out_png}", flush=True)
                    except Exception:
                        pass
                    try:
                        plt.show()
                    except Exception:
                        pass
                except Exception:
                    print(
                        "[wf][plot][warn] matplotlib nÃ£o estÃ¡ disponÃ­vel; grÃ¡fico nÃ£o foi exibido/salvo. "
                        "VocÃª ainda pode usar wf_equity_curve.csv para plotar.",
                        flush=True,
                    )
        except Exception as e:
            print(f"[wf][plot][warn] falhou ao gerar grÃ¡fico: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

