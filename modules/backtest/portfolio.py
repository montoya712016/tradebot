# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest de portfólio multi-ativo usando o WF mais recente.

Alinhado com o fluxo atual do single-symbol:
- contrato dinâmico por asset/timeframe
- cache de features coerente com o pipeline atual
- resolução correta do wf_* por asset
- overrides opcionais de tau por simulação
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import json
import time

import numpy as np
import pandas as pd

from backtest.sniper_walkforward import (
    load_period_models,
    predict_scores_walkforward,
    select_entry_mid,
    apply_threshold_overrides,
)
from backtest.sniper_portfolio import PortfolioConfig, SymbolData, simulate_portfolio
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL, _symbol_cache_paths, _cache_format
from trade_contract import DEFAULT_TRADE_CONTRACT, TradeContract
from plotting.plotting import plot_equity_and_correlation


_PLOT_CORR_KEEP_COLS = (
    "accept_ratio",
    "kept_avg_pair_corr",
    "kept_avg_market_corr",
    "open_max_corr",
    "open_weighted_mean_corr",
    "open_correlated_exposure",
    "post_weight",
)


def _prepare_corr_trace_for_plot(
    corr_trace: pd.DataFrame | None,
    *,
    max_points: int = 4000,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    if corr_trace is None or len(corr_trace) == 0:
        return None, None
    cdf = corr_trace.copy()
    if not isinstance(cdf.index, pd.DatetimeIndex):
        cdf.index = pd.to_datetime(cdf.index, errors="coerce")
    cdf = cdf[~cdf.index.isna()].sort_index()
    keep_cols = [c for c in _PLOT_CORR_KEEP_COLS if c in cdf.columns and pd.api.types.is_numeric_dtype(cdf[c])]
    if not keep_cols:
        return None, None
    cdf = cdf[keep_cols].copy()
    n = int(len(cdf))
    if n > int(max_points):
        step = int(np.ceil(float(n) / float(max_points)))
        cdf = cdf.iloc[::step].copy()
    return cdf, list(cdf.columns)


def _default_portfolio_cfg() -> PortfolioConfig:
    return PortfolioConfig(
        # Mais conservador:
        # - nÃ£o aloca 100% da carteira unica) usando o WF mais recente.
        # - limita tamanho mÃ¡ximo por trade
        # - limita posiÃ§Ãµes simultÃ¢neas
        max_positions=20,
        total_exposure=0.75,      # 75% do portfÃ³lio no mÃ¡ximo exposto
        max_trade_exposure=0.10,  # no mÃ¡ximo 10% por trade
        min_trade_exposure=0.03,  # ignora trades muito pequenos (<3%)
        exit_min_hold_bars=3,
        exit_confirm_bars=2,      # exige confirmaÃ§Ã£o extra para sair (menos churn)
        corr_filter_enabled=True,
        corr_window_bars=144,
        corr_min_obs=96,
        corr_max_with_market=0.80,
        corr_max_pair=0.85,
        corr_keep_top_n=1,
        corr_abs=True,
        corr_debug=False,
        corr_open_filter_enabled=True,
        corr_open_window_bars=144,
        corr_open_min_obs=96,
        corr_open_reduce_start=0.60,
        corr_open_hard_reject=0.92,
        corr_open_min_weight_mult=0.25,
    )


@dataclass
class PortfolioDemoSettings:
    asset_class: str = "crypto"
    run_dir: str | None = None  # None => usa wf_* mais recente em models_sniper/
    # Por padrao usamos 2 anos (para ficar consistente com outros backtests).
    # Se quiser 1 ano, mude para 365.
    days: int = 365 * 2
    max_symbols: int = 50
    # Para o demo nÃ£o precisa 5 anos; sÃ³ precisa o suficiente para cobrir `days` + warmup de features.
    total_days_cache: int = 365 * 2 + 180
    symbols: list[str] = field(default_factory=list)  # vazio => pega do meta.json
    exclude_symbols: list[str] = field(default_factory=list)
    cfg: PortfolioConfig = field(default_factory=_default_portfolio_cfg)
    save_plot: bool = True
    plot_out: str | None = None  # None => run_dir/plots_portfolio/portfolio_equity.html
    override_tau_entry: float | None = None
    candle_sec: int = 300
    contract: TradeContract | None = None
    long_only: bool = True
    require_feature_cache: bool = False
    rebuild_on_score_error: bool = True
    # Importante para portfÃ³lio realista/determinÃ­stico:
    # se True, usa a MESMA janela [end_global - days, end_global] para todos os sÃ­mbolos.
    # Se False, recorta "Ãºltimos days" por sÃ­mbolo (pode misturar perÃ­odos e variar muito).
    align_global_window: bool = True


@dataclass
class PreparedPortfolioData:
    run_dir: str
    contract: TradeContract
    candle_sec: int
    sym_data: dict[str, SymbolData]
    window_info: str
    tau_entry_default: float
    end_global: pd.Timestamp | None = None
    start_global: pd.Timestamp | None = None
    symbols_total: int = 0


def _best_run_contract() -> TradeContract:
    try:
        from config.trade_contract import (  # type: ignore
            CRYPTO_PIPELINE_CANDLE_SEC as _CRYPTO_PIPELINE_CANDLE_SEC,
            apply_crypto_pipeline_env as _apply_crypto_pipeline_env,
            build_default_crypto_contract as _build_default_crypto_contract,
        )

        candle_sec = _apply_crypto_pipeline_env(_CRYPTO_PIPELINE_CANDLE_SEC)
        return _build_default_crypto_contract(candle_sec)
    except Exception:
        return DEFAULT_TRADE_CONTRACT


def _default_flags_for_asset(asset_class: str) -> dict:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        try:
            from stocks import prepare_features_stocks as pfs  # type: ignore

            flags = dict(getattr(pfs, "FLAGS_STOCKS", {}))
            if flags:
                return flags
        except Exception:
            pass
    return dict(GLOBAL_FLAGS_FULL)


def _default_contract_for_asset(asset_class: str) -> TradeContract:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        try:
            from stocks.trade_contract import DEFAULT_TRADE_CONTRACT as STOCKS_CONTRACT  # type: ignore

            return STOCKS_CONTRACT
        except Exception:
            return DEFAULT_TRADE_CONTRACT
    return _best_run_contract()


def _find_latest_wf_dir(run_dir: str | None, asset_class: str | None = None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f"run_dir inválido: {p}")
        return p
    try:
        from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

        models_root = _models_root_for_asset(asset_class).resolve()
    except Exception:
        asset = str(asset_class or os.getenv("SNIPER_ASSET_CLASS", "crypto")).strip().lower()
        repo_root = Path(__file__).resolve().parents[2]
        models_root = (repo_root.parent / "models_sniper" / asset).resolve()
    if models_root.is_dir():
        wf_list = sorted([p for p in models_root.glob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if wf_list:
            return wf_list[-1]
    raise RuntimeError(f"Nenhum wf_* encontrado em {models_root}")


def _pick_symbols_from_meta(run_dir: Path, limit: int) -> list[str]:
    pd_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("period_") and p.name.endswith("d")])
    if not pd_dirs:
        return []
    pd_dirs.sort(key=lambda p: int(p.name.replace("period_", "").replace("d", "")))
    meta_path = pd_dirs[0] / "meta.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        syms = meta.get("symbols") or meta.get("symbols_used") or []
        if isinstance(syms, list) and syms:
            out = [str(s).upper() for s in syms]
            lim = int(limit)
            return out if lim <= 0 else out[:lim]
    except Exception:
        pass
    return []


def prepare_portfolio_data(settings: PortfolioDemoSettings | None = None) -> PreparedPortfolioData:
    settings = settings or PortfolioDemoSettings()
    asset = str(settings.asset_class or "crypto").lower()
    contract = settings.contract or _default_contract_for_asset(asset)
    candle_sec = int(max(1, int(getattr(contract, "timeframe_sec", settings.candle_sec) or settings.candle_sec)))

    run_dir = _find_latest_wf_dir(settings.run_dir, asset_class=asset)
    periods = load_period_models(run_dir)
    periods = apply_threshold_overrides(periods, tau_entry=settings.override_tau_entry)
    if bool(getattr(settings, "long_only", False)):
        filtered_periods = []
        for pm in periods:
            entry_models = dict(pm.entry_models or {})
            if "long" in entry_models:
                entry_models = {"long": entry_models["long"]}
            entry_cols_map = dict(pm.entry_cols_map or {})
            if "long" in entry_cols_map:
                entry_cols_map = {"long": list(entry_cols_map["long"])}
            entry_calib_map = dict(pm.entry_calib_map or {})
            if "long" in entry_calib_map:
                entry_calib_map = {"long": dict(entry_calib_map["long"])}
            tau_entry_map = dict(pm.tau_entry_map or {})
            if "long" in tau_entry_map:
                tau_entry_map = {"long": float(tau_entry_map["long"])}
            filtered_periods.append(
                replace(
                    pm,
                    entry_model=(entry_models.get("long") or pm.entry_model),
                    entry_models=entry_models,
                    entry_cols_map=entry_cols_map,
                    entry_calib_map=entry_calib_map,
                    tau_entry_map=tau_entry_map,
                )
            )
        periods = filtered_periods
        print("[backtest-portfolio] mode=long_only (short model disabled)", flush=True)

    symbols = [s.strip().upper() for s in (settings.symbols or []) if str(s).strip()]
    if not symbols:
        symbols = _pick_symbols_from_meta(run_dir, settings.max_symbols)
    exclude_symbols = {str(s).strip().upper() for s in (settings.exclude_symbols or []) if str(s).strip()}
    if exclude_symbols:
        symbols = [s for s in symbols if s not in exclude_symbols]
    max_symbols = int(settings.max_symbols)
    if max_symbols > 0:
        symbols = symbols[: max_symbols]
    # ordem estÃ¡vel (impacta apenas empates); ajuda reprodutibilidade
    symbols = sorted(symbols)
    if not symbols:
        raise RuntimeError("Sem sÃ­mbolos (defina settings.symbols ou verifique meta.json do WF)")

    flags = _default_flags_for_asset(asset)
    flags["_quiet"] = True
    cache_map = ensure_feature_cache(
        symbols,
        total_days=int(settings.total_days_cache),
        contract=contract,
        flags=flags,
        asset_class=asset,
        parallel=False,
        allow_build=not bool(getattr(settings, "require_feature_cache", False)),
    )
    symbols = [s for s in symbols if s in cache_map]
    if not symbols:
        raise RuntimeError("Nenhum sÃ­mbolo restou apÃ³s cache (ver logs [cache])")

    end_by_sym: dict[str, pd.Timestamp] = {}
    fmt = _cache_format()
    for sym in symbols:
        p = cache_map[sym]
        data_path, meta_path = _symbol_cache_paths(sym, p.parent, fmt)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            meta = {}
        end_ts_raw = meta.get("end_ts_utc")
        if end_ts_raw:
            try:
                end_by_sym[sym] = pd.to_datetime(end_ts_raw)
                continue
            except Exception:
                pass
        df0 = pd.read_parquet(p) if str(p).lower().endswith(".parquet") else pd.read_pickle(p)
        if df0 is None or df0.empty:
            continue
        end_by_sym[sym] = pd.to_datetime(df0.index.max())
        del df0

    if not end_by_sym:
        raise RuntimeError("Nenhum sÃ­mbolo com dados no cache para backtest")

    if settings.align_global_window:
        # usa o menor end_ts para garantir que TODOS tÃªm dados atÃ© o mesmo fim
        end_global = min(end_by_sym.values())
        start_global = end_global - pd.Timedelta(days=int(settings.days))
    else:
        end_global = None
        start_global = None

    sym_data: dict[str, SymbolData] = {}
    use_symbols = [s for s in symbols if s in end_by_sym]
    n_syms_total = len(use_symbols)
    for k, sym in enumerate(sorted(use_symbols), start=1):
        p = cache_map[sym]
        df0 = pd.read_parquet(p) if str(p).lower().endswith(".parquet") else pd.read_pickle(p)
        if df0 is None or df0.empty:
            continue
        idx = pd.to_datetime(df0.index)
        if settings.align_global_window and (end_global is not None) and (start_global is not None):
            m = (idx >= start_global) & (idx <= end_global)
            df = df0.loc[m].copy()
        else:
            end_ts = pd.to_datetime(df0.index.max())
            start_ts = end_ts - pd.Timedelta(days=int(settings.days))
            df = df0.loc[idx >= start_ts].copy()

        if len(df) < 1000:
            continue

        t_sym = time.perf_counter()
        try:
            p_entry_map, p_danger, p_exit, used, pid = predict_scores_walkforward(df, periods=periods, return_period_id=True)
        except Exception as e:
            # fallback: tenta rebuildar o cache do sÃ­mbolo 1x (parquet pode estar corrompido)
            print(f"[scores] WARN {sym}: {type(e).__name__}: {e} -> tentando rebuild do cache", flush=True)
            try:
                if not bool(getattr(settings, "rebuild_on_score_error", True)):
                    raise
                cache_map2 = ensure_feature_cache(
                    [sym],
                    total_days=int(settings.total_days_cache),
                    contract=contract,
                    flags=flags,
                    asset_class=asset,
                    parallel=False,
                    refresh=True,
                )
                p2 = cache_map2.get(sym)
                if p2:
                    df_retry = pd.read_parquet(p2) if str(p2).lower().endswith(".parquet") else pd.read_pickle(p2)
                    idx2 = pd.to_datetime(df_retry.index)
                    if settings.align_global_window and (end_global is not None) and (start_global is not None):
                        m2 = (idx2 >= start_global) & (idx2 <= end_global)
                        df = df_retry.loc[m2].copy()
                    else:
                        end_ts2 = pd.to_datetime(df_retry.index.max())
                        start_ts2 = end_ts2 - pd.Timedelta(days=int(settings.days))
                        df = df_retry.loc[idx2 >= start_ts2].copy()
                    p_entry_map, p_danger, p_exit, used, pid = predict_scores_walkforward(df, periods=periods, return_period_id=True)
                else:
                    raise
            except Exception as e2:
                print(f"[scores] ERROR {sym}: {type(e2).__name__}: {e2} (skip)", flush=True)
                continue
        p_entry = select_entry_mid(p_entry_map)
        dt_sym = time.perf_counter() - t_sym
        # progresso: ajuda a entender "onde estÃ¡ travado"
        if (k <= 3) or (k == n_syms_total) or (k % 5 == 0):
            print(f"[scores] {k}/{n_syms_total} {sym} rows={len(df):,} sec={dt_sym:.2f}".replace(",", "."), flush=True)
        keep_cols = [c for c in ("close", "high", "low") if c in df.columns]
        df_keep = df[keep_cols].copy()
        for c in keep_cols:
            try:
                df_keep[c] = pd.to_numeric(df_keep[c], errors="coerce").astype(np.float32, copy=False)
            except Exception:
                pass
        sym_data[sym] = SymbolData(
            df=df_keep,
            p_entry=np.asarray(p_entry, dtype=np.float32),
            p_danger=np.asarray(p_danger, dtype=np.float32),
            p_exit=np.asarray(p_exit, dtype=np.float32),
            tau_entry=float(used.tau_entry),
            tau_danger=1.0,
            tau_add=float(used.tau_add),
            tau_danger_add=1.0,
            tau_exit=1.0,
            period_id=pid,
            periods=periods,
        )
        del df0
        del df
        del df_keep

    if not sym_data:
        raise RuntimeError("Nenhum sÃ­mbolo com dados suficientes para backtest")

    if settings.align_global_window and (end_global is not None) and (start_global is not None):
        win_info = f"window={start_global.date()}..{end_global.date()}"
    else:
        # sÃ³ para indicar que nÃ£o estÃ¡ alinhado (pode misturar perÃ­odos)
        min_end = min(end_by_sym.values()) if end_by_sym else None
        max_end = max(end_by_sym.values()) if end_by_sym else None
        win_info = f"window=per-symbol end_min={min_end.date() if min_end is not None else 'NA'} end_max={max_end.date() if max_end is not None else 'NA'}"

    return PreparedPortfolioData(
        run_dir=str(run_dir),
        contract=contract,
        candle_sec=candle_sec,
        sym_data=sym_data,
        window_info=win_info,
        tau_entry_default=float(periods[0].tau_entry),
        end_global=end_global,
        start_global=start_global,
        symbols_total=len(sym_data),
    )


def run_prepared_portfolio(
    prepared: PreparedPortfolioData,
    *,
    cfg: PortfolioConfig,
    days: int,
    override_tau_entry: float | None = None,
    save_plot: bool = True,
    plot_out: str | None = None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    tau_entry = float(override_tau_entry) if override_tau_entry is not None else float(prepared.tau_entry_default)
    sym_data = {
        sym: replace(sd, tau_entry=tau_entry)
        for sym, sd in prepared.sym_data.items()
    }

    res = simulate_portfolio(sym_data, cfg=cfg, contract=prepared.contract, candle_sec=prepared.candle_sec)
    eq_end = float(res.equity_curve.iloc[-1]) if len(res.equity_curve) else 1.0
    ret_total = eq_end - 1.0
    dt = time.perf_counter() - t0

    print(
        f"PORTF symbols={len(sym_data)} days={days} "
        f"max_pos={cfg.max_positions} total_exp={cfg.total_exposure:.2f} "
        f"tau_entry={tau_entry:.2f} "
        f"eq={eq_end:.4f} ret={ret_total:+.2%} max_dd={res.max_dd:.2%} "
        f"trades={len(res.trades)} sec={dt:.2f} "
        f"wf={Path(prepared.run_dir).name} {prepared.window_info}"
    )

    plot_path: str | None = None
    if save_plot and len(res.equity_curve):
        out = Path(plot_out).expanduser().resolve() if plot_out else (Path(prepared.run_dir) / "plots_portfolio" / "portfolio_equity.html")
        title = (
            f"Portfolio equity | syms={len(sym_data)} days={days} "
            f"tau={tau_entry:.2f} "
            f"ret={ret_total:+.2%} maxDD={res.max_dd:.2%} trades={len(res.trades)}"
        )
        plot_equity_and_correlation(
            res.equity_curve,
            corr_df=None,
            corr_columns=None,
            title=title,
            save_path=out,
            show=False,
        )
        plot_path = str(out)
        print(f"[plot] salvo em: {out}", flush=True)
        try:
            (out.parent / "portfolio_equity.csv").write_text(res.equity_curve.to_csv(), encoding="utf-8")
        except Exception:
            pass
    return {
        "run_dir": str(prepared.run_dir),
        "eq_end": float(eq_end),
        "ret_total": float(ret_total),
        "plot_path": plot_path,
        "window_info": prepared.window_info,
        "symbols": list(sym_data.keys()),
        "result": res,
        "tau_entry": tau_entry,
    }


def run(settings: PortfolioDemoSettings | None = None) -> dict[str, object]:
    settings = settings or PortfolioDemoSettings()
    prepared = prepare_portfolio_data(settings)
    return run_prepared_portfolio(
        prepared,
        cfg=settings.cfg,
        days=int(settings.days),
        override_tau_entry=settings.override_tau_entry,
        save_plot=bool(settings.save_plot),
        plot_out=settings.plot_out,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()

