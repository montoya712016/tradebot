# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest single-symbol (Entry + Danger) usando cache de features.

Regras:
- parâmetros definidos em código (sem ENV)
- usa cache parquet/pickle (features+labels) para evitar recalcular features
- simula walk-forward real (modelo escolhido por train_end_utc no WF)
- plota com plotly: candles, faixas de trades, probabilidades, sinais e equity
"""

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def _ensure_modules_on_sys_path() -> None:
    """
    Permite executar este arquivo direto (ex.: `python modules/backtest/single_symbol.py`)
    sem depender de PYTHONPATH.
    """
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from backtest.sniper_walkforward import (
    load_period_models,
    predict_scores_walkforward,
    simulate_sniper_from_scores,
    select_entry_mid,
    apply_threshold_overrides,
)
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL
from trade_contract import DEFAULT_TRADE_CONTRACT, TradeContract
from plotting.plotting import plot_backtest_single


def _best_run_contract() -> TradeContract:
    """
    Contrato alinhado com o melhor WF recente (janela 120m, min_profit 3.2%, alpha 0.5, EMA 120 / offset 0.2%).
    """
    base = DEFAULT_TRADE_CONTRACT
    return TradeContract(
        timeframe_sec=60,
        entry_label_windows_minutes=(120,),
        entry_label_min_profit_pcts=(0.032,),
        entry_label_weight_alpha=0.5,
        exit_ema_span=120,
        exit_ema_init_offset_pct=0.002,
        fee_pct_per_side=base.fee_pct_per_side,
        slippage_pct=base.slippage_pct,
        max_adds=base.max_adds,
        add_spacing_pct=base.add_spacing_pct,
        add_sizing=base.add_sizing,
        risk_max_cycle_pct=base.risk_max_cycle_pct,
        dd_intermediate_limit_pct=base.dd_intermediate_limit_pct,
        danger_drop_pct=base.danger_drop_pct,
        danger_recovery_pct=base.danger_recovery_pct,
        danger_timeout_hours=base.danger_timeout_hours,
        danger_fast_minutes=base.danger_fast_minutes,
        danger_drop_pct_critical=base.danger_drop_pct_critical,
        danger_stabilize_recovery_pct=base.danger_stabilize_recovery_pct,
        danger_stabilize_bars=base.danger_stabilize_bars,
    )


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


@dataclass
class SingleSymbolDemoSettings:
    asset_class: str = "crypto"
    symbol: str = "ADAUSDT"
    # janela parecida com o WF campeão (aprox. 6 anos)
    days: int = 6 * 365 + 30
    candle_sec: int = 60
    # Se quiser fixar um WF específico, preencha; senão, pega o wf_* mais recente.
    run_dir: str | None = None
    # Cache (tamanho total que será carregado/garantido em disco)
    total_days_cache: int = 365 * 6 + 30
    # Saída/execução
    exit_min_hold_bars: int = 0
    exit_confirm_bars: int = 2
    # Plot (html com plotly)
    save_plot: bool = True
    plot_out: str = "data/generated/plots/single_symbol_plot.html"
    plot_candles: bool = True
    # Diagnóstico (prints)
    print_signal_diagnostics: bool = True
    # Thresholds são definidos manualmente em config/thresholds.py.
    override_tau_entry: float | None = None
    disable_entry_calibration: bool = False
    long_only: bool = False
    # Contrato usado para cache/simulação
    contract: TradeContract | None = None


def _find_latest_wf_dir(run_dir: str | None, asset_class: str | None = None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f"run_dir inválido: {p}")
        return p
    # usa paths oficiais do projeto (workspace/models_sniper)
    try:
        from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

        models_root = _models_root_for_asset(asset_class).resolve()
    except Exception:
        # fallback (layout inesperado)
        asset = str(asset_class or "crypto").lower()
        models_root = (Path(__file__).resolve().parents[2].parent / "models_sniper" / asset).resolve()
    if models_root.is_dir():
        wf_list = sorted([p for p in models_root.glob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if wf_list:
            return wf_list[-1]
    raise RuntimeError(f"Nenhum wf_* encontrado em {models_root} (verifique treino/paths)")


def run(settings: SingleSymbolDemoSettings | None = None) -> None:
    settings = settings or SingleSymbolDemoSettings()
    t0 = time.perf_counter()

    asset = str(settings.asset_class or "crypto").lower()
    symbol = settings.symbol.strip().upper()
    if asset != "stocks" and not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    run_dir = _find_latest_wf_dir(settings.run_dir, asset_class=asset)
    periods = load_period_models(run_dir)
    if bool(getattr(settings, "disable_entry_calibration", False)):
        periods = [
            replace(
                pm,
                entry_calib={"type": "identity"},
                entry_calib_map={k: {"type": "identity"} for k in (pm.entry_calib_map or {}).keys()},
            )
            for pm in periods
        ]
        print("[backtest-single] entry calibration disabled (identity)", flush=True)
    # aplica overrides (opcional)
    periods = apply_threshold_overrides(
        periods,
        tau_entry=settings.override_tau_entry,
    )
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
        print("[backtest-single] mode=long_only (short model disabled)", flush=True)

    # garante cache do símbolo e carrega df (features+labels+ohlc)
    contract = settings.contract or _default_contract_for_asset(asset)
    flags = _default_flags_for_asset(asset)
    flags["_quiet"] = True
    cache_map = ensure_feature_cache(
        [symbol],
        total_days=int(settings.total_days_cache),
        contract=contract,
        flags=flags,
        asset_class=asset,
    )
    if symbol not in cache_map:
        raise RuntimeError(f"Cache indisponível para {symbol} (ver logs [cache])")

    p = cache_map[symbol]
    df = pd.read_parquet(p) if str(p).lower().endswith(".parquet") else pd.read_pickle(p)
    if df.empty:
        raise RuntimeError("df vazio")

    end_ts = pd.to_datetime(df.index.max())
    start_ts = end_ts - pd.Timedelta(days=int(settings.days))
    df = df.loc[pd.to_datetime(df.index) >= start_ts].copy()
    if len(df) < 1000:
        raise RuntimeError(f"Poucos candles para simular: rows={len(df)}")

    # scores WF (sem vazamento)
    p_entry_map, _p_danger_unused, _p_exit_unused, used, pid = predict_scores_walkforward(df, periods=periods, return_period_id=True)
    p_long = np.asarray(p_entry_map.get("long", np.full(len(df), np.nan, dtype=np.float32)), dtype=np.float32)
    p_short = np.asarray(p_entry_map.get("short", np.full(len(df), np.nan, dtype=np.float32)), dtype=np.float32)
    if bool(getattr(settings, "long_only", False)):
        p_short = np.full(len(df), np.nan, dtype=np.float32)
    p_entry = select_entry_mid(p_entry_map)
    tau_entry = float(settings.override_tau_entry) if settings.override_tau_entry is not None else float(used.tau_entry)
    p_danger = np.zeros(len(p_entry), dtype=np.float32)

    # `simulate_sniper_from_scores` recebe um `PeriodModel` em `thresholds`.
    # Se houver override, cria uma cópia do período usado com thresholds alterados.
    thresholds = used
    if settings.override_tau_entry is not None:
        thresholds = replace(
            used,
            tau_entry=float(tau_entry),
            # mantem consistencia dos thresholds derivados
            tau_add=float(used.tau_add),
        )

    # sinais usados pela estrategia (para plot/diagnostico)
    pe_long = np.asarray(p_long, dtype=np.float64)
    pe_short = np.asarray(p_short, dtype=np.float64)
    entry_sig = (pe_long >= float(tau_entry)) if bool(getattr(settings, "long_only", False)) else ((pe_long >= float(tau_entry)) | (pe_short >= float(tau_entry)))
    entry_ok = entry_sig

    res = simulate_sniper_from_scores(
        df,
        p_entry=p_long,
        p_entry_short=(None if bool(getattr(settings, "long_only", False)) else p_short),
        p_danger=p_danger,
        thresholds=thresholds,
        periods=periods,
        period_id=pid,
        contract=contract,
        candle_sec=int(settings.candle_sec),
        exit_min_hold_bars=int(settings.exit_min_hold_bars),
        exit_confirm_bars=int(settings.exit_confirm_bars),
    )

    eq_end = float(res.equity_curve[-1]) if len(res.equity_curve) else 1.0
    ret_total = eq_end - 1.0
    dt = time.perf_counter() - t0
    win_rate = float(getattr(res, "win_rate", 0.0) or 0.0)
    pf = float(getattr(res, "profit_factor", 0.0) or 0.0)
    pf_s = ("inf" if np.isinf(pf) else f"{pf:.2f}")
    print(
        f"SINGLE sym={symbol} days={settings.days} trades={len(res.trades)} "
        f"eq={eq_end:.4f} ret={ret_total:+.2%} max_dd={float(res.max_dd):.2%} "
        f"win={win_rate:.2%} pf={pf_s} sec={dt:.2f}"
    )

    ema_exit = np.full(len(df), np.nan, dtype=np.float32)
    try:
        from trade_contract import exit_ema_span_from_window

        span = exit_ema_span_from_window(contract, int(settings.candle_sec))
        alpha = 2.0 / float(span + 1) if span > 0 else 0.0
        offset = float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0)
        close = df["close"].to_numpy(np.float64, copy=False)
        idx = pd.to_datetime(df.index)
        for t in res.trades:
            try:
                entry_ts = pd.to_datetime(t.entry_ts)
                exit_ts = pd.to_datetime(t.exit_ts)
                entry_i = int(idx.get_indexer([entry_ts], method="nearest")[0])
                exit_i = int(idx.get_indexer([exit_ts], method="nearest")[0])
            except Exception:
                continue
            if entry_i < 0 or exit_i < entry_i:
                continue
            ema = float(close[entry_i]) * (1.0 - offset)
            ema_exit[entry_i] = ema
            for i in range(entry_i + 1, exit_i + 1):
                ema = ema + (alpha * (float(close[i]) - ema))
                ema_exit[i] = ema
    except Exception:
        ema_exit = None

    if settings.save_plot:
        plot_backtest_single(
            df,
            trades=res.trades,
            equity=np.asarray(res.equity_curve, dtype=np.float64),
            p_entry=np.asarray(p_entry, dtype=np.float64),
            p_entry_long=np.asarray(p_long, dtype=np.float64),
            p_entry_short=np.asarray(p_short, dtype=np.float64),
            entry_sig=np.asarray(entry_ok, dtype=bool),
            tau_entry=tau_entry,
            title=f"{symbol} | days={settings.days} | ret={ret_total:+.2%} | trades={len(res.trades)}",
            save_path=settings.plot_out,
            show=True,
            ema_exit=ema_exit,
            plot_probs=True,
            plot_signals=False,
            plot_candles=bool(settings.plot_candles),
        )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
