# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest single-symbol (Entry + Danger) usando cache de features.

Regras:
- parÃ¢metros definidos em cÃ³digo (sem ENV)
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
            for cand in (p.parent, p):
                sp = str(cand)
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
    Contrato alinhado com o pipeline crypto atual.
    O timeframe efetivo vem de `crypto.trade_contract.CRYPTO_PIPELINE_CANDLE_SEC`.
    """
    try:
        from crypto.trade_contract import (  # type: ignore
            CRYPTO_PIPELINE_CANDLE_SEC as _CRYPTO_PIPELINE_CANDLE_SEC,
            apply_crypto_pipeline_env as _apply_crypto_pipeline_env,
            build_default_crypto_contract as _build_default_crypto_contract,
        )

        candle_sec = _apply_crypto_pipeline_env(_CRYPTO_PIPELINE_CANDLE_SEC)
        return _build_default_crypto_contract(candle_sec)
    except Exception:
        base = DEFAULT_TRADE_CONTRACT
        return TradeContract(
            timeframe_sec=300,
            entry_label_windows_minutes=(240,),
            entry_label_min_profit_pcts=(0.01,),
            entry_label_weight_alpha=base.entry_label_weight_alpha,
            exit_ema_span=24,
            exit_ema_init_offset_pct=0.005,
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


_CRYPTO_DEFAULT_CANDLE_SEC = int(getattr(_best_run_contract(), "timeframe_sec", 300) or 300)


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
    # janela parecida com o WF campeÃ£o (aprox. 6 anos)
    days: int = 6 * 365 + 30
    candle_sec: int = _CRYPTO_DEFAULT_CANDLE_SEC
    # Se quiser fixar um WF especÃ­fico, preencha; senÃ£o, pega o wf_* mais recente.
    run_dir: str | None = None
    # Cache (tamanho total que serÃ¡ carregado/garantido em disco)
    total_days_cache: int = 365 * 6 + 30
    # SaÃ­da/execuÃ§Ã£o
    exit_min_hold_bars: int = 0
    exit_confirm_bars: int = 2
    exit_span_center_smooth: float = 0.90
    exit_span_window_pct: float = 0.20
    exit_span_window_steps: float = 2.0
    exit_span_rate_limit_pct: float = 0.10
    use_exit_model: bool = False
    # Plot (html com plotly)
    save_plot: bool = True
    plot_out: str = "data/generated/plots/single_symbol_plot.html"
    plot_candles: bool = True
    # DiagnÃ³stico (prints)
    print_signal_diagnostics: bool = True
    # Thresholds sÃ£o definidos manualmente em config/thresholds.py.
    override_tau_entry: float | None = None
    force_period_days: tuple[int, ...] = ()
    disable_entry_calibration: bool = False
    long_only: bool = False
    # Contrato usado para cache/simulaÃ§Ã£o
    contract: TradeContract | None = None


def _find_latest_wf_dir(run_dir: str | None, asset_class: str | None = None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f"run_dir invÃ¡lido: {p}")
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
    force_period_days = tuple(
        int(x) for x in (getattr(settings, "force_period_days", ()) or ()) if int(x) >= 0
    )
    if force_period_days:
        force_set = {int(x) for x in force_period_days}
        periods = [pm for pm in periods if int(pm.period_days) in force_set]
        if not periods:
            raise RuntimeError(f"Nenhum period_* correspondente a force_period_days={sorted(force_set)}")
        print(
            f"[backtest-single] forced periods={','.join(str(int(pm.period_days)) for pm in periods)}",
            flush=True,
        )
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

    # garante cache do sÃ­mbolo e carrega df (features+labels+ohlc)
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
        raise RuntimeError(f"Cache indisponÃ­vel para {symbol} (ver logs [cache])")

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
    p_entry_map, _p_danger_unused, p_exit_wf, used, pid = predict_scores_walkforward(df, periods=periods, return_period_id=True)
    p_long = np.asarray(p_entry_map.get("long", np.full(len(df), np.nan, dtype=np.float32)), dtype=np.float32)
    p_short = np.asarray(p_entry_map.get("short", np.full(len(df), np.nan, dtype=np.float32)), dtype=np.float32)
    if bool(getattr(settings, "long_only", False)):
        p_short = np.full(len(df), np.nan, dtype=np.float32)
    p_entry = select_entry_mid(p_entry_map)
    tau_entry = float(settings.override_tau_entry) if settings.override_tau_entry is not None else float(used.tau_entry)
    p_danger = np.zeros(len(p_entry), dtype=np.float32)
    # Diagnostico de saturacao do entry.
    def _print_entry_diag(name: str, arr: np.ndarray, tau: float) -> None:
        try:
            x = np.asarray(arr, dtype=np.float64)
            m = np.isfinite(x)
            n = int(m.sum())
            if n <= 0:
                print(f"[backtest-single] {name}: sem valores finitos", flush=True)
                return
            xf = x[m]
            q50 = float(np.nanquantile(xf, 0.50))
            q95 = float(np.nanquantile(xf, 0.95))
            q90 = float(np.nanquantile(xf, 0.90))
            q99 = float(np.nanquantile(xf, 0.99))
            q995 = float(np.nanquantile(xf, 0.995))
            q999 = float(np.nanquantile(xf, 0.999))
            q9995 = float(np.nanquantile(xf, 0.9995))
            q9999 = float(np.nanquantile(xf, 0.9999))
            hit_tau = float(np.mean(xf >= float(tau)))
            hit_08 = float(np.mean(xf >= 0.80))
            hit_09 = float(np.mean(xf >= 0.90))
            hit_095 = float(np.mean(xf >= 0.95))
            hit_099 = float(np.mean(xf >= 0.99))
            hit_0995 = float(np.mean(xf >= 0.995))
            hit_0999 = float(np.mean(xf >= 0.999))
            print(
                (
                    f"[backtest-single] {name}: rows={n}/{len(x)} "
                    f"q50={q50:.4f} q90={q90:.4f} q95={q95:.4f} q99={q99:.4f} q99.5={q995:.4f} q99.9={q999:.4f} "
                    f"q99.95={q9995:.4f} q99.99={q9999:.4f} "
                    f"ge_tau={hit_tau:.2%} ge_0.80={hit_08:.2%} ge_0.90={hit_09:.2%} ge_0.95={hit_095:.2%} "
                    f"ge_0.99={hit_099:.2%} ge_0.995={hit_0995:.2%} ge_0.999={hit_0999:.2%}"
                ),
                flush=True,
            )
        except Exception:
            pass

    _print_entry_diag("p_long", p_long, tau_entry)
    if not bool(getattr(settings, "long_only", False)):
        _print_entry_diag("p_short", p_short, tau_entry)
        try:
            ml = np.isfinite(p_long)
            ms = np.isfinite(p_short)
            m = ml & ms
            if int(m.sum()) > 10:
                corr = float(np.corrcoef(np.asarray(p_long[m], dtype=np.float64), np.asarray(p_short[m], dtype=np.float64))[0, 1])
                same = float(np.mean(np.abs(np.asarray(p_long[m], dtype=np.float64) - np.asarray(p_short[m], dtype=np.float64)) <= 1e-6))
                print(f"[backtest-single] entry_l_vs_s: corr={corr:.4f} equal_eps={same:.2%}", flush=True)
        except Exception:
            pass
    try:
        calib_long = dict((used.entry_calib_map or {}).get("long") or used.entry_calib or {"type": "identity"})
        calib_short = dict((used.entry_calib_map or {}).get("short") or used.entry_calib or {"type": "identity"})
        print(
            f"[backtest-single] calib long={str(calib_long.get('type', 'identity'))} short={str(calib_short.get('type', 'identity'))}",
            flush=True,
        )
    except Exception:
        pass

    use_exit_model = bool(getattr(settings, "use_exit_model", False))
    p_exit_input = (np.asarray(p_exit_wf, dtype=np.float32) if use_exit_model else None)

    # DiagnÃ³stico: confirma se o regresssor de exit estÃ¡ ativo no perÃ­odo usado.
    if use_exit_model:
        try:
            px = np.asarray(p_exit_wf, dtype=np.float64)
            finite = np.isfinite(px)
            n_fin = int(finite.sum())
            if n_fin > 0:
                q50 = float(np.nanquantile(px[finite], 0.50))
                q90 = float(np.nanquantile(px[finite], 0.90))
                print(
                    f"[backtest-single] exit_model ativo: rows={n_fin}/{len(px)} span_p50={q50:.1f} span_p90={q90:.1f}",
                    flush=True,
                )
            else:
                print("[backtest-single] exit_model inativo: p_exit sem valores finitos", flush=True)
        except Exception:
            pass
    else:
        print("[backtest-single] mode=entry_only (exit model disabled; fixed contract span)", flush=True)

    # `simulate_sniper_from_scores` recebe um `PeriodModel` em `thresholds`.
    # Se houver override, cria uma cÃ³pia do perÃ­odo usado com thresholds alterados.
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
        p_exit=p_exit_input,
        thresholds=thresholds,
        periods=periods,
        period_id=pid,
        contract=contract,
        candle_sec=int(settings.candle_sec),
        exit_min_hold_bars=int(settings.exit_min_hold_bars),
        exit_confirm_bars=int(settings.exit_confirm_bars),
        exit_span_center_smooth=float(settings.exit_span_center_smooth),
        exit_span_window_pct=float(settings.exit_span_window_pct),
        exit_span_window_steps=float(settings.exit_span_window_steps),
        exit_span_rate_limit_pct=float(settings.exit_span_rate_limit_pct),
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
    try:
        tr_ret = np.asarray([float(getattr(t, "r_net", 0.0) or 0.0) for t in (res.trades or [])], dtype=np.float64)
        if tr_ret.size > 0:
            pos = tr_ret[tr_ret > 0.0]
            neg = tr_ret[tr_ret < 0.0]
            gross_pos = float(np.sum(pos)) if pos.size else 0.0
            order = np.argsort(-tr_ret)
            best_idx = int(order[0])
            keep_wo_best = np.ones(tr_ret.size, dtype=bool)
            keep_wo_best[best_idx] = False
            eq_wo_best = float(np.prod(1.0 + tr_ret[keep_wo_best])) if int(keep_wo_best.sum()) > 0 else 1.0
            top3_n = int(min(3, tr_ret.size))
            keep_wo_top3 = np.ones(tr_ret.size, dtype=bool)
            keep_wo_top3[order[:top3_n]] = False
            eq_wo_top3 = float(np.prod(1.0 + tr_ret[keep_wo_top3])) if int(keep_wo_top3.sum()) > 0 else 1.0
            top1_share = (float(tr_ret[best_idx]) / gross_pos) if gross_pos > 1e-12 and tr_ret[best_idx] > 0.0 else 0.0
            top3_share = (float(np.sum(np.clip(tr_ret[order[:top3_n]], 0.0, None))) / gross_pos) if gross_pos > 1e-12 else 0.0
            q25_tr = float(np.nanquantile(tr_ret, 0.25))
            q50_tr = float(np.nanquantile(tr_ret, 0.50))
            q75_tr = float(np.nanquantile(tr_ret, 0.75))
            avg_win = float(np.mean(pos)) if pos.size else 0.0
            avg_loss = float(np.mean(neg)) if neg.size else 0.0
            payoff = (avg_win / abs(avg_loss)) if abs(avg_loss) > 1e-12 else np.inf
            expectancy = float(np.mean(tr_ret))
            print(
                (
                    f"[backtest-single] trade_dist: mean={float(np.mean(tr_ret)):+.2%} "
                    f"p25={q25_tr:+.2%} p50={q50_tr:+.2%} p75={q75_tr:+.2%} "
                    f"best={float(np.max(tr_ret)):+.2%} worst={float(np.min(tr_ret)):+.2%}"
                ),
                flush=True,
            )
            print(
                (
                    f"[backtest-single] trade_edge: wins={int(pos.size)} losses={int(neg.size)} "
                    f"avg_win={avg_win:+.2%} avg_loss={avg_loss:+.2%} "
                    f"payoff={('inf' if np.isinf(payoff) else f'{payoff:.2f}')} expectancy={expectancy:+.2%}"
                ),
                flush=True,
            )
            print(
                (
                    f"[backtest-single] trade_conc: top1_share={top1_share:.2%} top3_share={top3_share:.2%} "
                    f"ret_wo_best={eq_wo_best - 1.0:+.2%} ret_wo_top3={eq_wo_top3 - 1.0:+.2%}"
                ),
                flush=True,
            )
    except Exception:
        pass
    try:
        su = np.asarray(getattr(res, "exit_span_curve", None), dtype=np.float64)
        m = np.isfinite(su)
        if int(m.sum()) > 0:
            q10 = float(np.nanquantile(su[m], 0.10))
            q50 = float(np.nanquantile(su[m], 0.50))
            q90 = float(np.nanquantile(su[m], 0.90))
            print(f"[backtest-single] span_used p10={q10:.1f} p50={q50:.1f} p90={q90:.1f}", flush=True)
    except Exception:
        pass

    ema_exit = None
    span_used = None
    try:
        if getattr(res, "ema_exit_curve", None) is not None:
            ema_exit = np.asarray(res.ema_exit_curve, dtype=np.float32)
        if getattr(res, "exit_span_curve", None) is not None:
            span_used = np.asarray(res.exit_span_curve, dtype=np.float32)
    except Exception:
        ema_exit = None
        span_used = None

    if settings.save_plot:
        plot_backtest_single(
            df,
            trades=res.trades,
            equity=np.asarray(res.equity_curve, dtype=np.float64),
            p_entry=np.asarray(p_entry, dtype=np.float64),
            p_entry_long=np.asarray(p_long, dtype=np.float64),
            p_entry_short=np.asarray(p_short, dtype=np.float64),
            p_exit=(np.asarray(p_exit_wf, dtype=np.float64) if use_exit_model else None),
            p_exit_used=(np.asarray(span_used, dtype=np.float64) if span_used is not None else None),
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

