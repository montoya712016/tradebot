# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest single-symbol (5 modelos: Entry + Danger + Exit + ProfitReg + TimeReg) usando cache de features.

Regras:
- parâmetros definidos em código (sem ENV)
- usa cache parquet/pickle (features+labels) para evitar recalcular features
- simula walk-forward real (modelo escolhido por train_end_utc no WF)
- salva plot com: preço + faixas de trades, probs (3 modelos) e equity
"""

from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward_5, simulate_sniper_from_scores
from backtest.sniper_walkforward import apply_threshold_overrides
from dataclasses import replace
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL
from trade_contract import DEFAULT_TRADE_CONTRACT
from config.thresholds import DEFAULT_THRESHOLD_OVERRIDES


@dataclass
class SingleSymbolDemoSettings:
    symbol: str = "ADAUSDT"
    days: int = 2160
    candle_sec: int = 60
    # Se quiser fixar um WF específico, preencha; senão, pega o wf_* mais recente.
    run_dir: str | None = None
    # Cache (tamanho total que será carregado/garantido em disco)
    total_days_cache: int = 365 * 5
    # Saída/execução
    tp_hard_when_exit: bool = False
    exit_min_hold_bars: int = 3
    exit_confirm_bars: int = 1
    # Plot
    save_plot: bool = True
    plot_out: str = "out_sniper_single.png"
    # Diagnóstico (prints)
    print_signal_diagnostics: bool = True
    # Por padrão, usa os thresholds treinados em cada período do WF (meta.json).
    # Se quiser forçar valores fixos, preencha estes overrides.
    override_tau_entry: float | None = None
    override_tau_danger: float | None = None
    override_tau_exit: float | None = None
    # Estratégia 5-modelos (gating + exit por alvo previsto)
    min_expected_profit_pct: float = 0.006  # 0.6% (depois de custos ainda pode valer a pena)
    tp_target_frac_of_expected: float = 0.85
    exit_on_expected_time: bool = True
    exit_on_danger: bool = True


def _find_latest_wf_dir(run_dir: str | None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f"run_dir inválido: {p}")
        return p
    # usa paths oficiais do projeto (workspace/models_sniper)
    try:
        from utils.paths import models_root as _models_root  # type: ignore

        models_root = _models_root().resolve()
    except Exception:
        # fallback (layout inesperado)
        models_root = (Path(__file__).resolve().parents[2].parent / "models_sniper").resolve()

    if models_root.is_dir():
        wf_list = sorted([p for p in models_root.glob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if wf_list:
            return wf_list[-1]
    raise RuntimeError(f"Nenhum wf_* encontrado em {models_root} (verifique treino/paths)")


def _plot_result(
    df: pd.DataFrame,
    *,
    equity: np.ndarray,
    trades,
    p_entry: np.ndarray,
    p_danger: np.ndarray,
    p_exit: np.ndarray,
    p_profit: np.ndarray,
    p_time: np.ndarray,
    tau_entry: float,
    tau_danger: float,
    tau_exit: float,
    min_expected_profit_pct: float,
    timeout_bars: int,
    title: str,
    out_path: str | None,
) -> None:
    idx = pd.to_datetime(df.index)
    close = df["close"].to_numpy(np.float64, copy=False)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.2, 1.2, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[1, 0], sharex=ax0)
    axr = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax1 = fig.add_subplot(gs[3, 0], sharex=ax0)

    ax0.plot(idx, close, linewidth=1.0, color="black", alpha=0.9)
    ax0.set_title(title)
    ax0.grid(True, alpha=0.25)

    # faixas dos trades
    for t in trades:
        et = pd.to_datetime(t.entry_ts)
        xt = pd.to_datetime(t.exit_ts)
        r_net = float(getattr(t, "r_net", 0.0) or 0.0)
        color = "green" if r_net > 1e-12 else "red"
        ax0.axvspan(et, xt, color=color, alpha=0.18)

        # marcadores entry/exit (ajuda a ver o que "parece positivo" vs o que foi de fato)
        try:
            ei = int(idx.get_indexer([et], method="nearest")[0])
            xi = int(idx.get_indexer([xt], method="nearest")[0])
            ax0.scatter([idx[ei]], [close[ei]], marker="^", s=28, color=color, alpha=0.85)
            ax0.scatter([idx[xi]], [close[xi]], marker="v", s=28, color=color, alpha=0.85)
        except Exception:
            pass

        # Se o trade parece "subiu" (exit > entry) mas net < 0, anota (caso que você descreveu)
        try:
            entry_px = float(getattr(t, "entry_price", np.nan))
            exit_px = float(getattr(t, "exit_price", np.nan))
            gross_entry = (exit_px / entry_px - 1.0) if (np.isfinite(entry_px) and entry_px > 0 and np.isfinite(exit_px)) else np.nan
            costs = getattr(t, "costs", None)
            reason = str(getattr(t, "reason", ""))
            if np.isfinite(gross_entry) and (gross_entry > 0) and (r_net < 0):
                msg = f"net={r_net:+.2%} (gross(entry)={gross_entry:+.2%})"
                if costs is not None:
                    msg += f" costs~{float(costs):.2%}"
                if reason:
                    msg += f" {reason}"
                ax0.annotate(
                    msg,
                    xy=(xt, exit_px),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color="darkred",
                    alpha=0.9,
                )
        except Exception:
            pass

    # probabilidades + thresholds
    axp.plot(idx, p_entry, color="#1f77b4", alpha=0.9, linewidth=0.9, label="p_entry")
    axp.plot(idx, p_danger, color="#d62728", alpha=0.9, linewidth=0.9, label="p_danger")
    axp.plot(idx, p_exit, color="#2ca02c", alpha=0.9, linewidth=0.9, label="p_exit")
    axp.axhline(float(tau_entry), color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.7)
    axp.axhline(float(tau_danger), color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
    axp.axhline(float(tau_exit), color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.7)
    axp.set_ylabel("Prob")
    axp.legend(loc="upper left", ncol=3, fontsize=9)
    axp.grid(True, alpha=0.25)

    # regressões (profit esperado e tempo esperado)
    # - p_profit: fração (ex.: 0.02 = 2%)
    # - p_time: barras (1m => 60 barras = 1h)
    axr.plot(idx, p_profit, color="#9467bd", alpha=0.9, linewidth=0.9, label="profit_expected_pct")
    axr.axhline(float(min_expected_profit_pct), color="#9467bd", linestyle="--", linewidth=0.8, alpha=0.7)
    axr.set_ylabel("Profit exp. (frac)")
    axr.grid(True, alpha=0.25)
    axr2 = axr.twinx()
    axr2.plot(idx, p_time, color="#ff7f0e", alpha=0.85, linewidth=0.9, label="time_expected_bars")
    axr2.axhline(float(timeout_bars), color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.55)
    axr2.set_ylabel("Time exp. (bars)")

    # legenda combinada (axr + axr2)
    h1, l1 = axr.get_legend_handles_labels()
    h2, l2 = axr2.get_legend_handles_labels()
    axr.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=9)

    ax1.plot(idx, equity, linewidth=1.2, color="#1f77b4")
    ax1.set_ylabel("Equity")
    ax1.set_xlabel("Tempo")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    if out_path:
        Path(out_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
    plt.show()
    plt.close(fig)


def run(settings: SingleSymbolDemoSettings | None = None) -> None:
    settings = settings or SingleSymbolDemoSettings()
    t0 = time.perf_counter()

    symbol = settings.symbol.strip().upper()
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    run_dir = _find_latest_wf_dir(settings.run_dir)
    periods = load_period_models(run_dir)
    # aplica overrides (opcional)
    periods = apply_threshold_overrides(
        periods,
        tau_entry=settings.override_tau_entry,
        tau_danger=settings.override_tau_danger,
        tau_exit=settings.override_tau_exit,
    )

    # garante cache do símbolo e carrega df (features+labels+ohlc)
    cache_map = ensure_feature_cache(
        [symbol],
        total_days=int(settings.total_days_cache),
        contract=DEFAULT_TRADE_CONTRACT,
        flags=dict(GLOBAL_FLAGS_FULL, **{"_quiet": True}),
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
    p_entry, p_danger, p_exit, p_profit, p_time, used, pid = predict_scores_walkforward_5(
        df, periods=periods, return_period_id=True
    )
    tau_entry = float(settings.override_tau_entry) if settings.override_tau_entry is not None else float(used.tau_entry)
    tau_danger = float(settings.override_tau_danger) if settings.override_tau_danger is not None else float(used.tau_danger)
    tau_exit = float(settings.override_tau_exit) if settings.override_tau_exit is not None else float(getattr(used, "tau_exit", 1.0))

    # `simulate_sniper_from_scores` recebe um `PeriodModel` em `thresholds`.
    # Se houver override, cria uma cópia do período usado com thresholds alterados.
    thresholds = used
    if (settings.override_tau_entry is not None) or (settings.override_tau_danger is not None) or (settings.override_tau_exit is not None):
        thresholds = replace(
            used,
            tau_entry=float(tau_entry),
            tau_danger=float(tau_danger),
            tau_exit=float(tau_exit),
            # mantém consistência dos thresholds derivados
            tau_add=float(used.tau_add),
            tau_danger_add=float(used.tau_danger_add),
        )

    if settings.print_signal_diagnostics:
        pe = np.asarray(p_entry, dtype=np.float64)
        pdg = np.asarray(p_danger, dtype=np.float64)
        pex = np.asarray(p_exit, dtype=np.float64)
        ppr = np.asarray(p_profit, dtype=np.float64)
        ptt = np.asarray(p_time, dtype=np.float64)
        pe_q = np.nanquantile(pe, [0.5, 0.9, 0.95, 0.99])
        pd_q = np.nanquantile(pdg, [0.5, 0.9, 0.95, 0.99])
        px_q = np.nanquantile(pex, [0.5, 0.9, 0.95, 0.99])
        pr_q = np.nanquantile(ppr, [0.5, 0.9, 0.99])
        tt_q = np.nanquantile(ptt, [0.5, 0.9, 0.99])
        print(f"[diag] tau_entry={tau_entry:.4f} tau_danger={tau_danger:.4f} tau_exit={tau_exit:.4f}")
        print(f"[diag] p_entry q50/q90/q95/q99 = {pe_q}")
        print(f"[diag] p_danger q50/q90/q95/q99 = {pd_q}")
        print(f"[diag] p_exit q50/q90/q95/q99 = {px_q}")
        print(f"[diag] profit_pred_pct q50/q90/q99 = {pr_q}")
        print(f"[diag] time_pred_bars q50/q90/q99 = {tt_q}")
        # Importante: a regra é "entra se p_danger < tau_danger" (tau_danger alto = mais permissivo)
        m_valid = np.isfinite(pe) & np.isfinite(pdg)
        m_entry = m_valid & (pe >= float(tau_entry))
        pass_all = float(np.mean(pdg[m_valid] < float(tau_danger))) if np.any(m_valid) else float("nan")
        pass_when_entry = float(np.mean(pdg[m_entry] < float(tau_danger))) if np.any(m_entry) else float("nan")
        print(f"[diag] danger_pass_rate(all)={pass_all:.2%} | danger_pass_rate(when p_entry>=tau_entry)={pass_when_entry:.2%}")

    res = simulate_sniper_from_scores(
        df,
        p_entry=p_entry,
        p_danger=p_danger,
        p_exit=p_exit,
        p_profit=p_profit,
        p_time=p_time,
        thresholds=thresholds,
        periods=periods,
        period_id=pid,
        contract=DEFAULT_TRADE_CONTRACT,
        candle_sec=int(settings.candle_sec),
        tp_hard_when_exit=bool(settings.tp_hard_when_exit),
        exit_min_hold_bars=int(settings.exit_min_hold_bars),
        exit_confirm_bars=int(settings.exit_confirm_bars),
        use_profit_time=True,
        min_expected_profit_pct=float(settings.min_expected_profit_pct),
        tp_target_frac_of_expected=float(settings.tp_target_frac_of_expected),
        exit_on_expected_time=bool(settings.exit_on_expected_time),
        exit_on_danger=bool(settings.exit_on_danger),
    )

    eq_end = float(res.equity_curve[-1]) if len(res.equity_curve) else 1.0
    ret_total = eq_end - 1.0
    dt = time.perf_counter() - t0
    print(
        f"SINGLE sym={symbol} days={settings.days} trades={len(res.trades)} "
        f"eq={eq_end:.4f} ret={ret_total:+.2%} max_dd={float(res.max_dd):.2%} sec={dt:.2f}"
    )

    if settings.save_plot:
        timeout_bars = DEFAULT_TRADE_CONTRACT.timeout_bars(int(settings.candle_sec))
        _plot_result(
            df,
            equity=np.asarray(res.equity_curve, dtype=np.float64),
            trades=res.trades,
            p_entry=np.asarray(p_entry, dtype=np.float64),
            p_danger=np.asarray(p_danger, dtype=np.float64),
            p_exit=np.asarray(p_exit, dtype=np.float64),
            p_profit=np.asarray(p_profit, dtype=np.float64),
            p_time=np.asarray(p_time, dtype=np.float64),
            tau_entry=tau_entry,
            tau_danger=tau_danger,
            tau_exit=tau_exit,
            min_expected_profit_pct=float(settings.min_expected_profit_pct),
            timeout_bars=int(timeout_bars),
            title=f"{symbol} | days={settings.days} | ret={ret_total:+.2%} | trades={len(res.trades)}",
            out_path=settings.plot_out,
        )


def main() -> None:
    run()


if __name__ == "__main__":
    main()

