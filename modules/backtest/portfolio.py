# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest multi-cripto (carteira unica) usando o WF mais recente.

Regras:
- parÃ¢metros definidos em cÃ³digo (sem depender de ENV)
- usa cache parquet de features sniper quando disponÃ­vel
- gera plot da equity (e drawdown) ao final
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward, select_entry_mid
from backtest.sniper_walkforward import apply_threshold_overrides
from backtest.sniper_portfolio import PortfolioConfig, SymbolData, simulate_portfolio
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL
from trade_contract import DEFAULT_TRADE_CONTRACT
from config.thresholds import DEFAULT_THRESHOLD_OVERRIDES


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
    )


@dataclass
class PortfolioDemoSettings:
    run_dir: str | None = None  # None => usa wf_* mais recente em models_sniper/
    # Por padrao usamos 2 anos (para ficar consistente com outros backtests).
    # Se quiser 1 ano, mude para 365.
    days: int = 365 * 2
    max_symbols: int = 50
    # Para o demo nÃ£o precisa 5 anos; sÃ³ precisa o suficiente para cobrir `days` + warmup de features.
    total_days_cache: int = 365 * 2 + 180
    symbols: list[str] = field(default_factory=list)  # vazio => pega do meta.json
    cfg: PortfolioConfig = field(default_factory=_default_portfolio_cfg)
    save_plot: bool = True
    plot_out: str | None = None  # None => run_dir/plots_portfolio/portfolio_equity.png
    # Importante para portfÃ³lio realista/determinÃ­stico:
    # se True, usa a MESMA janela [end_global - days, end_global] para todos os sÃ­mbolos.
    # Se False, recorta "Ãºltimos days" por sÃ­mbolo (pode misturar perÃ­odos e variar muito).
    align_global_window: bool = True


def _find_latest_wf_dir(run_dir: str | None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if p.is_dir():
            return p
    # backtest -> modules -> repo_root(parent do repo) tem models_sniper/
    repo_root = Path(__file__).resolve().parents[2]
    models_root = (repo_root.parent / "models_sniper").resolve()
    if models_root.is_dir():
        wf_list = sorted([p for p in models_root.glob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if wf_list:
            return wf_list[-1]
    wf_list = sorted([p for p in Path.cwd().rglob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if wf_list:
        return wf_list[-1]
    raise RuntimeError("Nenhum wf_* encontrado (verifique models_sniper/)")


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
            return out[: max(0, int(limit))]
    except Exception:
        pass
    return []


def _plot_equity_curve(eq: pd.Series, *, out_path: Path, title: str) -> None:
    if eq is None or len(eq) == 0:
        return
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index)
    dd = (eq / eq.cummax()) - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(eq.index, eq.to_numpy(), color="#1f77b4", linewidth=1.5)
    ax1.set_title(title)
    ax1.set_ylabel("Equity")
    ax1.grid(True, alpha=0.2)

    ax2.fill_between(dd.index, dd.to_numpy(), 0.0, color="#d62728", alpha=0.25)
    ax2.plot(dd.index, dd.to_numpy(), color="#d62728", linewidth=1.0)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Tempo")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(settings: PortfolioDemoSettings | None = None) -> None:
    settings = settings or PortfolioDemoSettings()
    t0 = time.perf_counter()

    run_dir = _find_latest_wf_dir(settings.run_dir)
    periods = load_period_models(run_dir)
    # aplica overrides globais (simulaÃ§Ã£o)
    periods = apply_threshold_overrides(
        periods,
        tau_entry=DEFAULT_THRESHOLD_OVERRIDES.tau_entry,
        tau_danger=DEFAULT_THRESHOLD_OVERRIDES.tau_danger,
        tau_exit=DEFAULT_THRESHOLD_OVERRIDES.tau_exit,
    )

    symbols = [s.strip().upper() for s in (settings.symbols or []) if str(s).strip()]
    if not symbols:
        symbols = _pick_symbols_from_meta(run_dir, settings.max_symbols)
    symbols = symbols[: settings.max_symbols]
    # ordem estÃ¡vel (impacta apenas empates); ajuda reprodutibilidade
    symbols = sorted(symbols)
    if not symbols:
        raise RuntimeError("Sem sÃ­mbolos (defina settings.symbols ou verifique meta.json do WF)")

    cache_map = ensure_feature_cache(
        symbols,
        total_days=int(settings.total_days_cache),
        contract=DEFAULT_TRADE_CONTRACT,
        flags=dict(GLOBAL_FLAGS_FULL, **{"_quiet": True}),
        parallel=True,
        max_workers=32,
    )
    symbols = [s for s in symbols if s in cache_map]
    if not symbols:
        raise RuntimeError("Nenhum sÃ­mbolo restou apÃ³s cache (ver logs [cache])")

    # carrega primeiro todos os dfs para poder alinhar a janela global
    raw_dfs: dict[str, pd.DataFrame] = {}
    end_by_sym: dict[str, pd.Timestamp] = {}
    for sym in symbols:
        p = cache_map[sym]
        df0 = pd.read_parquet(p) if str(p).lower().endswith(".parquet") else pd.read_pickle(p)
        if df0 is None or df0.empty:
            continue
        end_ts = pd.to_datetime(df0.index.max())
        raw_dfs[sym] = df0
        end_by_sym[sym] = end_ts

    if not raw_dfs:
        raise RuntimeError("Nenhum sÃ­mbolo com dados no cache para backtest")

    if settings.align_global_window:
        # usa o menor end_ts para garantir que TODOS tÃªm dados atÃ© o mesmo fim
        end_global = min(end_by_sym.values())
        start_global = end_global - pd.Timedelta(days=int(settings.days))
    else:
        end_global = None
        start_global = None

    sym_data: dict[str, SymbolData] = {}
    n_syms_total = len(raw_dfs)
    for k, sym in enumerate(sorted(raw_dfs.keys()), start=1):
        # read df (pode ter cache corrompido em execuÃ§Ãµes anteriores)
        df0 = raw_dfs[sym]
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
                cache_map2 = ensure_feature_cache(
                    [sym],
                    total_days=int(settings.total_days_cache),
                    contract=DEFAULT_TRADE_CONTRACT,
                    flags=dict(GLOBAL_FLAGS_FULL, **{"_quiet": True}),
                    parallel=False,
                    refresh=True,
                )
            p_entry = select_entry_mid(p_entry_map)
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
        dt_sym = time.perf_counter() - t_sym
        # progresso: ajuda a entender "onde estÃ¡ travado"
        if (k <= 3) or (k == n_syms_total) or (k % 5 == 0):
            print(f"[scores] {k}/{n_syms_total} {sym} rows={len(df):,} sec={dt_sym:.2f}".replace(",", "."), flush=True)
        sym_data[sym] = SymbolData(
            df=df,
            p_entry=p_entry,
            p_danger=p_danger,
            p_exit=p_exit,
            tau_entry=float(used.tau_entry),
            tau_danger=float(used.tau_danger),
            tau_add=float(used.tau_add),
            tau_danger_add=float(used.tau_danger_add),
            tau_exit=float(getattr(used, "tau_exit", 1.0)),
            period_id=pid,
            periods=periods,
        )
            p_entry = select_entry_mid(p_entry_map)

    if not sym_data:
        raise RuntimeError("Nenhum sÃ­mbolo com dados suficientes para backtest")

    res = simulate_portfolio(sym_data, cfg=settings.cfg, contract=DEFAULT_TRADE_CONTRACT, candle_sec=60)
    eq_end = float(res.equity_curve.iloc[-1]) if len(res.equity_curve) else 1.0
    ret_total = eq_end - 1.0
    dt = time.perf_counter() - t0

    if settings.align_global_window and (end_global is not None) and (start_global is not None):
        win_info = f"window={start_global.date()}..{end_global.date()}"
    else:
        # sÃ³ para indicar que nÃ£o estÃ¡ alinhado (pode misturar perÃ­odos)
        min_end = min(end_by_sym.values()) if end_by_sym else None
        max_end = max(end_by_sym.values()) if end_by_sym else None
        win_info = f"window=per-symbol end_min={min_end.date() if min_end is not None else 'NA'} end_max={max_end.date() if max_end is not None else 'NA'}"

    print(
        f"PORTF symbols={len(sym_data)} days={settings.days} "
        f"max_pos={settings.cfg.max_positions} total_exp={settings.cfg.total_exposure:.2f} "
        f"eq={eq_end:.4f} ret={ret_total:+.2%} max_dd={res.max_dd:.2%} "
        f"trades={len(res.trades)} sec={dt:.2f} "
        f"wf={Path(run_dir).name} {win_info}"
    )

    if settings.save_plot and len(res.equity_curve):
        out = Path(settings.plot_out).expanduser().resolve() if settings.plot_out else (run_dir / "plots_portfolio" / "portfolio_equity.png")
        title = f"Portfolio equity | syms={len(sym_data)} days={settings.days} ret={ret_total:+.2%} maxDD={res.max_dd:.2%} trades={len(res.trades)}"
        _plot_equity_curve(res.equity_curve, out_path=out, title=title)
        print(f"[plot] salvo em: {out}", flush=True)
        try:
            (out.parent / "portfolio_equity.csv").write_text(res.equity_curve.to_csv(), encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    run()


if __name__ == "__main__":
    main()

