# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Pipeline completo hÃ­brido (WF):
1) Exporta sinais supervisionados long/short
2) Treina RL (DQN) em walk-forward temporal global
3) Avalia RL vs baseline supervisionado
4) Roda backtests agregados (supervised-only e supervised+RL)
"""

from pathlib import Path
import argparse
import json
import sys
import time
from typing import Iterable


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            repo_root = p.parent
            for cand in (repo_root, p):
                sp = str(cand)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from supervised.inference_long_short import SignalExportConfig, export_long_short_signals  # type: ignore
from rl.train_rl import TrainRLConfig, train_rl_walkforward  # type: ignore
from rl.evaluate_rl import EvaluateConfig, evaluate_rl_run  # type: ignore
from backtest.backtest_supervised_only import run as run_sup_only  # type: ignore
from backtest.backtest_supervised_plus_rl import run as run_sup_plus_rl  # type: ignore


def _default_storage_root() -> Path:
    # .../tradebot/modules/rl/run_hybrid_wf_pipeline.py -> .../astra/models_sniper/hybrid_rl
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


def run_pipeline(
    *,
    signals_cfg: SignalExportConfig,
    rl_cfg: TrainRLConfig,
    eval_cfg: EvaluateConfig,
    base_out_dir: Path,
) -> dict:
    def _fmt_secs(dt: float) -> str:
        return f"{dt:.2f}s"

    t0_all = time.perf_counter()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    signals_path = export_long_short_signals(signals_cfg)
    dt_signals = time.perf_counter() - t0
    print(f"[pipeline][time] export_signals={_fmt_secs(dt_signals)} path={signals_path}", flush=True)

    rl_cfg.signals_path = str(signals_path)
    t0 = time.perf_counter()
    run_dir = train_rl_walkforward(rl_cfg)
    dt_train = time.perf_counter() - t0
    print(f"[pipeline][time] train_rl={_fmt_secs(dt_train)} run_dir={run_dir}", flush=True)

    eval_cfg.signals_path = str(signals_path)
    eval_cfg.run_dir = str(run_dir)
    t0 = time.perf_counter()
    eval_csv = evaluate_rl_run(eval_cfg)
    dt_eval = time.perf_counter() - t0
    print(f"[pipeline][time] evaluate_rl={_fmt_secs(dt_eval)} csv={eval_csv}", flush=True)

    t0 = time.perf_counter()
    sup_only_json = run_sup_only(
        signals_path=str(signals_path),
        out_json=str(base_out_dir / "backtest_supervised_only.json"),
        min_rows_symbol=int(rl_cfg.min_rows_symbol),
        edge_threshold=float(eval_cfg.edge_threshold),
        strength_threshold=float(eval_cfg.strength_threshold),
        fee_rate=float(rl_cfg.fee_rate),
        slippage_rate=float(rl_cfg.slippage_rate),
        small_size=float(rl_cfg.small_size),
        big_size=float(rl_cfg.big_size),
        cooldown_bars=int(rl_cfg.cooldown_bars),
        min_reentry_gap_bars=int(rl_cfg.min_reentry_gap_bars),
        min_hold_bars=int(rl_cfg.min_hold_bars),
        edge_entry_threshold=float(rl_cfg.edge_entry_threshold),
        strength_entry_threshold=float(rl_cfg.strength_entry_threshold),
    )
    dt_sup_only = time.perf_counter() - t0
    print(f"[pipeline][time] backtest_supervised_only={_fmt_secs(dt_sup_only)} out={sup_only_json}", flush=True)

    t0 = time.perf_counter()
    sup_plus_rl_json = run_sup_plus_rl(
        signals_path=str(signals_path),
        checkpoint=None,
        run_dir=str(run_dir),
        fold_id=-1,
        out_json=str(base_out_dir / "backtest_supervised_plus_rl.json"),
        min_rows_symbol=int(rl_cfg.min_rows_symbol),
        fee_rate=float(rl_cfg.fee_rate),
        slippage_rate=float(rl_cfg.slippage_rate),
        small_size=float(rl_cfg.small_size),
        big_size=float(rl_cfg.big_size),
        cooldown_bars=int(rl_cfg.cooldown_bars),
        min_reentry_gap_bars=int(rl_cfg.min_reentry_gap_bars),
        min_hold_bars=int(rl_cfg.min_hold_bars),
        edge_entry_threshold=float(rl_cfg.edge_entry_threshold),
        strength_entry_threshold=float(rl_cfg.strength_entry_threshold),
        device=str(eval_cfg.device),
    )
    dt_sup_rl = time.perf_counter() - t0
    print(f"[pipeline][time] backtest_supervised_plus_rl={_fmt_secs(dt_sup_rl)} out={sup_plus_rl_json}", flush=True)

    out = {
        "signals_path": str(signals_path),
        "rl_run_dir": str(run_dir),
        "evaluation_csv": str(eval_csv),
        "backtest_supervised_only": str(sup_only_json),
        "backtest_supervised_plus_rl": str(sup_plus_rl_json),
    }
    summary_path = base_out_dir / "pipeline_summary.json"
    total_s = time.perf_counter() - t0_all
    out["timings_seconds"] = {
        "export_signals": float(dt_signals),
        "train_rl": float(dt_train),
        "evaluate_rl": float(dt_eval),
        "backtest_supervised_only": float(dt_sup_only),
        "backtest_supervised_plus_rl": float(dt_sup_rl),
        "total": float(total_s),
    }
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pipeline][time] total={_fmt_secs(total_s)}", flush=True)
    out["summary_path"] = str(summary_path)
    return out


def _parse_args(argv: Iterable[str] | None = None):
    storage_root = _default_storage_root()
    default_signals_out = storage_root / "supervised_signals.parquet"
    default_rl_out_dir = storage_root / "rl_runs" / "default"
    default_pipeline_out = storage_root / "reports"
    default_symbols = [
        "ADAUSDT",
        "XLMUSDT",
        "STXUSDT",
        "ATOMUSDT",
        "DOGEUSDT",
        "SHIBUSDT",
        "HBARUSDT",
        "SUIUSDT",
        "APEUSDT",
        "XRPUSDT",
    ]
    ap = argparse.ArgumentParser(description="Pipeline hÃ­brido WF: regressÃ£o long/short + RL")
    ap.add_argument("--asset-class", default="crypto", choices=["crypto", "stocks"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", nargs="*", default=default_symbols)
    ap.add_argument("--symbols-limit", type=int, default=0)
    ap.add_argument("--days", type=int, default=1100)
    ap.add_argument("--total-days-cache", type=int, default=0)
    ap.add_argument("--signals-out", default=str(default_signals_out))

    ap.add_argument("--rl-out-dir", default=str(default_rl_out_dir))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-days", type=int, default=720)
    ap.add_argument("--valid-days", type=int, default=365)
    ap.add_argument("--step-days", type=int, default=365)
    ap.add_argument("--embargo-minutes", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--min-rows-symbol", type=int, default=800)
    ap.add_argument("--single-split-mode", action="store_true", default=True)
    ap.add_argument("--wf-mode", dest="single_split_mode", action="store_false")
    ap.add_argument("--holdout-days", type=int, default=365)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--episodes-per-epoch", type=int, default=48)
    ap.add_argument("--episode-bars", type=int, default=10080)
    ap.add_argument("--no-random-starts", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--train-steps-per-epoch", type=int, default=4000)
    ap.add_argument("--eval-every-epochs", type=int, default=5)
    ap.add_argument("--expert-mix-start", type=float, default=0.6)
    ap.add_argument("--expert-mix-end", type=float, default=0.0)
    ap.add_argument("--expert-mix-decay-epochs", type=int, default=10)
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-rate", type=float, default=0.0001)
    ap.add_argument("--small-size", type=float, default=0.2)
    ap.add_argument("--big-size", type=float, default=0.5)
    ap.add_argument("--cooldown-bars", type=int, default=8)
    ap.add_argument("--min-reentry-gap-bars", type=int, default=10080)
    ap.add_argument("--min-hold-bars", type=int, default=240)
    ap.add_argument("--edge-entry-threshold", type=float, default=0.40)
    ap.add_argument("--strength-entry-threshold", type=float, default=0.70)
    ap.add_argument("--turnover-penalty", type=float, default=0.0007)
    ap.add_argument("--dd-penalty", type=float, default=0.20)
    ap.add_argument("--regret-penalty", type=float, default=0.02)
    ap.add_argument("--idle-penalty", type=float, default=0.0)
    ap.add_argument("--target-trades-per-week", type=float, default=1.0)
    ap.add_argument("--trade-rate-penalty", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--edge-threshold", type=float, default=0.2)
    ap.add_argument("--strength-threshold", type=float, default=0.2)
    ap.add_argument("--pipeline-out", default=str(default_pipeline_out))
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    ns = _parse_args(argv)
    signals_cfg = SignalExportConfig(
        asset_class=str(ns.asset_class),
        run_dir=ns.run_dir,
        symbols=tuple(ns.symbols or ()),
        symbols_file=ns.symbols_file,
        symbols_limit=int(ns.symbols_limit),
        days=int(ns.days),
        total_days_cache=int(ns.total_days_cache),
        out_path=str(ns.signals_out),
    )
    rl_cfg = TrainRLConfig(
        signals_path=str(ns.signals_out),
        out_dir=str(ns.rl_out_dir),
        seed=int(ns.seed),
        train_days=int(ns.train_days),
        valid_days=int(ns.valid_days),
        step_days=int(ns.step_days),
        embargo_minutes=int(ns.embargo_minutes),
        max_folds=int(ns.max_folds),
        min_rows_symbol=int(ns.min_rows_symbol),
        single_split_mode=bool(ns.single_split_mode),
        holdout_days=int(ns.holdout_days),
        epochs=int(ns.epochs),
        episodes_per_epoch=int(ns.episodes_per_epoch),
        episode_bars=int(ns.episode_bars),
        random_starts=bool(not ns.no_random_starts),
        batch_size=int(ns.batch_size),
        train_steps_per_epoch=int(ns.train_steps_per_epoch),
        eval_every_epochs=int(ns.eval_every_epochs),
        expert_mix_start=float(ns.expert_mix_start),
        expert_mix_end=float(ns.expert_mix_end),
        expert_mix_decay_epochs=int(ns.expert_mix_decay_epochs),
        fee_rate=float(ns.fee_rate),
        slippage_rate=float(ns.slippage_rate),
        small_size=float(ns.small_size),
        big_size=float(ns.big_size),
        cooldown_bars=int(ns.cooldown_bars),
        min_reentry_gap_bars=int(ns.min_reentry_gap_bars),
        min_hold_bars=int(ns.min_hold_bars),
        edge_entry_threshold=float(ns.edge_entry_threshold),
        strength_entry_threshold=float(ns.strength_entry_threshold),
        turnover_penalty=float(ns.turnover_penalty),
        dd_penalty=float(ns.dd_penalty),
        regret_penalty=float(ns.regret_penalty),
        idle_penalty=float(ns.idle_penalty),
        target_trades_per_week=float(ns.target_trades_per_week),
        trade_rate_penalty=float(ns.trade_rate_penalty),
        device=str(ns.device),
    )
    eval_cfg = EvaluateConfig(
        signals_path=str(ns.signals_out),
        run_dir=str(ns.rl_out_dir),
        min_rows_symbol=int(ns.min_rows_symbol),
        edge_threshold=float(ns.edge_threshold),
        strength_threshold=float(ns.strength_threshold),
        device=str(ns.device),
    )
    out = run_pipeline(
        signals_cfg=signals_cfg,
        rl_cfg=rl_cfg,
        eval_cfg=eval_cfg,
        base_out_dir=Path(str(ns.pipeline_out)).expanduser().resolve(),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


