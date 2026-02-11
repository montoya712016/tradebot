# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Loop automatico de treinos RL curtos (1-3 epocas) com ajuste iterativo simples.

Objetivo:
- executar muitas tentativas curtas
- ler metricas do summary por tentativa
- ajustar hiperparametros para melhorar score/qualidade de trade

Saidas:
- history.jsonl
- best_summary.json
- best_config.json
- loop_state.json
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import copy
import json
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


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

from rl.train_rl import TrainRLConfig, train_rl_walkforward  # type: ignore


def _default_storage_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_base_cfg(signals_path: Path, out_base: Path, seed: int, device: str) -> TrainRLConfig:
    return TrainRLConfig(
        signals_path=str(signals_path),
        out_dir=str(out_base / "seed"),
        seed=int(seed),
        train_days=720,
        holdout_days=180,
        single_split_mode=True,
        max_folds=1,
        min_rows_symbol=800,
        epochs=2,
        episodes_per_epoch=6,
        episode_bars=2_160,
        random_starts=True,
        batch_size=512,
        replay_size=120_000,
        train_steps_per_epoch=300,
        eval_every_epochs=1,
        eval_symbols_per_epoch=4,
        eval_tail_bars=120_000,
        eval_full_every_epochs=3,
        hidden_dim=96,
        target_update_steps=250,
        epsilon_start=0.45,
        epsilon_end=0.04,
        epsilon_decay_epochs=4,
        expert_mix_start=0.80,
        expert_mix_end=0.10,
        expert_mix_decay_epochs=6,
        fee_rate=0.0005,
        slippage_rate=0.0001,
        small_size=0.08,
        big_size=0.18,
        cooldown_bars=6,
        min_reentry_gap_bars=180,
        min_hold_bars=90,
        max_hold_bars=0,
        edge_entry_threshold=0.50,
        strength_entry_threshold=0.82,
        force_close_on_adverse=True,
        edge_exit_threshold=0.22,
        strength_exit_threshold=0.18,
        allow_scale_in=False,
        turnover_penalty=0.0007,
        dd_penalty=0.45,
        dd_level_penalty=0.10,
        dd_soft_limit=0.12,
        dd_excess_penalty=3.0,
        dd_hard_limit=0.22,
        dd_hard_penalty=0.08,
        hold_bar_penalty=0.0,
        hold_soft_bars=360,
        hold_excess_penalty=0.001,
        hold_regret_penalty=0.09,
        regret_penalty=0.03,
        idle_penalty=0.0,
        target_trades_per_week=1.0,
        trade_rate_penalty=0.18,
        trade_rate_under_penalty=0.07,
        score_ret_weight=0.25,
        score_calmar_weight=0.55,
        score_efficiency_weight=0.30,
        score_giveback_penalty=0.45,
        score_hold_hours_target=4.0,
        score_hold_hours_penalty=0.04,
        score_hold_hours_min=2.0,
        score_hold_hours_under_penalty=0.05,
        score_hold_hours_excess_cap=72.0,
        score_dd_soft_limit=0.12,
        score_dd_excess_penalty=4.0,
        state_use_uncertainty=True,
        state_use_vol_long=False,
        state_use_shock=False,
        device=str(device),
    )


def _guided_adjust(cfg: TrainRLConfig, last: dict[str, Any], rng: np.random.Generator) -> TrainRLConfig:
    c = copy.deepcopy(cfg)

    # pequena exploracao aleatoria
    c.edge_entry_threshold = _clip(c.edge_entry_threshold + float(rng.normal(0.0, 0.025)), 0.22, 0.82)
    c.strength_entry_threshold = _clip(c.strength_entry_threshold + float(rng.normal(0.0, 0.025)), 0.45, 1.15)
    c.turnover_penalty = _clip(c.turnover_penalty * float(np.exp(rng.normal(0.0, 0.10))), 0.00015, 0.02)
    c.hold_regret_penalty = _clip(c.hold_regret_penalty * float(np.exp(rng.normal(0.0, 0.14))), 0.01, 1.2)
    c.hold_excess_penalty = _clip(c.hold_excess_penalty * float(np.exp(rng.normal(0.0, 0.14))), 0.0004, 0.06)
    c.hold_bar_penalty = _clip(c.hold_bar_penalty * float(np.exp(rng.normal(0.0, 0.10))), 0.0, 0.002)
    c.score_giveback_penalty = _clip(c.score_giveback_penalty * float(np.exp(rng.normal(0.0, 0.14))), 0.05, 4.0)
    c.dd_penalty = _clip(c.dd_penalty * float(np.exp(rng.normal(0.0, 0.08))), 0.05, 2.00)
    c.dd_level_penalty = _clip(c.dd_level_penalty * float(np.exp(rng.normal(0.0, 0.08))), 0.00, 1.20)
    c.expert_mix_start = _clip(c.expert_mix_start + float(rng.normal(0.0, 0.03)), 0.25, 0.95)
    c.expert_mix_end = _clip(c.expert_mix_end + float(rng.normal(0.0, 0.02)), 0.00, 0.35)
    c.epsilon_start = _clip(c.epsilon_start + float(rng.normal(0.0, 0.04)), 0.10, 0.90)
    c.epsilon_end = _clip(c.epsilon_end + float(rng.normal(0.0, 0.02)), 0.01, 0.30)
    c.edge_exit_threshold = _clip(c.edge_exit_threshold + float(rng.normal(0.0, 0.01)), 0.04, 0.70)
    c.strength_exit_threshold = _clip(c.strength_exit_threshold + float(rng.normal(0.0, 0.03)), 0.05, 0.70)
    c.min_reentry_gap_bars = int(round(_clip(float(c.min_reentry_gap_bars + int(rng.normal(0.0, 8.0))), 0.0, 600.0)))
    c.min_hold_bars = int(round(_clip(float(c.min_hold_bars + int(rng.normal(0.0, 3.0))), 0.0, 120.0)))
    c.hold_soft_bars = int(round(_clip(float(c.hold_soft_bars + int(rng.normal(0.0, 40.0))), 120.0, 1800.0)))
    c.score_hold_hours_target = _clip(c.score_hold_hours_target + float(rng.normal(0.0, 0.5)), 2.0, 24.0)
    c.score_hold_hours_min = _clip(c.score_hold_hours_min + float(rng.normal(0.0, 0.35)), 0.5, 12.0)
    c.score_hold_hours_under_penalty = _clip(c.score_hold_hours_under_penalty * float(np.exp(rng.normal(0.0, 0.12))), 0.005, 0.50)
    c.eval_symbols_per_epoch = int(round(_clip(float(c.eval_symbols_per_epoch + int(rng.normal(0.0, 1.5))), 2.0, 8.0)))
    c.eval_tail_bars = int(round(_clip(float(c.eval_tail_bars + int(rng.normal(0.0, 20000.0))), 60_000.0, 300_000.0)))

    trw = float(last.get("best_valid_trades_per_week_mean", 0.0))
    ret = float(last.get("best_valid_ret_total_mean", 0.0))
    dd = float(last.get("best_valid_max_dd_mean", 0.0))
    gb = float(last.get("best_valid_avg_giveback_mean", 0.0))
    eff = float(last.get("best_valid_trade_efficiency_mean", 0.0))
    hold_l = float(last.get("best_valid_avg_hold_long_hours_mean", last.get("best_valid_avg_hold_hours_mean", 0.0)))
    hold_s = float(last.get("best_valid_avg_hold_short_hours_mean", last.get("best_valid_avg_hold_hours_mean", 0.0)))
    hold_h = max(0.0, max(hold_l, hold_s))

    # controle de frequencia
    if trw > (c.target_trades_per_week * 1.7):
        c.edge_entry_threshold = _clip(c.edge_entry_threshold + 0.03, 0.22, 0.85)
        c.strength_entry_threshold = _clip(c.strength_entry_threshold + 0.03, 0.45, 1.25)
        c.turnover_penalty = _clip(c.turnover_penalty * 1.15, 0.00015, 0.02)
        c.min_reentry_gap_bars = int(_clip(float(c.min_reentry_gap_bars + 12), 0.0, 600.0))
        c.epsilon_start = _clip(c.epsilon_start - 0.05, 0.10, 0.90)
        c.expert_mix_start = _clip(c.expert_mix_start + 0.04, 0.25, 0.95)
        c.edge_exit_threshold = _clip(c.edge_exit_threshold + 0.01, 0.04, 0.50)
    if trw > (c.target_trades_per_week * 3.0):
        c.edge_entry_threshold = _clip(c.edge_entry_threshold + 0.05, 0.22, 0.90)
        c.strength_entry_threshold = _clip(c.strength_entry_threshold + 0.05, 0.45, 1.30)
        c.turnover_penalty = _clip(c.turnover_penalty * 1.25, 0.00015, 0.03)
        c.min_reentry_gap_bars = int(_clip(float(c.min_reentry_gap_bars + 30), 0.0, 900.0))
        c.edge_exit_threshold = _clip(c.edge_exit_threshold + 0.02, 0.04, 0.55)
    elif trw < (c.target_trades_per_week * 0.55):
        c.edge_entry_threshold = _clip(c.edge_entry_threshold - 0.03, 0.22, 0.85)
        c.strength_entry_threshold = _clip(c.strength_entry_threshold - 0.03, 0.45, 1.25)
        c.min_reentry_gap_bars = int(_clip(float(c.min_reentry_gap_bars - 10), 0.0, 600.0))
        c.expert_mix_start = _clip(c.expert_mix_start + 0.03, 0.25, 0.90)
        c.epsilon_start = _clip(c.epsilon_start + 0.02, 0.10, 0.90)
        c.edge_exit_threshold = _clip(c.edge_exit_threshold - 0.01, 0.04, 0.50)

    # qualidade de trade: se devolve lucro, aperta penalizacao de giveback
    if gb > 0.0015 or eff < 0.10:
        c.hold_regret_penalty = _clip(c.hold_regret_penalty * 1.20, 0.01, 1.2)
        c.hold_excess_penalty = _clip(c.hold_excess_penalty * 1.18, 0.0004, 0.06)
        c.score_giveback_penalty = _clip(c.score_giveback_penalty * 1.15, 0.05, 4.0)

    # se trades longos demais, aumenta pressao para encerrar melhor
    if hold_h > (c.score_hold_hours_target * 1.5):
        c.hold_regret_penalty = _clip(c.hold_regret_penalty * 1.18, 0.01, 1.2)
        c.hold_excess_penalty = _clip(c.hold_excess_penalty * 1.20, 0.0004, 0.06)
        c.hold_bar_penalty = _clip(c.hold_bar_penalty * 1.15 + 0.000002, 0.0, 0.002)
        c.score_hold_hours_penalty = _clip(c.score_hold_hours_penalty * 1.10, 0.005, 0.40)
        c.min_hold_bars = int(_clip(float(c.min_hold_bars - 2), 0.0, 120.0))
    if hold_h > 0.0 and hold_h < (0.8 * c.score_hold_hours_min):
        # trades muito curtos: evita scalp e reduz gatilhos de saida precoce.
        c.min_hold_bars = int(_clip(float(c.min_hold_bars + 20), 0.0, 240.0))
        c.min_reentry_gap_bars = int(_clip(float(c.min_reentry_gap_bars + 15), 0.0, 900.0))
        c.edge_exit_threshold = _clip(c.edge_exit_threshold + 0.03, 0.04, 0.70)
        c.strength_exit_threshold = _clip(c.strength_exit_threshold - 0.02, 0.05, 0.70)
        c.score_hold_hours_under_penalty = _clip(c.score_hold_hours_under_penalty * 1.20, 0.005, 0.50)

    # risco
    if dd > 0.18:
        c.dd_penalty = _clip(c.dd_penalty * 1.12, 0.05, 1.5)
        c.dd_level_penalty = _clip(c.dd_level_penalty * 1.12, 0.0, 1.2)
        c.big_size = _clip(c.big_size * 0.92, 0.08, 0.40)
    if ret < 0.0:
        c.small_size = _clip(c.small_size * 0.95, 0.04, 0.25)
        c.big_size = _clip(c.big_size * 0.95, 0.08, 0.40)
        c.edge_entry_threshold = _clip(c.edge_entry_threshold - 0.01, 0.22, 0.85)
        c.strength_entry_threshold = _clip(c.strength_entry_threshold - 0.01, 0.45, 1.25)

    return c


def _randomize_short_budget(cfg: TrainRLConfig, rng: np.random.Generator, min_epochs: int, max_epochs: int) -> TrainRLConfig:
    c = copy.deepcopy(cfg)
    c.epochs = int(rng.integers(int(min_epochs), int(max_epochs) + 1))
    c.episodes_per_epoch = int(rng.integers(6, 10))
    c.episode_bars = int(rng.choice([1_440, 2_160, 2_880]))
    c.train_steps_per_epoch = int(rng.choice([240, 300, 360, 420]))
    c.eval_every_epochs = 1
    c.eval_symbols_per_epoch = int(rng.choice([3, 4, 5, 6]))
    c.eval_tail_bars = int(rng.choice([80_000, 100_000, 120_000, 150_000]))
    c.eval_full_every_epochs = int(max(1, c.epochs))
    return c


def _read_fold_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "fold_000" / "summary.json"
    if not p.exists():
        return {}
    return _load_json(p)


def _infer_span_days(signals_path: Path) -> float:
    try:
        df = pd.read_parquet(signals_path, columns=["symbol"])
    except Exception:
        df = pd.read_parquet(signals_path)
    if len(df.index) == 0:
        return 0.0
    ts = pd.to_datetime(df.index)
    span = (pd.Timestamp(ts.max()) - pd.Timestamp(ts.min())).total_seconds() / (24.0 * 3600.0)
    return float(max(0.0, span))


def _fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def run_loop(
    *,
    hours: float,
    signals_path: Path,
    out_base: Path,
    seed: int,
    device: str,
    min_epochs: int,
    max_epochs: int,
    max_iters: int,
) -> None:
    if not signals_path.exists():
        raise FileNotFoundError(f"signals_path nao encontrado: {signals_path}")

    out_base.mkdir(parents=True, exist_ok=True)
    history_path = out_base / "history.jsonl"
    state_path = out_base / "loop_state.json"
    best_summary_path = out_base / "best_summary.json"
    best_cfg_path = out_base / "best_config.json"

    t0 = time.time()
    t_end = t0 + max(0.1, float(hours)) * 3600.0
    rng = np.random.default_rng(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))

    base_cfg = _build_base_cfg(signals_path, out_base, seed=int(seed), device=device)
    span_days = _infer_span_days(signals_path)
    if span_days > 0 and span_days < float(base_cfg.train_days + base_cfg.holdout_days + 7):
        if span_days < 10.0:
            hold = 1
            train = 1
        else:
            hold = int(max(1, min(365, int(span_days * 0.35))))
            train = int(max(1, min(720, int(span_days - hold - 2))))
        if train > 0 and hold > 0:
            base_cfg.train_days = int(train)
            base_cfg.holdout_days = int(hold)
            print(
                f"[auto-loop] ajustando janela para span={span_days:.1f}d -> train_days={train} holdout_days={hold}",
                flush=True,
            )
    best_cfg = copy.deepcopy(base_cfg)
    best_score = -1e30
    best_summary: dict[str, Any] = {}
    if best_cfg_path.exists():
        try:
            prev_cfg = _load_json(best_cfg_path)
            if isinstance(prev_cfg, dict) and prev_cfg:
                merged = asdict(base_cfg)
                merged.update(prev_cfg)
                fields = set(TrainRLConfig.__dataclass_fields__.keys())
                kwargs = {k: merged[k] for k in merged.keys() if k in fields}
                best_cfg = TrainRLConfig(**kwargs)
                print(f"[auto-loop] resume cfg from {best_cfg_path}", flush=True)
        except Exception:
            pass
    if best_summary_path.exists():
        try:
            prev_sum = _load_json(best_summary_path)
            if isinstance(prev_sum, dict) and prev_sum:
                best_summary = dict(prev_sum)
                best_score = float(prev_sum.get("best_score", best_score))
                print(f"[auto-loop] resume best_score={best_score:+.4f} from {best_summary_path}", flush=True)
        except Exception:
            pass
    last_summary: dict[str, Any] = {}
    it = 0

    print(
        f"[auto-loop] start hours={hours:.2f} signals={signals_path} out={out_base} "
        f"epochs=[{min_epochs},{max_epochs}] device={device}",
        flush=True,
    )

    while time.time() < t_end:
        if int(max_iters) > 0 and it >= int(max_iters):
            break
        it += 1

        cfg = _guided_adjust(best_cfg, last_summary, rng) if last_summary else copy.deepcopy(best_cfg)
        cfg = _randomize_short_budget(cfg, rng, min_epochs=min_epochs, max_epochs=max_epochs)
        cfg.seed = int(seed + it * 17)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg.out_dir = str((out_base / f"iter_{it:04d}_{ts}").resolve())

        t_run = time.time()
        try:
            run_dir = Path(train_rl_walkforward(cfg)).resolve()
        except RuntimeError as e:
            msg = str(e).lower()
            if "split unico invalido" not in msg:
                raise
            # fallback adicional se janela ainda for grande para o arquivo atual
            span_days2 = _infer_span_days(signals_path)
            if span_days2 < 10.0:
                hold2 = 1
                train2 = 1
            else:
                hold2 = int(max(1, min(120, int(span_days2 * 0.30))))
                train2 = int(max(1, min(365, int(span_days2 - hold2 - 2))))
            cfg.train_days = int(train2)
            cfg.holdout_days = int(hold2)
            print(
                f"[auto-loop] retry split train_days={cfg.train_days} holdout_days={cfg.holdout_days}",
                flush=True,
            )
            run_dir = Path(train_rl_walkforward(cfg)).resolve()
        dt = time.time() - t_run
        summary = _read_fold_summary(run_dir)
        if not summary:
            summary = {"best_score": -1e9}
        score = float(summary.get("best_score", -1e9))
        ret = float(summary.get("best_valid_ret_total_mean", 0.0))
        dd = float(summary.get("best_valid_max_dd_mean", 0.0))
        tw = float(summary.get("best_valid_trades_per_week_mean", 0.0))
        h = float(summary.get("best_valid_avg_hold_hours_mean", 0.0))
        hl = float(summary.get("best_valid_avg_hold_long_hours_mean", h))
        hs = float(summary.get("best_valid_avg_hold_short_hours_mean", h))
        eff = float(summary.get("best_valid_trade_efficiency_mean", 0.0))
        gb = float(summary.get("best_valid_avg_giveback_mean", 0.0))

        is_best = bool(score > best_score)
        if is_best:
            best_score = score
            best_summary = dict(summary)
            best_cfg = copy.deepcopy(cfg)
            _write_json(best_summary_path, best_summary)
            _write_json(best_cfg_path, asdict(best_cfg))

        row = {
            "iter": int(it),
            "run_dir": str(run_dir),
            "seconds": float(dt),
            "score": float(score),
            "ret": float(ret),
            "dd": float(dd),
            "trades_per_week": float(tw),
            "hold_h": float(h),
            "hold_long_h": float(hl),
            "hold_short_h": float(hs),
            "efficiency": float(eff),
            "giveback": float(gb),
            "is_best": bool(is_best),
            "epochs": int(cfg.epochs),
            "episodes_per_epoch": int(cfg.episodes_per_epoch),
            "episode_bars": int(cfg.episode_bars),
            "train_steps_per_epoch": int(cfg.train_steps_per_epoch),
            "edge_thr": float(cfg.edge_entry_threshold),
            "strength_thr": float(cfg.strength_entry_threshold),
            "hold_regret_penalty": float(cfg.hold_regret_penalty),
            "turnover_penalty": float(cfg.turnover_penalty),
            "dd_penalty": float(cfg.dd_penalty),
            "dd_level_penalty": float(cfg.dd_level_penalty),
            "small_size": float(cfg.small_size),
            "big_size": float(cfg.big_size),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(history_path, row)
        last_summary = dict(summary)

        _write_json(
            state_path,
            {
                "iter": int(it),
                "elapsed_h": float((time.time() - t0) / 3600.0),
                "remaining_h": float(max(0.0, (t_end - time.time()) / 3600.0)),
                "last_run_dir": str(run_dir),
                "last_summary": summary,
                "best_score": float(best_score),
                "best_summary": best_summary,
            },
        )

        print(
            f"[auto-loop] iter={it} t={dt:.1f}s score={score:+.4f} ret={_fmt_pct(ret)} dd={dd:.2%} "
            f"tw={tw:.2f} hold(h)={h:.2f} L/S={hl:.2f}/{hs:.2f} eff={eff:+.3f} gb={gb:.4f} best={int(is_best)} "
            f"ep={cfg.epochs}x{cfg.episodes_per_epoch}",
            flush=True,
        )

    print(
        f"[auto-loop] end iters={it} elapsed_h={(time.time()-t0)/3600.0:.2f} best_score={best_score:+.4f} "
        f"best_path={best_summary_path}",
        flush=True,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _default_storage_root()
    ap = argparse.ArgumentParser(description="Loop automatico de treinos RL curtos")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--signals", default=str(root / "supervised_signals.parquet"))
    ap.add_argument("--out-base", default=str(root / "rl_runs" / "auto_loop"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-epochs", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=0, help="0=sem limite, usa somente --hours")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse_args(argv)
    run_loop(
        hours=float(ns.hours),
        signals_path=Path(str(ns.signals)).expanduser().resolve(),
        out_base=Path(str(ns.out_base)).expanduser().resolve(),
        seed=int(ns.seed),
        device=str(ns.device),
        min_epochs=int(ns.min_epochs),
        max_epochs=int(ns.max_epochs),
        max_iters=int(ns.max_iters),
    )


if __name__ == "__main__":
    main()
