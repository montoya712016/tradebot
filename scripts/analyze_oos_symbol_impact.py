import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "modules"))

from backtest.portfolio import PortfolioDemoSettings, prepare_portfolio_data, run_prepared_portfolio  # noqa: E402
from config.trade_contract import apply_crypto_pipeline_env  # noqa: E402
from run_oos_walkforward import (  # noqa: E402
    build_cfg_from_row,
    build_contract_from_row,
)


def _to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_segment_rows(report_dir: Path, mode: str) -> pd.DataFrame:
    seg_path = report_dir / f"walkforward_oos_segments_{mode}.csv"
    if not seg_path.exists():
        raise FileNotFoundError(f"missing segment summary: {seg_path}")
    df = pd.read_csv(seg_path)
    if df.empty:
        raise RuntimeError("segment summary is empty")
    return df


def _find_selected_row(fair_root: Path, segment_row: pd.Series) -> pd.Series:
    source_step = int(segment_row["source_step"])
    step_dir = fair_root / f"step_{source_step}d"
    csv_path = step_dir / "explore_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing explore_runs.csv for step {source_step}: {csv_path}")
    df = pd.read_csv(csv_path)
    mask = (
        (df.get("stage") == "backtest")
        & (df.get("status") == "ok")
        & (df.get("label_id").astype(str) == str(segment_row["label_id"]))
        & (df.get("model_id").astype(str) == str(segment_row["model_id"]))
        & (df.get("backtest_id").astype(str) == str(segment_row["backtest_id"]))
    )
    hits = df.loc[mask].copy()
    if hits.empty:
        raise RuntimeError(
            f"could not find selected individual for step={source_step} "
            f"{segment_row['label_id']}/{segment_row['model_id']}/{segment_row['backtest_id']}"
        )
    return hits.iloc[0]


def _calc_max_dd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    dd = equity / peaks - 1.0
    return float(abs(dd.min()))


def _run_segment(segment_row: pd.Series, selected_row: pd.Series) -> list[dict]:
    candle_sec = apply_crypto_pipeline_env(300)
    contract = build_contract_from_row(selected_row, candle_sec)
    cfg = build_cfg_from_row(selected_row)
    tau_entry = _to_float(segment_row.get("tau_entry"), _to_float(selected_row.get("tau_entry"), 0.5))
    target_step = int(segment_row["target_step"])
    run_dir = str(segment_row.get("selected_train_run_dir") or "").strip()
    if not run_dir:
        raise RuntimeError(f"missing train run dir for source step {segment_row['source_step']}")

    os.environ["SNIPER_REMOVE_TAIL_DAYS"] = str(target_step)
    settings = PortfolioDemoSettings(
        asset_class="crypto",
        run_dir=run_dir,
        days=180,
        max_symbols=0,
        cfg=cfg,
        save_plot=False,
        override_tau_entry=tau_entry,
        candle_sec=candle_sec,
        contract=contract,
        long_only=True,
        align_global_window=True,
    )
    prepared = prepare_portfolio_data(settings)
    res = run_prepared_portfolio(
        prepared,
        cfg=cfg,
        days=180,
        override_tau_entry=tau_entry,
        save_plot=False,
        plot_out="",
    )
    out = []
    for tr in list(getattr(res["result"], "trades", []) or []):
        pnl_w = float(getattr(tr, "r_net", 0.0) or 0.0) * float(getattr(tr, "weight", 0.0) or 0.0)
        if pnl_w <= -0.999999:
            log_contrib = np.nan
        else:
            log_contrib = float(np.log1p(pnl_w))
        out.append(
            {
                "mode": str(segment_row.get("mode") or "reuse"),
                "source_step": int(segment_row["source_step"]),
                "test_start_step": int(segment_row["test_start_step"]),
                "target_step": int(segment_row["target_step"]),
                "label_id": str(segment_row["label_id"]),
                "model_id": str(segment_row["model_id"]),
                "backtest_id": str(segment_row["backtest_id"]),
                "symbol": str(getattr(tr, "symbol", "") or "").upper(),
                "entry_ts": pd.to_datetime(getattr(tr, "entry_ts", None)),
                "exit_ts": pd.to_datetime(getattr(tr, "exit_ts", None)),
                "r_net": float(getattr(tr, "r_net", 0.0) or 0.0),
                "weight": float(getattr(tr, "weight", 0.0) or 0.0),
                "pnl_w": pnl_w,
                "log_contrib": log_contrib,
                "reason": str(getattr(tr, "reason", "") or ""),
                "num_adds": int(getattr(tr, "num_adds", 0) or 0),
            }
        )
    return out


def _aggregate(trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = trades_df.copy()
    base["exit_date"] = pd.to_datetime(base["exit_ts"]).dt.normalize()
    base["is_win"] = (base["pnl_w"] > 0.0).astype(int)
    base["is_loss"] = (base["pnl_w"] < 0.0).astype(int)
    base["loss_abs"] = np.where(base["pnl_w"] < 0.0, -base["pnl_w"], 0.0)
    base["gain_abs"] = np.where(base["pnl_w"] > 0.0, base["pnl_w"], 0.0)
    base["segment_key"] = (
        base["source_step"].astype(str)
        + "->"
        + base["test_start_step"].astype(str)
        + "->"
        + base["target_step"].astype(str)
    )

    sym_seg = (
        base.groupby(["symbol", "segment_key", "source_step", "test_start_step", "target_step"], as_index=False)
        .agg(
            trades=("pnl_w", "size"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pnl_sum=("pnl_w", "sum"),
            log_sum=("log_contrib", "sum"),
            worst_trade=("pnl_w", "min"),
            best_trade=("pnl_w", "max"),
            loss_abs=("loss_abs", "sum"),
            gain_abs=("gain_abs", "sum"),
        )
    )
    sym_seg["segment_return"] = np.expm1(sym_seg["log_sum"])
    sym_seg["win_rate"] = np.where(sym_seg["trades"] > 0, sym_seg["wins"] / sym_seg["trades"], np.nan)
    sym_seg["negative_segment"] = (sym_seg["segment_return"] < 0.0).astype(int)
    sym_seg["positive_segment"] = (sym_seg["segment_return"] > 0.0).astype(int)

    sym_day = (
        base.groupby(["symbol", "exit_date"], as_index=False)
        .agg(log_sum=("log_contrib", "sum"))
        .sort_values(["symbol", "exit_date"])
    )

    dd_map = {}
    for sym, sdf in sym_day.groupby("symbol"):
        eq = np.exp(sdf["log_sum"].cumsum())
        dd_map[sym] = _calc_max_dd(pd.Series(eq, index=sdf["exit_date"]))

    summary = (
        base.groupby("symbol", as_index=False)
        .agg(
            trades=("pnl_w", "size"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            total_pnl_w=("pnl_w", "sum"),
            mean_pnl_w=("pnl_w", "mean"),
            median_pnl_w=("pnl_w", "median"),
            total_log_contrib=("log_contrib", "sum"),
            worst_trade=("pnl_w", "min"),
            best_trade=("pnl_w", "max"),
            loss_abs=("loss_abs", "sum"),
            gain_abs=("gain_abs", "sum"),
            active_days=("exit_date", "nunique"),
        )
    )

    seg_stats = (
        sym_seg.groupby("symbol", as_index=False)
        .agg(
            segments_active=("segment_key", "nunique"),
            neg_segments=("negative_segment", "sum"),
            pos_segments=("positive_segment", "sum"),
            mean_segment_return=("segment_return", "mean"),
            median_segment_return=("segment_return", "median"),
            worst_segment_return=("segment_return", "min"),
            best_segment_return=("segment_return", "max"),
            mean_segment_log=("log_sum", "mean"),
        )
    )

    summary = summary.merge(seg_stats, on="symbol", how="left")
    summary["approx_compounded_return"] = np.expm1(summary["total_log_contrib"])
    summary["win_rate"] = np.where(summary["trades"] > 0, summary["wins"] / summary["trades"], np.nan)
    summary["loss_share_abs"] = np.where(
        (summary["loss_abs"] + summary["gain_abs"]) > 0.0,
        summary["loss_abs"] / (summary["loss_abs"] + summary["gain_abs"]),
        np.nan,
    )
    summary["neg_segment_frac"] = np.where(
        summary["segments_active"] > 0,
        summary["neg_segments"] / summary["segments_active"],
        np.nan,
    )
    summary["pos_segment_frac"] = np.where(
        summary["segments_active"] > 0,
        summary["pos_segments"] / summary["segments_active"],
        np.nan,
    )
    summary["symbol_max_dd_proxy"] = summary["symbol"].map(dd_map).fillna(0.0)
    summary["catastrophic_score"] = (
        (-summary["total_log_contrib"]).clip(lower=0.0) * 0.45
        + summary["neg_segment_frac"].fillna(0.0) * 0.25
        + summary["symbol_max_dd_proxy"].fillna(0.0) * 0.20
        + summary["loss_share_abs"].fillna(0.0) * 0.10
    )
    summary = summary.sort_values(
        ["catastrophic_score", "total_log_contrib", "neg_segment_frac", "symbol_max_dd_proxy"],
        ascending=[False, True, False, False],
        na_position="last",
    )
    return summary, sym_seg


def main():
    mode = str(os.getenv("WF_OOS_ANALYSIS_MODE", "reuse") or "reuse").strip().lower()
    fair_root = repo_root / "data" / "generated" / "fair_wf_explore"
    report_dir = fair_root / "robustness_report"
    segments = _load_segment_rows(report_dir, mode)

    trade_rows = []
    for _, seg in segments.iterrows():
        selected_row = _find_selected_row(fair_root, seg)
        seg_trades = _run_segment(seg, selected_row)
        trade_rows.extend(seg_trades)
        print(
            f"[segment] source={int(seg['source_step'])} test={int(seg['test_start_step'])}->{int(seg['target_step'])} "
            f"trades={len(seg_trades)}"
        )

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        raise RuntimeError("no trades extracted from OOS segments")

    summary_df, per_segment_df = _aggregate(trades_df)
    trades_path = report_dir / f"walkforward_oos_symbol_trades_{mode}.csv"
    summary_path = report_dir / f"walkforward_oos_symbol_summary_{mode}.csv"
    segment_path = report_dir / f"walkforward_oos_symbol_segments_{mode}.csv"

    trades_df.sort_values(["exit_ts", "symbol", "entry_ts"], inplace=True, na_position="last")
    per_segment_df.sort_values(["symbol", "source_step", "test_start_step"], inplace=True, na_position="last")
    summary_df.to_csv(summary_path, index=False)
    per_segment_df.to_csv(segment_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print(f"[ok] trades saved to {trades_path}")
    print(f"[ok] summary saved to {summary_path}")
    print(f"[ok] segment summary saved to {segment_path}")
    print("")
    print("Top catastrophic candidates:")
    cols = [
        "symbol",
        "trades",
        "segments_active",
        "approx_compounded_return",
        "total_pnl_w",
        "neg_segment_frac",
        "symbol_max_dd_proxy",
        "catastrophic_score",
    ]
    print(summary_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
