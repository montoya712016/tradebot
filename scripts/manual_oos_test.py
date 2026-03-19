import os
import sys
from pathlib import Path

# Ensure modules are on sys.path
repo_root = Path(r"d:\astra\tradebot")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "modules"))

from backtest.portfolio import (
    PortfolioDemoSettings,
    _default_portfolio_cfg,
    prepare_portfolio_data,
    run_prepared_portfolio,
)
from config.trade_contract import build_default_crypto_contract, apply_crypto_pipeline_env

def run_manual_oos():
    # PARAMETERS FROM BEST TRIAL (step_1440d)
    # label_id: label_003, model_id: model_003, bt_017
    run_dir = r"D:\astra\models_sniper\crypto\wf_124"
    tau_entry = 0.6
    max_positions = 101
    total_exposure = 1.18
    corr_enabled = True
    
    # OOS PERIOD (Next 180 days after T-1440)
    # T-1440 to T-1260 was IS.
    # T-1260 to T-1080 is OOS.
    tail = 1080
    days = 180
    
    os.environ["SNIPER_REMOVE_TAIL_DAYS"] = str(tail)
    
    candle_sec = apply_crypto_pipeline_env(300)
    contract = build_default_crypto_contract(candle_sec)
    
    cfg = _default_portfolio_cfg()
    cfg.max_positions = int(max_positions)
    cfg.total_exposure = float(total_exposure)
    cfg.corr_filter_enabled = corr_enabled
    cfg.corr_open_filter_enabled = corr_enabled
    # Use defaults for other explore params
    cfg.corr_max_with_market = 0.9
    cfg.corr_max_pair = 0.9
    cfg.corr_open_reduce_start = 0.7
    cfg.corr_open_hard_reject = 0.9
    cfg.corr_open_min_weight_mult = 0.5

    plot_out = str(repo_root / "data" / "generated" / "fair_wf_explore" / "manual_oos_step_1260d.html")
    
    settings = PortfolioDemoSettings(
        asset_class="crypto",
        run_dir=run_dir,
        days=days,
        max_symbols=0, # All symbols
        cfg=cfg,
        save_plot=True,
        plot_out=plot_out,
        override_tau_entry=tau_entry,
        candle_sec=candle_sec,
        contract=contract,
        long_only=True,
        align_global_window=True
    )
    
    print(f"\n{'='*60}")
    print(f"RUNNING MANUAL OOS TEST")
    print(f"Individual: label_003/model_003/bt_017")
    print(f"Period: Tail={tail}d (T-1260 to T-1080)")
    print(f"{'='*60}\n")
    
    res = run_prepared_portfolio(prepare_portfolio_data(settings), cfg=cfg, days=days, override_tau_entry=tau_entry, save_plot=True, plot_out=plot_out)
    
    ret = res.get("ret_total", 0.0)
    dd = res.get("max_dd", 0.0)
    trades = len(getattr(res["result"], "trades", []))
    
    print(f"\nOOS RESULT:")
    print(f"Return: {ret:+.2%}")
    print(f"Max DD: {dd:.2%}")
    print(f"Trades: {trades}")
    print(f"Plot saved: {plot_out}")
    
    with open("oos_results.txt", "w") as f:
        f.write(f"Return: {ret:+.2%}\n")
        f.write(f"Max DD: {dd:.2%}\n")
        f.write(f"Trades: {trades}\n")
        f.write(f"Plot: {plot_out}\n")

if __name__ == "__main__":
    run_manual_oos()
