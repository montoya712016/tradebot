# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Pipeline de treino (walk-forward) do Sniper com parÃ¢metros em cÃ³digo (sem ENV).

Recomendacao: execute via arquivo:
`python modules/train/train_sniper_wf.py`
"""

from dataclasses import dataclass, field
import os
import sys
from pathlib import Path

def _ensure_modules_on_sys_path() -> None:
    """
    Permite executar este arquivo direto (ex.: `python modules/train/train_sniper_wf.py`)
    sem depender de PYTHONPATH.

    Regra do repo: o pacote fica em `modules/`, entÃ£o precisamos que `modules/`
    esteja no sys.path para `import prepare_features...`.
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
from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore

# Defaults de performance/telemetria para rodar com "Run" sem flags externas.
os.environ.setdefault("SNIPER_CACHE_PROGRESS_EVERY_S", "3")
os.environ.setdefault("SNIPER_CACHE_WORKERS", "6")
os.environ.setdefault("PF_PREFER_PURE_FOR_THREADS", "0")

# lazy import do sniper_trainer para evitar travar no import em ambientes instáveis


@dataclass
class TrainSniperWFSettings:
    # modo/asset
    asset_class: str = "crypto"  # "crypto" ou "stocks"
    symbols_file: str | None = None
    symbols: tuple[str, ...] = field(default_factory=tuple)
    # dataset/cache
    # 0 => histÃ³rico completo (recomendado se vocÃª quer offsets grandes, ex.: 4+ anos)
    total_days: int = 0
    # offsets (dias "antes do fim") para cada perÃ­odo do walk-forward.
    # Se `offsets_days` estiver vazio, ele Ã© gerado automaticamente via range:
    # - step de 90 dias
    # - horizonte de 4 anos (â‰ˆ 1460 dias)
    # Ex.: 90, 180, ..., 1440
    offsets_step_days: int = 90
    # Aumente para cobrir mais janelas (mais perÃ­odos no wf_*).
    # Ex.: 6 anos com step de 90d => ~24 perÃ­odos (90, 180, ..., 2160).
    offsets_years: int = 6
    offsets_days: tuple[int, ...] = field(default_factory=tuple)
    # Universo maior deixa o wf_* mais robusto, mas aumenta tempo de cache/treino.
    # 0 => sem limite (usa todas elegÃ­veis por market cap)
    max_symbols: int = 0
    # filtro por market cap (USD)
    mcap_min_usd: float = 100_000_000.0
    mcap_max_usd: float = 150_000_000_000.0
    use_feature_cache: bool = True
    # se um perÃ­odo ficar com poucos sÃ­mbolos com dados suficientes, pula (evita modelo "escasso")
    min_symbols_used_per_period: int = 30

    # sizing (use RAM/VRAM)
    max_rows_entry: int = 6_000_000
    # se True, monta pool completo e amostra por modelo no treino
    entry_pool_full: bool = False
    entry_pool_dir: str | None = None
    entry_pool_prefiltered: bool = True
    # modelo para regressao
    entry_model_type: str = "xgb"
    # regressao: janela de target (min). Se 0, usa contrato
    entry_reg_window_min: int = 0
    # regressao: pesos por amostra
    entry_reg_weight_alpha: float = 4.0
    entry_reg_weight_power: float = 1.2
    # regressao: bins para balanceamento (labels 0..100)
    entry_reg_balance_bins: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 80.0)
    # modo do label: "split_0_100" (long/short)
    entry_label_mode: str = "split_0_100"
    entry_label_scale: float = 100.0
    entry_sides: tuple[str, ...] = ("long", "short")
    # regressao: balanceamento por distancia do zero (0 = uniforme)
    entry_reg_balance_distance_power: float = 1.0
    # regressao: fração mínima por bin (relativo ao bin mais pesado)
    entry_reg_balance_min_frac: float = 0.1
    # aborta treino se RAM ultrapassar esse percentual (<=0 desativa)
    abort_ram_pct: float = 99.0

    # device

    xgb_device: str = "cuda:0"  # "cpu" se quiser forÃ§ar

    # thresholds são definidos manualmente em config/thresholds.py
    # params xgb
    entry_params: dict = field(default_factory=dict)
    # opcional: flags de features custom (senão usa o default por asset)
    feature_flags: dict | None = None
    # opcional: diretório para cache de features
    feature_cache_dir: str | None = None
    # contrato (se None, usa o DEFAULT_TRADE_CONTRACT do pacote)
    contract: TradeContract | None = None
    # opcional: retomar um wf_XXX já existente
    run_dir: str | None = None

    def __post_init__(self) -> None:
        # Se o usuÃ¡rio nÃ£o definiu offsets explicitamente, gera automaticamente.
        if not self.offsets_days:
            step = int(self.offsets_step_days)
            years = int(self.offsets_years)
            horizon = int(365 * max(1, years))
            if step <= 0:
                raise ValueError("offsets_step_days deve ser > 0")
            # range inclui apenas mÃºltiplos do step; para 4 anos (1460), termina em 1440.
            self.offsets_days = tuple(range(step, horizon + 1, step))


def run(settings: TrainSniperWFSettings | None = None) -> str:
    settings = settings or TrainSniperWFSettings()
    print("[train-wf] run(): importing sniper_trainer...", flush=True)
    from train.sniper_trainer import (  # type: ignore[import]
        TrainConfig,
        train_sniper_models,
        DEFAULT_ENTRY_PARAMS,
    )
    print("[train-wf] run(): sniper_trainer imported", flush=True)

    try:
        print(
            f"[train-wf] offsets_years={int(settings.offsets_years)} step={int(settings.offsets_step_days)}d "
            f"periodos={len(settings.offsets_days) if settings.offsets_days else 'auto'} max_symbols={int(settings.max_symbols)} "
            f"total_days={int(settings.total_days)} device={settings.xgb_device}",
            flush=True,
        )
    except Exception:
        pass

    entry_params = dict(settings.entry_params)
    entry_params["device"] = settings.xgb_device
    entry_params.setdefault("eval_metric", "rmse")
    print(f"[train-wf] entry_metric={entry_params.get('eval_metric')}", flush=True)
    contract_obj = settings.contract or DEFAULT_TRADE_CONTRACT

    cfg = TrainConfig(
        total_days=int(settings.total_days),
        offsets_days=tuple(int(x) for x in settings.offsets_days),
        mcap_min_usd=float(settings.mcap_min_usd),
        mcap_max_usd=float(settings.mcap_max_usd),
        max_symbols=int(settings.max_symbols),
        min_symbols_used_per_period=int(settings.min_symbols_used_per_period),
        entry_params=entry_params,
        max_rows_entry=int(settings.max_rows_entry),
        entry_pool_full=bool(settings.entry_pool_full),
        entry_pool_dir=Path(settings.entry_pool_dir) if settings.entry_pool_dir else None,
        entry_pool_prefiltered=bool(settings.entry_pool_prefiltered),
        entry_model_type=str(settings.entry_model_type or "xgb"),
        entry_reg_window_min=int(settings.entry_reg_window_min),
        entry_reg_weight_alpha=float(settings.entry_reg_weight_alpha),
        entry_reg_weight_power=float(settings.entry_reg_weight_power),
        entry_reg_balance_bins=tuple(float(x) for x in (settings.entry_reg_balance_bins or ())),
        entry_reg_balance_distance_power=float(settings.entry_reg_balance_distance_power),
        entry_reg_balance_min_frac=float(settings.entry_reg_balance_min_frac),
        entry_label_mode=str(settings.entry_label_mode or "split_0_100"),
        entry_label_scale=float(settings.entry_label_scale),
        entry_sides=tuple(str(s) for s in (settings.entry_sides or ())),
        use_feature_cache=bool(settings.use_feature_cache),
        asset_class=str(settings.asset_class or "crypto"),
        symbols=tuple(settings.symbols),
        symbols_file=Path(settings.symbols_file) if settings.symbols_file else None,
        feature_flags=settings.feature_flags,
        feature_cache_dir=Path(settings.feature_cache_dir) if settings.feature_cache_dir else None,
        contract=contract_obj,
        run_dir=Path(settings.run_dir) if settings.run_dir else None,
        abort_ram_pct=float(settings.abort_ram_pct),
    )
    run_dir = train_sniper_models(cfg)
    return str(run_dir)


def main() -> None:
    run_dir = run()
    print(f"[train-wf] run_dir: {run_dir}")


if __name__ == "__main__":
    main()

