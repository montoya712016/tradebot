# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Pipeline de treino (walk-forward) do Sniper com parÃ¢metros em cÃ³digo (sem ENV).

Recomendacao: execute via arquivo:
`python modules/train/train_sniper_wf.py`
"""

from dataclasses import dataclass, field
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

from train.sniper_trainer import (  # type: ignore[import]
    TrainConfig,
    train_sniper_models,
    DEFAULT_ENTRY_PARAMS,
    DEFAULT_DANGER_PARAMS,
    DEFAULT_EXIT_PARAMS,
    DEFAULT_PROFIT_PARAMS,
    DEFAULT_TIME_PARAMS,
)


@dataclass
class TrainSniperWFSettings:
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
    use_feature_cache: bool = True
    # se um perÃ­odo ficar com poucos sÃ­mbolos com dados suficientes, pula (evita modelo "escasso")
    min_symbols_used_per_period: int = 30

    # sizing (use RAM/VRAM)
    max_rows_entry: int = 4_000_000
    max_rows_danger: int = 2_000_000
    max_rows_exit: int = 2_000_000
    max_rows_profit: int = 2_000_000
    max_rows_time: int = 2_000_000
    bins_profit: int = 12
    bins_time: int = 12
    entry_ratio_neg_per_pos: float = 6.0
    danger_ratio_neg_per_pos: float = 4.0
    exit_ratio_neg_per_pos: float = 4.0

    # device
    xgb_device: str = "cuda:0"  # "cpu" se quiser forÃ§ar

    # thresholds (taxa alvo no VAL)
    entry_target_pred_pos_frac: float = 0.010
    exit_target_pred_pos_frac: float = 0.020
    entry_tau_min: float = 0.60
    entry_tau_max: float = 0.90

    # params xgb
    entry_params: dict = field(default_factory=lambda: dict(DEFAULT_ENTRY_PARAMS))
    danger_params: dict = field(default_factory=lambda: dict(DEFAULT_DANGER_PARAMS))
    exit_params: dict = field(default_factory=lambda: dict(DEFAULT_EXIT_PARAMS))
    profit_params: dict = field(default_factory=lambda: dict(DEFAULT_PROFIT_PARAMS))
    time_params: dict = field(default_factory=lambda: dict(DEFAULT_TIME_PARAMS))

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
    danger_params = dict(settings.danger_params)
    exit_params = dict(settings.exit_params)
    profit_params = dict(settings.profit_params)
    time_params = dict(settings.time_params)
    entry_params["device"] = settings.xgb_device
    danger_params["device"] = settings.xgb_device
    exit_params["device"] = settings.xgb_device
    profit_params["device"] = settings.xgb_device
    time_params["device"] = settings.xgb_device

    cfg = TrainConfig(
        total_days=int(settings.total_days),
        offsets_days=tuple(int(x) for x in settings.offsets_days),
        max_symbols=int(settings.max_symbols),
        min_symbols_used_per_period=int(settings.min_symbols_used_per_period),
        entry_params=entry_params,
        danger_params=danger_params,
        exit_params=exit_params,
        profit_params=profit_params,
        time_params=time_params,
        max_rows_entry=int(settings.max_rows_entry),
        max_rows_danger=int(settings.max_rows_danger),
        max_rows_exit=int(settings.max_rows_exit),
        max_rows_profit=int(settings.max_rows_profit),
        max_rows_time=int(settings.max_rows_time),
        bins_profit=int(settings.bins_profit),
        bins_time=int(settings.bins_time),
        entry_ratio_neg_per_pos=float(settings.entry_ratio_neg_per_pos),
        danger_ratio_neg_per_pos=float(settings.danger_ratio_neg_per_pos),
        exit_ratio_neg_per_pos=float(settings.exit_ratio_neg_per_pos),
        use_feature_cache=bool(settings.use_feature_cache),
        entry_target_pred_pos_frac=float(settings.entry_target_pred_pos_frac),
        exit_target_pred_pos_frac=float(settings.exit_target_pred_pos_frac),
        entry_tau_min=float(settings.entry_tau_min),
        entry_tau_max=float(settings.entry_tau_max),
    )
    run_dir = train_sniper_models(cfg)
    return str(run_dir)


def main() -> None:
    run_dir = run()
    print(f"âœ“ run_dir: {run_dir}")


if __name__ == "__main__":
    main()

