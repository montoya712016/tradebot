# -*- coding: utf-8 -*-
"""
Orquestrador interno do prepare_features (modular).

Objetivo imediato (pedidos):
- Calcular features escolhidas
- Calcular labels do contrato (sniper)
- Plotar
"""
from typing import Dict, Iterable, Any
import numpy as np, pandas as pd
# Suporte a execução como pacote OU como script direto (fallback para import absoluto)
try:
    from . import pf_config as cfg
    from .features import make_features
    from .labels import apply_timing_regression_labels
    from .plotting import plot_all
except Exception:
    import sys
    import pathlib

    # execução direta: precisamos que repo_root e `modules/` estejam no sys.path
    _HERE = pathlib.Path(__file__).resolve()
    for _p in _HERE.parents:
        if _p.name.lower() == "modules":
            _sp = str(_p)
            if _sp not in sys.path:
                sys.path.insert(0, _sp)
            break
    # adiciona repo_root para resolver imports como `core`
    _repo_root = _HERE
    for _p in _HERE.parents:
        if _p.name.lower() == "tradebot":
            _repo_root = _p
            break
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)
    from prepare_features import pf_config as cfg  # type: ignore[import]
    from prepare_features.features import make_features  # type: ignore[import]
    from prepare_features.labels import apply_timing_regression_labels  # type: ignore[import]
    from prepare_features.plotting import plot_all  # type: ignore[import]

try:
    from trade_contract import TradeContract  # type: ignore
except Exception:
    from trade_contract import TradeContract  # type: ignore[import]

# Conjuntos de features reconhecidos
FEATURE_KEYS = [
    "shitidx","atr","rsi","slope","vol","ci","cum_logret",
    "keltner","cci","adx","time_since","zlog","slope_reserr",
    "vol_ratio","regime","liquidity","rev_speed","vol_z","shadow","range_ratio",
    # novos blocos de contexto recente
    "runs","hh_hl","ema_cross","breakout","mom_short","wick_stats",
]


def default_flags(*, label: bool = True) -> Dict[str, bool]:
    """Retorna um dicionário de flags com todas as features ligadas e 'label' conforme solicitado."""
    d: Dict[str, bool] = {k: True for k in FEATURE_KEYS}
    d["label"] = bool(label)
    return d


# Template pronto para uso (todas as features ligadas; label=True)
FLAGS_DEFAULT: Dict[str, bool] = default_flags(label=True)

# Exemplo pronto (compatível com o estilo antigo) — pode importar direto como FLAGS
FLAGS: Dict[str, bool] = {
    "shitidx":      True,
    "atr":          True,
    "rsi":          True,
    "slope":        True,
    "vol":          True,
    "ci":           True,
    "cum_logret":   True,
    "keltner":      True,
    "cci":          True,
    "adx":          True,
    "time_since":   True,
    "zlog":         True,
    "slope_reserr": True,
    "vol_ratio":    True,
    "regime":       True,
    "liquidity":    True,
    "rev_speed":    True,
    "vol_z":        True,
    "shadow":       True,
    "range_ratio":  True,
    "runs":         True,
    "hh_hl":        True,
    "ema_cross":    True,
    "breakout":     True,
    "mom_short":    True,
    "wick_stats":   True,
    "label":        True,
    "plot_candles": True,
}

# Flags para rodar somente labels (sem features).
FLAGS_LABEL_ONLY: Dict[str, bool] = {k: False for k in FEATURE_KEYS}
FLAGS_LABEL_ONLY["label"] = False
FLAGS_LABEL_ONLY["plot_candles"] = False

# Parâmetros padrão (somente MySQL)
DEFAULT_SYMBOL: str = "ADAUSDT"
DEFAULT_DAYS: int = 90
DEFAULT_REMOVE_TAIL_DAYS: int = 0
DEFAULT_CANDLE_SEC: int = 60
DEFAULT_U_THRESHOLD: float = 0.0
DEFAULT_GREY_ZONE: float | None = None

def build_flags(enable: Iterable[str] | None = None, *, label: bool = True) -> Dict[str, bool]:
    """Convenience: liga somente as chaves fornecidas (True) e mantém o resto False.
    Equivalente ao estilo antigo quando você quer ativar por lista.
    """
    base = default_flags(label=label)
    for k in (enable or []):
        k2 = str(k).lower().strip()
        if k2 in base:
            base[k2] = True
    return base


def normalize_flags(flags: Dict[str, bool] | None = None, *, enable: Iterable[str] | None = None, label: bool = True) -> Dict[str, bool]:
    """Normaliza para o formato completo estilo antigo (todas as chaves presentes True/False).
    Prioriza 'flags' se fornecido; senão usa a lista 'enable'.
    """
    if flags is not None:
        base = default_flags(label=label)
        # Preserve "meta flags" (controle de logs/diagnósticos etc) que NÃO fazem parte do set de features.
        # Isso é crítico para permitir `_quiet=True` atravessar a pipeline até `make_features`.
        meta: Dict[str, Any] = {}
        for k, v in flags.items():
            k2 = str(k).lower().strip()
            if k2 in base:
                base[k2] = bool(v)
            elif k2 in {"quiet", "_quiet"}:
                # suportar ambas as grafias
                meta["quiet"] = bool(v)
                meta["_quiet"] = bool(v)
            elif k2 in {"feature_timings", "_feature_timings"}:
                meta["feature_timings"] = bool(v)
                meta["_feature_timings"] = bool(v)
            elif str(k).startswith("_"):
                # permita passar chaves internas sem quebrar (ex.: _quiet, _debug, etc.)
                meta[str(k)] = v
        # garante 'label' explícito (pode ter sido passado dentro de flags)
        if "label" in flags:
            base["label"] = bool(flags["label"])
        if meta:
            base.update(meta)
        return base
    # sem flags: usa enable
    return build_flags(enable=enable, label=label)


def _infer_candle_sec(idx: pd.DatetimeIndex) -> int:
    if len(idx) < 2:
        return 60
    try:
        dt = float((idx[1] - idx[0]).total_seconds())
        if not np.isfinite(dt) or dt <= 0.0:
            return 60
        return int(round(dt))
    except Exception:
        return 60


def run(
    df_ohlc: pd.DataFrame,
    *,
    enable: Iterable[str] | None = None,
    flags: Dict[str, bool] | None = None,
    plot: bool = True,
    plot_candles: bool = False,
    plot_path: str | None = None,
    u_threshold: float = 0.0,
    grey_zone: float | None = None,
    show: bool = True,
    verbose_features: bool = True,
    apply_precision: bool = False,
    default_decimals: int = 5,
    trade_contract: TradeContract | None = None,
    mark_gaps: bool = True,
) -> pd.DataFrame:
    """Pipeline simples: features selecionadas + timing labels + plot opcional.

    Retorna o DataFrame (mutado) com colunas calculadas.
    """
    # Checagem mínima
    required = {"open","high","low","close"}
    if not required.issubset(df_ohlc.columns):
        raise ValueError("df_ohlc precisa conter colunas open/high/low/close")

    flags = normalize_flags(flags, enable=enable, label=True)

    # Snapshot de colunas (para identificar apenas as novas features)
    _cols_before = set(df_ohlc.columns)
    # Features (só se alguma estiver ligada)
    features_on = any(bool(flags.get(k, False)) for k in FEATURE_KEYS)
    if features_on:
        make_features(df_ohlc, flags, verbose=verbose_features)
    _new_feat_cols = [c for c in df_ohlc.columns if c not in _cols_before]
    # (stats printing removido por padrão para evitar ruído)
    # Opcional: reduzir precisão/armazenamento das features (pós-cálculo)
    if apply_precision:
        try:
            precision_rules: list[tuple[str, int]] = [
                # prefixo, casas decimais desejadas
                ("atr_pct_", 4), ("vol_pct_", 4), ("keltner_", 4),
                ("slope_pct_", 5), ("slope_reserr_pct_", 4),
                ("vol_ratio_pct_", 4), ("pct_from_", 4), ("range_ratio_", 4),
                ("rev_speed_", 5), ("zlog_", 4), ("signed_vol_z", 4), ("vol_z", 4),
                ("cci_", 1), ("adx_", 1), ("rsi_", 1),
                ("log_volume_ema", 5), ("liquidity_ratio", 4),
            ]
            time_since_prefix = ("time_since_",)
            for c in _new_feat_cols:
                try:
                    # time_since_* são inteiros pequenos: converte para uint16 com clamp
                    if any(c.startswith(p) for p in time_since_prefix):
                        arr = df_ohlc[c].to_numpy()
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        arr = np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
                        df_ohlc[c] = arr
                        continue
                    # regras por prefixo
                    dec = None
                    for pref, d in precision_rules:
                        if c.startswith(pref):
                            dec = int(d); break
                    if dec is None:
                        dec = int(default_decimals)
                    arr = df_ohlc[c].to_numpy(np.float32, copy=False)
                    # substituir inf/NaN
                    arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
                    # arredondar
                    with np.errstate(invalid="ignore"):
                        arr = np.round(arr, decimals=dec).astype(np.float32, copy=False)
                    df_ohlc[c] = arr
                except Exception:
                    # fallback: ignora coluna problemática
                    continue
        except Exception:
            pass

    # Labels de regressão (timing)
    if bool(flags.get("label", True)):
        candle_sec = _infer_candle_sec(df_ohlc.index)
        # Timing regression labels
        apply_timing_regression_labels(
            df_ohlc,
            candle_sec=int(candle_sec),
        )

    # Plot
    if plot:
        candle_sec = _infer_candle_sec(df_ohlc.index)
        plot_all(
            df_ohlc.dropna(subset=["open","high","low","close"]),
            flags,
            u_threshold=float(u_threshold),
            candle_sec=int(candle_sec),
            plot_candles=bool(plot_candles),
            grey_zone=grey_zone,
            show=show,
            save_path=plot_path,
            mark_gaps=bool(mark_gaps),
        )

    return df_ohlc


def run_from_flags_dict(
    df_ohlc: pd.DataFrame,
    FLAGS: Dict[str, bool],
    *,
    plot_candles: bool | None = None,
    plot: bool = True,
    plot_path: str | None = None,
    u_threshold: float = 0.0,
    grey_zone: float | None = None,
    show: bool = True,
    verbose_features: bool = True,
    trade_contract: TradeContract | None = None,
    mark_gaps: bool = True,
) -> pd.DataFrame:
    """Compat com o estilo antigo usando um único dict FLAGS (True/False por feature).

    Extras aceitos dentro de FLAGS (se presentes):
      - label (bool)
      - plot_candles (bool) — alternativo ao parâmetro homônimo
    """
    label_on = bool(FLAGS.get("label", True))
    flags = normalize_flags(FLAGS, label=label_on)
    if plot_candles is None:
        plot_candles = bool(FLAGS.get("plot_candles", False))
    return run(
        df_ohlc,
        flags=flags,
        plot=plot,
        plot_candles=bool(plot_candles),
        plot_path=plot_path,
        u_threshold=u_threshold,
        grey_zone=grey_zone,
        show=show,
        verbose_features=verbose_features,
        trade_contract=trade_contract,
        mark_gaps=mark_gaps,
    )


__all__ = [
    "cfg",
    "FEATURE_KEYS",
    "FLAGS_DEFAULT",
    "FLAGS",
    "default_flags",
    "build_flags",
    "normalize_flags",
    "make_features",
    "apply_timing_regression_labels",
    "plot_all",
    "run",
    "run_from_flags_dict",
]


if __name__ == "__main__":
    # Execução declarativa (somente MySQL): tudo em código, sem flags/CLI
    try:
        try:
            from .data import load_ohlc_1m_series, to_ohlc_from_1m
        except Exception:
            from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m  # type: ignore[import]
        raw_1m = load_ohlc_1m_series(DEFAULT_SYMBOL, int(DEFAULT_DAYS), remove_tail_days=int(DEFAULT_REMOVE_TAIL_DAYS))
        if raw_1m.empty:
            print("Sem dados retornados do MySQL.", flush=True)
        df_ohlc = to_ohlc_from_1m(raw_1m, int(DEFAULT_CANDLE_SEC))
        label_only = True
        flags_to_use = FLAGS_LABEL_ONLY if label_only else FLAGS
        flags_to_use["label"] = True
        flags_to_use["plot_candles"] = True
        df = run_from_flags_dict(
            df_ohlc,
            flags_to_use,
            plot=True,
            u_threshold=float(DEFAULT_U_THRESHOLD),
            grey_zone=DEFAULT_GREY_ZONE,
            show=True,
        )
        try:
            print(f"OK: rows={len(df):,} | cols={len(df.columns):,}".replace(',', '.'), flush=True)
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"Erro: {e}", flush=True)
        except Exception:
            pass
