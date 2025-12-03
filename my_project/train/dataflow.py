# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, List, Any
import numpy as np, pandas as pd
import os, time
from pathlib import Path
try:
    import psutil  # watchdog de RAM
except Exception:
    psutil = None
import multiprocessing as mp

# Imports do prepare_features com fallback absoluto
try:
    from ..prepare_features.prepare_features import run as pf_run, FLAGS as PF_FLAGS, DEFAULT_SYMBOL, DEFAULT_DAYS, DEFAULT_REMOVE_TAIL_DAYS, DEFAULT_CANDLE_SEC, build_flags, FEATURE_KEYS
    from ..prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
    from ..prepare_features.dataset import build_entry_dataset_indices, balance_both_sides
    from ..prepare_features.pf_config import WEIGHT_RULE
except Exception:
    from my_project.prepare_features.prepare_features import run as pf_run, FLAGS as PF_FLAGS, DEFAULT_SYMBOL, DEFAULT_DAYS, DEFAULT_REMOVE_TAIL_DAYS, DEFAULT_CANDLE_SEC, build_flags, FEATURE_KEYS
    from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
    from my_project.prepare_features.dataset import build_entry_dataset_indices, balance_both_sides
    from my_project.prepare_features.pf_config import WEIGHT_RULE

# Flags globais padrão (sem my_project.config): liga famílias conhecidas e mantém label/pivots True
GLOBAL_FLAGS_FULL = build_flags(enable=FEATURE_KEYS, label=True)
GLOBAL_FLAGS_FULL["pivots"] = True

def ram_str() -> str:
    try:
        if psutil:
            vm = psutil.virtual_memory()
            return f"{vm.used/(1024**3):.1f}GB/{vm.total/(1024**3):.1f}GB"
    except Exception:
        pass
    return "n/a"

def _pin_num_threads(n: int = 1) -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(n)))
        os.environ.setdefault("MKL_NUM_THREADS", str(int(n)))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(int(n)))
        os.environ.setdefault("NUMEXPR_MAX_THREADS", str(int(n)))
    except Exception:
        pass

# Defaults locais substituindo train.config
RATIO_NEG_PER_POS: float = 3.0
_WEIGHT_CFG = WEIGHT_RULE or {}
U_WEIGHT_MAX: float = float(_WEIGHT_CFG.get("u_weight_max", 10.0))
ALPHA_POS: float = float(_WEIGHT_CFG.get("alpha_pos", 2.0))
BETA_NEG: float = float(_WEIGHT_CFG.get("beta_neg", 3.0))
GAMMA_EASE: float = float(_WEIGHT_CFG.get("gamma_ease", 0.6))
NEG_FLOOR: float = float(_WEIGHT_CFG.get("neg_floor", 0.5))
MP_USE: bool = True
MP_MAX_PROCS: int = 32
MP_RESERVE_GB: float = 2.0
MP_MIN_FREE_GB: float = 1.0
MP_SAFETY_FACTOR: float = 2.0


def _list_feature_columns(df: pd.DataFrame) -> List[str]:
    """Seleciona colunas modeláveis: remove OHLC/labels/regras/meta, mantém numéricas."""
    drop_prefixes = (
        "rev_buy_rule", "peak_rule", "U_compra", "U_venda", "U_total",
        "exit_", "pivot_", "rev_buy_candidate",
    )
    exclude_exact = {"open","high","low","close","volume","gap_after",
                     "y","y_long","y_short","dd_pct","ru_up_pct"}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _gather_X_y_w(df: pd.DataFrame, feat_cols: List[str], pos_idx: np.ndarray, neg_idx: np.ndarray, *, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monta X, y, w, ts e u (U_total) com penalizações por U baseadas no lado (buy/short)."""
    Xp = df.iloc[pos_idx][feat_cols].to_numpy(np.float32)
    Xn = df.iloc[neg_idx][feat_cols].to_numpy(np.float32)
    X = np.vstack([Xp, Xn]) if Xn.size else Xp.copy()
    y = np.concatenate([np.ones(len(pos_idx), np.int32), np.zeros(len(neg_idx), np.int32)]) if len(neg_idx) else np.ones(len(pos_idx), np.int32)
    ts = df.index.to_numpy()
    tsp = ts[pos_idx] if len(pos_idx) else np.empty((0,), dtype='datetime64[ns]')
    tsn = ts[neg_idx] if len(neg_idx) else np.empty((0,), dtype='datetime64[ns]')
    ts_all = np.concatenate([tsp, tsn]) if tsn.size else tsp

    # Pesos
    U = df["U_total"].astype(float).to_numpy()
    u_pos = np.clip((U[pos_idx] if len(pos_idx) else np.array([],float)), -U_WEIGHT_MAX, U_WEIGHT_MAX)
    u_neg = np.clip((U[neg_idx] if len(neg_idx) else np.array([],float)), -U_WEIGHT_MAX, U_WEIGHT_MAX)
    # guarda U concatenado (na mesma ordem do X empilhado)
    u_all = (np.concatenate([u_pos, u_neg]) if len(neg_idx) else u_pos).astype(np.float32, copy=False)
    # Normalização 0..1
    s_pos = np.clip(u_pos / max(1e-6, U_WEIGHT_MAX), 0.0, 1.0)
    if side == "long":
        # positivos com U alto (bom) => reforço
        w_pos = 1.0 + float(ALPHA_POS) * s_pos
        # negativos com U < 0 (ruins para buy) => penalização; U > 0 => aliviar
        s_neg_bad = np.clip((-u_neg) / max(1e-6, U_WEIGHT_MAX), 0.0, 1.0)
        w_neg = 1.0 + float(BETA_NEG) * s_neg_bad
        s_neg_good = np.clip(u_neg / max(1e-6, U_WEIGHT_MAX), 0.0, 1.0)
        w_neg *= (1.0 - float(GAMMA_EASE) * s_neg_good)
        w_neg = np.maximum(w_neg, float(NEG_FLOOR))
    else:
        # short: usa -U
        s_pos_sh = np.clip((-u_pos) / max(1e-6, U_WEIGHT_MAX), 0.0, 1.0)
        w_pos = 1.0 + float(ALPHA_POS) * s_pos_sh
        s_neg_bad = np.clip(u_neg / max(1e-6, U_WEIGHT_MAX), 0.0, 1.0)  # U>0 é ruim para short
        w_neg = 1.0 + float(BETA_NEG) * s_neg_bad
        s_neg_good = np.clip((-u_neg) / max(1e-6, U_WEIGHT_MAX), 0.0, 1.0)
        w_neg *= (1.0 - float(GAMMA_EASE) * s_neg_good)
        w_neg = np.maximum(w_neg, float(NEG_FLOOR))

    w = (np.concatenate([w_pos, w_neg]).astype(np.float32) if len(neg_idx) else w_pos.astype(np.float32))
    return X, y.astype(np.int32), w, ts_all, u_all


def prepare_block(symbol: str, *, total_days: int, remove_tail_days: int, flags: Dict[str, bool] | None = None) -> dict:
    """Produz X,y,w por lado para um bloco (T-remove_tail_days)."""
    raw = load_ohlc_1m_series(symbol, int(total_days), remove_tail_days=int(remove_tail_days))
    df_ohlc = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
    use_flags = (flags or {**GLOBAL_FLAGS_FULL, "pivots": True, "label": True})
    df = pf_run(df_ohlc, flags=use_flags, plot=False)

    idxs = build_entry_dataset_indices(
        df,
        loose_drop=0.02, loose_rise=0.02, loose_u_hi=2.0, loose_u_lo=-2.0,
        lookback_min=120, pre_off_min=(5, 15, 30), post_off_min=(5, 15, 30), min_dist_min=3,
    )
    balanced = balance_both_sides(idxs, ratio_neg_per_pos=RATIO_NEG_PER_POS, bucket_weights=(0.5,0.3,0.15,0.05), seed=42)
    # Inclui pontos do lado oposto como negativos (ajuda o modelo a evitar o outro lado)
    try:
        neg_long_extra = np.setdiff1d(idxs.get("pos_short", np.empty(0, dtype=np.int64)), balanced["long"]["pos"], assume_unique=False)
        neg_short_extra= np.setdiff1d(idxs.get("pos_long",  np.empty(0, dtype=np.int64)), balanced["short"]["pos"], assume_unique=False)
        balanced["long"]["neg"]  = np.unique(np.concatenate([balanced["long"]["neg"],  neg_long_extra]))
        balanced["short"]["neg"] = np.unique(np.concatenate([balanced["short"]["neg"], neg_short_extra]))
    except Exception:
        pass

    feat_cols = _list_feature_columns(df)
    # LONG
    X_long, y_long, w_long, ts_long, u_long = _gather_X_y_w(df, feat_cols, balanced["long"]["pos"], balanced["long"]["neg"], side="long")
    # SHORT
    X_short, y_short, w_short, ts_short, u_short = _gather_X_y_w(df, feat_cols, balanced["short"]["pos"], balanced["short"]["neg"], side="short")

    return dict(
        symbol=symbol,
        remove_tail_days=int(remove_tail_days),
        feature_cols=feat_cols,
        long=dict(X=X_long, y=y_long, w=w_long, ts=ts_long, u=u_long),
        short=dict(X=X_short, y=y_short, w=w_short, ts=ts_short, u=u_short),
        counts=dict(
            pos_long=int(balanced["long"]["pos"].size), neg_long=int(balanced["long"]["neg"].size),
            pos_short=int(balanced["short"]["pos"].size), neg_short=int(balanced["short"]["neg"].size),
        ),
    )


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ===================== Modo Multiprocessos com watchdog de RAM =====================

def _np_save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def _to_int64_ts(a: np.ndarray) -> np.ndarray:
    try:
        if np.issubdtype(a.dtype, np.datetime64):
            return a.astype('datetime64[ns]').astype('int64')
    except Exception:
        pass
    # tenta via pandas
    try:
        return pd.to_datetime(a).to_numpy(dtype='datetime64[ns]').astype('int64')
    except Exception:
        return a.astype('int64', copy=False)


def _process_symbol_worker(
    sym: str,
    sym_id: int,
    total_days: int,
    remove_tail_days: int,
    use_flags: Dict[str, bool],
    feat_master: List[str] | None,
    parts_dir: str,
    ratio_neg_per_pos: float | int,
) -> Dict[str, Any]:
    """
    Worker independente (novo processo): calcula features, seleciona índices e salva partes no disco.
    Retorna somente metadados e estatísticas, evitando grandes transferências de arrays.
    """
    try:
        _pin_num_threads()
    except Exception:
        pass
    peak_gb = 0.0
    try:
        pself = psutil.Process(os.getpid()) if psutil else None
    except Exception:
        pself = None

    t0 = time.time()
    raw = load_ohlc_1m_series(sym, int(total_days), remove_tail_days=0)
    if raw.empty:
        return {"symbol": sym, "sym_id": int(sym_id), "rows_ohlc": 0, "error": "sem_dados"}
    df_all = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
    if int(remove_tail_days) > 0 and len(df_all) > 0:
        cutoff = df_all.index[-1] - pd.Timedelta(days=int(remove_tail_days))
        df_ohlc = df_all[df_all.index < cutoff]
    else:
        df_ohlc = df_all
    rows_ohlc = int(len(df_ohlc))
    if pself:
        try:
            peak_gb = max(peak_gb, pself.memory_info().rss / (1024**3))
        except Exception:
            pass

    df = pf_run(df_ohlc, flags=use_flags, plot=False)
    if df is None or len(df) == 0:
        return {"symbol": sym, "sym_id": int(sym_id), "rows_ohlc": rows_ohlc, "error": "pf_vazio"}
    if pself:
        try:
            peak_gb = max(peak_gb, pself.memory_info().rss / (1024**3))
        except Exception:
            pass

    feat_cols_here: List[str] = [
        c for c in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[c])
            and c not in {"open","high","low","close","volume","gap_after",
                          "y","y_long","y_short","dd_pct","ru_up_pct"}
            and not c.startswith(("rev_buy_rule","peak_rule","U_compra","U_venda","U_total","exit_","pivot_","rev_buy_candidate"))
        )
    ]
    if feat_master is None:
        feat_master = list(feat_cols_here)
    else:
        df = _ensure_columns(df, feat_master)
    # índices balanceados
    idxs = build_entry_dataset_indices(
        df,
        loose_drop=0.02, loose_rise=0.02, loose_u_hi=2.0, loose_u_lo=-2.0,
        lookback_min=120, pre_off_min=(5, 15, 30), post_off_min=(5, 15, 30), min_dist_min=3,
    )
    balanced = balance_both_sides(idxs, ratio_neg_per_pos=ratio_neg_per_pos, bucket_weights=(0.5,0.3,0.15,0.05), seed=42)
    try:
        neg_long_extra = np.setdiff1d(idxs.get("pos_short", np.empty(0, dtype=np.int64)), balanced["long"]["pos"], assume_unique=False)
        neg_short_extra= np.setdiff1d(idxs.get("pos_long",  np.empty(0, dtype=np.int64)), balanced["short"]["pos"], assume_unique=False)
        balanced["long"]["neg"]  = np.unique(np.concatenate([balanced["long"]["neg"],  neg_long_extra]))
        balanced["short"]["neg"] = np.unique(np.concatenate([balanced["short"]["neg"], neg_short_extra]))
    except Exception:
        pass

    # monta X/y/w/ts e salva
    Xl, yl, wl, tsl, ul = _gather_X_y_w(df, feat_master, balanced["long"]["pos"], balanced["long"]["neg"], side="long")
    Xs, ys, ws, tss, us = _gather_X_y_w(df, feat_master, balanced["short"]["pos"], balanced["short"]["neg"], side="short")
    # converte ts para int64 para ser portátil
    tsl64 = _to_int64_ts(tsl)
    tss64 = _to_int64_ts(tss)
    # salva partes
    pd_long = Path(parts_dir) / "long"
    pd_short = Path(parts_dir) / "short"
    name = f"{int(sym_id):05d}_{sym}"
    _np_save(pd_long / f"{name}_X.npy", Xl)
    _np_save(pd_long / f"{name}_y.npy", yl.astype(np.int32, copy=False))
    _np_save(pd_long / f"{name}_w.npy", wl.astype(np.float32, copy=False))
    _np_save(pd_long / f"{name}_ts.npy", tsl64.astype(np.int64, copy=False))
    _np_save(pd_long / f"{name}_u.npy", ul.astype(np.float32, copy=False))
    _np_save(pd_long / f"{name}_sym.npy", np.full(yl.shape, int(sym_id), dtype=np.uint16))

    _np_save(pd_short / f"{name}_X.npy", Xs)
    _np_save(pd_short / f"{name}_y.npy", ys.astype(np.int32, copy=False))
    _np_save(pd_short / f"{name}_w.npy", ws.astype(np.float32, copy=False))
    _np_save(pd_short / f"{name}_ts.npy", tss64.astype(np.int64, copy=False))
    _np_save(pd_short / f"{name}_u.npy", us.astype(np.float32, copy=False))
    _np_save(pd_short / f"{name}_sym.npy", np.full(ys.shape, int(sym_id), dtype=np.uint16))

    if pself:
        try:
            peak_gb = max(peak_gb, pself.memory_info().rss / (1024**3))
        except Exception:
            pass
    last_ts = df.index[-1]
    dt = time.time() - t0
    return {
        "symbol": sym,
        "sym_id": int(sym_id),
        "rows_ohlc": rows_ohlc,
        "feat_cols": feat_cols_here,
        "last_ts": last_ts,
        "counts": {
            "pl": int(balanced['long']['pos'].size),
            "nl": int(balanced['long']['neg'].size),
            "ps": int(balanced['short']['pos'].size),
            "ns": int(balanced['short']['neg'].size),
        },
        "peak_gb": float(peak_gb),
        "secs": float(dt),
    }

def _worker_wrapper(q: mp.Queue, args: tuple):
    try:
        res = _process_symbol_worker(*args)
    except Exception as e:
        try:
            sym = args[0]
            sym_id = int(args[1])
        except Exception:
            sym, sym_id = "?", -1
        res = {"error": str(e), "symbol": sym, "sym_id": sym_id}
    try:
        q.put(res)
    except Exception:
        pass


def prepare_block_multi(symbols: list[str], *, total_days: int, remove_tail_days: int, flags: Dict[str, bool] | None = None, ratio_neg_per_pos: int | float = RATIO_NEG_PER_POS, seed: int = 42, ohlc_cache: dict[str, pd.DataFrame] | None = None) -> dict:
    """Versão multi-símbolo: agrega X,y,w por lado e alinha colunas de features.
    Também retorna âncoras de tempo úteis para salvar no run.
    """
    # Se MP habilitado e existir variável de destino de partes, usamos scheduler
    out_parts_dir_env = os.getenv("TRAIN_PARTS_DIR", "")
    use_mp = bool(MP_USE)
    parts_dir: Path | None = (Path(out_parts_dir_env) if out_parts_dir_env else None)
    if use_mp and parts_dir is not None and len(symbols) > 1:
        return _prepare_block_multi_mp(
            symbols=symbols,
            total_days=total_days,
            remove_tail_days=remove_tail_days,
            flags=flags,
            ratio_neg_per_pos=ratio_neg_per_pos,
            seed=seed,
            parts_dir=parts_dir,
        )

    rng = np.random.default_rng(int(seed))
    feat_master: list[str] | None = None
    Xl_list: list[np.ndarray] = []; yl_list: list[np.ndarray] = []; wl_list: list[np.ndarray] = []
    Xs_list: list[np.ndarray] = []; ys_list: list[np.ndarray] = []; ws_list: list[np.ndarray] = []
    sym_l_list: list[np.ndarray] = []; sym_s_list: list[np.ndarray] = []
    last_ts_list: list[pd.Timestamp] = []

    # acumuladores globais
    tot_pos_l = tot_neg_l = tot_pos_s = tot_neg_s = 0
    n_total = max(1, len(symbols))
    # mapeia símbolo -> id pequeno para reduzir RAM (uint16)
    sym_to_id: dict[str, int] = {s: i for i, s in enumerate(symbols)}
    _sym_dtype = (np.uint16 if len(symbols) <= np.iinfo(np.uint16).max else np.uint32)

    cache = (ohlc_cache or {})
    for idx, sym in enumerate(symbols, start=1):
        print(
            f"[dataflow] carregando {sym} | days={int(total_days)} | tail={int(remove_tail_days)} "
            f"[{idx}/{n_total} {(idx/n_total):.0%}]",
            flush=True,
        )
        if sym in cache:
            df_full = cache[sym]
            # corta cauda no OHLC já pronto
            if int(remove_tail_days) > 0 and len(df_full) > 0:
                cutoff = df_full.index[-1] - pd.Timedelta(days=int(remove_tail_days))
                df_ohlc = df_full[df_full.index < cutoff]
            else:
                df_ohlc = df_full
        else:
            raw = load_ohlc_1m_series(sym, int(total_days), remove_tail_days=0)
            if raw.empty:
                print(f"[dataflow] {sym}: sem dados 1m — pulando", flush=True)
                continue
            df_all = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
            cache[sym] = df_all
            if int(remove_tail_days) > 0 and len(df_all) > 0:
                cutoff = df_all.index[-1] - pd.Timedelta(days=int(remove_tail_days))
                df_ohlc = df_all[df_all.index < cutoff]
            else:
                df_ohlc = df_all
        print(f"[dataflow] {sym}: ohlc rows={len(df_ohlc):,}".replace(',', '.'), flush=True)
        use_flags = (flags or {**GLOBAL_FLAGS_FULL, "pivots": True, "label": True})
        try:
            df = pf_run(df_ohlc, flags=use_flags, plot=False)
        except Exception as e:
            print(f"[dataflow] {sym}: erro em prepare_features.run: {e}", flush=True)
            continue
        if len(df) == 0:
            print(f"[dataflow] {sym}: df vazio após PF — pulando", flush=True)
            continue
        last_ts_list.append(df.index[-1])

        idxs = build_entry_dataset_indices(
            df,
            loose_drop=0.02, loose_rise=0.02, loose_u_hi=2.0, loose_u_lo=-2.0,
            lookback_min=120, pre_off_min=(5, 15, 30), post_off_min=(5, 15, 30), min_dist_min=3,
        )
        # contagens rápidas
        try:
            n_buy = int(df.get("rev_buy_rule", pd.Series(False, index=df.index)).astype(bool).sum())
        except Exception:
            n_buy = 0
        try:
            n_peak = int(df.get("peak_rule", pd.Series(False, index=df.index)).astype(bool).sum())
        except Exception:
            n_peak = 0
        print((
            f"[dataflow] {sym}: marks buy={n_buy:,} peak={n_peak:,} | near_long={idxs['neg_near_long'].size:,} near_short={idxs['neg_near_short'].size:,}"
        ).replace(',', '.'), flush=True)

        balanced = balance_both_sides(idxs, ratio_neg_per_pos=ratio_neg_per_pos, bucket_weights=(0.5,0.3,0.15,0.05), seed=42)

        feat_cols = _list_feature_columns(df)
        if feat_master is None:
            feat_master = list(feat_cols)
        else:
            # assegura presença de todas as colunas da master
            df = _ensure_columns(df, feat_master)

        X_long, y_long, w_long, ts_long = _gather_X_y_w(df, feat_master, balanced["long"]["pos"], balanced["long"]["neg"], side="long")
        X_short, y_short, w_short, ts_short = _gather_X_y_w(df, feat_master, balanced["short"]["pos"], balanced["short"]["neg"], side="short")

        pl = balanced['long']['pos'].size; nl = balanced['long']['neg'].size
        ps = balanced['short']['pos'].size; ns = balanced['short']['neg'].size
        print((
            f"[dataflow] {sym}: long pos={pl:,} neg={nl:,} | short pos={ps:,} neg={ns:,} | feats={len(feat_master)}"
        ).replace(',', '.'), flush=True)

        # totais acumulados
        tot_pos_l += int(pl); tot_neg_l += int(nl)
        tot_pos_s += int(ps); tot_neg_s += int(ns)
        print((
            f"[dataflow] total acum.: long pos={tot_pos_l:,} neg={tot_neg_l:,} | short pos={tot_pos_s:,} neg={tot_neg_s:,}"
        ).replace(',', '.'), flush=True)
        # uso de RAM após concluir o símbolo
        try:
            print(f"[dataflow] ram uso: {ram_str()}", flush=True)
        except Exception:
            pass

        if X_long.size:
            Xl_list.append(X_long); yl_list.append(y_long); wl_list.append(w_long)
            sym_l_list.append(np.full(len(y_long), sym_to_id.get(sym, 0), dtype=_sym_dtype))
        if X_short.size:
            Xs_list.append(X_short); ys_list.append(y_short); ws_list.append(w_short)
            sym_s_list.append(np.full(len(y_short), sym_to_id.get(sym, 0), dtype=_sym_dtype))
        # anexa timestamps
        if X_long.size:
            try:
                tl = np.asarray(ts_long, dtype='datetime64[ns]')
            except Exception:
                tl = pd.to_datetime(ts_long).to_numpy()
            if 'tsl_list' not in locals():
                tsl_list = []
            tsl_list.append(tl)
        if X_short.size:
            try:
                ts_ = np.asarray(ts_short, dtype='datetime64[ns]')
            except Exception:
                ts_ = pd.to_datetime(ts_short).to_numpy()
            if 'tss_list' not in locals():
                tss_list = []
            tss_list.append(ts_)

        # Libera memória temporária (OHLC/DF) cedo
        try:
            del df, df_ohlc
        except Exception:
            pass
        try:
            del raw
        except Exception:
            pass
        if (idx % 2) == 0:
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass

    feat_master = feat_master or []
    Xl = (np.vstack(Xl_list) if Xl_list else np.empty((0, len(feat_master)), np.float32))
    yl = (np.concatenate(yl_list).astype(np.int32) if yl_list else np.empty((0,), np.int32))
    wl = (np.concatenate(wl_list).astype(np.float32) if wl_list else np.empty((0,), np.float32))
    Xs = (np.vstack(Xs_list) if Xs_list else np.empty((0, len(feat_master)), np.float32))
    ys = (np.concatenate(ys_list).astype(np.int32) if ys_list else np.empty((0,), np.int32))
    ws = (np.concatenate(ws_list).astype(np.float32) if ws_list else np.empty((0,), np.float32))
    tsl = (np.concatenate(tsl_list) if 'tsl_list' in locals() else np.empty((0,), dtype='datetime64[ns]'))
    tss = (np.concatenate(tss_list) if 'tss_list' in locals() else np.empty((0,), dtype='datetime64[ns]'))
    syml = (np.concatenate(sym_l_list) if sym_l_list else np.empty((0,), dtype=_sym_dtype))
    syms = (np.concatenate(sym_s_list) if sym_s_list else np.empty((0,), dtype=_sym_dtype))

    # Âncoras de tempo: usa o menor último timestamp para evitar extrapolação
    T_anchor = (min(last_ts_list) if last_ts_list else None)

    return dict(
        feature_cols=feat_master,
        long=dict(X=Xl, y=yl, w=wl, ts=tsl, sym_id=syml),
        short=dict(X=Xs, y=ys, w=ws, ts=tss, sym_id=syms),
        anchor_end_utc=(T_anchor.isoformat() if T_anchor is not None else None),
        sym_map=symbols,
    )

def _prepare_block_multi_mp(*, symbols: list[str], total_days: int, remove_tail_days: int, flags: Dict[str, bool] | None, ratio_neg_per_pos: int | float, seed: int, parts_dir: Path) -> dict:
    """
    Scheduler com controle dinâmico de processos baseado na RAM disponível.
    1) Escolhe um piloto (preferência XLMUSDT) para definir feat_master e calibrar pico.
    2) Dispara workers respeitando reserva de RAM e fator de segurança sobre pico medido.
    3) Retorna metadados e diretório de partes para montagem posterior.
    """
    rng = np.random.default_rng(int(seed))
    parts_dir = Path(parts_dir)
    (parts_dir / "long").mkdir(parents=True, exist_ok=True)
    (parts_dir / "short").mkdir(parents=True, exist_ok=True)

    # mapeia símbolos para ids compactos
    sym_to_id: dict[str, int] = {s: i for i, s in enumerate(symbols)}

    # reordena símbolos para calibrar com XLM primeiro, se existir
    syms = list(symbols)
    if "XLMUSDT" in syms:
        syms.insert(0, syms.pop(syms.index("XLMUSDT")))

    use_flags = (flags or {**GLOBAL_FLAGS_FULL, "pivots": True, "label": True})

    # piloto sequencial (no processo pai) para obter feat_master e pico
    pilot = syms[0]
    print(f"[dataflow][mp] piloto: {pilot}", flush=True)
    meta_pilot = _process_symbol_worker(
        pilot, sym_to_id[pilot], int(total_days), int(remove_tail_days), use_flags, None, str(parts_dir), ratio_neg_per_pos
    )
    if meta_pilot.get("error"):
        raise RuntimeError(f"Piloto falhou: {pilot} -> {meta_pilot.get('error')}")
    feat_master: List[str] = list(meta_pilot.get("feat_cols", []))
    last_ts_list: list[pd.Timestamp] = []
    if isinstance(meta_pilot.get("last_ts"), pd.Timestamp):
        last_ts_list.append(meta_pilot["last_ts"])
    peak_per_proc_gb = max(0.1, float(meta_pilot.get("peak_gb", 0.5)))  # fallback mínimo
    print(f"[dataflow][mp] pico estimado por proc={peak_per_proc_gb:.2f} GB | feats={len(feat_master)}", flush=True)

    # estado global/contadores para logs
    tot_pl = int(meta_pilot["counts"]["pl"]); tot_nl = int(meta_pilot["counts"]["nl"])
    tot_ps = int(meta_pilot["counts"]["ps"]); tot_ns = int(meta_pilot["counts"]["ns"])
    print((
        f"[dataflow] {pilot}: long pos={tot_pl:,} neg={tot_nl:,} | short pos={tot_ps:,} neg={tot_ns:,} | feats={len(feat_master)}"
    ).replace(',', '.'), flush=True)
    try:
        print(f"[dataflow] ram uso: {ram_str()}", flush=True)
    except Exception:
        pass

    # fila de pendentes (demais símbolos), exclui piloto
    pending = syms[1:]
    running: dict[int, tuple[mp.Process, mp.Queue, str]] = {}
    finished: list[dict] = [meta_pilot]

    def _allowed_concurrency() -> int:
        # baseado em memória livre e pico calibrado
        if not psutil:
            return MP_MAX_PROCS
        vm = psutil.virtual_memory()
        free_gb = vm.available / (1024**3)
        # número máximo possível adicional
        max_by_mem = int(max(0, (free_gb - float(MP_RESERVE_GB)) / (peak_per_proc_gb * float(MP_SAFETY_FACTOR))))
        return max(1, min(int(MP_MAX_PROCS), max_by_mem))

    def _spawn(sym: str) -> None:
        q: mp.Queue = mp.Queue(maxsize=1)
        args = (sym, sym_to_id[sym], int(total_days), int(remove_tail_days), use_flags, feat_master, str(parts_dir), ratio_neg_per_pos)
        p = mp.Process(target=_worker_wrapper, args=(q, args), daemon=True)
        p.start()
        running[p.pid] = (p, q, sym)

    # wrapper para retornar metadados via queue
    # loop principal
    while pending or running:
        # tenta spawnar dentro do limite atual
        allow = _allowed_concurrency()
        while pending and len(running) < allow:
            # checa margem mínima
            if psutil:
                vm = psutil.virtual_memory()
                free_gb = vm.available / (1024**3)
                if free_gb < float(MP_MIN_FREE_GB) + peak_per_proc_gb * float(MP_SAFETY_FACTOR):
                    break
            sym = pending.pop(0)
            print(f"[dataflow][mp] spawn {sym} | running={len(running)+1}/{allow}", flush=True)
            _spawn(sym)

        # coleta resultados de quem terminou
        time.sleep(0.1)
        done_pids: list[int] = []
        for pid, (proc, q, sym) in list(running.items()):
            if not proc.is_alive():
                try:
                    res = q.get(timeout=1.0)
                except Exception:
                    res = {"error": "no_result", "symbol": sym, "sym_id": sym_to_id[sym]}
                proc.join(timeout=1.0)
                done_pids.append(pid)
                finished.append(res)
                if not res.get("error"):
                    c = res.get("counts")
                    if not isinstance(c, dict):
                        print(f"[dataflow][mp] {sym}: resultado sem 'counts' — pulando símbolo.", flush=True)
                        continue
                    tot_pl += int(c["pl"]); tot_nl += int(c["nl"])
                    tot_ps += int(c["ps"]); tot_ns += int(c["ns"])
                    try:
                        lt = res.get("last_ts")
                        if isinstance(lt, pd.Timestamp):
                            last_ts_list.append(lt)
                        pk = float(res.get("peak_gb", peak_per_proc_gb))
                        # atualiza pico por processo (pode diminuir/aumentar) com média robusta
                        peak_per_proc_gb = max(0.1, 0.7 * peak_per_proc_gb + 0.3 * pk)
                    except Exception:
                        pass
                    print((
                        f"[dataflow] {sym}: long pos={c['pl']:,} neg={c['nl']:,} | short pos={c['ps']:,} neg={c['ns']:,} | feats={len(feat_master)}"
                    ).replace(',', '.'), flush=True)
                    print((
                        f"[dataflow] total acum.: long pos={tot_pl:,} neg={tot_nl:,} | short pos={tot_ps:,} neg={tot_ns:,}"
                    ).replace(',', '.'), flush=True)
                else:
                    print(f"[dataflow][mp] {sym}: erro={res.get('error')}", flush=True)
                try:
                    print(f"[dataflow] ram uso: {ram_str()}", flush=True)
                except Exception:
                    pass
        for pid in done_pids:
            running.pop(pid, None)

    T_anchor = (min(last_ts_list).isoformat() if last_ts_list else None)
    return {
        "feature_cols": feat_master,
        "long": {"X": np.empty((0, len(feat_master)), np.float32), "y": np.empty((0,), np.int32), "w": np.empty((0,), np.float32), "ts": np.empty((0,), dtype='datetime64[ns]'), "sym_id": np.empty((0,), np.uint16)},
        "short": {"X": np.empty((0, len(feat_master)), np.float32), "y": np.empty((0,), np.int32), "w": np.empty((0,), np.float32), "ts": np.empty((0,), dtype='datetime64[ns]'), "sym_id": np.empty((0,), np.uint16)},
        "anchor_end_utc": T_anchor,
        "sym_map": symbols,
        "parts_dir": str(parts_dir),
    }
