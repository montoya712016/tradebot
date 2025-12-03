# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np

# Fallback de imports (execução direta)
try:
    from ..config import SAVE_DIR
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _PKG_ROOT = _Path(__file__).resolve().parents[1]  # .../my_project
    _WORKSPACE = _PKG_ROOT.parent
    if str(_WORKSPACE) not in _sys.path:
        _sys.path.insert(0, str(_WORKSPACE))
    from my_project.config import SAVE_DIR


# Imports centralizados
try:
    from .load_models import find_latest_run, find_latest_run_any, build_X_from_features
except Exception:
    from my_project.test.load_models import find_latest_run, find_latest_run_any, build_X_from_features


def _map_importances_to_names(imp_map: dict[str, float], feat_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in imp_map.items():
        try:
            if k.startswith('f'):
                idx = int(k[1:])
            else:
                idx = int(k)
            if 0 <= idx < len(feat_cols):
                out[feat_cols[idx]] = out.get(feat_cols[idx], 0.0) + float(v)
        except Exception:
            # ignora chaves não mapeáveis
            continue
    return out


def _aggregate_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    agg: dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    return agg


def main():
    # Parâmetros (edite aqui)
    RUN_HINT: str | None = None           # 'market_cap_150B_100M' ou caminho wf_XXX ou None
    SIDES = ['long', 'short']             # opções: 'long', 'short'
    PERIODS_OVERRIDE: list[int] | None = [90]  # usa apenas 90d por padrão
    TOPK = 30
    IMPORTANCE_TYPES = ['gain', 'weight', 'cover', 'total_gain', 'total_cover']  # tenta todos; nem todos disponíveis em toda versão
    # Permutation sensitivity opcional (sem y): mede |Δp| médio ao permutar colunas, para UM modelo/período/lado
    PERM_SAMPLE = False
    PERM_SYMBOL = 'ADAUSDT'
    PERM_DAYS = 90
    PERM_SIDE = 'long'  # 'long' ou 'short'
    PERM_TOP_FROM = 'gain'  # de qual importância pegar o top para testar permuta
    PERM_TOPK = 25
    PERM_SAMPLE_MAX_ROWS = 20000

    # Localiza run
    if RUN_HINT:
        p = Path(RUN_HINT)
        if p.exists() and p.is_dir() and p.name.startswith('wf_'):
            run_dir, all_periods = p, find_latest_run(p.parent)[1]
        else:
            run_dir, all_periods = find_latest_run(SAVE_DIR / RUN_HINT)
    else:
        try:
            run_dir, all_periods = find_latest_run(SAVE_DIR / 'market_cap_150B_100M')
        except Exception:
            run_dir, all_periods = find_latest_run_any(SAVE_DIR)
    periods = (PERIODS_OVERRIDE or all_periods)
    print(f"[fi] run_dir={run_dir}")
    print(f"[fi] periods={periods}")

    # Carrega cols
    feat_cols_path = run_dir / 'dataset' / 'feature_columns.json'
    if not feat_cols_path.exists():
        raise RuntimeError("feature_columns.json não encontrado")
    feat_cols: list[str] = json.loads(feat_cols_path.read_text(encoding='utf-8'))

    # Coleta importâncias de modelos por período/lado
    import xgboost as xgb
    side_to_models = {
        'long': 'model_buy.json',
        'short': 'model_short.json',
    }
    results: dict[str, dict[str, dict[str, float]]] = {}  # side -> imp_type -> name->val (agregado)
    for side in SIDES:
        side_res: dict[str, list[dict[str, float]]] = {t: [] for t in IMPORTANCE_TYPES}
        for per in periods:
            pd_dir = run_dir / f'period_{int(per)}d' / 'entry_models'
            model_path = pd_dir / side_to_models.get(side, '')
            if not model_path.exists():
                continue
            t0 = time.time()
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            dt = time.time() - t0
            print(f"[fi] loaded {side} {per}d in {dt:.2f}s")
            # cada tipo
            for imp_type in IMPORTANCE_TYPES:
                try:
                    raw = bst.get_score(importance_type=imp_type)
                    mapped = _map_importances_to_names(raw, feat_cols)
                    if mapped:
                        side_res[imp_type].append(mapped)
                except Exception:
                    # ignora tipos não suportados
                    continue
        # agrega por tipo
        results[side] = {}
        for imp_type, lst in side_res.items():
            agg = _aggregate_dicts(lst) if lst else {}
            results[side][imp_type] = agg

    # Exibe top-K por lado/tipo
    def _print_top(side: str, imp_type: str, top: int):
        m = results.get(side, {}).get(imp_type, {})
        if not m:
            print(f"[fi] {side} {imp_type}: (vazio)")
            return
        tot = float(sum(m.values())) or 1.0
        items = sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:int(top)]
        print(f"\n[fi] TOP {int(top)} — side={side} type={imp_type}")
        for name, val in items:
            share = 100.0 * float(val) / tot
            print(f"   {name:<36s}  {val:12.6f}  ({share:6.2f}%)")

    for side in SIDES:
        for imp_type in IMPORTANCE_TYPES:
            _print_top(side, imp_type, TOPK)

    # ===== SUGESTÃO DE ARRAYS PARA pf_config.py (baseada no 'gain' agregado long+short) =====
    def _suggest_pf_arrays():
        import re
        combined = {}
        for side in results.keys():
            gain_map = results.get(side, {}).get('gain', {})
            for k, v in gain_map.items():
                combined[k] = combined.get(k, 0.0) + float(v)
        items = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)

        # Pruning heurístico para reduzir o conjunto mantendo a maior parte do ganho
        MAX_GLOBAL = 60           # limite global de features
        MIN_GAIN_SHARE = 0.90     # cobrir ao menos 90% do ganho agregado
        DEFAULT_FAM_LIMIT = 3
        FAMILY_LIMITS = {
            'minmax': 6, 'keltner_center': 3, 'keltner_width': 2, 'keltner_pos': 2,
            'vol': 4, 'atr': 3, 'cci': 2, 'adx': 3, 'zlog': 1,
            'slope': 2, 'slope_reserr': 3, 'vol_ratio': 1, 'ema_pairs': 2, 'rsi_ema': 1, 'logret': 1,
            'runs': 4, 'hh_hl': 2, 'ema_cross': 2, 'breakout': 3, 'slope_diff': 2, 'wick_stats': 2,
        }
        def _family_of(name: str) -> str:
            if re.match(r'^(pct_from_(min|max)|time_since_(min|max))_\d+$', name): return 'minmax'
            if re.match(r'^keltner_center_pct_\d+$', name): return 'keltner_center'
            if re.match(r'^keltner_halfwidth_pct_\d+$', name): return 'keltner_width'
            if re.match(r'^keltner_pos_\d+$', name): return 'keltner_pos'
            if re.match(r'^vol_pct_\d+$', name): return 'vol'
            if re.match(r'^atr_pct_\d+$', name): return 'atr'
            if re.match(r'^cci_\d+$', name): return 'cci'
            if re.match(r'^adx_\d+$', name): return 'adx'
            if re.match(r'^zlog_\d+m$', name): return 'zlog'
            if re.match(r'^slope_pct_\d+$', name): return 'slope'
            if re.match(r'^slope_reserr_pct_\d+$', name): return 'slope_reserr'
            if re.match(r'^vol_ratio_pct_\d+_\d+$', name): return 'vol_ratio'
            if re.match(r'^shitidx_pct_\d+_\d+$', name): return 'ema_pairs'
            if re.match(r'^rsi_ema\d+_\d+$', name): return 'rsi_ema'
            if re.match(r'^cum_ret_pct_\d+$', name): return 'logret'
            if re.match(r'^(run_up_cnt|run_dn_cnt)_\d+$', name): return 'runs'
            if re.match(r'^(hh_cnt|hl_cnt)_\d+$', name): return 'hh_hl'
            if re.match(r'^(bars_above_ema|bars_below_ema|bars_since_cross)_\d+$', name): return 'ema_cross'
            if re.match(r'^(break_high|break_low|bars_since_bhigh|bars_since_blow)_\d+$', name): return 'breakout'
            if re.match(r'^slope_diff_\d+_\d+$', name): return 'slope_diff'
            if re.match(r'^wick_lower_mean_\d+$', name): return 'wick_stats'
            return 'misc'
        total_gain = float(sum(v for _, v in items)) or 1.0
        selected: list[tuple[str, float]] = []
        fam_counts: dict[str, int] = {}
        gain_acc = 0.0
        for name, val in items:
            fam = _family_of(name)
            lim = FAMILY_LIMITS.get(fam, DEFAULT_FAM_LIMIT)
            if fam_counts.get(fam, 0) >= lim:
                continue
            selected.append((name, float(val)))
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            gain_acc += float(val)
            if len(selected) >= MAX_GLOBAL and (gain_acc / total_gain) >= MIN_GAIN_SHARE:
                break
        selected_set = {n for n, _ in selected}
        # coletores
        minmax: dict[int, float] = {}
        k_center: dict[int, float] = {}
        k_width: dict[int, float] = {}
        k_pos: dict[int, float] = {}
        vol_min: dict[int, float] = {}
        atr_min: dict[int, float] = {}
        cci_min: dict[int, float] = {}
        adx_min: dict[int, float] = {}
        zlog_min: dict[int, float] = {}
        slope_min: dict[int, float] = {}
        slope_res: dict[int, float] = {}
        vol_ratio_pairs: dict[tuple[int, int], float] = {}
        ema_pairs: dict[tuple[int, int], float] = {}
        rsi_ema_pairs: dict[tuple[int, int], float] = {}
        logret_min: dict[int, float] = {}
        # NOVOS blocos (alinhados com prepare_features/pf_config.py)
        run_windows: dict[int, float] = {}
        hhhl_windows: dict[int, float] = {}
        ema_confirm_spans: dict[int, float] = {}
        break_lookback: dict[int, float] = {}
        slope_diff_pairs: dict[tuple[int, int], float] = {}
        wick_mean_windows: dict[int, float] = {}
        # parse
        for name, val in items:
            if name not in selected_set:
                continue
            m = re.match(r'^pct_from_(min|max)_(\d+)$', name)
            if m:
                w = int(m.group(2)); minmax[w] = minmax.get(w, 0.0) + val; continue
            m = re.match(r'^time_since_(min|max)_(\d+)$', name)
            if m:
                w = int(m.group(2)); minmax[w] = minmax.get(w, 0.0) + val; continue
            m = re.match(r'^keltner_center_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); k_center[w] = k_center.get(w, 0.0) + val; continue
            m = re.match(r'^keltner_halfwidth_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); k_width[w] = k_width.get(w, 0.0) + val; continue
            m = re.match(r'^keltner_pos_(\d+)$', name)
            if m:
                w = int(m.group(1)); k_pos[w] = k_pos.get(w, 0.0) + val; continue
            m = re.match(r'^vol_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); vol_min[w] = vol_min.get(w, 0.0) + val; continue
            m = re.match(r'^atr_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); atr_min[w] = atr_min.get(w, 0.0) + val; continue
            m = re.match(r'^cci_(\d+)$', name)
            if m:
                w = int(m.group(1)); cci_min[w] = cci_min.get(w, 0.0) + val; continue
            m = re.match(r'^adx_(\d+)$', name)
            if m:
                w = int(m.group(1)); adx_min[w] = adx_min.get(w, 0.0) + val; continue
            m = re.match(r'^zlog_(\d+)m$', name)
            if m:
                w = int(m.group(1)); zlog_min[w] = zlog_min.get(w, 0.0) + val; continue
            m = re.match(r'^slope_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); slope_min[w] = slope_min.get(w, 0.0) + val; continue
            m = re.match(r'^slope_reserr_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); slope_res[w] = slope_res.get(w, 0.0) + val; continue
            m = re.match(r'^vol_ratio_pct_(\d+)_(\d+)$', name)
            if m:
                a, b = int(m.group(1)), int(m.group(2)); vol_ratio_pairs[(a, b)] = vol_ratio_pairs.get((a, b), 0.0) + val; continue
            m = re.match(r'^shitidx_pct_(\d+)_(\d+)$', name)
            if m:
                a, b = int(m.group(1)), int(m.group(2)); ema_pairs[(a, b)] = ema_pairs.get((a, b), 0.0) + val; continue
            m = re.match(r'^rsi_ema(\d+)_(\d+)$', name)
            if m:
                a, b = int(m.group(1)), int(m.group(2)); rsi_ema_pairs[(a, b)] = rsi_ema_pairs.get((a, b), 0.0) + val; continue
            m = re.match(r'^cum_ret_pct_(\d+)$', name)
            if m:
                w = int(m.group(1)); logret_min[w] = logret_min.get(w, 0.0) + val; continue
            # ===== Novos blocos adicionados =====
            m = re.match(r'^run_up_cnt_(\d+)$', name) or re.match(r'^run_dn_cnt_(\d+)$', name)
            if m:
                w = int(m.group(1)); run_windows[w] = run_windows.get(w, 0.0) + val; continue
            m = re.match(r'^hh_cnt_(\d+)$', name) or re.match(r'^hl_cnt_(\d+)$', name)
            if m:
                w = int(m.group(1)); hhhl_windows[w] = hhhl_windows.get(w, 0.0) + val; continue
            m = re.match(r'^bars_above_ema_(\d+)$', name) or re.match(r'^bars_below_ema_(\d+)$', name) or re.match(r'^bars_since_cross_(\d+)$', name)
            if m:
                w = int(m.group(1)); ema_confirm_spans[w] = ema_confirm_spans.get(w, 0.0) + val; continue
            m = re.match(r'^break_high_(\d+)$', name) or re.match(r'^break_low_(\d+)$', name) or re.match(r'^bars_since_bhigh_(\d+)$', name) or re.match(r'^bars_since_blow_(\d+)$', name)
            if m:
                w = int(m.group(1)); break_lookback[w] = break_lookback.get(w, 0.0) + val; continue
            m = re.match(r'^slope_diff_(\d+)_(\d+)$', name)
            if m:
                a, b = int(m.group(1)), int(m.group(2)); slope_diff_pairs[(a, b)] = slope_diff_pairs.get((a, b), 0.0) + val; continue
            m = re.match(r'^wick_lower_mean_(\d+)$', name)
            if m:
                w = int(m.group(1)); wick_mean_windows[w] = wick_mean_windows.get(w, 0.0) + val; continue
        # escolhe top por família com limites
        def top_keys(d: dict[int, float], kmax: int): 
            return sorted(d.keys(), key=lambda x: d[x], reverse=True)[:int(kmax)]
        def top_pair_keys(d: dict[tuple[int,int], float], kmax: int): 
            return sorted(d.keys(), key=lambda x: d[x], reverse=True)[:int(kmax)]
        MINMAX_MIN = sorted(top_keys(minmax, 6))
        KELTNER_CENTER_MIN = sorted(top_keys(k_center, 4))
        KELTNER_WIDTH_MIN = sorted(top_keys(k_width, 3))
        KELTNER_POS_MIN = sorted(top_keys(k_pos, 2))
        VOL_MIN = sorted(top_keys(vol_min, 5))
        ATR_MIN = sorted(top_keys(atr_min, 3))
        CCI_MIN = sorted(top_keys(cci_min, 2))
        ADX_MIN = sorted(top_keys(adx_min, 3))
        ZLOG_MIN = sorted(top_keys(zlog_min, 1))
        SLOPE_MIN = sorted(top_keys(slope_min, 2))
        SLOPE_RESERR_MIN = sorted(top_keys(slope_res, 4))
        VOL_RATIO_PAIRS = sorted(top_pair_keys(vol_ratio_pairs, 1))
        EMA_PAIRS = sorted(top_pair_keys(ema_pairs, 2))
        EMA_WINDOWS = sorted({w for pair in EMA_PAIRS for w in pair})
        RSI_EMA_PAIRS = sorted(top_pair_keys(rsi_ema_pairs, 1))
        LOGRET_MIN = sorted(top_keys(logret_min, 1))
        RUN_WINDOWS_MIN = sorted(top_keys(run_windows, 3))
        HHHL_WINDOWS_MIN = sorted(top_keys(hhhl_windows, 2))
        EMA_CONFIRM_SPANS_MIN = sorted(top_keys(ema_confirm_spans, 2))
        BREAK_LOOKBACK_MIN = sorted(top_keys(break_lookback, 3))
        SLOPE_DIFF_PAIRS_MIN = sorted(top_pair_keys(slope_diff_pairs, 2))
        WICK_MEAN_WINDOWS_MIN = sorted(top_keys(wick_mean_windows, 2))
        # imprime bloco pronto para colar
        def _fmt(lst): 
            return "[" + ", ".join(str(int(x)) for x in lst) + "]"
        def _fmt_pairs(pairs): 
            return "[" + ", ".join(f"({int(a)}, {int(b)})" for a,b in pairs) + "]"
        print("\n\n# ===== SUGESTÃO pf_config.py (pruning aplicado) =====")
        print(f"EMA_WINDOWS = {_fmt(EMA_WINDOWS)}")
        print(f"EMA_PAIRS = {_fmt_pairs(EMA_PAIRS)}")
        print(f"ATR_MIN = {_fmt(ATR_MIN)}")
        print(f"VOL_MIN = {_fmt(VOL_MIN)}")
        print(f"CI_MIN = []")
        print(f"LOGRET_MIN = {_fmt(LOGRET_MIN)}")
        print(f"KELTNER_WIDTH_MIN  = {_fmt(KELTNER_WIDTH_MIN)}")
        print(f"KELTNER_CENTER_MIN = {_fmt(KELTNER_CENTER_MIN)}")
        print(f"KELTNER_POS_MIN    = {_fmt(KELTNER_POS_MIN)}")
        print(f"KELTNER_Z_MIN      = []")
        print(f"RSI_PRICE_MIN = []")
        print(f"RSI_EMA_PAIRS = {_fmt_pairs(RSI_EMA_PAIRS)}")
        print(f"SLOPE_MIN   = {_fmt(SLOPE_MIN)}")
        print(f"CCI_MIN     = {_fmt(CCI_MIN)}")
        print(f"ADX_MIN     = {_fmt(ADX_MIN)}")
        print(f"ZLOG_MIN    = {_fmt(ZLOG_MIN)}")
        print(f"MINMAX_MIN  = {_fmt(MINMAX_MIN)}")
        print(f"SLOPE_RESERR_MIN = {_fmt(SLOPE_RESERR_MIN)}")
        print(f"REV_WINDOWS = [30, 60, 120]")
        print(f"VOL_RATIO_PAIRS  = {_fmt_pairs(VOL_RATIO_PAIRS)}")
        # novos blocos sugeridos
        print(f"RUN_WINDOWS_MIN = {_fmt(RUN_WINDOWS_MIN)}")
        print(f"HHHL_WINDOWS_MIN = {_fmt(HHHL_WINDOWS_MIN)}")
        print(f"EMA_CONFIRM_SPANS_MIN = {_fmt(EMA_CONFIRM_SPANS_MIN)}")
        print(f"BREAK_LOOKBACK_MIN = {_fmt(BREAK_LOOKBACK_MIN)}")
        print(f"SLOPE_DIFF_PAIRS_MIN = {_fmt_pairs(SLOPE_DIFF_PAIRS_MIN)}")
        print(f"WICK_MEAN_WINDOWS_MIN = {_fmt(WICK_MEAN_WINDOWS_MIN)}")
        print("# ===== FIM SUGESTÃO =====\n")

    _suggest_pf_arrays()

    # Permutation sensitivity para um único modelo/período/lado (sem rótulos)
    if PERM_SAMPLE:
        try:
            import xgboost as xgb
            # escolhe período alvo
            per_target = (PERIODS_OVERRIDE[0] if PERIODS_OVERRIDE else (periods[0] if periods else 90))
            pd_dir = run_dir / f'period_{int(per_target)}d' / 'entry_models'
            model_file = ('model_buy.json' if PERM_SIDE == 'long' else 'model_short.json')
            model_path = pd_dir / model_file
            if not model_path.exists():
                print(f"[perm] modelo não encontrado: {model_path}")
                return
            bst = xgb.Booster(); bst.load_model(str(model_path))
            # features
            from my_project.prepare_features.prepare_features import run as pf_run
            from my_project.prepare_features.prepare_features import DEFAULT_CANDLE_SEC
            from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
            raw = load_ohlc_1m_series(PERM_SYMBOL, int(PERM_DAYS), remove_tail_days=0)
            ohlc = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
            df = pf_run(ohlc, flags={'label': False, 'pivots': False}, plot=False)
            import pandas as _pd, numpy as _np
            feat_cols_path = run_dir / 'dataset' / 'feature_columns.json'
            feat_cols = json.loads(feat_cols_path.read_text(encoding='utf-8'))
            Xdf = build_X_from_features(df, feat_cols)
            if Xdf.empty:
                print("[perm] X vazio após build_X.")
                return
            if len(Xdf) > int(PERM_SAMPLE_MAX_ROWS):
                Xdf = Xdf.iloc[-int(PERM_SAMPLE_MAX_ROWS):]
            D = xgb.DMatrix(Xdf.to_numpy(_np.float32))
            p0 = bst.predict(D)
            # pega top features pela importância escolhida
            top_dict = results.get(PERM_SIDE, {}).get(PERM_TOP_FROM, {})
            if not top_dict:
                print(f"[perm] sem importâncias para {PERM_SIDE}/{PERM_TOP_FROM}.")
                return
            top_list = sorted(top_dict.items(), key=lambda kv: kv[1], reverse=True)[:int(PERM_TOPK)]
            name_to_idx = {name: i for i, name in enumerate(feat_cols)}
            # mede sensibilidade média |Δp|
            print(f"\n[perm] Sensibilidade média |Δp| por permutação ({PERM_SIDE} {int(per_target)}d, top {int(PERM_TOPK)} de '{PERM_TOP_FROM}'):")
            for fname, _ in top_list:
                j = name_to_idx.get(fname, None)
                if j is None:
                    continue
                Xp = Xdf.to_numpy(_np.float32).copy()
                # permuta a coluna j
                rnd = _np.random.default_rng(123)
                rnd.shuffle(Xp[:, j])
                Dp = xgb.DMatrix(Xp)
                pp = bst.predict(Dp)
                delta = float(_np.mean(_np.abs(pp - p0)))
                print(f"   {fname:<36s}  |Δp|={delta:.6f}")
        except Exception as e:
            print(f"[perm] falhou: {e}")


if __name__ == '__main__':
    main()


