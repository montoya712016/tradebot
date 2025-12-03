# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import time, json
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise


# Cache global de boosters por (run_dir_resolvido, período_em_dias)
# Formato: (buy_model, short_model, ureg_model | None)
_GLOBAL_MODELS: dict[tuple[str, int], tuple[xgb.Booster, xgb.Booster, xgb.Booster | None]] = {}


def ensure_boosters_for_period(run_dir: Path, period_days: int) -> tuple[xgb.Booster, xgb.Booster, xgb.Booster | None] | None:
    """
    Carrega boosters buy/short/ureg para o período especificado, preferindo .ubj (binário).
    Se apenas .json existir, carrega e salva .ubj para acelerar execuções futuras.
    Trata arquivos .ubj vazios ou corrompidos com fallback automático para .json.
    Retorna (buy_model, short_model, ureg_model | None).
    """
    pd_dir = Path(run_dir) / f'period_{int(period_days)}d'
    buy_ubj = pd_dir / 'entry_models' / 'model_buy.ubj'
    sho_ubj = pd_dir / 'entry_models' / 'model_short.ubj'
    ureg_ubj = pd_dir / 'entry_models' / 'model_ureg.ubj'
    buy_json = pd_dir / 'entry_models' / 'model_buy.json'
    sho_json = pd_dir / 'entry_models' / 'model_short.json'
    ureg_json = pd_dir / 'entry_models' / 'model_ureg.json'
    
    b_ureg = None
    
    # Tenta carregar .ubj primeiro (mais rápido), mas verifica se não está vazio/corrompido
    if buy_ubj.exists() and sho_ubj.exists():
        # Verifica se os arquivos têm tamanho válido (> 0 bytes)
        try:
            if buy_ubj.stat().st_size > 0 and sho_ubj.stat().st_size > 0:
                try:
                    b_buy = xgb.Booster()
                    b_buy.load_model(str(buy_ubj))
                    b_sho = xgb.Booster()
                    b_sho.load_model(str(sho_ubj))
                    # Tenta carregar ureg se disponível
                    if ureg_ubj.exists() and ureg_ubj.stat().st_size > 0:
                        try:
                            b_ureg = xgb.Booster()
                            b_ureg.load_model(str(ureg_ubj))
                        except Exception:
                            b_ureg = None
                    elif ureg_json.exists():
                        try:
                            b_ureg = xgb.Booster()
                            b_ureg.load_model(str(ureg_json))
                            try:
                                b_ureg.save_model(str(ureg_ubj))
                            except Exception:
                                pass
                        except Exception:
                            b_ureg = None
                    return (b_buy, b_sho, b_ureg)
                except Exception as e:
                    # .ubj corrompido ou inválido, tenta deletar e usar .json
                    try:
                        buy_ubj.unlink(missing_ok=True)
                        sho_ubj.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            # Erro ao verificar tamanho, tenta carregar mesmo assim
            try:
                b_buy = xgb.Booster()
                b_buy.load_model(str(buy_ubj))
                b_sho = xgb.Booster()
                b_sho.load_model(str(sho_ubj))
                if ureg_ubj.exists():
                    try:
                        b_ureg = xgb.Booster()
                        b_ureg.load_model(str(ureg_ubj))
                    except Exception:
                        b_ureg = None
                return (b_buy, b_sho, b_ureg)
            except Exception:
                # Falhou, tenta deletar e usar .json
                try:
                    buy_ubj.unlink(missing_ok=True)
                    sho_ubj.unlink(missing_ok=True)
                except Exception:
                    pass
    
    # Fallback para .json
    if buy_json.exists() and sho_json.exists():
        try:
            b_buy = xgb.Booster()
            b_buy.load_model(str(buy_json))
            b_sho = xgb.Booster()
            b_sho.load_model(str(sho_json))
            # Tenta carregar ureg se disponível
            if ureg_json.exists():
                try:
                    b_ureg = xgb.Booster()
                    b_ureg.load_model(str(ureg_json))
                    try:
                        b_ureg.save_model(str(ureg_ubj))
                    except Exception:
                        pass
                except Exception:
                    b_ureg = None
            # Tenta salvar .ubj para acelerar próximas execuções
            try:
                b_buy.save_model(str(buy_ubj))
                b_sho.save_model(str(sho_ubj))
            except Exception:
                pass
            return (b_buy, b_sho, b_ureg)
        except Exception as e:
            # .json também está corrompido ou inválido
            return None
    
    return None


def _predict_with_best(bst: xgb.Booster, data_np: np.ndarray) -> np.ndarray:
    """
    Predição robusta:
    - Tenta inplace_predict (mais rápido, evita problemas de C-API)
    - Fallback para DMatrix + predict(validate_features=False)
    """
    try:
        return bst.inplace_predict(data_np).astype(np.float32)
    except Exception:
        try:
            dm = xgb.DMatrix(data_np)
            return bst.predict(dm, validate_features=False).astype(np.float32)
        except Exception:
            # último fallback: cria DMatrix explicitamente e chama predict simples
            dm = xgb.DMatrix(data_np)
            return bst.predict(dm).astype(np.float32)


def preload_boosters(run_dir: Path, periods: list[int], print_timing: bool = True) -> dict[int, bool]:
    """
    Pré-carrega boosters no cache global (_GLOBAL_MODELS) para os períodos desejados.
    Retorna dict {period: loaded_now(bool)} indicando se foi carregado nesta chamada.
    Continua mesmo se alguns períodos falharem (arquivos corrompidos/ausentes).
    """
    loaded_now: dict[int, bool] = {}
    run_key = str(Path(run_dir).resolve())
    for p in sorted(set(int(x) for x in periods)):
        key = (run_key, int(p))
        if key in _GLOBAL_MODELS:
            loaded_now[int(p)] = False
            continue
        t0 = time.time()
        try:
            triple = ensure_boosters_for_period(run_dir, int(p))
            if triple is None:
                if print_timing:
                    print(f"[timing] preload {int(p)}d: FALHOU (modelos não encontrados ou corrompidos)", flush=True)
                continue
            _GLOBAL_MODELS[key] = triple
            dt = time.time() - t0
            loaded_now[int(p)] = True
            if print_timing:
                print(f"[timing] preload {int(p)}d: {dt:.2f}s", flush=True)
        except Exception as e:
            # Erro ao carregar período, continua para os próximos
            if print_timing:
                print(f"[timing] preload {int(p)}d: ERRO - {type(e).__name__}: {e}", flush=True)
            continue
    return loaded_now


def predict_buy_short_proba_for_segments(
    Xdf: pd.DataFrame,
    run_dir: Path,
    periods_avail: list[int],
    print_timing: bool = True,
    symbol: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[pd.Timestamp, pd.Timestamp, int]], list[tuple[int, float]]]:
    """
    Predição baseada nos cutoffs de treinamento por período/símbolo.
    
    Para cada timestamp t nos dados:
    - Verifica cada período (do menor ao maior)
    - Usa o MENOR período cujo cutoff de treino está ANTES de t
    - Isso garante que nunca usamos um modelo em dados que ele viu no treinamento
    
    Exemplo:
    - cutoff_90d = 2025-08-08, cutoff_180d = 2025-05-10, cutoff_270d = 2025-02-09
    - Para t = 2025-09-01: usa 90d (t > cutoff_90d)
    - Para t = 2025-06-01: usa 180d (t > cutoff_180d, mas t < cutoff_90d)
    - Para t = 2025-03-01: usa 270d (t > cutoff_270d, mas t < cutoff_180d)
    
    Retorna p_buy/p_sho/u_pred, spans agregados por período e tempos por período.
    """
    idx = Xdf.index
    if len(idx) == 0:
        return (np.full(0, np.nan, np.float32),
                np.full(0, np.nan, np.float32),
                np.full(0, np.nan, np.float32),
                [], [])
    
    # Carrega cutoffs de treinamento por período
    per_cutoffs: dict[int, pd.Timestamp] = {}
    periods_sorted = sorted(int(p) for p in periods_avail)
    
    # Tenta carregar cutoffs específicos por símbolo
    if symbol is not None:
        for p in periods_sorted:
            jpath = Path(run_dir) / f'period_{int(p)}d' / 'dataset_cutoffs_long.json'
            if jpath.exists():
                try:
                    j = json.loads(jpath.read_text(encoding='utf-8'))
                    cut_s = j.get(symbol)
                    if cut_s:
                        per_cutoffs[int(p)] = pd.to_datetime(cut_s)
                except Exception:
                    pass
    
    # Fallback: lê anchor_end e calcula cutoffs aproximados
    if not per_cutoffs:
        try:
            from my_project.test.load_models import read_anchor_end
        except Exception:
            try:
                from ..test.load_models import read_anchor_end
            except Exception:
                def read_anchor_end(rd: Path) -> pd.Timestamp | None:
                    try:
                        ta = Path(rd) / "time_anchors.json"
                        if not ta.exists():
                            return None
                        j = json.loads(ta.read_text(encoding="utf-8"))
                        if isinstance(j, dict) and j:
                            k = next(iter(j.keys()))
                            v = j.get(k) or {}
                            s = v.get("anchor_end_utc")
                            if s:
                                return pd.to_datetime(s)
                    except Exception:
                        return None
                    return None
        
        anchor_end = read_anchor_end(run_dir)
        if anchor_end is not None:
            # Cutoff aproximado: anchor_end - período
            for p in periods_sorted:
                per_cutoffs[int(p)] = anchor_end - pd.Timedelta(days=int(p))
    
    # Se ainda não temos cutoffs, usa lógica antiga como fallback
    if not per_cutoffs:
        print("[predict] AVISO: sem cutoffs disponíveis, usando fallback por anchor_end", flush=True)
        # Fallback para lógica antiga baseada em anchor_end
        try:
            from my_project.test.load_models import read_anchor_end
        except Exception:
            try:
                from ..test.load_models import read_anchor_end
            except Exception:
                def read_anchor_end(rd: Path) -> pd.Timestamp | None:
                    return None
        anchor_end = read_anchor_end(run_dir)
        data_end = pd.to_datetime(idx[-1])
        reference_end = anchor_end if anchor_end is not None else data_end
        days_to_ref = (reference_end - pd.to_datetime(idx)).days
        p_needed = ((np.maximum(0, days_to_ref) + 89) // 90) * 90
        p_needed = np.clip(p_needed, 90, 720).astype(int)
        periods_set = set(int(p) for p in periods_avail)
        p_needed = np.array([int(p) if int(p) in periods_set else -1 for p in p_needed], dtype=int)
    else:
        # NOVA LÓGICA: mapeia cada timestamp para o menor período cujo cutoff está ANTES do timestamp
        idx_ts = pd.to_datetime(idx)
        p_needed = np.full(len(idx), -1, dtype=int)
        
        for i, t in enumerate(idx_ts):
            # Percorre períodos do menor ao maior
            for p in periods_sorted:
                cutoff = per_cutoffs.get(int(p))
                if cutoff is None:
                    continue
                # Se o timestamp está APÓS o cutoff deste período, pode usar este modelo
                if t > cutoff:
                    p_needed[i] = int(p)
                    break  # Usa o menor período válido
        
        # Log de diagnóstico
        valid_count = np.sum(p_needed > 0)
        total_count = len(p_needed)
        if print_timing:
            print(f"[predict] cutoffs carregados para {len(per_cutoffs)} períodos | {valid_count}/{total_count} timestamps com modelo válido", flush=True)
            for p in periods_sorted:
                c = per_cutoffs.get(int(p))
                mask_p = (p_needed == int(p))
                n_p = int(np.sum(mask_p))
                if c is not None:
                    print(f"[predict]   {int(p):3d}d: cutoff={c.strftime('%Y-%m-%d %H:%M')} | {n_p} velas", flush=True)

    p_buy = np.full(len(Xdf), np.nan, np.float32)
    p_sho = np.full(len(Xdf), np.nan, np.float32)
    u_pred = np.full(len(Xdf), np.nan, np.float32)
    used_spans: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    per_times: list[tuple[int, float]] = []
    run_key = str(Path(run_dir).resolve())
    
    # Para cada período presente no p_needed, prever sobre as velas correspondentes
    for p_sel in sorted(set(int(x) for x in p_needed if int(x) > 0)):
        mask = (p_needed == int(p_sel))
        if not np.any(mask):
            continue
        key = (run_key, int(p_sel))
        if key not in _GLOBAL_MODELS:
            t0 = time.time()
            loaded = ensure_boosters_for_period(run_dir, int(p_sel))
            if loaded is None:
                continue
            _GLOBAL_MODELS[key] = loaded
            dt = time.time() - t0
            per_times.append((int(p_sel), dt))
            if print_timing:
                print(f"[timing] model {int(p_sel)}d loaded: {dt:.2f}s", flush=True)
        buy_model, sho_model, ureg_model = _GLOBAL_MODELS[key]
        data_np = Xdf.loc[mask].to_numpy(np.float32)
        p_buy[mask] = _predict_with_best(buy_model, data_np)
        p_sho[mask] = _predict_with_best(sho_model, data_np)
        # Regressor de U (opcional)
        if ureg_model is not None:
            try:
                u_pred[mask] = _predict_with_best(ureg_model, data_np)
            except Exception:
                pass
        # span agregada do sub-índice (primeiro ao último)
        sub_idx = idx[mask]
        if len(sub_idx) > 0:
            used_spans.append((pd.to_datetime(sub_idx[0]), pd.to_datetime(sub_idx[-1]), int(p_sel)))
    return p_buy, p_sho, u_pred, used_spans, per_times




