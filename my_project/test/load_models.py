# -*- coding: utf-8 -*-
"""
Funções centralizadas para carregamento de modelos XGBoost e utilitários relacionados.
"""
from __future__ import annotations
from pathlib import Path
import json

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None


def load_buy_short_models(run_dir: Path, period_days: int) -> tuple[xgb.Booster | None, xgb.Booster | None]:
    """
    Carrega modelos binários (buy/short) para o período especificado.
    Prefere .ubj (binário), fallback para .json.
    Retorna (buy_model, short_model) ou (None, None) se não encontrado.
    """
    if xgb is None:
        return None, None
    
    pd_dir = Path(run_dir) / f'period_{int(period_days)}d' / 'entry_models'
    buy_ubj = pd_dir / 'model_buy.ubj'
    sho_ubj = pd_dir / 'model_short.ubj'
    buy_json = pd_dir / 'model_buy.json'
    sho_json = pd_dir / 'model_short.json'
    
    # Tenta .ubj primeiro
    if buy_ubj.exists() and sho_ubj.exists():
        try:
            if buy_ubj.stat().st_size > 0 and sho_ubj.stat().st_size > 0:
                try:
                    b_buy = xgb.Booster()
                    b_buy.load_model(str(buy_ubj))
                    b_sho = xgb.Booster()
                    b_sho.load_model(str(sho_ubj))
                    return b_buy, b_sho
                except Exception:
                    # Corrompido, tenta deletar e usar .json
                    try:
                        buy_ubj.unlink(missing_ok=True)
                        sho_ubj.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Fallback para .json
    if buy_json.exists() and sho_json.exists():
        try:
            b_buy = xgb.Booster()
            b_buy.load_model(str(buy_json))
            b_sho = xgb.Booster()
            b_sho.load_model(str(sho_json))
            # Tenta salvar .ubj para próxima vez
            try:
                b_buy.save_model(str(buy_ubj))
                b_sho.save_model(str(sho_ubj))
            except Exception:
                pass
            return b_buy, b_sho
        except Exception:
            pass
    
    return None, None


def load_ureg_model(run_dir: Path, period_days: int) -> xgb.Booster | None:
    """
    Carrega modelo regressor de U para o período especificado.
    Prefere .ubj (binário), fallback para .json.
    Retorna o modelo ou None se não encontrado.
    """
    if xgb is None:
        return None
    
    pd_dir = Path(run_dir) / f'period_{int(period_days)}d' / 'entry_models'
    ureg_ubj = pd_dir / 'model_ureg.ubj'
    ureg_json = pd_dir / 'model_ureg.json'
    
    # Tenta .ubj primeiro
    if ureg_ubj.exists():
        try:
            if ureg_ubj.stat().st_size > 0:
                try:
                    b_ureg = xgb.Booster()
                    b_ureg.load_model(str(ureg_ubj))
                    return b_ureg
                except Exception:
                    # Corrompido, tenta deletar e usar .json
                    try:
                        ureg_ubj.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Fallback para .json
    if ureg_json.exists():
        try:
            b_ureg = xgb.Booster()
            b_ureg.load_model(str(ureg_json))
            # Tenta salvar .ubj para próxima vez
            try:
                b_ureg.save_model(str(ureg_ubj))
            except Exception:
                pass
            return b_ureg
        except Exception:
            pass
    
    return None


def load_all_models(run_dir: Path, period_days: int) -> tuple[xgb.Booster | None, xgb.Booster | None, xgb.Booster | None]:
    """
    Carrega todos os modelos (buy, short, ureg) para o período especificado.
    Retorna (buy_model, short_model, ureg_model | None).
    """
    b_buy, b_sho = load_buy_short_models(run_dir, period_days)
    b_ureg = load_ureg_model(run_dir, period_days)
    return b_buy, b_sho, b_ureg


def get_valid_periods(
    run_dir: Path,
    symbol: str,
    data_start: pd.Timestamp | None,
    data_end: pd.Timestamp | None,
    periods_available: list[int],
) -> list[int]:
    """
    Determina quais períodos são válidos baseado nos cutoffs de treinamento e nos timestamps dos dados.
    
    Args:
        run_dir: Diretório do run de treinamento
        symbol: Símbolo sendo avaliado
        data_start: Timestamp inicial dos dados
        data_end: Timestamp final dos dados
        periods_available: Lista de períodos disponíveis (ex: [90, 180, 270, ...])
    
    Returns:
        Lista de períodos válidos que podem ser usados (têm dados após o cutoff)
    """
    if pd is None:
        return periods_available
    
    # Carrega cutoffs por período
    per_cut: dict[int, pd.Timestamp] = {}
    last_train_ts0 = None
    
    try:
        dmeta_path = run_dir / 'dataset' / 'meta_long.parquet'
        smap_path = run_dir / 'dataset' / 'sym_map.json'
        
        if dmeta_path.exists() and smap_path.exists():
            dmeta = pd.read_parquet(dmeta_path)
            sym_map = json.loads(Path(smap_path).read_text(encoding='utf-8'))
            try:
                sym_id = int(sym_map.index(symbol))
            except Exception:
                sym_id = None
            
            if sym_id is not None:
                last_train_ts0 = pd.to_datetime(dmeta.loc[dmeta['sym_id'] == sym_id, 'ts']).max()
        
        # Carrega cutoffs de cada período
        for p in periods_available:
            jpath = run_dir / f'period_{int(p)}d' / 'dataset_cutoffs_long.json'
            cutoff = None
            
            if jpath.exists():
                try:
                    j = json.loads(jpath.read_text(encoding='utf-8'))
                    cut_s = j.get(symbol)
                    if cut_s:
                        cutoff = pd.to_datetime(cut_s)
                except Exception:
                    cutoff = None
            
            # Fallback: usa last_train_ts0 - p se não encontrou cutoff específico
            if cutoff is None and last_train_ts0 is not None:
                cutoff = last_train_ts0 - pd.Timedelta(days=int(p))
            
            if cutoff is not None:
                per_cut[int(p)] = cutoff
    
    except Exception:
        pass
    
    # Corrige cutoffs corrompidos no futuro usando fallback
    if data_end is not None and last_train_ts0 is not None:
        for p in list(per_cut.keys()):
            c = pd.to_datetime(per_cut[p])
            if pd.isna(c) or (c >= data_end):
                per_cut[p] = last_train_ts0 - pd.Timedelta(days=int(p))
    
    # Filtra períodos válidos
    periods_valid: list[int] = []
    
    for p in periods_available:
        cutoff = per_cut.get(int(p))
        
        if cutoff is None:
            # Sem cutoff conhecido, assume que pode usar
            periods_valid.append(int(p))
            continue
        
        cutoff = pd.to_datetime(cutoff)
        
        # Se o cutoff está no futuro em relação aos dados, NÃO há dados após o cutoff — não usar
        if data_end is not None and cutoff >= data_end:
            continue
        
        # Se o cutoff está antes do início dos dados, todos os dados são válidos — OK usar
        if data_start is not None and cutoff < data_start:
            periods_valid.append(int(p))
            continue
        
        # Cutoff está dentro do range dos dados — verifica se há dados após o cutoff
        if data_end is not None and cutoff < data_end:
            # Há dados após o cutoff — OK usar (mas será aplicado corte depois)
            periods_valid.append(int(p))
            continue
    
    return sorted(periods_valid)


def find_latest_run(base_dir: Path) -> tuple[Path, list[int]]:
    """
    Encontra o último run (wf_*) em um diretório base de market_cap.
    
    Args:
        base_dir: Diretório base (ex: models_classifier/market_cap_150B_100M)
    
    Returns:
        Tupla (run_dir, list[periods])
    """
    base = Path(base_dir)
    runs = sorted([p for p in base.glob('wf_*') if p.is_dir()])
    if not runs:
        raise RuntimeError(f"Nenhum run encontrado em {base}")
    run = runs[-1]
    periods = []
    for p in run.iterdir():
        if p.is_dir() and p.name.startswith('period_') and p.name.endswith('d'):
            try:
                val = int(p.name.replace('period_', '').replace('d', ''))
                periods.append(val)
            except Exception:
                pass
    periods = sorted(periods)
    if not periods:
        raise RuntimeError(f"Nenhum período encontrado em {run}")
    return run, periods


def find_latest_run_any(save_root: Path) -> tuple[Path, list[int]]:
    """
    Procura o wf_* mais recente em qualquer label market_cap_* dentro de save_root.
    
    Args:
        save_root: Diretório raiz (ex: models_classifier)
    
    Returns:
        Tupla (run_dir, list[periods])
    """
    candidates: list[Path] = []
    for base in Path(save_root).glob('market_cap_*'):
        if not base.is_dir():
            continue
        for wf in base.glob('wf_*'):
            if wf.is_dir():
                candidates.append(wf)
    if not candidates:
        raise RuntimeError(f"Nenhum run encontrado em {save_root}")
    # escolhe por mtime
    run = max(candidates, key=lambda p: p.stat().st_mtime)
    # extrai períodos
    periods = []
    for p in run.iterdir():
        if p.is_dir() and p.name.startswith('period_') and p.name.endswith('d'):
            try:
                periods.append(int(p.name.replace('period_','').replace('d','')))
            except Exception:
                pass
    periods = sorted(periods)
    if not periods:
        raise RuntimeError(f"Nenhum período encontrado em {run}")
    return run, periods


def build_X_from_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Constrói DataFrame de features a partir de um DataFrame completo e lista de colunas.
    
    Args:
        df: DataFrame com todas as features
        feat_cols: Lista de nomes de colunas a extrair
    
    Returns:
        DataFrame apenas com as colunas especificadas (sem NaN)
    """
    if pd is None or np is None:
        raise RuntimeError("pandas e numpy são necessários")
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = (df[c].astype(np.float32) if c in df.columns else np.nan)
    return X.dropna()


def load_training_cutoffs(
    run_dir: Path,
    symbol: str,
    periods: list[int],
) -> tuple[dict[int, pd.Timestamp], pd.Timestamp | None]:
    """
    Carrega cutoffs de treinamento por período/símbolo.
    
    Args:
        run_dir: Diretório do run de treinamento
        symbol: Símbolo sendo avaliado
        periods: Lista de períodos para carregar cutoffs
    
    Returns:
        Tupla (dict[period -> cutoff_timestamp], last_train_ts0 | None)
    """
    if pd is None:
        return {}, None
    
    per_cut: dict[int, pd.Timestamp] = {}
    last_train_ts0 = None
    
    try:
        dmeta_path = run_dir / 'dataset' / 'meta_long.parquet'
        smap_path = run_dir / 'dataset' / 'sym_map.json'
        if dmeta_path.exists() and smap_path.exists():
            dmeta = pd.read_parquet(dmeta_path)
            sym_map = json.loads(Path(smap_path).read_text(encoding='utf-8'))
            try:
                sym_id = int(sym_map.index(symbol))
            except Exception:
                sym_id = None
            if sym_id is not None:
                last_train_ts0 = pd.to_datetime(dmeta.loc[dmeta['sym_id'] == sym_id, 'ts']).max()
        
        for p in periods:
            jpath = run_dir / f'period_{int(p)}d' / 'dataset_cutoffs_long.json'
            cutoff = None
            if jpath.exists():
                try:
                    j = json.loads(jpath.read_text(encoding='utf-8'))
                    cut_s = j.get(symbol)
                    if cut_s:
                        cutoff = pd.to_datetime(cut_s)
                except Exception:
                    cutoff = None
            if cutoff is None and last_train_ts0 is not None:
                cutoff = last_train_ts0 - pd.Timedelta(days=int(p))
            if cutoff is not None:
                per_cut[int(p)] = cutoff
    except Exception:
        pass
    
    return per_cut, last_train_ts0


def load_feature_columns(run_dir: Path, periods: list[int]) -> list[str]:
    """Carrega feature_columns.json do run_dir."""
    fc = run_dir / "dataset" / "feature_columns.json"
    if fc.exists():
        return json.loads(fc.read_text(encoding="utf-8"))
    for p in sorted(periods):
        f = run_dir / f"period_{int(p)}d" / "feature_columns.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    raise RuntimeError("feature_columns.json não encontrado no run.")


def read_anchor_end(run_dir: Path) -> pd.Timestamp | None:
    """Lê anchor_end_utc do time_anchors.json do run_dir."""
    if pd is None:
        return None
    try:
        ta = Path(run_dir) / "time_anchors.json"
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
        pass
    return None


def get_data_window_end(days: int, skip_days: int) -> pd.Timestamp | None:
    """
    Obtém o timestamp final da janela de dados que será carregada.
    Com days=730 e skip_days=365, retorna o timestamp de 365 dias atrás.
    """
    if pd is None:
        return None
    try:
        try:
            from my_project.prepare_features.data import _now_ms_from_env
        except Exception:
            try:
                from ..prepare_features.data import _now_ms_from_env
            except Exception:
                return None
        now_ms = _now_ms_from_env()
        now_s = now_ms / 1000.0
        # O final da janela é 'skip_days' dias atrás
        end_s = now_s - (skip_days * 86400)
        return pd.to_datetime(int(end_s * 1000), unit='ms')
    except Exception:
        return None


def choose_run(
    save_root: Path,
    run_hint: str | None = None,
    *,
    target_end: pd.Timestamp | None = None,
    default_label: str = "market_cap_150B_50M",
) -> tuple[Path, list[int]]:
    """
    Seleciona o wf_* mais apropriado.
    - Se run_hint fornecido, usa-o.
    - Caso contrário, se target_end fornecido, escolhe wf_* cujo time_anchors.anchor_end_utc esteja mais próximo de target_end.
      (Isso alinha os cutoffs com a janela de avaliação, evitando zero OOS.)
    - Fallback: último wf_* do label padrão.
    
    Args:
        save_root: Diretório raiz (ex: models_classifier)
        run_hint: Hint para o run (label ou caminho direto)
        target_end: Timestamp alvo para alinhar o run
        default_label: Label padrão se não encontrar
    
    Returns:
        Tupla (run_dir, list[periods])
    """
    if run_hint:
        p = Path(run_hint)
        if p.exists() and p.is_dir() and p.name.startswith("wf_"):
            rd, periods = p, find_latest_run(p.parent)[1]
        else:
            # trata como label
            base_dir = Path(save_root) / run_hint
            rd, periods = find_latest_run(base_dir)
    else:
        # Prioriza candidatos do default_label primeiro
        default_base = Path(save_root) / default_label
        candidates: list[tuple[Path, pd.Timestamp | None]] = []
        
        # Primeiro tenta encontrar candidatos no default_label
        if default_base.exists() and default_base.is_dir():
            for wf in default_base.glob("wf_*"):
                if not wf.is_dir():
                    continue
                candidates.append((wf, read_anchor_end(wf)))
        
        # Se não encontrou candidatos no default_label, busca em todos os labels
        if not candidates:
            for base in Path(save_root).glob("market_cap_*"):
                if not base.is_dir():
                    continue
                for wf in base.glob("wf_*"):
                    if not wf.is_dir():
                        continue
                    candidates.append((wf, read_anchor_end(wf)))
        
        # Filtra por target_end se disponível
        if target_end is not None and candidates:
            # Ordena por proximidade do anchor_end ao target_end
            ranked = sorted(
                candidates,
                key=lambda t: (pd.NaT if t[1] is None else abs((t[1] - target_end).total_seconds()))
            )
            rd = ranked[0][0]
            periods = find_latest_run(rd.parent)[1]
        else:
            try:
                rd, periods = find_latest_run(Path(save_root) / default_label)
            except Exception:
                rd, periods = find_latest_run_any(Path(save_root))
    want = [i*90 for i in range(1, 9)]  # 90..720
    periods = [p for p in want if p in periods]
    if not periods:
        raise RuntimeError(f"Run sem períodos 90..720: {rd}")
    return rd, periods

