# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, Dict, List
import os, json, sys
from pathlib import Path
import numpy as np
import shutil
import gc
import pandas as pd
_pd = pd  # alias para usos existentes no código

# Imports locais (novo projeto sem my_project.config / utils)
try:
	from .dataflow import prepare_block_multi
except Exception:
	# execução direta como script: adiciona raiz ao sys.path e usa import absoluto
	_WS = Path(__file__).resolve().parents[2]
	if str(_WS) not in sys.path:
		sys.path.insert(0, str(_WS))
	from my_project.train.dataflow import prepare_block_multi
try:
	from ..prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
	from ..prepare_features.prepare_features import DEFAULT_CANDLE_SEC
except Exception:
	_WS = Path(__file__).resolve().parents[2]
	if str(_WS) not in sys.path:
		sys.path.insert(0, str(_WS))
	from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
	from my_project.prepare_features.prepare_features import DEFAULT_CANDLE_SEC

# Defaults substituindo antigo config/utils
OFFSETS_DAYS: list[int] = [90, 180, 270, 360, 450, 540, 630, 720]
RATIO_NEG_PER_POS: float = 3.0
ENTRY_XGB_PARAMS: dict = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'aucpr'],
    
    # Mantemos o modelo conservador com FP custando mais caro,
    # mas vamos deixar o otimizador explorar mais rounds.
    'scale_pos_weight': 0.3,
    
    # Aumentamos ligeiramente a profundidade para 10, mas compensamos
    # com mais amostragem aleatória para manter generalização.
    'max_depth': 10,
    
    # Learning rate menor => modelos precisam de mais rounds e tendem a generalizar melhor.
    'eta': 0.02,
    
    # Mais aleatoriedade nas árvores para ajudar generalização.
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bynode': 0.80,
    
    'min_child_weight': 3.0,
    
    # Regularização levemente reforçada para segurar o aumento de profundidade.
    'lambda': 1.8,
    'alpha': 0.10,
    'gamma': 0.10,
    'max_bin': 512,
}

# Mais rounds para aproveitar o eta menor; early stopping segura se não houver ganho.
# - 4000 rounds máx (~2x tempo), mas early stop interrompe antes se estabilizar.
ENTRY_XGB_ROUNDS: int = 4000

# Early stopping com janela maior para permitir que o eta menor converja.
ENTRY_XGB_EARLY: int = 400
SAVE_DIR: Path = Path(__file__).resolve().parents[2] / "models_classifier"
# GPU: habilita por padrão (se XGBoost tiver suporte). Pode forçar via env: XGB_USE_GPU=0/1 e XGB_GPU_ID
USE_GPU: bool = (os.getenv("XGB_USE_GPU", "1") != "0")
GPU_DEVICE_ID: int = int(os.getenv("XGB_GPU_ID", "0"))
RESUME_RUN_DEFAULT: Path | None = None

def _check_gpu_available() -> bool:
    """Verifica se GPU está disponível para XGBoost."""
    try:
        import xgboost as xgb
        
        # Primeiro, verifica se CUDA está disponível no sistema
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print(f"[trainer] nvidia-smi não disponível (CUDA não instalado ou GPU não detectada)", flush=True)
                return False
        except FileNotFoundError:
            print(f"[trainer] nvidia-smi não encontrado (CUDA não instalado)", flush=True)
            return False
        except Exception as e:
            print(f"[trainer] Erro ao verificar nvidia-smi: {e}", flush=True)
            # Continua tentando mesmo assim
        
        # Tenta criar um DMatrix simples com parâmetros GPU para verificar disponibilidade
        try:
            # Verifica se GPU está disponível (XGBoost 3.0.3+ usa tree_method='hist' com device='cuda:X')
            test_params = {
                'objective': 'binary:logistic',
                'tree_method': 'hist',  # Na versão 3.0.3+, usar 'hist' com device='cuda'
                'device': f'cuda:{int(GPU_DEVICE_ID)}',  # Especifica GPU diretamente no device
                'max_depth': 3,
            }
            # Cria um dataset mínimo para testar
            import numpy as np
            X_test = np.random.rand(10, 5).astype(np.float32)
            y_test = np.random.randint(0, 2, 10).astype(np.int32)
            dtest = xgb.DMatrix(X_test, label=y_test)
            # Tenta treinar com 1 round para verificar se GPU funciona
            try:
                bst_test = xgb.train(test_params, dtest, num_boost_round=1, verbose_eval=False)
                del bst_test, dtest
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'hist', 'not supported', 'not available']):
                    print(f"[trainer] GPU não disponível no XGBoost: {e}", flush=True)
                    print(f"[trainer] Dica: instale xgboost com suporte GPU: pip install xgboost[gpu] ou compile com CUDA", flush=True)
                    return False
                # Se o erro não é relacionado a GPU, assume que GPU está OK mas houve outro problema
                print(f"[trainer] Aviso: erro não relacionado a GPU durante teste: {e}", flush=True)
                return True
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'hist', 'not supported', 'not available']):
                print(f"[trainer] GPU não disponível no XGBoost: {e}", flush=True)
                print(f"[trainer] Dica: instale xgboost com suporte GPU: pip install xgboost[gpu] ou compile com CUDA", flush=True)
                return False
            print(f"[trainer] Erro ao configurar teste GPU: {e}", flush=True)
            return False
    except Exception as e:
        print(f"[trainer] Erro ao verificar GPU: {e}", flush=True)
        return False

# Verifica GPU na inicialização
_GPU_AVAILABLE = False
if USE_GPU:
    _GPU_AVAILABLE = _check_gpu_available()
    if _GPU_AVAILABLE:
        print(f"[trainer] GPU detectada e disponível (device_id={GPU_DEVICE_ID})", flush=True)
    else:
        print(f"[trainer] GPU solicitada mas não disponível; usando CPU", flush=True)
else:
    print(f"[trainer] GPU desabilitada via XGB_USE_GPU=0; usando CPU", flush=True)

def next_run_dir(base_dir: Path, prefix: str = "wf_") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base_dir.glob(f"{prefix}*") if p.is_dir()]
    max_idx = 0
    for p in existing:
        name = p.name
        try:
            idx = int(name.replace(prefix, ""))
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    return base_dir / f"{prefix}{max_idx+1:03d}"

# (mantido vazio — agora usamos defaults acima)


def _parse_market_caps(path: Path) -> dict[str, float]:
    m: dict[str, float] = {}
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return m
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        sym, val = line.split(":", 1)
        sym = sym.strip().upper()
        v = val.strip().replace(".", "").replace(",", "")
        try:
            m[sym] = float(int(v))
        except Exception:
            continue
    return m


def _format_range_label(min_usd: float, max_usd: float) -> str:
    def _fmt(v: float) -> str:
        T = 1_000_000_000_000.0; B = 1_000_000_000.0; M = 1_000_000.0
        if v >= T:  return f"{int(round(v/T))}T"
        if v >= B:  return f"{int(round(v/B))}B"
        if v >= M:  return f"{int(round(v/M))}M"
        return f"{int(v)}"
    # Usa ordem decrescente para manter convenção já existente (ex.: market_cap_1T_100B)
    hi, lo = (max(min_usd, max_usd), min(min_usd, max_usd))
    return f"market_cap_{_fmt(hi)}_{_fmt(lo)}"


def _load_anchor_from_dataset(dataset_dir: Path) -> str | None:
    """
    Reconstroi anchor_end_utc a partir dos metadados existentes.
    Usa o menor dos últimos timestamps por símbolo (long/short).
    """
    try:
        ts_candidates: list[pd.Timestamp] = []
        for meta in ("meta_long.parquet", "meta_short.parquet"):
            mpath = dataset_dir / meta
            if not mpath.exists():
                continue
            df = pd.read_parquet(mpath, columns=["sym_id", "ts"])
            g = df.groupby("sym_id")["ts"].max()
            if len(g):
                ts_candidates.append(g.min())
        if not ts_candidates:
            return None
        anchor = min(ts_candidates)
        return pd.to_datetime(anchor).isoformat()
    except Exception:
        return None


def _train_binary_xgb(X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
                      X_va: np.ndarray, y_va: np.ndarray, w_va: np.ndarray,
                      *, use_gpu: bool, gpu_id: int):
    import xgboost as xgb  # import tardio para evitar conflitos na inicialização
    # Usa GPU apenas se realmente disponível
    use_gpu_effective = bool(use_gpu) and _GPU_AVAILABLE
    
    def _make_params(_use_gpu: bool) -> dict:
        p = dict(ENTRY_XGB_PARAMS)
        if _use_gpu:
            # XGBoost 3.0.3+: usar tree_method='hist' com device='cuda:X' (não usar gpu_id separado)
            p.update({'tree_method': 'hist', 'device': f'cuda:{int(gpu_id)}'})
        else:
            # CPU: usar 'hist' para reduzir uso de RAM durante treinamento
            p.update({'tree_method': 'hist', 'device': 'cpu'})
            # Limita uso de memória durante construção do histograma (útil para datasets grandes)
            # Este parâmetro ajuda a evitar picos de RAM no início do treinamento
            p['max_bin'] = min(p.get('max_bin', 512), 256)  # Reduz max_bin temporariamente para CPU
        return p
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dva = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    params = _make_params(use_gpu_effective)
    try:
        from tqdm import tqdm
        class _TQDM(xgb.callback.TrainingCallback):
            def __init__(self, total):
                self.p = tqdm(total=total, desc=('xgb-gpu' if use_gpu_effective else 'xgb'), leave=False)
            def after_iteration(self, model, epoch, evals_log):
                self.p.update(1); return False
            def after_training(self, model):
                self.p.close(); return model
        cb = [_TQDM(ENTRY_XGB_ROUNDS), xgb.callback.EarlyStopping(rounds=ENTRY_XGB_EARLY)]
        bst = xgb.train(params, dtr, num_boost_round=ENTRY_XGB_ROUNDS, evals=[(dtr,'train'),(dva,'val')], callbacks=cb, verbose_eval=False)
    except Exception as e:
        if use_gpu_effective:
            # fallback robusto para CPU se GPU falhou durante treinamento
            error_msg = str(e).lower()
            print(f"[trainer] Erro durante treinamento com GPU: {e}", flush=True)
            print("[trainer] Voltando ao CPU para binário.", flush=True)
            params = _make_params(False)
            bst = xgb.train(params, dtr, num_boost_round=ENTRY_XGB_ROUNDS, evals=[(dtr,'train'),(dva,'val')], early_stopping_rounds=ENTRY_XGB_EARLY, verbose_eval=100)
        else:
            # fallback: prints a cada 100 rounds
            print(f"[trainer] Erro durante treinamento: {e}", flush=True)
            bst = xgb.train(params, dtr, num_boost_round=ENTRY_XGB_ROUNDS, evals=[(dtr,'train'),(dva,'val')], early_stopping_rounds=ENTRY_XGB_EARLY, verbose_eval=100)
    return bst

def _train_regression_xgb(X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
                          X_va: np.ndarray, y_va: np.ndarray, w_va: np.ndarray,
                          *, use_gpu: bool, gpu_id: int):
    import xgboost as xgb
    # Usa GPU apenas se realmente disponível
    use_gpu_effective = bool(use_gpu) and _GPU_AVAILABLE
    
    def _make_params(_use_gpu: bool) -> dict:
        p = dict(ENTRY_XGB_PARAMS)
        p.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse'})
        if _use_gpu:
            # XGBoost 3.0.3+: usar tree_method='hist' com device='cuda:X' (não usar gpu_id separado)
            p.update({'tree_method': 'hist', 'device': f'cuda:{int(gpu_id)}'})
        else:
            # CPU: usar 'hist' para reduzir uso de RAM durante treinamento
            p.update({'tree_method': 'hist', 'device': 'cpu'})
            # Limita uso de memória durante construção do histograma
            p['max_bin'] = min(p.get('max_bin', 512), 256)  # Reduz max_bin temporariamente para CPU
        return p
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dva = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    params = _make_params(use_gpu_effective)
    try:
        from tqdm import tqdm
        class _TQDM(xgb.callback.TrainingCallback):
            def __init__(self, total): self.p = tqdm(total=total, desc=('xgb-reg-gpu' if use_gpu_effective else 'xgb-reg'), leave=False)
            def after_iteration(self, model, epoch, evals_log): self.p.update(1); return False
            def after_training(self, model): self.p.close(); return model
        cb = [_TQDM(ENTRY_XGB_ROUNDS), xgb.callback.EarlyStopping(rounds=ENTRY_XGB_EARLY)]
        bst = xgb.train(params, dtr, num_boost_round=ENTRY_XGB_ROUNDS, evals=[(dtr,'train'),(dva,'val')], callbacks=cb, verbose_eval=False)
    except Exception as e:
        if use_gpu_effective:
            # fallback robusto para CPU se GPU falhou durante treinamento
            error_msg = str(e).lower()
            print(f"[trainer] Erro durante treinamento com GPU: {e}", flush=True)
            print("[trainer] Voltando ao CPU para regressão.", flush=True)
            params = _make_params(False)
            bst = xgb.train(params, dtr, num_boost_round=ENTRY_XGB_ROUNDS, evals=[(dtr,'train'),(dva,'val')], early_stopping_rounds=ENTRY_XGB_EARLY, verbose_eval=100)
        else:
            print(f"[trainer] Erro durante treinamento: {e}", flush=True)
            bst = xgb.train(params, dtr, num_boost_round=ENTRY_XGB_ROUNDS, evals=[(dtr,'train'),(dva,'val')], early_stopping_rounds=ENTRY_XGB_EARLY, verbose_eval=100)
    return bst


def train_entry_models_groups(*,
    mcap_min_usd: float,
    mcap_max_usd: float,
    total_days: int,
    offsets_days: Iterable[int] = OFFSETS_DAYS,
    symbols_file: Path | None = None,
    ratio_neg_per_pos: int | float = RATIO_NEG_PER_POS,
) -> Path:
    """Treina modelos de ENTRADA (buy/short) por blocos T-X para um grupo de market cap.

    Salva em models_classifier/<label>/wf_XXX/:
      - period_<Xd>/entry_models/model_buy.json, model_short.json
      - period_<Xd>/feature_columns.json
      - time_anchors.json (mapa offset->datas) no run_dir
    """
    base_dir = SAVE_DIR / _format_range_label(mcap_min_usd, mcap_max_usd)
    base_dir.mkdir(parents=True, exist_ok=True)

    resume_run = RESUME_RUN_DEFAULT
    dataset_ready = False
    if resume_run:
        run_dir = Path(resume_run).resolve()
        if not run_dir.exists():
            raise RuntimeError(f"RESUME_RUN_DEFAULT aponta para {run_dir}, mas o diretório não existe.")
        if not run_dir.is_dir():
            raise RuntimeError(f"RESUME_RUN_DEFAULT ({run_dir}) não é um diretório.")
        dataset_ready = (run_dir / "dataset").exists()
        print(f"[trainer] retomando em run_dir existente: {run_dir} | dataset_pronto={dataset_ready}", flush=True)
    else:
        run_dir = next_run_dir(base_dir, prefix="wf_")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[trainer] output dir: {run_dir}", flush=True)

    # símbolos pelo arquivo top_market_cap.txt
    if symbols_file is None:
        symbols_file = Path(__file__).resolve().parents[1] / "top_market_cap.txt"
    cap_map = _parse_market_caps(symbols_file)
    lo = float(min(mcap_min_usd, mcap_max_usd)); hi = float(max(mcap_min_usd, mcap_max_usd))
    symbols: List[str] = []
    EXCLUDE = {"BTCUSDT", "ETHUSDT"}
    for k, v in cap_map.items():
        if v >= lo and v <= hi:
            s = k
            if not s.endswith("USDT"):
                s = s + "USDT"
            if s in EXCLUDE:
                continue
            symbols.append(s)
    if dataset_ready:
        try:
            smap_path = run_dir / "dataset" / "sym_map.json"
            if smap_path.exists():
                symbols = json.loads(smap_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not symbols:
        raise RuntimeError("Nenhum símbolo no intervalo de market cap.")
    print(f"[trainer] símbolos selecionados ({len(symbols)}): faixa USD [{lo:.0f}, {hi:.0f}] do arquivo {symbols_file}", flush=True)

    # offsets definidos em config (já incluem 0..720, passo 90)
    offs = sorted(set(int(x) for x in offsets_days))
    anchors: dict[str, dict] = {}
    print(f"[trainer] offsets: {offs}", flush=True)

    feat_cols_base: list[str] = []
    base_anchor: str | None = None

    # 1) Constrói dataset base (sem corte de cauda), salvando ts e símbolo
    if dataset_ready:
        print("[trainer] dataset já existe; pulando coleta/preparo e usando arquivos do disco.", flush=True)
        feat_path = run_dir / 'dataset' / 'feature_columns.json'
        feat_cols_base = json.loads(feat_path.read_text(encoding='utf-8'))
        base_anchor = _load_anchor_from_dataset(run_dir / 'dataset')
    else:
        print("[trainer] construindo dataset base (sem corte de cauda)...", flush=True)
        parts_dir = run_dir / 'dataset_parts'
        # Sinaliza ao dataflow onde gravar partes (modo MP)
        os.environ["TRAIN_PARTS_DIR"] = str(parts_dir)
        base = prepare_block_multi(symbols, total_days=int(total_days), remove_tail_days=0, ratio_neg_per_pos=ratio_neg_per_pos)
        # Monta dataset final a partir das partes (streaming, baixo uso de RAM)
        (run_dir / 'dataset').mkdir(exist_ok=True)

    def _assemble_from_parts(parts_dir: Path, out_dir: Path, feat_cols: list[str]) -> tuple[list[str], Path]:
        from numpy.lib.format import open_memmap
        import pandas as _pd
        parts_dir = Path(parts_dir)
        long_dir = parts_dir / "long"
        short_dir = parts_dir / "short"
        # lista partes
        long_X_files = sorted([p for p in long_dir.glob("*_X.npy")])
        short_X_files = sorted([p for p in short_dir.glob("*_X.npy")])
        # total de linhas
        def _total_rows(files, side_dir):
            tot = 0
            for fx in files:
                base = fx.name[:-6]  # remove '_X.npy'
                y = np.load(side_dir / f"{base}_y.npy", mmap_mode='r')
                tot += int(y.shape[0])
            return tot
        n_feats = int(len(feat_cols))
        nL = _total_rows(long_X_files, long_dir)
        nS = _total_rows(short_X_files, short_dir)
        print(f"[trainer] montando base: long={nL:,} linhas | short={nS:,} linhas | feats={n_feats}".replace(',', '.'), flush=True)
        # cria memmaps .npy já no formato final
        Xl_mm = open_memmap(out_dir / 'X_long.npy', mode='w+', dtype=np.float32, shape=(nL, n_feats))
        Xs_mm = open_memmap(out_dir / 'X_short.npy', mode='w+', dtype=np.float32, shape=(nS, n_feats))
        # meta acumuladores
        tsl_all: list[np.ndarray] = []; syml_all: list[np.ndarray] = []; yl_all: list[np.ndarray] = []; wl_all: list[np.ndarray] = []; ul_all: list[np.ndarray] = []
        tss_all: list[np.ndarray] = []; syms_all: list[np.ndarray] = []; ys_all: list[np.ndarray] = []; ws_all: list[np.ndarray] = []; us_all: list[np.ndarray] = []
        # preenche long
        off = 0
        for fx in long_X_files:
            base = fx.name[:-6]
            X = np.load(fx, mmap_mode='r')
            y = np.load(long_dir / f"{base}_y.npy", mmap_mode='r')
            w = np.load(long_dir / f"{base}_w.npy", mmap_mode='r')
            ts = np.load(long_dir / f"{base}_ts.npy", mmap_mode='r')
            u_path = long_dir / f"{base}_u.npy"
            u = (np.load(u_path, mmap_mode='r') if u_path.exists() else np.zeros_like(y, dtype=np.float32))
            sy = np.load(long_dir / f"{base}_sym.npy", mmap_mode='r')
            n = int(y.shape[0])
            Xl_mm[off:off+n, :] = X
            yl_all.append(np.asarray(y, dtype=np.int32)); wl_all.append(np.asarray(w, dtype=np.float32))
            tsl_all.append(np.asarray(ts, dtype=np.int64)); syml_all.append(np.asarray(sy, dtype=np.uint16))
            ul_all.append(np.asarray(u, dtype=np.float32))
            off += n
        # preenche short
        off = 0
        for fx in short_X_files:
            base = fx.name[:-6]
            X = np.load(fx, mmap_mode='r')
            y = np.load(short_dir / f"{base}_y.npy", mmap_mode='r')
            w = np.load(short_dir / f"{base}_w.npy", mmap_mode='r')
            ts = np.load(short_dir / f"{base}_ts.npy", mmap_mode='r')
            u_path = short_dir / f"{base}_u.npy"
            u = (np.load(u_path, mmap_mode='r') if u_path.exists() else np.zeros_like(y, dtype=np.float32))
            sy = np.load(short_dir / f"{base}_sym.npy", mmap_mode='r')
            n = int(y.shape[0])
            Xs_mm[off:off+n, :] = X
            ys_all.append(np.asarray(y, dtype=np.int32)); ws_all.append(np.asarray(w, dtype=np.float32))
            tss_all.append(np.asarray(ts, dtype=np.int64)); syms_all.append(np.asarray(sy, dtype=np.uint16))
            us_all.append(np.asarray(u, dtype=np.float32))
            off += n
        # salva metas em parquet
        dfl = _pd.DataFrame({
            'ts': _pd.to_datetime(np.concatenate(tsl_all).astype('int64'), unit='ns'),
            'sym_id': np.concatenate(syml_all).astype(np.uint16, copy=False),
            'y': np.concatenate(yl_all).astype(np.int32, copy=False),
            'w': np.concatenate(wl_all).astype(np.float32, copy=False),
            'u': np.concatenate(ul_all).astype(np.float32, copy=False),
        })
        dfs = _pd.DataFrame({
            'ts': _pd.to_datetime(np.concatenate(tss_all).astype('int64'), unit='ns'),
            'sym_id': np.concatenate(syms_all).astype(np.uint16, copy=False),
            'y': np.concatenate(ys_all).astype(np.int32, copy=False),
            'w': np.concatenate(ws_all).astype(np.float32, copy=False),
            'u': np.concatenate(us_all).astype(np.float32, copy=False),
        })
        dfl.to_parquet(out_dir / 'meta_long.parquet', index=False)
        dfs.to_parquet(out_dir / 'meta_short.parquet', index=False)
        return feat_cols, out_dir

    if not dataset_ready:
        feat_cols_base = list(base['feature_cols'])
        # monta (streaming) e salva metadados
        feat_cols_base, _ = _assemble_from_parts(parts_dir, run_dir / 'dataset', feat_cols_base)
        # limpa partes para liberar disco
        try:
            shutil.rmtree(parts_dir, ignore_errors=True)
        except Exception:
            pass

        # salva metadados adicionais
        (run_dir / 'dataset' / 'feature_columns.json').write_text(json.dumps(feat_cols_base, ensure_ascii=False, indent=2), encoding='utf-8')
        # salva mapa de s?mbolos (usado para nomear cutoffs por per?odo); ser? removido ao final
        (run_dir / 'dataset' / 'sym_map.json').write_text(json.dumps(base.get('sym_map', symbols), ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[trainer] dataset base salvo temporariamente em {run_dir / 'dataset'}", flush=True)
        base_anchor = base.get('anchor_end_utc')

    # 2) Para cada período, filtra o dataset base por cauda por símbolo
    for tail in offs:
        try:
            print(f"[trainer] === período T-{int(tail)}d ===", flush=True)
            # carrega base do disco usando memory mapping para evitar pico de RAM
            Xl_all = np.load(run_dir / 'dataset' / 'X_long.npy', mmap_mode='r')
            Xs_all = np.load(run_dir / 'dataset' / 'X_short.npy', mmap_mode='r')
            dfl = _pd.read_parquet(run_dir / 'dataset' / 'meta_long.parquet')
            dfs = _pd.read_parquet(run_dir / 'dataset' / 'meta_short.parquet')
            # cutoff por símbolo (treino antes do cutoff; removemos completamente os últimos X=taiLd dias do treino)
            cut_by_sym_l = dfl.groupby('sym_id')['ts'].max() - _pd.to_timedelta(int(tail), unit='D')
            cut_by_sym_s = dfs.groupby('sym_id')['ts'].max() - _pd.to_timedelta(int(tail), unit='D')
            cutL = dfl['sym_id'].map(cut_by_sym_l)
            cutS = dfs['sym_id'].map(cut_by_sym_s)
            # máscaras de treino (somente antes do cutoff); validação será aleatória dentro do treino
            mask_l_tr = dfl['ts'] < cutL
            mask_s_tr = dfs['ts'] < cutS
            # salva cutoffs por período/símbolo para verificação OOS futura
            try:
                period_dir = run_dir / f"period_{int(tail)}d"
                period_dir.mkdir(parents=True, exist_ok=True)
                # id->symbol
                try:
                    sym_map_list = json.loads((run_dir / 'dataset' / 'sym_map.json').read_text(encoding='utf-8'))
                except Exception:
                    sym_map_list = symbols
                id_to_sym = {int(i): str(sym_map_list[i]) for i in range(len(sym_map_list))}
                cut_long = {id_to_sym.get(int(k), str(k)): (None if _pd.isna(v) else _pd.to_datetime(v).isoformat()) for k, v in cut_by_sym_l.items()}
                cut_short = {id_to_sym.get(int(k), str(k)): (None if _pd.isna(v) else _pd.to_datetime(v).isoformat()) for k, v in cut_by_sym_s.items()}
                (period_dir / 'dataset_cutoffs_long.json').write_text(json.dumps(cut_long, ensure_ascii=False, indent=2), encoding='utf-8')
                (period_dir / 'dataset_cutoffs_short.json').write_text(json.dumps(cut_short, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass
            # Cria pools de treino (faz cópia explícita apenas dos dados necessários)
            # Isso evita manter arrays gigantes na RAM durante todo o processamento
            mask_l_tr_arr = mask_l_tr.values
            mask_s_tr_arr = mask_s_tr.values
            Xl_pool = np.array(Xl_all[mask_l_tr_arr], dtype=np.float32, copy=True)  # Cópia explícita
            yl_pool = dfl.loc[mask_l_tr, 'y'].to_numpy(np.int32)
            wl_pool = dfl.loc[mask_l_tr, 'w'].to_numpy(np.float32)
            Ul_pool = dfl.loc[mask_l_tr, 'u'].to_numpy(np.float32) if 'u' in dfl.columns else np.zeros_like(wl_pool, dtype=np.float32)
            Xs_pool = np.array(Xs_all[mask_s_tr_arr], dtype=np.float32, copy=True)  # Cópia explícita
            ys_pool = dfs.loc[mask_s_tr, 'y'].to_numpy(np.int32)
            ws_pool = dfs.loc[mask_s_tr, 'w'].to_numpy(np.float32)
            Us_pool = dfs.loc[mask_s_tr, 'u'].to_numpy(np.float32) if 'u' in dfs.columns else np.zeros_like(ws_pool, dtype=np.float32)
            feat_cols = feat_cols_base
            
            # Libera arrays grandes da memória imediatamente após criar os pools
            del Xl_all, Xs_all
            gc.collect()
            dims_l = f"pool({int(Xl_pool.shape[0])}x{int(Xl_pool.shape[1])})"
            dims_s = f"pool({int(Xs_pool.shape[0])}x{int(Xs_pool.shape[1])})"
            print((
                f"[trainer] T-{int(tail)}d | long: {dims_l} pos={(yl_pool==1).sum():,} neg={(yl_pool==0).sum():,} | "
                f"short: {dims_s} pos={(ys_pool==1).sum():,} neg={(ys_pool==0).sum():,} | feats={len(feat_cols)}"
            ).replace(',', '.'), flush=True)

            # –– Verificação de corte por símbolo (resumo apenas)
            used_l = dfl.loc[mask_l_tr, ['sym_id','ts']].groupby('sym_id')['ts'].max()
            used_s = dfs.loc[mask_s_tr, ['sym_id','ts']].groupby('sym_id')['ts'].max()
            last_l = dfl.groupby('sym_id')['ts'].max()
            syms_all = sorted(list(set(last_l.index.astype(int))))
            violations = 0
            for s in syms_all:
                try:
                    cutL = cut_by_sym_l.loc[s]; cutS = cut_by_sym_s.loc[s]
                except Exception:
                    cutL = _pd.NaT; cutS = _pd.NaT
                uL = used_l.get(s, _pd.NaT)
                uS = used_s.get(s, _pd.NaT)
                okL = (uL is _pd.NaT) or (uL < cutL)
                okS = (uS is _pd.NaT) or (uS < cutS)
                if (not okL) or (not okS):
                    violations += 1
            if violations > 0:
                print(f"[trainer] T-{int(tail)}d WARNING: {violations}/{len(syms_all)} símbolos com violação de corte!", flush=True)
            else:
                print(f"[trainer] T-{int(tail)}d cutoff OK ({len(syms_all)} símbolos)", flush=True)

            # ================================================================
            # GROUP SPLIT por blocos temporais (evita temporal leakage)
            # ================================================================
            # Problema do split aleatório:
            # - Timestamp T de ADA no train, T de XLM no val
            # - São MUITO similares (mercado move junto) → leakage
            #
            # Solução: agrupar por bloco temporal (ex: 1 hora)
            # - Todos os símbolos do mesmo bloco vão juntos
            # - Gap de 1 bloco entre train e val
            # ================================================================
            BLOCK_SIZE_MINUTES = 60  # 1 hora por bloco
            GAP_BLOCKS = 2  # Gap de 2 blocos entre train e val
            
            rng = np.random.default_rng(42 + tail)
            
            def _split_by_temporal_blocks(X, y, w, timestamps, block_minutes=60, gap_blocks=2, train_frac=0.8):
                """
                Split train/val por blocos temporais para evitar leakage.
                
                1. Agrupa timestamps em blocos (ex: 1 hora)
                2. Shuffle dos blocos (não dos pontos individuais)
                3. 80% dos blocos para train, 20% para val
                4. Gap de N blocos entre train e val (descartados)
                """
                n = y.size
                if n == 0:
                    return (X, y, w, X, y, w)
                
                # Converte timestamps para blocos (minutos desde epoch / block_size)
                ts_arr = _pd.to_datetime(timestamps).values.astype('datetime64[m]').astype(np.int64)
                block_ids = ts_arr // block_minutes
                
                # Lista de blocos únicos
                unique_blocks = np.unique(block_ids)
                n_blocks = len(unique_blocks)
                
                if n_blocks < 5:
                    # Fallback: poucos blocos, usa split simples
                    idx = np.arange(n, dtype=np.int64)
                    rng.shuffle(idx)
                    cutn = max(1, int(train_frac * n))
                    tr = idx[:cutn]; va = idx[cutn:]
                    return X[tr], y[tr], w[tr], X[va], y[va], w[va]
                
                # Shuffle dos blocos
                block_order = unique_blocks.copy()
                rng.shuffle(block_order)
                
                # Calcula split: train | gap | val
                n_train_blocks = int(n_blocks * train_frac)
                n_gap = min(gap_blocks, n_blocks - n_train_blocks - 1)
                n_val_blocks = n_blocks - n_train_blocks - n_gap
                
                train_blocks = set(block_order[:n_train_blocks])
                # gap_blocks são descartados (block_order[n_train_blocks:n_train_blocks+n_gap])
                val_blocks = set(block_order[n_train_blocks + n_gap:])
                
                # Cria máscaras
                mask_train = np.array([b in train_blocks for b in block_ids])
                mask_val = np.array([b in val_blocks for b in block_ids])
                
                tr_idx = np.where(mask_train)[0]
                va_idx = np.where(mask_val)[0]
                
                # Log
                n_gap_samples = n - len(tr_idx) - len(va_idx)
                print(f"[split] blocos: {n_blocks} total | {n_train_blocks} train | {n_gap} gap | {n_val_blocks} val | gap_samples={n_gap_samples}", flush=True)
                
                return X[tr_idx], y[tr_idx], w[tr_idx], X[va_idx], y[va_idx], w[va_idx]
            
            # Extrai timestamps do pool
            ts_l = dfl.loc[mask_l_tr, 'ts'].values
            ts_s = dfs.loc[mask_s_tr, 'ts'].values
            
            Xl_tr, yl_tr, wl_tr, Xl_va, yl_va, wl_va = _split_by_temporal_blocks(
                Xl_pool, yl_pool, wl_pool, ts_l, 
                block_minutes=BLOCK_SIZE_MINUTES, gap_blocks=GAP_BLOCKS
            )
            Xs_tr, ys_tr, ws_tr, Xs_va, ys_va, ws_va = _split_by_temporal_blocks(
                Xs_pool, ys_pool, ws_pool, ts_s,
                block_minutes=BLOCK_SIZE_MINUTES, gap_blocks=GAP_BLOCKS
            )

            # treina
            bst_buy = _train_binary_xgb(Xl_tr, yl_tr, wl_tr, Xl_va, yl_va, wl_va, use_gpu=bool(USE_GPU), gpu_id=int(GPU_DEVICE_ID))
            bst_sho = _train_binary_xgb(Xs_tr, ys_tr, ws_tr, Xs_va, ys_va, ws_va, use_gpu=bool(USE_GPU), gpu_id=int(GPU_DEVICE_ID))
            
            # Loga informações sobre os modelos treinados
            try:
                # Tenta obter número de rounds treinados
                buy_rounds = getattr(bst_buy, 'best_ntree_limit', None)
                if buy_rounds is None:
                    try:
                        buy_rounds = bst_buy.num_boosted_rounds()
                    except Exception:
                        buy_rounds = 'n/a'
                
                sho_rounds = getattr(bst_sho, 'best_ntree_limit', None)
                if sho_rounds is None:
                    try:
                        sho_rounds = bst_sho.num_boosted_rounds()
                    except Exception:
                        sho_rounds = 'n/a'
                
                print(f"[trainer] T-{int(tail)}d treinado | buy rounds={buy_rounds}/{ENTRY_XGB_ROUNDS} | short rounds={sho_rounds}/{ENTRY_XGB_ROUNDS}", flush=True)
            except Exception as e:
                print(f"[trainer] T-{int(tail)}d aviso: nao foi possivel obter info dos modelos: {e}", flush=True)

            # ================= Regressão de U (único modelo) =================
            # Usa o pool combinado de long+short; pesos herdados (concatenação simples)
            Xr_pool = np.vstack([Xl_pool, Xs_pool]) if Xs_pool.size else Xl_pool
            ur_pool = np.concatenate([Ul_pool, Us_pool], axis=0) if Us_pool.size else Ul_pool
            wr_pool = np.concatenate([wl_pool, ws_pool], axis=0) if ws_pool.size else wl_pool
            ts_r = np.concatenate([ts_l, ts_s], axis=0) if ts_s.size else ts_l
            Xr_tr, ur_tr, wr_tr, Xr_va, ur_va, wr_va = _split_by_temporal_blocks(
                Xr_pool, ur_pool, wr_pool, ts_r,
                block_minutes=BLOCK_SIZE_MINUTES, gap_blocks=GAP_BLOCKS
            )
            bst_ureg  = _train_regression_xgb(Xr_tr, ur_tr, wr_tr, Xr_va, ur_va, wr_va, use_gpu=bool(USE_GPU), gpu_id=int(GPU_DEVICE_ID))

            # salva
            period_dir = run_dir / f"period_{int(tail)}d"
            period_dir.mkdir(parents=True, exist_ok=True)
            (period_dir / "entry_models").mkdir(exist_ok=True)
            bst_buy.save_model(str(period_dir / "entry_models" / "model_buy.json"))
            bst_sho.save_model(str(period_dir / "entry_models" / "model_short.json"))
            # Tenta salvar também em formato binário (.ubj) para modelos maiores
            try:
                bst_buy.save_model(str(period_dir / "entry_models" / "model_buy.ubj"))
                bst_sho.save_model(str(period_dir / "entry_models" / "model_short.ubj"))
            except Exception:
                pass
            # salva regressor de U
            try:
                bst_ureg.save_model(str(period_dir / "entry_models" / "model_ureg.json"))
                bst_ureg.save_model(str(period_dir / "entry_models" / "model_ureg.ubj"))
            except Exception:
                pass
            (period_dir / "feature_columns.json").write_text(json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Loga tamanho dos arquivos salvos
            try:
                buy_json_size = os.path.getsize(period_dir / "entry_models" / "model_buy.json") / (1024*1024)
                sho_json_size = os.path.getsize(period_dir / "entry_models" / "model_short.json") / (1024*1024)
                buy_ubj_size = os.path.getsize(period_dir / "entry_models" / "model_buy.ubj") / (1024*1024) if (period_dir / "entry_models" / "model_buy.ubj").exists() else 0
                sho_ubj_size = os.path.getsize(period_dir / "entry_models" / "model_short.ubj") / (1024*1024) if (period_dir / "entry_models" / "model_short.ubj").exists() else 0
                print(f"[trainer] modelos salvos em {period_dir / 'entry_models'}", flush=True)
                print(f"[trainer] T-{int(tail)}d tamanhos salvos | buy: JSON={buy_json_size:.2f}MB UBJ={buy_ubj_size:.2f}MB | short: JSON={sho_json_size:.2f}MB UBJ={sho_ubj_size:.2f}MB", flush=True)
            except Exception:
                print(f"[trainer] modelos salvos em {period_dir / 'entry_models'}", flush=True)

            anchors[str(int(tail))] = {"anchor_end_utc": base_anchor}
            
            # Libera memória explicitamente após cada período
            # Nota: Xl_all e Xs_all já foram liberados anteriormente
            del dfl, dfs
            del Xl_pool, yl_pool, wl_pool, Ul_pool
            del Xs_pool, ys_pool, ws_pool, Us_pool
            del Xl_tr, yl_tr, wl_tr, Xl_va, yl_va, wl_va
            del Xs_tr, ys_tr, ws_tr, Xs_va, ys_va, ws_va
            del Xr_pool, ur_pool, wr_pool, Xr_tr, ur_tr, wr_tr, Xr_va, ur_va, wr_va
            del bst_buy, bst_sho, bst_ureg
            gc.collect()
            print(f"[trainer] T-{int(tail)}d concluído e memória liberada.", flush=True)
        except Exception as e:
            import traceback
            print(f"[trainer] ERRO no período T-{int(tail)}d: {type(e).__name__}: {e}", flush=True)
            print(f"[trainer] traceback:", flush=True)
            traceback.print_exc()
            print(f"[trainer] continuando para o próximo período...", flush=True)
            continue

    (run_dir / "time_anchors.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")
    # Limpeza: remover dataset persistido (mantém apenas modelos e JSONs importantes)
    if not dataset_ready:
        try:
            shutil.rmtree(run_dir / 'dataset_parts', ignore_errors=True)
        except Exception:
            pass
        try:
            shutil.rmtree(run_dir / 'dataset', ignore_errors=True)
        except Exception:
            pass
    return run_dir


if __name__ == "__main__":
    # Grupo: 150B..100M
    out = train_entry_models_groups(mcap_min_usd=50_000_000.0, mcap_max_usd=150_000_000_000.0, total_days=365*10)
    print(f"✓ modelos salvos em: {out}")
