# -*- coding: utf-8 -*-
"""
Teste simples para verificar se GPU está disponível para XGBoost.
"""
import sys
import numpy as np

print("=" * 60)
print("TESTE DE GPU PARA XGBOOST")
print("=" * 60)

# 1. Verifica nvidia-smi
print("\n[1] Verificando nvidia-smi...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("[OK] nvidia-smi disponivel")
        # Mostra algumas linhas do output
        lines = result.stdout.split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"  {line}")
    else:
        print("[ERRO] nvidia-smi retornou erro")
        print(f"  stderr: {result.stderr[:200]}")
except FileNotFoundError:
    print("[ERRO] nvidia-smi nao encontrado (CUDA nao instalado ou nao no PATH)")
except Exception as e:
    print(f"[ERRO] Erro ao executar nvidia-smi: {e}")

# 2. Verifica import do XGBoost
print("\n[2] Verificando import do XGBoost...")
try:
    import xgboost as xgb
    print(f"[OK] XGBoost importado com sucesso")
    print(f"  Versao: {xgb.__version__}")
except Exception as e:
    print(f"[ERRO] Erro ao importar XGBoost: {e}")
    sys.exit(1)

# 3. Verifica se tree_method='gpu_hist' está disponível
print("\n[3] Verificando suporte a GPU no XGBoost...")
try:
    # Tenta criar parâmetros com GPU
    # Na versão 3.0.3+, não usar gpu_id quando device='cuda' já está especificado
    test_params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda:0',  # Especifica GPU diretamente no device
        'max_depth': 3,
    }
    print(f"  Tentando tree_method='gpu_hist'...")
    
    # Cria dataset de teste pequeno
    X_test = np.random.rand(100, 10).astype(np.float32)
    y_test = np.random.randint(0, 2, 100).astype(np.int32)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print(f"  Dataset criado: {X_test.shape}")
    
    # Tenta treinar com GPU
    print(f"  Tentando treinar modelo com GPU...")
    bst_test = xgb.train(test_params, dtest, num_boost_round=5, verbose_eval=False)
    print("[OK] GPU FUNCIONANDO! Treinamento com GPU bem-sucedido.")
    
    # Testa predição também
    preds = bst_test.predict(dtest)
    print(f"[OK] Predicao funcionando. Shape: {preds.shape}")
    
    del bst_test, dtest
    
except Exception as e:
    error_msg = str(e).lower()
    print(f"[ERRO] Erro ao usar GPU: {e}")
    
    if any(kw in error_msg for kw in ['gpu', 'cuda', 'device', 'hist', 'not supported', 'not available']):
        print("\n  Diagnóstico: GPU não está disponível no XGBoost")
        print("  Possíveis causas:")
        print("    1. XGBoost não foi compilado com suporte GPU")
        print("    2. CUDA não está instalado corretamente")
        print("    3. Versão do CUDA não é compatível")
        print("\n  Soluções:")
        print("    - Instalar XGBoost com GPU: pip install xgboost[gpu]")
        print("    - Ou compilar XGBoost com CUDA manualmente")
        print("    - Verificar versão do CUDA: nvidia-smi")
    else:
        print(f"  Erro não relacionado a GPU: {e}")

# 4. Testa CPU como comparação
print("\n[4] Testando CPU (para comparação)...")
try:
    cpu_params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cpu',
        'max_depth': 3,
    }
    X_test = np.random.rand(100, 10).astype(np.float32)
    y_test = np.random.randint(0, 2, 100).astype(np.int32)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst_cpu = xgb.train(cpu_params, dtest, num_boost_round=5, verbose_eval=False)
    print("[OK] CPU funcionando normalmente")
    del bst_cpu, dtest
except Exception as e:
    print(f"[ERRO] Erro ao usar CPU: {e}")

print("\n" + "=" * 60)
print("TESTE CONCLUÍDO")
print("=" * 60)

