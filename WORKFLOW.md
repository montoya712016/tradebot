# Fair Universe Protocol: Exploration Workflow

This document outlines the professional, bias-free workflow for trading strategy exploration and validation.

## 1. Multi-Generation Exploration
Run the orchestrator to explore multiple historical milestones independently. This ensures each period is treated as a contemporaneous "Fair Universe".

```powershell
python scripts/run_independent_step_explores.py
```

1.  **Orquestrador de Exploração** (`run_independent_step_explores.py`):
    -   Executa **80.000 backtests** (10.000 por milestone).
    -   **Resumo Granular**: O sistema salva o progresso a cada backtest individual. Se você interromper (`Ctrl+C`), ele retoma exatamente de onde parou, sem perder modelos já treinados ou refreshes concluídos.
    -   **Monitoramento**: O Dashboard SPA (lançado automaticamente) permite acompanhar os scores em tempo real.

## 2. Rolling Walk-Forward Verification
Once all exploration steps (1440d down to 180d) are finished, run the verification script to generate the final robustness curve.

```powershell
python scripts/verify_rolling_wf.py
```

- **Logic**: For each 180-day OOS period, it selects the best model from the corresponding exploration pool *without hindsight bias*.
- **Output**: A combined equity curve representing the true historical expectation of the strategy.

## 3. Maintenance
- **scripts/explore.py**: The core worker script used by the orchestrator.
- **scripts/fair_dashboard.py**: The real-time monitor.
- **data/generated/fair_wf_explore/**: Target directory for all results.
