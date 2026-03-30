# modules/thresholds

Utilidades auxiliares para estudos de threshold e curvas de score.

## Conteúdo
- `grid_search.py` - buscas simples de limiar.
- `score_curves.py` - visualização de PnL/comportamento por threshold.
- `utils.py` - helpers de suavização, cortes percentuais e apoio a análise.

No workflow fair atual, o sweep principal de `tau_entry` já está embutido no explorer `v5`, então este pacote ficou mais como apoio analítico do que como parte obrigatória do pipeline.
