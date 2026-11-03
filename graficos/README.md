# Gráficos padronizados

## Arquivos

- `graficos.py` — as funções. Autocontido, não importa `comum.py`.
- `_exemplo_graficos.py` — mocks + geração de todos os PNGs. Descartável.

## O que cada modelo deve salvar em `resultados/<modelo>.json`

Campos mínimos consumidos pelos gráficos:

| campo | tipo | uso |
| --- | --- | --- |
| `y_teste` | list[int] | mesmo teste em todos os modelos |
| `probabilidades_teste_media` | list[float] | ROC, PR, KS, correlação |
| `teste_resumo_5_sementes.ks` | dict | barras com IC |
| `.media`, `.ic95_bootstrap` | float, list[2] | barras com IC |
| `metricas_teste_prob_media.{tn,fp,fn,tp}` | int | matriz de confusão |
| `importancias_optuna` | dict[str, float] | importâncias |

O `y_teste` precisa ser **idêntico** entre os sete. O `graficos.py` verifica
e aborta se divergir.


**Esse módulo, e em especial as funções de grade (plot_importancias_painel, plot_matrizes_confusao), foram desenhados para a análise comparativa dos sete modelos**

## Como usar em outro notebook

```python
from graficos import carregar_resultados, plot_roc_overlay
resultados = carregar_resultados("resultados")
fig, ax = plot_roc_overlay(resultados)
fig.savefig("roc.png", dpi=150, bbox_inches="tight")
