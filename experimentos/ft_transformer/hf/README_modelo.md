---
language: pt
library_name: pytorch
pipeline_tag: tabular-classification
tags:
- tabular-classification
- churn
- ft-transformer
- telco
---

# FT-Transformer para churn (Telco IBM)

Um FT-Transformer ([Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959)) que estima a probabilidade de um cliente de telecom cancelar o contrato. Foi treinado no dataset Telco Customer Churn (IBM), na disciplina IF702. A métrica principal é o **KS** (Kolmogorov-Smirnov) entre as probabilidades de quem cancela e de quem fica.

## Resultado no teste

O conjunto de teste (25% dos clientes, estratificado, semente 42) foi usado **uma única vez**, depois da escolha do modelo.

| Métrica | Valor |
|---|---|
| **KS** | **{ks}** (IC95 {ks_lo} a {ks_hi}, bootstrap) |
| AUROC | {auroc} |
| Recall / Precisão / F1 | {recall} / {precision} / {f1} |

As métricas de classe usam o limiar do KS ({limiar}), escolhido nas previsões fora da dobra do treino, nunca no teste.

## Como foi escolhido

- **Placar comum:** 3 repetições de validação cruzada em 5 dobras no treino. A parada é decidida em 20% internos de cada dobra, e a dobra externa só mede. Esta versão (`{rotulo}`) teve KS médio de **{ks_placar}** no placar, o maior entre as versões do FT-Transformer.
- **Hiperparâmetros:** encontrados pelo Optuna em divisões próprias, separadas das do placar.
- **Features:** além das colunas originais, {features}. Vieram da análise de erro das previsões fora da dobra e de seleção em divisões próprias.
- **Arquitetura:** `{arquitetura}`, com AdamW, sem oversampling.
- **Treino final:** treino oficial inteiro, {n_sementes} sementes. A previsão é a média das redes. Épocas por semente: {epocas}, decididas pela parada em 20% internos.

## Uso

```python
import pandas as pd
import churn_ft                                   # arquivo deste repositório

modelo = churn_ft.carregar(".")                   # pasta com config.json e pesos_semente*.pt
p = modelo.prever(pd.read_csv("clientes.csv"))    # probabilidade de churn
cancela = p >= modelo.limiar
```

O CSV precisa das colunas do dataset Telco Customer Churn (`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`). `customerID` e `Churn` são opcionais. Precisa de `torch`, `numpy` e `pandas`.

## Arquivos

- `config.json`: arquitetura, colunas, features, média e desvio do scaler, limiar e métricas;
- `pesos_semente*.pt`: os pesos de cada rede;
- `churn_ft.py`: o pré-processamento e o modelo, sem depender do resto do projeto.

## Limitações

O modelo foi treinado em cerca de 5 mil clientes de uma única base pública. As probabilidades refletem a proporção de churn dessa base (cerca de 26%) e não foram calibradas para outras populações.
