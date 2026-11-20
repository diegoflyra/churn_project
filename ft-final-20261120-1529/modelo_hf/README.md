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
| **KS** | **0.5487** (IC95 0.508 a 0.591, bootstrap) |
| AUROC | 0.8493 |
| Recall / Precisão / F1 | 0.760 / 0.542 / 0.633 |

As métricas de classe usam o limiar do KS (0.2802), escolhido nas previsões fora da dobra do treino, nunca no teste.

## Como foi escolhido

- **Placar comum:** 3 repetições de validação cruzada em 5 dobras no treino. A parada é decidida em 20% internos de cada dobra, e a dobra externa só mede. Esta versão (`f7_triagem`) teve KS médio de **0.5573** no placar, o maior entre as versões do FT-Transformer.
- **Hiperparâmetros:** encontrados pelo Optuna em divisões próprias, separadas das do placar.
- **Features:** além das colunas originais, `cliente_novo`, `delta_preco`, `tenure_1`, `cobranca_razao`, `antigo`, `caro_sem_backup`, `fibra_sem_protecao`, `atrito_pagamento`. Vieram da análise de erro das previsões fora da dobra e de seleção em divisões próprias.
- **Arquitetura:** `{"n_blocos": 1, "d_token": 192, "n_cabecas": 4, "drop_atencao": 0.49174646579703485, "drop_ffn": 0.33792357369236903, "drop_residual": 0.002815031260296183, "ffn_fator": 1.6205510930929652}`, com AdamW, sem oversampling.
- **Treino final:** treino oficial inteiro, 5 sementes. A previsão é a média das redes. Épocas por semente: 90, 36, 31, 24, 89, decididas pela parada em 20% internos.

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
