# Comparacao dos pre-processamentos

Gerado por `python experimentos/preprocessamentos.py`. Nao edite aqui: edite o `registro.json` de cada modelo.
As regras estao em `docs/estrutura_preprocessamento.md`.

## O que cada modelo usa hoje

| Modelo | Versao | Validacao | Variaveis criadas | Scaler | Oversampling | Extras | Assinatura do teste |
|---|---|---|---|---|---|---|---|
| gradient_boosting | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| kan | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| mitra | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| mlp | f4_tenure_1 | 5 dobras oficiais | sim | sim | sim | cliente_novo, delta_preco, tenure_1 | e90ba6fab761f037 |
| stab | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| tabkan | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| tabpfn_v2 | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |

## Evolucao por modelo

So compare numeros de KS com o mesmo protocolo. O teste nao aparece aqui: ele fica para a avaliacao final.

### gradient_boosting

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | - | - | - | - | base | - |

### kan

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | - | - | - | - | base | - |

### mitra

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | - | - | - | - | base | - |

### mlp

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | 0.5400 | - | - | 0.0028 | base | 5 divisoes novas do treino |
| b02_sem_oversampling | v0 | Remove o oversampling | 0.5413 | +0.0014 | 4/5 | 0.0028 | rejeitada | 5 divisoes novas do treino |
| b03_sem_criadas | v0 | Remove total_addons e PaymentMethod_Automatic | 0.5383 | -0.0016 | 1/5 | 0.0028 | rejeitada | 5 divisoes novas do treino |
| b04_cliente_novo | v0 | Adiciona cliente_novo (tenure <= 6) | 0.5491 | +0.0092 | 5/5 | 0.0028 | aceita | 5 divisoes novas do treino |
| b05_delta_preco | b04_cliente_novo | Adiciona delta_preco (preco atual menos media paga) | 0.5506 | +0.0014 | 4/5 | 0.0009 | aceita | 5 divisoes novas do treino |
| b06_mtm_fibra | b05_delta_preco | Adiciona contrato mensal x fibra | 0.5510 | +0.0004 | 4/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b07_preco_por_servico | b05_delta_preco | Adiciona preco por servico | 0.5498 | -0.0008 | 2/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b08_tenure_ate_3 | b05_delta_preco | Adiciona tenure_ate_3 (tenure <= 3) | 0.5526 | +0.0020 | 3/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b08_tenure_ate_12 | b05_delta_preco | Adiciona tenure_ate_12 (tenure <= 12) | 0.5489 | -0.0016 | 2/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b08_log_tenure | b05_delta_preco | Adiciona log_tenure (log1p de tenure) | 0.5505 | -0.0000 | 3/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b09_log_total | b05_delta_preco | Adiciona log_total (log1p de TotalCharges) | 0.5496 | -0.0010 | 1/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b09_log_mensal | b05_delta_preco | Adiciona log_mensal (log1p de MonthlyCharges) | 0.5502 | -0.0003 | 2/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b10_novo_sem_fibra_caro | b05_delta_preco | Adiciona regra: tenure <= 10, sem fibra, MonthlyCharges > 55 | 0.5498 | -0.0007 | 3/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b10_fibra_recem_chegado | b05_delta_preco | Adiciona regra: fibra e TotalCharges <= 120 | 0.5500 | -0.0005 | 2/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b10_antigo_sem_automatico | b05_delta_preco | Adiciona regra: tenure > 29, sem pagamento automatico | 0.5496 | -0.0010 | 2/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b11_sem_gender | b05_delta_preco | Retira gender | 0.5515 | +0.0009 | 3/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b11_sem_streaming | b05_delta_preco | Retira StreamingTV e StreamingMovies | 0.5503 | -0.0002 | 3/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| b11_sem_total_charges | b05_delta_preco | Retira TotalCharges | 0.5506 | +0.0000 | 3/5 | 0.0019 | rejeitada | 5 divisoes novas do treino |
| f4_tenure_ate_3 | b05_delta_preco | Adiciona tenure_ate_3 | 0.5505 | +0.0034 | 27/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_novo_mes_a_mes | b05_delta_preco | Adiciona novo_mes_a_mes | 0.5474 | +0.0003 | 18/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_n_protecao | b05_delta_preco | Adiciona n_protecao | 0.5474 | +0.0003 | 16/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_n_streaming | b05_delta_preco | Adiciona n_streaming | 0.5468 | -0.0003 | 13/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_fibra_sem_protecao | b05_delta_preco | Adiciona fibra_sem_protecao | 0.5472 | +0.0001 | 13/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_atrito_pagamento | b05_delta_preco | Adiciona atrito_pagamento | 0.5468 | -0.0002 | 16/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_sozinho | b05_delta_preco | Adiciona sozinho | 0.5463 | -0.0008 | 12/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_sem_gender | b05_delta_preco | Remove gender | 0.5481 | +0.0011 | 17/30 | 0.0006 | rejeitada | k-fold aninhado, 30 divisoes (fase 4) |
| f4_tenure_1 | b05_delta_preco | Adiciona tenure_1 (tenure <= 1) | 0.5520 | +0.0049 | 26/30 | 0.0006 | aceita | k-fold aninhado, 30 divisoes (fase 4) |
| p_fase4_sem_over | f4_tenure_1 | Remove o oversampling | 0.5505 | +0.0116 | empatou vs trial0 | - | rejeitada | placar comum 3x5 (sementes 101-103), parada em 20% internos |
| p_fase4_over_0.5 | f4_tenure_1 | Oversampling ate 0,5 da maioria | 0.5485 | +0.0110 | empatou vs trial0 | - | rejeitada | placar comum 3x5 (sementes 101-103), parada em 20% internos |
| p_fase4_over_0.75 | f4_tenure_1 | Oversampling ate 0,75 da maioria | 0.5513 | +0.0158 | ganhou vs trial0 | - | rejeitada | placar comum 3x5 (sementes 101-103), parada em 20% internos |
| p_sem_escala | v0 | Remove o StandardScaler | 0.5254 | +0.0026 | empatou vs trial0 | - | rejeitada | placar comum 3x5 (sementes 101-103), parada em 20% internos |
| p_so_tenure_1 | v0 | So tenure_1, sem cliente_novo e delta_preco | 0.5495 | +0.0131 | ganhou vs trial0 | - | rejeitada | placar comum 3x5 (sementes 101-103), parada em 20% internos |

### stab

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | 0.5453 | - | - | - | base | Melhor de 30 trials nas 5 dobras oficiais |

### tabkan

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | 0.5432 | - | - | - | base | Melhor de 30 trials nas 5 dobras oficiais |

### tabpfn_v2

| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |
|---|---|---|---|---|---|---|---|---|
| v0 | - | Pre-processamento base do grupo | - | - | - | - | base | - |

