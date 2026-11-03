# Comparacao dos pre-processamentos

Gerado por `python experimentos/preprocessamentos.py`. Nao edite aqui: edite o `registro.json` de cada modelo.
As regras estao em `experimentos/README.md`.

## O que cada modelo usa hoje

| Modelo | Versao | Validacao | Variaveis criadas | Scaler | Oversampling | Extras | Assinatura do teste |
|---|---|---|---|---|---|---|---|
| gradient_boosting | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| kan | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| mitra | v0 | 5 dobras oficiais | sim | sim | sim | - | e90ba6fab761f037 |
| mlp | b05_delta_preco | 5 dobras oficiais | sim | sim | sim | cliente_novo, delta_preco | e90ba6fab761f037 |
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

