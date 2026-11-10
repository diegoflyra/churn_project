# Experimentos

Cada modelo tem uma pasta própria e pode evoluir o próprio pré-processamento. Essa pasta guarda também o histórico dele:

```
experimentos/
├── README.md                este guia
├── PREPROCESSAMENTOS.md     comparação entre modelos (gerada)
├── preprocessamentos.py     confere os registros e gera a comparação
└── <modelo>/
    └── preprocessamento/
        ├── registro.json    o histórico: o que mudou, por quê e o resultado
        └── ...              o código ou notebook de cada versão
```

`if702-preprocessamento.ipynb` continua como a **versão base (`v0`)** de todos. Ele é o ponto de partida, não um controle central: cada modelo pode divergir dele, desde que registre a mudança.

## O que pode variar e o que não pode

**Pode variar por modelo:** variáveis criadas, escala, oversampling, seleção de variáveis e até a divisão de validação.

**Não pode variar:**

1. **O conjunto de teste.** Mesmos clientes para todos, vindos de `comum.dividir`. Confira com `comum.assinatura_teste()`.
2. **Sem vazamento.** Tudo que aprende com os dados (scaler, oversampling, codificadores) é ajustado só na parte de treino, e o teste não orienta nenhuma escolha.
3. **A métrica.** KS, com `comum.ks`.

Mudar a divisão de validação é permitido, mas os KS de validação deixam de ser comparáveis entre modelos. Por isso cada resultado registra o protocolo. A comparação entre modelos é feita no teste.

## Como registrar uma mudança

1. Coloque o código em `<modelo>/preprocessamento/`, em arquivo ou notebook. Se ele já estiver em outro arquivo da pasta do modelo, basta apontar para ele.
2. Acrescente uma versão ao `registro.json`, **inclusive se ela for rejeitada**: saber o que não funcionou também faz parte da evolução.
3. Rode `python experimentos/preprocessamentos.py` e faça o commit junto.

```json
{
  "id": "v1_cliente_novo",
  "parte_de": "v0",
  "mudanca": "Adiciona cliente_novo (tenure <= 6)",
  "motivo": "De onde veio a ideia",
  "altera": {"extras": ["cliente_novo"]},
  "codigo": "experimentos/mlp/blocos.py (CANDIDATAS)",
  "validacao": {"protocolo": "5 divisões novas do treino", "ks": 0.5491, "delta": 0.0092, "ganhou": "5/5", "regua": 0.0028},
  "decisao": "aceita",
  "evidencia": "resultados/mlp-20261103-0337/blocos/b04_cliente_novo.json"
}
```

- **`decisao`:** `base`, `em teste`, `aceita` ou `rejeitada`.
- **`altera`:** só o que muda em relação à versão anterior. Aceita `divisao_validacao`, `variaveis_criadas`, `scaler`, `oversampling` e `extras`.
- **`atual`:** no topo do arquivo, indica a versão que o modelo usa hoje.

## Processamentos para testar nos outros modelos

O MLP testou 20 features e o tamanho do oversampling (histórico em `mlp/HISTORICO.md`). O que funcionou está disponível para todos em `comum.py`, sem código novo:

```python
Xa, ya, Xv = comum.preparar(X_tr, y_tr, X_val, features=("cliente_novo", "delta_preco", "tenure_1"), over=True)
```

| Processamento | O que faz |
|---|---|
| `tenure_1` | 1 se tenure ≤ 1 (61% de churn no treino) |
| `cliente_novo` | 1 se tenure ≤ 6 |
| `delta_preco` | MonthlyCharges − TotalCharges/(tenure+1): preço atual menos a média já paga. Entra no scaler |
| `over=True` / `False` / `0.5` | Oversampling 1:1 (o padrão do grupo), nenhum ou a minoria até essa fração da maioria |

As três features são calculadas linha a linha, sem olhar o rótulo nem outros clientes, então não vazam. Sem argumentos, `comum.preparar` faz o mesmo de antes.

**Evidência no placar** (`python experimentos/sondar_processamentos.py`, resultados em `resultados/processamentos_comuns.json`). As features foram medidas em três famílias de modelo, cada uma contra a sua própria base do grupo:

| Processamento | MLP | Regressão logística | XGBoost |
|---|---|---|---|
| + `tenure_1` | **ganhou** +0,014 | empatou +0,009 | empatou +0,001 |
| + as três | **ganhou** +0,013 | **ganhou** +0,013 | empatou −0,001 |
| sem oversampling | KS igual; MSE de 0,163 para 0,134 | KS igual; MSE de 0,166 para 0,135 | KS igual; MSE de 0,156 para 0,136 |

**Leitura:**
- **As features ajudam modelos que recebem o tenure como número contínuo** (MLP, regressão logística e, provavelmente, KAN, TabKAN e STab), porque eles não conseguem fazer sozinhos o corte "cliente nos primeiros meses". Árvores, como XGBoost e gradient boosting, já fazem esse corte, e para elas as features não mudam nada.
- **O oversampling não muda o KS em nenhum dos três, mas piora o MSE e a entropia cruzada**, porque as probabilidades saem infladas. Para os modelos pré-treinados (TabPFN, Mitra), vale testar o oversampling do contexto pelo mesmo motivo.

**Como testar no seu modelo:**
1. Meça a versão atual e a versão com `features=...` no placar (`comum.medir_placar` e `comum.registrar_placar`).
2. Use `contra={"vs_trial0": (modelo, "trial0")}`.
3. Só adote se o veredito for **ganhou**.
