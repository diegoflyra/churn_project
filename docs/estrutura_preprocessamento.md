# Estrutura do pré-processamento

Cada modelo tem uma pasta própria em `experimentos/` e pode evoluir o próprio pré-processamento. Essa pasta guarda também o histórico dele:

```
docs/
├── estrutura_preprocessamento.md   este guia
└── PREPROCESSAMENTOS.md            comparação entre modelos (gerada)
experimentos/
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
  "evidencia": "resultados/mlp/mlp-20261103-0337/blocos/b04_cliente_novo.json"
}
```

- **`decisao`:** `base`, `em teste`, `aceita` ou `rejeitada`.
- **`altera`:** só o que muda em relação à versão anterior. Aceita `divisao_validacao`, `variaveis_criadas`, `scaler`, `oversampling` e `extras`.
- **`atual`:** no topo do arquivo, indica a versão que o modelo usa hoje.

## Processamentos para testar nos outros modelos

O que o MLP aprendeu sobre os dados, o que já foi medido em outros modelos e como testar no seu: **[`GUIA_DADOS.md`](GUIA_DADOS.md)**. Tudo sai de `comum.preparar(..., features=..., over=...)`, sem mexer no seu código além dessa linha.
