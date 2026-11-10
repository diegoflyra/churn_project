# Guia de dados para os outros modelos

O que o MLP aprendeu sobre os dados e como testar isso no seu modelo. Ninguém precisa reescrever o pré-processamento: está tudo em `comum.preparar`. Sem argumentos novos, ele faz o mesmo de antes, então nada muda no seu notebook até você decidir testar.

O processo do MLP está em [`experimentos/mlp/HISTORICO.md`](../experimentos/mlp/HISTORICO.md). Ali estão as cinco fases e as 20 features testadas.

## 1. O que está disponível

```python
Xa, ya, Xv = comum.preparar(X_tr, y_tr, X_val,
                            features=("cliente_novo", "delta_preco", "tenure_1"),   # opcional
                            over=True)                                              # True, False ou 0.5
```

| Opção | O que faz |
|---|---|
| `tenure_1` | 1 se tenure ≤ 1 (61% de churn no treino) |
| `cliente_novo` | 1 se tenure ≤ 6 |
| `delta_preco` | MonthlyCharges − TotalCharges/(tenure+1): o preço atual menos a média já paga. Entra no scaler |
| `over=True` | Oversampling 1:1, o padrão do grupo |
| `over=False` | Sem oversampling |
| `over=0.5` | A minoria vai até 50% da maioria |

As três features são calculadas linha a linha, sem olhar o rótulo nem outros clientes, então não há vazamento. O scaler e o oversampling continuam ajustados só no treino da dobra.

## 2. O que já foi medido

Cada modelo foi comparado com a sua própria base no placar comum (k-fold 3×5, veredito pareado). Os resultados estão em `resultados/processamentos_comuns.json` e podem ser gerados de novo com `python experimentos/sondar_processamentos.py`.

| Processamento | MLP | Regressão logística | XGBoost |
|---|---|---|---|
| + `tenure_1` | **ganhou** +0,014 | empatou +0,009 | empatou +0,001 |
| + as três | **ganhou** +0,013 | **ganhou** +0,013 | empatou −0,001 |
| sem oversampling | KS igual; MSE de 0,163 para 0,134 | KS igual; MSE de 0,166 para 0,135 | KS igual; MSE de 0,156 para 0,136 |

Duas conclusões:
- **As features ajudam quem recebe o tenure como número contínuo.** Esses modelos não fazem sozinhos o corte "cliente nos primeiros meses". As árvores já fazem esse corte, e para elas as features não mudam nada.
- **O oversampling não muda o KS de nenhum modelo, mas piora o MSE e a entropia cruzada.** As probabilidades saem infladas, porque o modelo aprende uma taxa de churn de 50% em vez de 26%. Isso pesa nas métricas do slide 14.

## 3. O que testar em cada modelo

| Modelo | Família | Testar | Expectativa |
|---|---|---|---|
| KAN | rede, entrada contínua | `features=` as três; `over=False` | Ganho parecido com o do MLP (~+0,01). MSE menor sem oversampling |
| TabKAN | rede, entrada contínua | `features=` as três; `over=False` | Igual ao KAN |
| STab | transformer, entrada contínua | `features=` as três; `over=False` | Provável ganho. Não medido |
| TabPFN v2 | pré-treinado, aprende pelo contexto | `over=False` no contexto; depois `features=` | Não medido. O oversampling repete clientes no contexto e pode distorcer as probabilidades |
| Mitra | pré-treinado, aprende pelo contexto | Igual ao TabPFN v2 | Não medido |
| Gradient boosting | árvores | Só `over=False` | As features não devem mudar o KS (como no XGBoost). O MSE melhora sem oversampling |

"Não medido" quer dizer que é uma hipótese, não um resultado. Só o placar decide.

## 4. Como testar, passo a passo

Cada teste custa uma medida no placar. O teste final não é usado aqui.

```python
import comum
X_tr, X_te, y_tr, y_te = comum.dividir(comum.carregar_df())
MODELO = "kan"                                         # o nome do seu modelo

def medir(**proc):
    def ajustar_prever(X, y, X_val, ajuste, parada, semente):
        Xa, ya, Xv = comum.preparar(X, y, X_val, **proc)   # a única linha que muda
        modelo = ...                                       # o seu treino, igual ao de hoje
        return modelo.predict_proba(Xv)[:, 1]
    return comum.medir_placar(X_tr, y_tr, ajustar_prever)

ass = comum.assinatura_treino(X_tr)
TRES = ("cliente_novo", "delta_preco", "tenure_1")
comum.registrar_placar(comum.RESULTADOS, MODELO, "base", config, medir(), y_tr, ass)
comum.registrar_placar(comum.RESULTADOS, MODELO, "tres_features", config, medir(features=TRES), y_tr, ass,
                       contra={"vs_base": (MODELO, "base")})
comum.registrar_placar(comum.RESULTADOS, MODELO, "sem_over", config, medir(over=False), y_tr, ass,
                       contra={"vs_base": (MODELO, "base")})
comum.tabela_placar()                                  # mostra o veredito de cada linha
```

Use a mesma `config` nas três linhas, para que só os dados mudem. Se o seu modelo usa early stopping, a parada vem de `X.iloc[parada]`, nunca de `X_val` ([`pre-proposta.md`](pre-proposta.md), item 2 do placar).

## 5. Como decidir e registrar

1. **Adote só com veredito `ganhou`.** `empatou` quer dizer que a diferença cabe no ruído, de cerca de ±0,01 de KS. Nesse caso, fique com a versão mais simples.
2. **`over=False` com `empatou` no KS também pode ser adotado, se o MSE cair.** É a mesma régua de KS com uma calibração melhor. Diga isso no motivo.
3. **Registre no `experimentos/<modelo>/preprocessamento/registro.json`**, inclusive o que foi rejeitado (formato em [`estrutura_preprocessamento.md`](estrutura_preprocessamento.md)). Depois rode `python experimentos/preprocessamentos.py`.
4. **No fim, use o teste uma vez**, na versão adotada: `comum.registrar_teste_placar(...)`. Esse é o KS oficial do modelo.

Um aviso para STab e TabKAN: `resultados/stab.json` e `resultados/tabkan.json` ainda vêm da régua antiga (5 dobras de uma divisão, sem placar). Para comparar com o MLP, meça a versão atual no placar com o rótulo `base` (passo 4) antes de testar qualquer mudança.
