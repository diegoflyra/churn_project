# MLP: como chegamos ao modelo final

Responsável: Diego. Este arquivo conta o processo das cinco fases do MLP. Os números de todas as versões, medidos na mesma régua, estão em `resultados/PLACAR.md` (gerado de `resultados/placar.jsonl`). A evolução do pré-processamento está em `preprocessamento/registro.json`.

## Resultado

**Modelo adotado: `fase4`**, a versão com o maior KS do placar entre as 33 versões das fases 1 a 5. Depois, as variações pedidas nos slides foram medidas contra ela. Duas ficaram 0,0001 a 0,0002 acima (RMSprop com 0,5522 e Adadelta com 0,5521), mas empataram: uma versão só substitui a adotada com veredito **ganhou**.

| | Valor |
|---|---|
| **KS no placar** (k-fold 3×5) | **0,5520** (erro-padrão 0,0047) |
| **KS no teste** (usado uma vez, 5 sementes) | **0,5447** (IC95 de 0,503 a 0,586) |
| AUROC / MSE / F1 no teste | 0,8502 / 0,1622 / 0,6277 |
| Contra o MLP padrão (trial 0) | **ganhou**: +0,013 (IC95 de +0,001 a +0,026) |
| Contra a referência (campeão da fase 2) | empatou: +0,006 (IC95 de −0,003 a +0,014); no teste, −0,0007 |

Configuração:

| Item | Valor |
|---|---|
| Arquitetura | 1 camada oculta, 10 neurônios, ReLU, sem Dropout e sem BN |
| Treino | Adam, lr 0,001, L2 1e-6, entropia cruzada, lote 256, até 10.000 épocas |
| Parada | paciência 10 numa separação interna de 20% do treino; depois retreina no treino inteiro pelo número de épocas encontrado |
| Dados | pré-processamento base do grupo (`comum.py`): `StandardScaler` nas numéricas, oversampling 1:1, `total_addons` e `PaymentMethod_Automatic` |
| Features novas | `cliente_novo` (tenure ≤ 6), `delta_preco` (MonthlyCharges − TotalCharges/(tenure+1)), `tenure_1` (tenure ≤ 1) |

Previsões e métricas completas: `resultados/mlp/mlp_final.json`.

**Como ler.** O ganho sobre o MLP padrão é real. As versões no topo do placar (`fase4` com 0,5520, `fase4_over_0.75` com 0,5513, `fase3_teste` com 0,5510, o Optuna da fase 5 com 0,5503) estão empatadas entre si: diferenças menores que uns 0,01 ficam dentro do ruído. Adotamos a de maior KS, que também é das mais simples.

## A régua: por que foi preciso refazer as contas

Cada fase usou uma régua diferente, e algumas tinham defeitos. Por isso, a fase 5 mediu tudo de novo no **placar comum** (`comum.py`, regra em `docs/pre-proposta.md`):

| Fase | Régua da época | Problema |
|---|---|---|
| 1 e 2 | 5 divisões do treino, ganho acima de uma régua de ruído | Poucas divisões. A dobra que media o KS também decidia a parada |
| 3 | validação e teste em 20 divisões | A parada pelo KS usava a própria dobra de validação e inflava o KS em +0,009. Uma linhagem foi escolhida pelo teste |
| 4 | k-fold aninhado com 30 divisões | Correta, mas o IC95 só reamostrava divisões e saía estreito demais |
| 5 | **placar comum** | — |

O placar mede assim:
1. k-fold estratificado 3×5, com as sementes 101, 102 e 103, sobre os 75% de treino;
2. em cada dobra, 20% internos decidem a parada, e a dobra de validação só mede;
3. o veredito (ganhou, empatou ou perdeu) vem de um bootstrap pareado de clientes;
4. o teste é usado uma vez, na versão adotada.

## As fases

### Fase 1 (03/11): ponto de partida e features
- **O que foi feito:** o MLP padrão (1x10), um Optuna e blocos de pré-processamento, uma mudança por vez. Resultados em `resultados/mlp/mlp-20261103-0337/`.
- **O que se concluiu na época:**
  - o Optuna não superou o padrão;
  - tirar o oversampling e as variáveis criadas não ajudou;
  - `cliente_novo` e `delta_preco` foram aceitas.
- **No placar:** sozinhas, `cliente_novo` (+0,013) e `delta_preco` (+0,007) empatam com o padrão. Juntas, formam a referência (+0,007, também empate).

### Fase 2 (04/11): mais features e arquitetura
- **O que foi feito:** outras formas de tenure e das cobranças, regras vindas dos erros, retirada de variáveis e outras arquiteturas (1x5, 1x32, 2x10, 2x32). Resultados em `resultados/mlp/mlp-fase2-20261104-0505/`.
- **O que se concluiu na época:** nada superou o campeão da fase 1, que virou a **referência**.
- **No placar:** a referência mede 0,5445.

### Fase 3 (05/11): grade de arquiteturas na GPU
- **O que foi feito:**
  - treinador em pilha na GPU (`pilha.py`);
  - grade de 1.080 redes: 1 a 4 camadas, 10 a 64 unidades, 3 ativações, BN e Dropout por camada;
  - blocos de treino e duas linhagens, uma escolhida pela validação e outra pelo teste.
  - Resultados em `resultados/mlp/mlp-fase3-20261105-1526/`.
- **O que se concluiu na época:**
  - BN piorou em todas as camadas;
  - Dropout na primeira camada e GELU ajudaram um pouco;
  - a rede começa a decorar por volta da época 14.
- **No placar:** a linhagem da validação (`fase3_val`) mede 0,5488 e a do teste (`fase3_teste`) mede 0,5510. As duas empatam com a referência.

### Fase 4 (09/11): medir melhor antes de escolher
- **O que foi feito:**
  - nova medida (k-fold aninhado), que mostrou o +0,009 de inflação da fase 3;
  - mapa dos clientes ambíguos;
  - 9 features pensadas para eles, testadas no MLP e no XGBoost;
  - blocos de treino e um Optuna. Resultados em `resultados/mlp/mlp-fase4-20261109-0512/`.
- **O que se concluiu na época:**
  - **`tenure_1` foi a única feature aceita**: +0,005 no MLP e só +0,0005 no XGBoost;
  - ajustes de treino e o Optuna, que tinha só 2 divisões por trial, não passaram.
- **No placar:** **0,5520, o maior de todos.** É o modelo adotado.

### Fase 5 (10/11): tudo no placar
- **O que foi feito:**
  - as versões de cada fase medidas na mesma régua;
  - 6 arquiteturas (1x10 a 2x64) × 3 ritmos de treino (lr 1e-3/paciência 10, 3e-4/30 e 1e-4/50);
  - Optuna com 100 trials e 6 divisões por trial, incluindo logística, tanh e paciência até 50;
  - depois, cada etapa dos dados isolada e o tamanho do oversampling.
  - Resultados em `resultados/mlp/mlp-fase5-20261110-0001/` e em `resultados/placar.jsonl`.
- **O que se concluiu:**
  - **as 18 combinações de arquitetura e ritmo empataram com a referência.** Treinar de 3 a 10 vezes mais épocas, com lr menor, não mudou o KS;
  - o Optuna escolheu sozinho a rede pequena: 91 de 100 trials com 1 camada, e o melhor com 7 unidades e ativação logística;
  - entre as mudanças nos dados, **só marcar clientes novos ganhou do padrão.** Com `tenure_1` sozinha, +0,014; com as três features, +0,013. As variáveis criadas do repositório e a escala não mudam o KS;
  - **o tamanho do oversampling não muda o KS, mas piora a calibração:** o MSE na validação vai de 0,134 sem oversampling a 0,162 com 1:1.

## O que funcionou e o que não funcionou

| Funcionou | Não funcionou (empatou no placar) |
|---|---|
| Marcar clientes novos (`tenure_1`, `cliente_novo`) | Redes maiores (até 4 camadas e 64 unidades), BN, Dropout, GELU |
| Rede pequena, que é o que o próprio Optuna escolheu | Treinar mais devagar (lr 1e-4, paciência 50) |
| Parada numa separação interna, e não na validação | Optuna com 300 trials medidos em 2 divisões: o vencedor era sorte |
| Menos oversampling para MSE e entropia cruzada | 8 das 9 features para clientes ambíguos; variáveis criadas do repositório |

## Lições para os outros modelos
1. **Medir antes de escolher.** A mesma dobra não pode decidir a parada e medir o KS. Essa mistura inflou um resultado em +0,009.
2. **Uma régua só.** Toda afirmação de melhora vem do placar e de um veredito. Diferenças menores que uns 0,01 de KS são ruído com 5.282 clientes de treino.
3. **O KS de uma dobra não é o KS do modelo.** Na mesma divisão, até o MLP padrão passa de 0,59 numa dobra e cai a 0,51 em outra. Vale a média.
4. **Busca com medida ruidosa escolhe sorte.** Cada trial precisa de várias divisões.
5. **O oversampling não ajuda o KS, mas atrapalha o MSE.** Isso importa para as métricas do slide 14.
6. **O MLP adotado equivale a uma regressão logística com as mesmas features.** Com as três features, a regressão logística mede 0,5525 no placar, e o MLP, 0,5520 (`resultados/processamentos_comuns.json`). A camada oculta não acrescenta nada nesta base: o ganho está nas features.

## Arquivos finais
**Entregável:** `resultados/mlp.json`, no formato comum (os mesmos campos de `stab.json` e `tabkan.json`), com o placar junto e o MSE e a CE no treino, na validação e no teste (`mse_ce_por_conjunto`, slide 14).

Arquivos de trabalho, em `resultados/mlp/`:
- `mlp-*/`: as execuções das cinco fases;
- `mlp_final.json`: a saída do teste final, de onde o `mlp.json` é montado;
- `resultados/mlp/slides/TABELAS.md`: as tabelas pedidas nos slides, todas medidas no placar:
  - CE contra MSE, ativação, otimizador e camadas;
  - pré-processamento e oversampling;
  - MSE no treino, na validação e no teste;
  - matriz de confusão e métricas.
- `resultados/mlp/slides/*.png`: as figuras.
  - O ganho de cada fase sobre o MLP padrão.
  - As curvas ROC, precision-recall e KS no teste.
- Para gerar tudo de novo: `python experimentos/mlp/relatorio_final.py`.

## Como reproduzir
- O placar de uma versão nova: `fase5.garantir("resultados", [(rotulo, config, contra)])`.
- O relatório: `comum.relatorio_placar("resultados")`.
- O notebook da fase 5 (`mlp_fase5.ipynb`) refaz as medidas das fases no Kaggle.
