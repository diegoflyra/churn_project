# Plano de implementação: frente do Rafael

21/10/2026 · Rafael Xavier

Este documento detalha o que cabe ao Rafael na [pré-proposta](../pre-proposta.md) e como cada peça será construída.
Os números citados aqui foram medidos em 21/10/2026 sobre a base real, reproduzindo o pré-processamento de `if702-preprocessamento.ipynb` fora do Kaggle.
Onde algo ainda não pôde ser medido, o texto diz isso e indica quando será medido.

## Escopo

Pela divisão de tarefas da pré-proposta, o Rafael é dono de:

| Frente | Entregável | Semana principal |
| --- | --- | --- |
| Base comum | `experimentos/comum.py` com dados, dobras, `preparar`, métricas, limiar do KS, bootstrap e gravação de resultados | 22/10 a 28/10 |
| Formato de resultados | Esquema de `experimentos/resultados/<modelo>.json`, com validação no próprio `comum.py` | 22/10 a 28/10 |
| XGBoost | `experimentos/xgboost.ipynb`, referência da comparação | 29/10 a 18/11 |
| Gradient boosting do scikit-learn | `experimentos/gradient_boosting.ipynb`, linha extra | 29/10 a 18/11 |
| Mitra v2 | `experimentos/mitra.ipynb` | 29/10 a 18/11 |
| Comparação | `experimentos/comparacao.ipynb`, que lê os sete JSON e gera tabelas e gráficos | 19/11 a 25/11 |
| Coordenação | Contratos entre as peças, revisão dos PRs, cronômetro do N de trials junto com o Gabriel A | todo o projeto |
| Apresentação | Pergunta 1 (qual modelo separa melhor e se a diferença é real) e pergunta 6, com o Gabriel B (erros e variáveis em comum) | 26/11 a 01/12 |

Os gráficos padronizados (ROC, PR, KS sobrepostos e importâncias) são do Gabriel B, mas moram no `comum.py`.
O modelo de estudo do Optuna (sampler, pruner, trial 0, SQLite) é do Diego e também mora no `comum.py`.
O Rafael escreve o contrato que as duas peças seguem e revisa os PRs delas.

## O que o repositório tem hoje

- `if702-preprocessamento.ipynb`: o pré-processamento completo, escrito para o Kaggle, com o caminho do CSV fixo em `/kaggle/input/...`.
- `experimentos/*.ipynb`: sete roteiros sem código, um por modelo, escritos no protocolo de blocos (uma mudança por vez).
- `estrutura_dos_experimentos.docx`: o resumo dos sete roteiros.
- Não existe ainda `comum.py`, `resultados/`, `comparacao.ipynb`, arquivo de dependências nem testes.

## Fatos medidos com a base real

Ambiente da medição: Python 3.12.10, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, scipy 1.18.1, optuna 5.0.0, kagglehub 1.0.3, autogluon.tabular 1.6.3, torch 2.13.0 (CPU), num notebook Windows sem GPU.

| Fato | Valor medido | Consequência no plano |
| --- | --- | --- |
| Download pelo `kagglehub` | Funciona sem `kaggle.json`; a versão atual do dataset é a 3 | `comum.py` baixa por `kagglehub.dataset_download(".../versions/3")`, com a versão fixa, e o mesmo código roda local, no Kaggle e no Colab |
| Base bruta | 7.043 × 20, 11 nulos em `TotalCharges` (já vem como `float64`) | O teste de regressão confere essas contagens |
| Base após codificação e variáveis criadas | 7.043 × 29 (28 variáveis e o alvo), churn 26,54% | Idem |
| Divisão treino/teste | Treino 5.282 × 28, teste 1.761 × 28 (467 churn e 1.294 não churn) | O bootstrap do teste trabalha com 1.761 clientes |
| Dobras | Treino de cada dobra com 6.208 linhas após oversampling (50% churn), validação com 1.056 ou 1.057 (26,5% churn) | O teste de regressão confere os tamanhos das cinco dobras |
| Cópias no treino da dobra | 1.996 das 6.208 linhas da dobra 0 são duplicatas do oversampling | Nenhum early stopping pode usar uma fração do treino da dobra como validação (ver gradient boosting) |
| Ordem do `RandomOverSampler` | As primeiras n linhas da saída são as originais, na ordem de entrada; o índice do pandas é perdido | `comum.py` guarda posições, não índices do pandas, e recupera o treino sem cópias como as primeiras n linhas |
| KS por `roc_curve` contra `scipy.stats.ks_2samp` | Diferença máxima de 5,6e-17 em três amostras sintéticas de 1.761 clientes, com empates | A implementação é `max(tpr - fpr)` de `roc_curve`; o teste unitário compara com o scipy |
| XGBoost, um trial de 5 dobras, taxa 0,1 | 2,6 s; KS médio 0,528 ± 0,026 | Um estudo de 100 trials do XGBoost leva minutos; ele não limita o N |
| XGBoost com `reg:squarederror` | Probabilidades de -0,42 a 1,07 | Entropia cruzada só é calculada depois de recortar em [1e-7, 1 - 1e-7]; o KS não muda, porque só depende da ordem |
| XGBoost no padrão (taxa 0,3), paciência 10, 20 e 50 | Mesmo KS (0,5207) e mesmas árvores (8 a 44) nas três paciências | A paciência fixa da pré-proposta (10) basta no XGBoost; a regra de subir para 20 não muda o resultado nele |
| Limiar do KS máximo por dobra | De 0,29 a 0,54 entre dobras (taxa 0,05, paciência 20) | O limiar usado no teste sai das probabilidades fora da dobra concatenadas (5.282 clientes), não de uma dobra só |
| Gradient boosting do scikit-learn com `n_iter_no_change` | KS 0,503 ± 0,023, 24 s por trial | A validação interna (`validation_fraction`) é cortada do treino com cópias, então não é usada |
| Gradient boosting com early stopping na validação da dobra | KS 0,526 ± 0,018, árvores de 90 a 212 | O early stopping é feito pela perda na validação da dobra, como no XGBoost |
| `criterion` do `GradientBoostingClassifier` | Default `"deprecated"` no scikit-learn 1.9.1 | `friedman_mse` do slide 28 não muda o ajuste e não conta entre os 8 hiperparâmetros |
| Bootstrap de 1.000 reamostras do KS | 0,63 s; numa dobra de 1.056 clientes o intervalo de 95% foi [0,502; 0,606] | Diferenças de 1 ou 2 pontos de KS são ruído, como a pré-proposta antecipa; o teste pareado é necessário |
| Optuna 5.0.0 | `MedianPruner(n_startup_trials=5, n_warmup_steps=0, ...)`; `enqueue_trial` fixa o trial 0; `get_param_importances` inclui parâmetros condicionais | O pruner "depois da segunda dobra" é `n_warmup_steps=1` com a dobra como passo; a taxa do fine-tuning do Mitra aparece nas importâncias |
| Mitra v2 no Hugging Face | `autogluon/mitra-classifier-2` existe, público, Apache-2.0 | O checkpoint v2 pode ser usado |
| `MitraClassifier` (autogluon.tabular 1.6.3) | `hf_model` default é `autogluon/mitra-classifier` (v1); `fine_tune=True`, `fine_tune_steps=50`, `lr=1e-4`, `warmup_steps=1000`, `patience=40`, `weight_decay` fixo em 0,1, precisão `bfloat16`, `max_samples_support=8192` | Ver a seção do Mitra |
| Aquecimento do Mitra | `LinearLR` de 1/1000 até 1 em 1.000 passos; com 50 passos a taxa chega só a 5% do valor nominal | `warmup_steps` fica fixo em 10 (receita v2); sem isso a taxa do fine-tuning quase não age |
| `fit` do Mitra sem `X_val` | Separa 20% do treino para validação, mesmo com `fine_tune=False` | O `fit` sempre recebe `X_val` e `y_val` da dobra |
| `predict_proba` do Mitra | Usa como contexto todo o `X` passado ao `fit` | O contexto é o treino inteiro da dobra (6.208 linhas com oversampling, 4.225 ou 4.226 sem), abaixo das 8.192 do teto |
| Checkpoint do fine-tuning do Mitra | Guarda o menor log loss na validação, incluindo o passo 0 (antes de ajustar) | O fine-tuning nunca piora o log loss da validação; isso é otimista do mesmo jeito que o early stopping, e entra na limitação já citada na pré-proposta |
| Mitra v2 em CPU | Ver a seção do Mitra | Ver a seção do Mitra |

## Divergências entre a pré-proposta e os roteiros

Os roteiros em `experimentos/` foram escritos antes da pré-proposta e ainda descrevem o protocolo de blocos.
A pré-proposta decidiu trocar isso pela busca do Optuna com orçamento comum, e manter os blocos só para o registro evolutivo do pré-processamento.
Os pontos abaixo seguem a pré-proposta e precisam ser refletidos nos notebooks quando cada um ganhar código.

| Ponto | Roteiro | Decisão |
| --- | --- | --- |
| Referência | `gradient_boosting.ipynb` escolhe a biblioteca no bloco 0 | XGBoost é a referência, num notebook próprio `xgboost.ipynb`; `gradient_boosting.ipynb` fica com o scikit-learn como linha extra |
| Mitra, passos do fine-tuning | Fixo em 50 | Varia entre os 8 hiperparâmetros, como na tabela da pré-proposta |
| Mitra, `warmup_steps` | Bloco próprio | Fixo em 10, pelo cálculo do aquecimento acima |
| Mitra, `use_random_transforms` | Bloco próprio | Fora da busca, no default (`False`), porque a tabela da pré-proposta já tem 8 itens |
| Mitra, `patience` | Bloco próprio | Fixo em 10, a paciência comum da pré-proposta |

## Frente 1: `comum.py`

### Onde fica

`experimentos/comum.py`, ao lado dos notebooks.
O Jupyter roda cada notebook com a pasta dele como diretório atual, então `import comum` funciona nos oito notebooks sem mexer em `sys.path`.
No Kaggle e no Colab o notebook começa clonando o repositório e entrando em `experimentos/`.

### Contrato

Todas as funções abaixo são as únicas formas de obter dados, dobras e métricas nos notebooks.

| Função | O que faz |
| --- | --- |
| `carregar_bruto()` | Baixa a versão 3 da base pelo `kagglehub` e devolve o `DataFrame` cru |
| `preprocessar(df, criar_variaveis=True)` | Faz o que o notebook de pré-processamento faz: preenche `TotalCharges`, codifica e cria `total_addons` e `PaymentMethod_Automatic`; devolve `X` e `y` |
| `dividir(X, y)` | 75/25 estratificado com `random_state=42` |
| `preparar(X_tr, y_tr, X_aval, padronizar=True, balancear=True)` | O mesmo `preparar` do notebook, com as duas etapas desligáveis |
| `gerar_dobras(X_train, y_train, **opcoes_preparar)` | As cinco dobras estratificadas com `random_state=42`; cada dobra traz `X_tr`, `y_tr`, `X_val`, `y_val`, as posições da validação no treino e `n_original` (linhas antes do oversampling) |
| `ks(y, p)` | KS como `max(tpr - fpr)` de `roc_curve` |
| `limiar_ks(y, p)` | O limiar onde o KS é máximo |
| `metricas(y, p, limiar)` | KS, AUROC, precision, recall, F1, matriz de confusão, MSE e entropia cruzada (com `p` recortado em [1e-7, 1 - 1e-7]) |
| `avaliar_cv(ajustar, dobras, trial=None)` | Para cada dobra chama `ajustar(dobra)`, que devolve uma função `prever(X) -> p`; mede KS, MSE e entropia cruzada no treino sem cópias e na validação; informa ao `trial` a média acumulada do KS a cada dobra e levanta `TrialPruned` quando o pruner pedir; devolve o KS por dobra, as perdas e as probabilidades fora da dobra |
| `bootstrap_ks(y, p, n=1000, semente=42)` | Intervalo de 95% por percentil |
| `bootstrap_pareado(y, p_modelo, p_referencia, n=1000, semente=42)` | Intervalo de 95% e p-valor da diferença de KS, com as mesmas reamostras para os dois modelos |
| `salvar_resultado(modelo, resultado)` e `carregar_resultados()` | Gravam e leem `resultados/<modelo>.json`, validando o esquema |

O treino "sem cópias" de uma dobra são as primeiras `n_original` linhas de `X_tr`, pela ordem medida do `RandomOverSampler`.
Medir MSE e entropia cruzada nele, e não no treino balanceado, deixa treino e validação com a mesma proporção de churn.

`criar_estudo(nome)` e o padrão de trial 0 vêm do Diego, e os gráficos vêm do Gabriel B, os dois sobre esse contrato.

### Como garantir que é igual ao notebook

`tests/test_comum.py`, com pytest, confere:

1. Os números da tabela de fatos: 7.043 × 20, 11 nulos, 28 variáveis, 1.761 no teste com 467 churn, 6.208 linhas no treino de cada dobra e 1.056 ou 1.057 na validação.
2. Que a lista de colunas é a mesma do notebook, na mesma ordem.
3. Que `ks` bate com `scipy.stats.ks_2samp` em amostras com empates.
4. Que `limiar_ks` aplicado devolve um ponto da curva com `tpr - fpr` igual ao KS.
5. Que nenhuma linha de validação aparece no treino da dobra (pelas posições).
6. Que o esquema de resultados rejeita um JSON sem campo obrigatório.

Os testes que baixam a base são marcados para pular quando não há rede.
`ruff` roda sobre `comum.py` e os testes.
As dependências ficam em `requirements.txt`, com as versões da tabela de fatos.

O notebook de pré-processamento continua como está, como documentação das decisões.
O README passa a apontar para `comum.py` como a fonte que os modelos usam.

## Frente 2: formato de `resultados/<modelo>.json`

Um arquivo por modelo, com estes campos obrigatórios:

| Campo | Conteúdo |
| --- | --- |
| `modelo`, `versao_esquema` | Nome curto (`xgboost`, `gb_sklearn`, `mitra`, `mlp`, `stab`, `kan`, `tabkan`, `tabpfn_v2`) e a versão do esquema |
| `ambiente` | Versões das bibliotecas, Python, CPU ou GPU usada |
| `busca` | N de trials, completos, podados, tempo de parede, sampler, pruner e o caminho do SQLite |
| `trial0` | Parâmetros, KS por dobra e KS médio da configuração padrão |
| `melhor` | Número do trial, parâmetros, KS por dobra, média e desvio, MSE e entropia cruzada em treino e validação por dobra |
| `limiar` | Valor do limiar de KS máximo nas probabilidades fora da dobra do melhor trial |
| `teste` | Sementes, e para a média delas: KS com intervalo de 95%, AUROC, precision, recall, F1, MSE e entropia cruzada (treino e teste) e matriz de confusão |
| `predicoes_teste` | Posições dos 1.761 clientes no teste, rótulos e a probabilidade de cada semente |
| `tempos` | Busca, treino final e predição no teste, em segundos |

Com 1.761 probabilidades por semente e 5 sementes, o arquivo fica na casa de 100 KB, o que cabe no Git.
Guardar as probabilidades permite à `comparacao.ipynb` refazer curvas, bootstrap pareado, correlação entre modelos, clientes que todos erram e a média simples dos sete sem rodar nenhum modelo de novo.

## Frente 3: XGBoost (referência)

`XGBClassifier` com `binary:logistic` e `XGBRegressor` com `reg:squarederror`, conforme o hiperparâmetro de objetivo.
O número de árvores sai do early stopping na validação da dobra (`n_estimators=10000`, `early_stopping_rounds=10`, métrica `logloss` ou `rmse` conforme o objetivo).
Peso de classe desligado, porque o oversampling já está nas dobras.

| Hiperparâmetro | Faixa | Trial 0 (padrão do XGBoost) |
| --- | --- | --- |
| `max_depth` | inteiro de 2 a 10 | 6 |
| `min_child_weight` | 0,1 a 20, log | 1 |
| `learning_rate` | 0,01 a 0,3, log | 0,3 |
| `subsample` | 0,5 a 1 | 1 |
| `colsample_bytree` | 0,5 a 1 | 1 |
| `reg_lambda` | 1e-3 a 100, log | 1 |
| `gamma` | 0 a 5 | 0 |
| `objective` | `binary:logistic` ou `reg:squarederror` | `binary:logistic` |

A taxa mínima é 0,01 porque, medido, a taxa 0,05 já pede de 71 a 217 árvores; 0,01 mantém o trial em segundos, e 0,001 empurraria o early stopping para perto do teto de 10.000.

Treino final: `preparar(X_train, y_train, X_test)` e `n_estimators` igual à média de `best_iteration + 1` nas cinco dobras do melhor trial, porque o treino inteiro não tem validação para o early stopping.
`subsample` e `colsample_bytree` abaixo de 1 tornam o modelo estocástico, então o treino final roda com 5 sementes.
SHAP do XGBoost final alimenta a pergunta 6.

## Frente 4: gradient boosting do scikit-learn (linha extra)

O early stopping não usa `n_iter_no_change`, pela medição acima.
O treino cresce com `warm_start=True` de 10 em 10 árvores e para quando o log loss na validação da dobra passa 10 rodadas sem cair, até 10.000 árvores.

| Hiperparâmetro | Faixa | Trial 0 (padrão do scikit-learn) |
| --- | --- | --- |
| `loss` | `log_loss` ou `exponential` | `log_loss` |
| `learning_rate` | 0,01 a 0,3, log | 0,1 |
| `max_depth` | inteiro de 2 a 10 | 3 |
| `min_samples_split` | inteiro de 2 a 50 | 2 |
| `min_samples_leaf` | inteiro de 1 a 50 | 1 |
| `subsample` | 0,5 a 1 | 1 |
| `max_features` | 0,3 a 1 | 1 (todas) |
| `min_impurity_decrease` | 0 a 0,05 | 0 |

`criterion` fica de fora porque está obsoleto no scikit-learn 1.9.1.
O scikit-learn não tem MSE como perda de `GradientBoostingClassifier`, então este modelo não entra na pergunta "entropia cruzada ou MSE".

## Frente 5: Mitra v2

`MitraClassifier(hf_model="autogluon/mitra-classifier-2", ...)`, sempre com `X_val` e `y_val` da dobra no `fit`.

Fixos: `warmup_steps=10`, `patience=10`, `use_random_transforms=False`, `metric="log_loss"`.

| Hiperparâmetro | Faixa | Trial 0 |
| --- | --- | --- |
| `n_estimators` | 1, 2, 4 ou 8 | 1 |
| `random_mirror_x` | sim ou não | sim (default) |
| `shuffle_features` | sim ou não | não (default) |
| `shuffle_classes` | sim ou não | não (default) |
| oversampling do contexto | sim ou não (`preparar(..., balancear=...)`) | sim |
| `fine_tune` | sim ou não | não |
| `lr` (só com fine-tuning) | 1e-6 a 1e-4, log | não sorteado |
| `fine_tune_steps` (só com fine-tuning) | 10 a 100, de 10 em 10 | não sorteado |

O trial 0 é o uso em contexto puro, porque o fine-tuning é o default do código mas não é o uso padrão do artigo.
O teto de `n_estimators` é 8, e não os 32 do artigo, até a medição de tempo dizer que cabe mais.

Treino final: contexto com `preparar(X_train, y_train, X_test, balancear=...)`.
Com oversampling são 7.760 linhas (3.880 não churn duplicadas até o mesmo número de churn), abaixo das 8.192 do teto de contexto.
Se o fine-tuning vencer, o treino inteiro não tem validação; o `fit` recebe o próprio treino sem cópias como `X_val`, e o checkpoint escolhe o menor log loss no treino.
Isso fica registrado como limitação no notebook.

O que ainda precisa ser medido na semana 1, no ambiente com GPU do Gabriel A:

1. O tempo de um trial de 5 dobras em contexto e com fine-tuning de 100 passos com `n_estimators=8`, para fixar N.
2. Se a GPU do Kaggle (P100) ou do Colab (T4) aceita o `autocast` em `bfloat16` que o código fixa; se não aceitar, o Mitra precisa de uma GPU que aceite ou de uma mudança de precisão, e o v1 continua como reserva.
3. Se duas execuções com a mesma semente dão as mesmas probabilidades, porque o código do AutoGluon diz que o `set_seed` foi removido na v1.4; o resultado decide se o Mitra roda com 5 sementes no teste.

Em CPU, no notebook local, o ajuste em contexto de uma única dobra com um estimador não terminou em mais de 10 minutos, contando o download de cerca de 290 MB dos pesos.
A execução foi interrompida porque o sistema ficou sem memória, então não há tempo de CPU medido.
Como um estudo pede 5 dobras por trial e N trials, a CPU local fica descartada para o Mitra, e a medição na GPU passa a ser o primeiro item da semana 1.

## Frente 6: `comparacao.ipynb`

Lê os oito JSON (sete modelos e a linha extra) e gera:

1. Tabela no teste: KS com intervalo, diferença para o XGBoost com intervalo e p-valor do bootstrap pareado, AUROC, F1, MSE e entropia cruzada.
2. Curvas ROC, PR e KS dos modelos sobrepostas.
3. KS do trial 0 contra o melhor trial e convergência do Optuna no mesmo eixo (pergunta do Diego).
4. Importâncias do Optuna agrupadas pelas quatro frentes (pergunta do Gabriel B).
5. Melhor KS com entropia cruzada e com MSE nos modelos treináveis (pergunta do Jeff).
6. KS contra tempo de parede (pergunta do Gabriel A).
7. Correlação entre as probabilidades, clientes que todos erram e a média simples dos sete (pergunta 6).

O notebook é escrito na semana 1 sobre JSON falsos gerados pelos testes, para que o formato seja testado antes de existir resultado real.

## Coordenação

| Com quem | O que precisa ser combinado | Até |
| --- | --- | --- |
| Diego | `criar_estudo` e o uso de `avaliar_cv(trial=...)` no estudo | 24/10 |
| Gabriel A | Ambiente com GPU para TabPFN e Mitra e a medição do trial mais lento | 28/10 |
| Gabriel B | Gráficos sobre as probabilidades do JSON; mesma forma de medir a perda do KAN e do TabKAN | 28/10 |
| Jefferson | Mesma forma de medir a perda do TabKAN e do KAN | 28/10 |
| Todos | Revisão do PR do `comum.py` antes de qualquer notebook usar | 26/10 |

## Cronograma do Rafael

| Período | Entregas |
| --- | --- |
| 22/10 a 28/10 | `comum.py`, testes, `requirements.txt`, esquema do JSON e `comparacao.ipynb` sobre JSON falsos; N fixado com a medição do Mitra |
| 29/10 a 04/11 | `xgboost.ipynb` e `gradient_boosting.ipynb` rodando trial 0 e um estudo curto; `mitra.ipynb` rodando trial 0 na GPU |
| 05/11 a 18/11 | Estudos completos dos três notebooks com N trials e gravação dos JSON |
| 19/11 a 25/11 | Teste final dos três, `comparacao.ipynb` com os oito JSON reais |
| 26/11 a 01/12 | Slides das perguntas 1 e 6 |

## Pendências fora do controle do Rafael

As perguntas para o professor da pré-proposta continuam abertas.
A que mexe direto nesta frente é a quinta: se XGBoost e gradient boosting do scikit-learn são dois modelos ou um só.
Até a resposta, os dois são feitos, e o scikit-learn aparece como linha extra na tabela.
