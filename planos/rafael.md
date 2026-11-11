# Plano de implementação: frente do Rafael

Criado em 21/10/2026 e revisado em 02/11/2026 · Rafael Xavier

Este documento detalha o que cabe ao Rafael na [pré-proposta](../pre-proposta.md) e como cada peça será construída.
A revisão de 02/11 incorpora a comparação com a branch `origin/jeff/stab-tabkan`, que já traz um `comum.py`.

## Situação em 02/11/2026

Nenhum entregável do Rafael existe ainda: nem código comum, nem modelo, nem comparação.
Só este plano existe, e a entrega da base comum (22/10 a 28/10) está atrasada.
O Jefferson já tem STab e TabKAN com resultados e escreveu um `comum.py` para isso, que passa a ser o ponto de partida.

## Escopo

Pela divisão de tarefas da pré-proposta, o Rafael é dono de:

| Frente | Entregável |
| --- | --- |
| Base comum | `comum.py`, evoluído a partir do que o Jefferson escreveu |
| Formato de resultados | Esquema de `resultados/<modelo>.json`, validado no próprio `comum.py` |
| XGBoost | `experimentos/xgboost.ipynb`, referência da comparação |
| Gradient boosting do scikit-learn | `experimentos/gradient_boosting.ipynb`, linha extra |
| Mitra v2 | `experimentos/mitra.ipynb` |
| Comparação | `experimentos/comparacao.ipynb`, que lê os JSON e gera tabelas e gráficos |
| Coordenação | Contratos entre as peças, revisão dos PRs e definição do N de trials com o Gabriel A |
| Apresentação | Pergunta 1 (qual modelo separa melhor e se a diferença é real) e pergunta 6, com o Gabriel B |

Os gráficos padronizados são do Gabriel B, e o modelo de estudo do Optuna é do Diego.
As duas peças moram no `comum.py`, e o Rafael revisa os PRs delas.

## Como ler os números

Cada fato abaixo está em uma de duas tabelas, porque a origem muda o quanto se pode confiar nele.
A primeira traz o que foi executado sobre a base real.
A segunda traz o que foi só lido no código de uma biblioteca, sem execução.

Nenhuma medida usou o conjunto de teste para decidir algo.
O teste só serviu para conferir o alinhamento das saídas do Jefferson e o valor de KS que ele gravou.
Os scripts das medidas foram descartáveis e não estão no repositório.

Ambiente: Python 3.12.10, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, scipy 1.18.1, optuna 5.0.0, kagglehub 1.0.3, em um notebook Windows sem GPU.

### Executado sobre a base real

| Fato | Valor medido | Consequência |
| --- | --- | --- |
| Download pelo `kagglehub` | Funciona sem `kaggle.json`, com a versão 3 fixada | É o caminho padrão do `comum.py` |
| Divisão | Treino 5.282 (1.402 churn, 26,543%), teste 1.761 (467 churn) | Os testes de regressão conferem essas contagens |
| Dobras | Treino com 6.208 linhas após oversampling, validação com 1.056 ou 1.057; 1.996 linhas da dobra 0 são cópias | Nenhuma parada antecipada usa fração do treino já balanceado |
| Treino final | 7.760 linhas com oversampling | Fica abaixo do teto de 8.192 linhas de contexto do Mitra |
| Ordem do `RandomOverSampler` | As primeiras n linhas são as originais, na ordem de entrada | O treino sem cópias é recuperado por posição |
| KS contra `scipy.stats.ks_2samp` | Diferença máxima de 5,6e-17, em três amostras sintéticas com empates | A fórmula `max(tpr - fpr)` está certa; falta repetir a conferência contra os gráficos dos slides |
| XGBoost em `float32` | KS 0,5281 (classificador) e 0,5315 (regressor), iguais aos obtidos com pandas | O formato do `comum.py` do Jefferson serve ao XGBoost |
| XGBoost com `reg:squarederror` | Previsões de -0,42 a 1,07 | Entropia cruzada e correção de prior recortam `p` antes do cálculo; o KS não muda |
| XGBoost, taxa e tempo de um trial de 5 dobras | Ver a tabela da Frente 3 | A taxa mínima da busca é 0,01 |
| Paciência 10 contra 20 no XGBoost | Idênticas com taxa 0,3 e 0,1; com 0,05, 0,01 e 0,001 o KS muda no máximo 0,0017 | Paciência fixa em 10, porque o efeito fica muito abaixo do desvio entre dobras (cerca de 0,02) |
| Limiar do KS máximo por dobra | De 0,29 a 0,54 entre dobras | O limiar sai das probabilidades fora da dobra concatenadas, como o Jefferson já faz |
| Parada do gradient boosting do scikit-learn | Ver a tabela da Frente 4 | A parada usa a validação da dobra |
| Crescimento com `warm_start`, uma árvore por vez | Dobra 0: 1,6 s com taxa 0,1 (101 árvores) e 22,4 s com taxa 0,01 (712 árvores) | Um trial com taxa 0,01 leva perto de 2 minutos; 100 trials cabem em cerca de 3 horas no pior caso medido |
| `criterion` do `GradientBoostingClassifier` | Default `"deprecated"` no scikit-learn 1.9.1 | `friedman_mse` do slide 28 não conta entre os 8 hiperparâmetros |
| Bootstrap de 1.000 reamostras do KS | 0,63 s; intervalo de [0,502; 0,606] em uma dobra de 1.056 clientes | Diferenças de 1 ou 2 pontos podem ser ruído, então o teste pareado é necessário |
| Correção de prior no XGBoost (validação, taxa 0,05) | Entropia cruzada de 0,4710 para 0,4283, MSE de 0,1575 para 0,1385, KS igual em 0,5243 | O JSON guarda as perdas brutas e as corrigidas |
| Perdas no treino com e sem cópias (mesmo modelo) | Entropia cruzada 0,3434 com cópias e 0,3635 sem; MSE 0,1046 e 0,1160 | As perdas de treino são medidas sem cópias |
| Ensemble das dobras contra retreino (XGBoost, com dobra externa como pseudo-teste) | Diferença média de +0,0044 com desvio de 0,0151 em 5 dobras externas; ensemble melhor em 3 de 5 | Sem vantagem sistemática demonstrada, mas não se misturam os dois procedimentos |

### Lido no código, sem execução

| Fato | Onde | Consequência |
| --- | --- | --- |
| `hf_model` default é `autogluon/mitra-classifier` (v1) | `sklearn_interface.py`, autogluon.tabular 1.6.3 | Passar o v2 explicitamente |
| `fine_tune=True`, `fine_tune_steps=50`, `lr=1e-4`, `warmup_steps=1000`, `patience=40`, `weight_decay=0,1`, `max_samples_support=8192` | Idem | Ver a Frente 5 |
| Precisão fixada em `bfloat16` no `autocast` | Idem e `trainer_finetune.py` | Pode não funcionar em toda GPU; precisa de teste |
| Aquecimento linear de 1/1000 até 1 em 1.000 passos | `get_scheduler.py` | Em 50 passos a taxa chega a 5% do nominal, então `warmup_steps` fica em 10 |
| Sem `X_val`, o `fit` separa 20% para validação, mesmo com `fine_tune=False` | `sklearn_interface.py` e `dataset_split.py` (primeira das 5 divisões estratificadas) | O `fit` sempre recebe a validação da dobra |
| `predict_proba` usa como contexto todo o `X` do `fit` | `sklearn_interface.py` | O contexto é o treino inteiro da dobra |
| O checkpoint guarda o menor log loss na validação, incluindo o passo 0 | `trainer_finetune.py` e `callbacks.py` | O fine-tuning não piora o log loss de validação, o que é otimista como qualquer parada antecipada |
| `set_seed` foi removido na v1.4, segundo comentário no código | `sklearn_interface.py` | A reprodutibilidade precisa de teste |

### Ainda sem medida

- **Mitra v2 em qualquer ambiente:** o modelo nunca gerou uma previsão.
  A única tentativa, em CPU e em uma dobra, passou de 10 minutos, contando instalação de pacotes e download de cerca de 290 MB de pesos, e foi interrompida quando a máquina ficou sem memória.
  Isso não dá um tempo de CPU; a escolha de rodar na GPU é uma decisão prudente, não um resultado medido.
- **`bfloat16` e reprodutibilidade do Mitra na GPU.**
- **Tempo do trial mais lento, que define o N.**
- **Tempos do XGBoost e do gradient boosting com hiperparâmetros caros:** as medidas usaram os padrões, então `max_depth` 10 pode custar mais.

## Comparação com a branch do Jefferson

Base da comparação: `origin/jeff/stab-tabkan`, commit `1246312`, de 01/11/2026.
O `comum.py` e os JSON foram lidos e executados; os modelos STab e TabKAN não foram treinados de novo.

### O que está certo e fica

| Item | Evidência |
| --- | --- |
| Pré-processamento | Executei as células do notebook e o `comum.py` sobre o mesmo CSV: colunas iguais (28), rótulos iguais e diferença máxima de 1,2e-7 nas variáveis, que vem do `float32` |
| Teste | Os `y_teste` dos dois JSON são iguais ao teste gerado com a base do Kaggle, posição a posição |
| `ks`, `limiar_ks`, `ce`, `mse`, `metricas`, `bootstrap_ks` | Mesma fórmula deste plano; recalculei o KS dos dois JSON e bati com o gravado (0,5438 e 0,5469) |
| Limiar | Vem das probabilidades fora da dobra, como este plano previa |
| Estudo | `TPESampler(seed=42)`, `MedianPruner(n_startup_trials=5, n_warmup_steps=1)`, média acumulada a cada dobra, corte só depois da segunda dobra, trial 0 enfileirado e SQLite |
| Campos úteis do JSON | `importancias`, `convergencia` e `por_perda` evitam que a comparação dependa de abrir os arquivos SQLite |
| Correção de prior | Não estava neste plano e passa a fazer parte dele |

### O que falta no `comum.py` dele

| Lacuna | Quem precisa | Mudança |
| --- | --- | --- |
| Sem CSV local o módulo falha (`FileNotFoundError` em `dados/telco_ibm.csv`, que não está no repositório) | Todos | `CHURN_CSV`, se definida, e senão `kagglehub` na versão 3 |
| `preparar` sempre padroniza e balanceia | Mitra e TabPFN (oversampling do contexto), Diego (registro evolutivo) | Parâmetros `padronizar` e `balancear` |
| Não dá para desligar as variáveis criadas | Diego | Parâmetro `criar_variaveis` |
| Sem bootstrap pareado (o dele está dentro de `avaliar_jeff.py`) | Comparação | `bootstrap_pareado` |
| `corrige_prior` está em `avaliar_jeff.py` e não recorta `p` | Todos, em especial XGBoost com MSE | Levar para o `comum.py`, com recorte |
| Sem laço de validação cruzada comum | XGBoost, gradient boosting, Mitra e MLP | `avaliar_cv` |
| Sem validação do JSON | Todos | `salvar_resultado` e `carregar_resultados` |
| Laço `for ... pass` sem efeito em `carregar_df` | Limpeza | Remover |

### Onde a branch diverge da pré-proposta

| Ponto | Pré-proposta | Branch | Encaminhamento |
| --- | --- | --- | --- |
| Avaliação final | Retreino no treino inteiro com `preparar`, com 5 sementes | Média de teste dos modelos das 5 dobras, em 5 sementes (25 modelos); o teste usa o scaler do treino inteiro, e cada modelo foi treinado com o scaler da própria dobra | Refazer com retreino, para todos seguirem o mesmo procedimento |
| Orçamento | Mesmo N para todos, proposta de 100 | 30 trials nos dois (21 completos e 9 podados no STab, 15 e 15 no TabKAN) | Fechar o N antes de novos estudos e refazer os dois com ele |
| Largura do STab | 64 a 512 | `[32, 64, 128]` em CPU e `[64, 128, 256]` em GPU | Usar a mesma faixa nos dois ambientes |
| Épocas | Até 10.000 com paciência | Tetos padrão de 30 épocas (`stab_pretreino`) e 100 passos (`tabkan_fit_predict`) nas assinaturas | Jefferson confirma se o teto limita os resultados |
| Perdas de treino | MSE em treino, validação e teste | Medidas no treino com cópias | Medir sem cópias |
| Reexecução | Notebooks e estudos salvos | Os notebooks abrem `resultados/<modelo>.db`, mas nenhum `.db` está na branch; eles também importam `comum` sem ajustar `sys.path` | Versionar os SQLite e padronizar a importação |
| Organização | Repositório único | Tudo dentro de uma pasta `churn_project/`, sem a pré-proposta nem este plano | Mover para a raiz antes do merge |

O experimento `ce_vs_mse` do Jefferson é mais detalhado que os demais vão ter.
A comparação entre modelos usa `por_perda`, que todos terão, e o `ce_vs_mse` fica como campo opcional.

## Frente 1: `comum.py`

### Decisão

O `comum.py` do Jefferson é a base, porque já funciona, gera os mesmos dados do notebook e tem dois modelos usando.
Os nomes e os retornos atuais são mantidos, para que `modelos_jeff.py` e `avaliar_jeff.py` continuem funcionando, e tudo o que entra é acréscimo.

O arquivo fica na raiz do repositório, como na organização dele, e os resultados em `resultados/`.
Isso substitui a versão anterior deste plano, que colocava tudo em `experimentos/`.
Cada notebook começa com `sys.path.insert(0, "..")`, e o `comum.py` resolve `resultados/` a partir do próprio arquivo, então o diretório atual não importa.

### Contrato

| Função | Situação | O que faz |
| --- | --- | --- |
| `carregar_df(path=None, criar_variaveis=True)` | Ajustar | Lê `path`, `CHURN_CSV` ou o `kagglehub` na versão 3, e pré-processa |
| `preparar(X_tr, y_tr, X_aval, padronizar=True, balancear=True)` | Ajustar | O `preparar` do notebook, com as etapas desligáveis |
| `carregar_dobras(path=None, **opcoes)` | Ajustar | Mantém o retorno `(folds, final, colunas)`, em `float32` |
| `posicoes_dobras(path=None)` | Nova | Devolve `(idx_tr, idx_val)` de cada dobra; `len(idx_tr)` é o tamanho do treino sem cópias |
| `ks`, `limiar_ks`, `ce`, `mse` | Pronta | Sem mudança de fórmula |
| `metricas(y, p, limiar)` | Pronta | Mantém as chaves atuais, como `confusao` |
| `corrige_prior(p, pi=TAXA_TREINO)` | Mover | Com recorte de `p` e a taxa do treino (0,26543), aplicada só a modelos treinados com oversampling |
| `bootstrap_ks(y, p, n=1000, seed=SEMENTE)` | Ajustar | Intervalo de 95% por percentil; o padrão da semente passa de 0 para 42 |
| `bootstrap_pareado(y, p_modelo, p_ref, n=1000, seed=SEMENTE)` | Nova | Intervalo de 95% e p-valor bilateral da diferença de KS, com as mesmas reamostras |
| `avaliar_cv(ajustar, folds, trial=None)` | Nova | `ajustar(dobra)` devolve `prever(X)`; reporta a média acumulada, corta só depois da segunda dobra e devolve KS, perdas sem cópias e probabilidades fora da dobra |
| `salvar_resultado(modelo, resultado)` e `carregar_resultados()` | Novas | Gravam e leem o JSON, validando o esquema e o alinhamento de `y_teste` |

`criar_estudo(nome)` vem do Diego, partindo de `estudo()` em `modelos_jeff.py`, que já segue a pré-proposta.
Os gráficos vêm do Gabriel B, sobre as probabilidades do JSON.

### Testes

`tests/test_comum.py`, com pytest, confere:

1. As contagens da tabela de fatos e a lista de colunas.
2. Que a saída é igual à das células do notebook, com tolerância de `float32`.
3. Que `ks` bate com `scipy.stats.ks_2samp` em amostras com empates.
4. Que `preparar(..., balancear=False)` não duplica linhas e que nenhuma posição de validação aparece no treino da dobra.
5. Que `corrige_prior` aceita previsões fora de [0, 1] e não muda o KS.
6. Que `bootstrap_pareado` de um modelo contra ele mesmo dá diferença zero.
7. Que o esquema rejeita um JSON sem campo obrigatório ou com `y_teste` desalinhado.

Os testes que baixam a base pulam quando não há rede.
`ruff` roda sobre `comum.py` e os testes, e as dependências ficam em `requirements.txt`, com as versões acima.
O notebook de pré-processamento continua como documentação, e o README passa a apontar para o `comum.py`.

## Frente 2: formato de `resultados/<modelo>.json`

O esquema parte dos nomes que o Jefferson já grava, para evitar retrabalho.

| Campo | Situação | Conteúdo |
| --- | --- | --- |
| `modelo`, `melhor_config`, `ks_dobras_melhor` | Existe | Nome curto, parâmetros e KS por dobra do melhor trial |
| `estudo` | Existe | `n_completos`, `n_podados`, `por_perda`, `importancias`, `ks_trial0`, `melhor_ks`, `convergencia` |
| `teste` | Existe | `ks`, `ks_ic95`, `auroc`, `ce`, `mse`, `ce_corr`, `mse_corr`, `precision`, `recall`, `f1`, `confusao`, `limiar` |
| `probabilidades_teste`, `y_teste` | Existe | Média das sementes e rótulos, na ordem fixa do teste |
| `ce_vs_mse` | Opcional | Experimento controlado, quando houver |
| `versao_esquema`, `ambiente` | Novo | Versão, bibliotecas, Python e CPU ou GPU |
| `estudo.n_trials`, `estudo.config_trial0`, `estudo.tempo_parede_s` | Novo | Orçamento, padrão e tempo real do estudo |
| `estudo.melhor_por_categoria` | Novo | Melhor KS por valor de cada variável categórica, para as perguntas de ativação e otimizador |
| `perdas_dobras` | Novo | MSE e entropia cruzada, brutas e corrigidas, em treino sem cópias e validação |
| `teste.ks_por_semente`, `probabilidades_por_semente` | Novo | Variação entre sementes |
| `avaliacao_final` | Novo | Procedimento usado, sempre `retreino` depois do acerto com o Jefferson |
| `tempos` | Novo | Treino final e predição no teste, em segundos |

Os JSON atuais têm cerca de 53 KB, com uma coluna de probabilidades.
Com 5 sementes, a estimativa fica perto de 250 KB por modelo, o que ainda cabe no Git.
As posições dos clientes não são gravadas: a ordem do teste é fixa e `salvar_resultado` confere `y_teste`.

## Frente 3: XGBoost (referência)

`XGBClassifier` com `binary:logistic` e `XGBRegressor` com `reg:squarederror`, conforme o hiperparâmetro de objetivo.
O número de árvores sai da parada antecipada na validação da dobra, com teto de 10.000, paciência 10 e métrica `logloss` ou `rmse`.
Peso de classe fica desligado, porque o oversampling já está nas dobras.

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

A taxa mínima vem desta medida, com os demais parâmetros no padrão e paciência 10:

| Taxa | KS médio | Árvores por dobra | Tempo do trial |
| --- | --- | --- | --- |
| 0,3 | 0,5207 | 9 a 45 | 1,2 s |
| 0,1 | 0,5281 | 33 a 102 | 2,5 s |
| 0,05 | 0,5243 | 72 a 194 | 4,0 s |
| 0,01 | 0,5321 | 304 a 1.013 | 17,1 s |
| 0,001 | 0,5296 | 2.939 a 4.810 | 117,6 s |

A taxa 0,001 custa 7 vezes mais que 0,01 e não melhora o KS, então fica fora da busca.
Nenhum caso chegou ao teto de 10.000 árvores.

Treino final: `final` do `carregar_dobras`, com `n_estimators` igual à média de `best_iteration + 1` nas cinco dobras do melhor trial e 5 sementes, porque o treino inteiro não tem validação para a parada.
SHAP do modelo final alimenta a pergunta 6.

## Frente 4: gradient boosting do scikit-learn (linha extra)

A parada não usa `n_iter_no_change`, pela medida abaixo, feita com taxa 0,1 e paciência 10:

| Parada | KS médio | Árvores por dobra | Validação da parada |
| --- | --- | --- | --- |
| `n_iter_no_change` | 0,5160 | 228 a 667 | 10% do treino já balanceado: 50% churn e com cópias |
| 10% separados antes do oversampling | 0,5419 | 54 a 97 | Sem cópias e com a proporção real |
| Validação da dobra | 0,5294 | 49 a 101 | A mesma que mede o KS |

A segunda linha ganha da primeira em todas as 5 dobras, com média de +0,0259, e as duas treinam em volume parecido de linhas.
A diferença está na validação da parada, mas a medida não separa o efeito das cópias do efeito da proporção de classes.

A escolha é a terceira linha, a mesma regra do XGBoost e dos demais modelos da pré-proposta.
A segunda teve KS maior, mas a diferença (0,0124) fica abaixo do desvio entre dobras (cerca de 0,019), e ela venceu só em 3 das 5 dobras.

O treino cresce com `warm_start=True`, uma árvore por vez, e para depois de 10 árvores sem queda do log loss na validação da dobra.
As previsões usam só as árvores até a melhor, por `staged_predict_proba`.

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

`criterion` fica de fora porque está obsoleto.
O scikit-learn não tem MSE como perda deste classificador, então o modelo não entra na pergunta "entropia cruzada ou MSE".

## Frente 5: Mitra v2

Nada desta frente foi executado até o fim, e os parâmetros abaixo vêm da pré-proposta e da leitura do código.

`MitraClassifier(hf_model="autogluon/mitra-classifier-2", ...)`, sempre com a validação da dobra no `fit`.
Os rótulos em `float32` do `comum.py` são convertidos para inteiro antes do `fit`, por precaução, já que o caminho do Mitra com `float32` não foi testado.

Fixos: `warmup_steps=10`, `patience=10`, `use_random_transforms=False`, `metric="log_loss"`.

| Hiperparâmetro | Faixa | Trial 0 |
| --- | --- | --- |
| `n_estimators` | 1, 2, 4 ou 8 | 1 |
| `random_mirror_x` | sim ou não | sim (default) |
| `shuffle_features` | sim ou não | não (default) |
| `shuffle_classes` | sim ou não | não (default) |
| oversampling do contexto | sim ou não (`carregar_dobras(balancear=...)`) | sim |
| `fine_tune` | sim ou não | não |
| `lr` (só com fine-tuning) | 1e-6 a 1e-4, log | não sorteado |
| `fine_tune_steps` (só com fine-tuning) | 10 a 100, de 10 em 10 | não sorteado |

Divergências em relação ao roteiro `mitra.ipynb`, seguindo a pré-proposta: `fine_tune_steps` varia, enquanto `warmup_steps`, `patience` e `use_random_transforms` ficam fixos.

Treino final: contexto com o treino inteiro, de 7.760 linhas com oversampling ou 5.282 sem, conforme o melhor trial.
Se o fine-tuning vencer, o `fit` recebe o próprio treino sem cópias como `X_val`, e o checkpoint escolhe o menor log loss no treino.
Essa escolha fica registrada como limitação no notebook.

Primeiras medidas, no ambiente com GPU do Gabriel A:

1. Um trial de 5 dobras em contexto, e outro com fine-tuning de 100 passos e `n_estimators=8`, para fixar o N.
2. Se a GPU disponível aceita o `autocast` em `bfloat16`; se não aceitar, é preciso outra GPU ou outra precisão, com o v1 como reserva.
3. Se duas execuções com a mesma semente dão as mesmas probabilidades, o que decide se o Mitra roda com 5 sementes no teste.

## Frente 6: `comparacao.ipynb`

Lê os oito JSON (sete modelos e a linha extra) e gera:

1. Tabela no teste: KS com intervalo, diferença para o XGBoost com intervalo e p-valor do bootstrap pareado, AUROC, F1, MSE e entropia cruzada, brutas e corrigidas.
2. Curvas ROC, PR e KS sobrepostas.
3. KS do trial 0 contra o melhor trial e convergência no mesmo eixo.
4. Importâncias agrupadas pelas quatro frentes, com o aviso de que parâmetros condicionais pouco sorteados podem aparecer com importância 0, como `lr_adam` e `base_fourier` no TabKAN.
5. Melhor KS com entropia cruzada e com MSE nos modelos treináveis.
6. KS contra tempo de parede.
7. Correlação entre as probabilidades, clientes que todos erram e a média simples dos modelos.

O desenvolvimento começa sobre os JSON do Jefferson, convertidos ao esquema final, e sobre JSON falsos gerados nos testes.

## Coordenação

As datas abaixo são propostas e ainda não foram combinadas com ninguém.

| Com quem | O que combinar | Até |
| --- | --- | --- |
| Jefferson | Adotar o `comum.py` dele como base, mover a pasta para a raiz e ajustar o que está na tabela de divergências | 03/11 |
| Diego | Promover `estudo()` a `criar_estudo` e combinar o que muda para o registro evolutivo | 03/11 |
| Gabriel A | Ambiente com GPU e medição do trial mais lento, para fixar o N | 04/11 |
| Gabriel B | Gráficos sobre as probabilidades do JSON e mesma forma de medir a perda entre KAN e TabKAN | 04/11 |
| Todos | Revisão do PR do `comum.py` e do esquema, antes de novos estudos | 04/11 |

## Decisões que dependem do Rafael ou da equipe

1. **Avaliação final:** este plano segue a pré-proposta (retreino) e pede ao Jefferson para refazer a dele.
   A medida com XGBoost não mostrou vantagem sistemática do ensemble, mas vale só para esse modelo e essa configuração, e as redes podem se comportar de outro jeito.
   A alternativa é a equipe inteira adotar o ensemble e mudar a pré-proposta; o que não pode é cada modelo usar um procedimento.
2. **N de trials:** a pré-proposta sugere 100, a branch do Jefferson usou 30, e o valor só fecha depois da medição do Mitra.
3. **XGBoost e gradient boosting como um ou dois modelos:** é a quinta pergunta para o professor, ainda aberta.
   Até a resposta, os dois são feitos, e o scikit-learn aparece como linha extra.

## Cronograma do Rafael

A primeira etapa está atrasada e entra junto com a criação dos modelos.
As demais seguem as datas da pré-proposta.

| Período | Entregas |
| --- | --- |
| 02/11 a 04/11 | `comum.py`, testes, `requirements.txt` e esquema; `xgboost.ipynb` e `gradient_boosting.ipynb` rodando o trial 0; primeira medição do Mitra na GPU e definição do N |
| 05/11 a 18/11 | Estudos completos dos três notebooks e rascunho da `comparacao.ipynb` |
| 19/11 a 25/11 | Teste final dos três e `comparacao.ipynb` com os oito JSON reais |
| 26/11 a 01/12 | Slides das perguntas 1 e 6 |
