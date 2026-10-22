# Pré-proposta de Projeto: Previsão de Churn
 
21/10/2026 · Rafael Xavier
 
A equipe propõe comparar sete classificadores na previsão de churn da base Telco do Kaggle, com o KS como métrica principal e o XGBoost como referência. Todos os modelos recebem a mesma busca de hiperparâmetros (8 hiperparâmetros e o mesmo número de trials no Optuna), e os resultados são discutidos juntos, organizados por perguntas que atravessam os sete modelos. O pré-processamento já está pronto no repositório; falta o código de treino e avaliação.
 
## Identificação
 
- **Disciplina:** IF702, Redes Neurais (CIn-UFPE), prof. Germano C. Vasconcelos
- **Tema:** previsão de churn (abandono) em telecom, classificação binária
- **Repositório de partida:** churn_project (pré-processamento + roteiros dos sete experimentos)
- **Equipe:** Diego Lyra, Gabriel Almeida, Gabriel Bezerra, Jefferson Marcelo e Rafael Xavier
## Problema, objetivos e base de dados
 
O objetivo é prever se um cliente vai cancelar o serviço e descobrir qual dos sete modelos separa melhor quem cancela de quem fica, medido pelo KS no conjunto de teste.
 
**Base:** [Customers churned in telecom services](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services) (Kaggle), 7.043 clientes, 19 variáveis independentes e o alvo `Churn`, com cerca de 26,5% de cancelamentos.
 
Objetivos específicos:
 
1. Treinar e ajustar os sete classificadores pedidos pelo professor sobre as mesmas dobras e o mesmo teste.
2. Dar a todos os modelos a mesma busca extensiva de hiperparâmetros: 8 hiperparâmetros por modelo e o mesmo número de trials no Optuna.
3. Investigar topologias e hiperparâmetros (camadas, unidades, taxa de aprendizagem, ativação, otimizador, dropout, regularização) dentro dessa busca.
4. Comparar entropia cruzada e MSE como função de treinamento em todos os modelos treináveis e medir o MSE em treino, validação e teste.
5. Registrar como cada etapa do pré-processamento muda o desempenho.
6. Discutir os sete modelos em conjunto, numa apresentação com todo o grupo organizada por perguntas comuns, e não modelo a modelo.
**O que o repositório já tem** (`if702-preprocessamento.ipynb`): os 11 `TotalCharges` ausentes preenchidos com 0 (clientes com `tenure` = 0), checagem de inconsistências e de outliers por IQR (nenhum encontrado), codificação binária e one-hot, as variáveis criadas `total_addons` e `PaymentMethod_Automatic`, divisão 75/25 estratificada e cinco dobras estratificadas. Em cada dobra o `StandardScaler` e o `RandomOverSampler` são ajustados só no treino da dobra, o que evita o vazamento de dados que o professor proíbe.
 
## Metodologia
 
Todos os modelos usam as mesmas cinco dobras de `if702-preprocessamento.ipynb`, o mesmo teste e o mesmo código de métricas. A busca de hiperparâmetros é conjunta, feita pelo Optuna com orçamento igual para todos (abaixo), e o teste só é usado no fim, com a configuração congelada. O protocolo de blocos do `estrutura_dos_experimentos.docx` (uma mudança por vez) fica para o registro evolutivo do pré-processamento, porque mudar um hiperparâmetro por vez daria a cada modelo uma busca de tamanho diferente (de cerca de 8 blocos no MLP a cerca de 17 no TabPFN v2).
 
### Modelos
 
| Modelo | Tipo | Biblioteca prevista | Ponto de atenção |
| --- | --- | --- | --- |
| XGBoost | Árvores em boosting (referência) | xgboost | Recomendado pelo professor para começar, junto com o MLP |
| Gradient boosting (linha extra) | Árvores em boosting | scikit-learn | Parâmetros do slide 28 (`log_loss`, antes `deviance`; `friedman_mse`) |
| MLP | Rede treinada do zero | Keras | 1 ou 2 camadas, começando com 1 camada e 10 unidades; Adadelta entra, L-BFGS sai (o Keras não tem) |
| STab | Dois MLPs com pesos compartilhados | Implementação do artigo | O artigo não tem atenção; confirmar com o professor |
| TabPFN v2 | Transformer pré-treinado, em contexto | tabpfn | Carregar o checkpoint v2 explicitamente; oversampling do contexto como hiperparâmetro |
| KAN 2.0 | Rede de Kolmogorov-Arnold | pykan 0.2.8 | Ativações pedidas não existem como base; usar silu |
| TabKAN | Famílias de KAN para tabela | Código do artigo | O artigo já usa esta base (Blastchar) |
| Mitra v2 | Modelo de fundação da Amazon | AutoGluon | Fine-tuning e oversampling do contexto como hiperparâmetros |
 
Os sete modelos são desenvolvidos em paralelo, cada um por seu responsável, sobre a mesma base comum. MLP e XGBoost, recomendados pelo professor como ponto de partida, servem de checagem rápida de que o pipeline comum funciona.
 
### Métricas
 
O KS (Kolmogorov-Smirnov) é a métrica principal e o critério da busca: cada trial vale o KS médio nas cinco dobras. Todos os modelos também reportam MSE e entropia cruzada (treino, validação e teste), matriz de confusão, AUROC, precision, recall, F1, curva ROC e curva precision-recall. Matriz de confusão, precision, recall e F1 usam o limiar do ponto de KS máximo, escolhido na validação e aplicado no teste.
 
A função de KS será conferida contra `scipy.stats.ks_2samp` e contra os gráficos dos slides antes de qualquer resultado ser usado. Todo esse código fica num módulo comum, `comum.py`, que os sete notebooks importam.
 
### Busca de hiperparâmetros com orçamento comum
 
Para que a comparação seja justa, os sete modelos recebem a mesma busca: **8 hiperparâmetros variados por modelo**, distribuídos nas mesmas quatro frentes, e **o mesmo número de trials no Optuna**.
 
| Modelo | Arquitetura / capacidade | Otimização | Regularização | Perda ou dados |
| --- | --- | --- | --- | --- |
| MLP | camadas {1, 2}; unidades {4 a 128}; ativação {logística, tanh, ReLU} | otimizador {Adam, RMSprop, SGD+momento, Adadelta}; taxa {1e-4 a 1e-1, log} | dropout {0 a 0,5}; L2 {1e-6 a 1e-2, log} | perda {CE, MSE} |
| STab | profundidade do codificador {1 a 4}; largura {64 a 512}; ativação {logística, tanh, ReLU} | otimizador {Adam, SGD}; taxa | p da máscara; weight decay | perda da cabeça {CE, MSE} |
| KAN 2.0 | largura; profundidade; intervalos da grade; ordem do spline | otimizador {LBFGS, Adam}; taxa | penalidade `lamb` | perda {CE, MSE} |
| TabKAN | família; profundidade; largura; parâmetro da base | otimizador {LBFGS, Adam}; taxa | L2 | perda {CE, MSE} |
| XGBoost | `max_depth`; `min_child_weight` | `learning_rate`; `subsample` | `colsample_bytree`; `reg_lambda`; `gamma` | objetivo {`binary:logistic` (CE), `reg:squarederror` (MSE)} |
| TabPFN v2 | `n_estimators`; temperatura do softmax; média antes do softmax | fine-tuning {não, sim}; taxa do fine-tuning | balanceamento de probabilidades; tamanho do contexto | oversampling do contexto {não, sim} |
| Mitra v2 | `n_estimators`; espelho de colunas | fine-tuning {não, sim}; taxa do fine-tuning; passos do fine-tuning | embaralhar colunas; embaralhar classes | oversampling do contexto {não, sim} |
 
Os nomes exatos dos argumentos de TabPFN, Mitra e TabKAN serão conferidos na versão instalada. Taxa e passos de fine-tuning só contam quando o fine-tuning está ligado. O gradient boosting do scikit-learn, como linha extra, varia os parâmetros do slide 28 com o mesmo orçamento.
 
**Fixos e iguais para todos:** as mesmas 5 dobras, o mesmo teste e a mesma semente; no máximo 10.000 épocas, com paciência 10, subindo para 20 se a parada for precoce; lote fixo nos modelos treináveis. No XGBoost, o número de árvores sai do early stopping e não conta entre os 8.
 
**Orçamento:**
 
1. Mesmo N de trials para todos, com `TPESampler(seed=42)`. A proposta é N = 100. Na primeira semana, um trial do modelo mais lento (provável: Mitra com fine-tuning) é cronometrado; se 100 não couber, N baixa para todos, nunca só para um.
2. Cada trial roda as 5 dobras e devolve o KS médio. O mesmo `MedianPruner` corta, depois da segunda dobra, trials abaixo da mediana.
3. O trial 0 é sempre a configuração padrão do modelo (no MLP, 1 camada com 10 unidades), para medir quanto a busca ganhou.
4. O tempo de parede de cada estudo é registrado.
5. Os estudos ficam salvos em SQLite, o que serve de backup automático e permite retomar.
A investigação de ativação, otimizador, camadas e CE contra MSE pedida nos slides sai do próprio estudo, sem rodadas extras: o melhor KS para cada valor de cada variável categórica e a importância dos hiperparâmetros calculada pelo Optuna.
 
### Avaliação final
 
A melhor configuração de cada modelo é treinada no treino inteiro com `preparar` e avaliada uma única vez no teste. Modelos estocásticos rodam com 5 sementes. O KS no teste sai com intervalo de 95% por bootstrap (1.000 reamostras), e a diferença de cada modelo para o XGBoost é testada por bootstrap pareado, porque com cerca de 1.760 clientes no teste diferenças de 1 ou 2 pontos de KS podem ser ruído.
 
O early stopping usa a mesma dobra de validação que escolhe a configuração, o que deixa o KS de validação um pouco otimista. O teste separado corrige isso, e a apresentação cita o ponto como limitação.
 
## Discussão integrada dos resultados
 
O projeto é apresentado como uma comparação única, e não como sete projetos. A integração começa no código:
 
- **`comum.py`:** dados, dobras, `preparar`, métricas, limiar do KS e gráficos. Nenhum notebook reimplementa isso.
- **Formato único de saída:** cada notebook grava `resultados/<modelo>.json` com a melhor configuração, o KS por dobra, as métricas no teste com intervalo, os tempos e as probabilidades previstas no teste.
- **`comparacao.ipynb`:** lê os sete arquivos e gera todas as tabelas e gráficos da apresentação.
A apresentação é organizada por perguntas que atravessam os sete modelos. Cada pessoa apresenta uma pergunta, falando de todos os modelos.
 
| Pergunta | Evidência | Apresenta |
| --- | --- | --- |
| Qual modelo separa melhor quem cancela? A diferença é real? | KS no teste com intervalo de 95%; curvas ROC, PR e KS dos sete sobrepostas | Rafael |
| Quanto a busca ajudou cada modelo? | KS do trial 0 (padrão) contra o melhor trial; convergência do Optuna dos sete no mesmo eixo | Diego |
| Quais hiperparâmetros importam, e isso se repete entre famílias? | Importâncias do Optuna agrupadas pelas quatro frentes; melhor KS por ativação e por otimizador | Gabriel B |
| Entropia cruzada ou MSE? | Melhor KS com CE e com MSE nos cinco modelos treináveis, lado a lado | Jeff |
| Modelos pré-treinados valem o custo? | KS contra tempo de parede dos sete; efeito do fine-tuning e do oversampling no contexto | Gabriel A |
| Os modelos erram nos mesmos clientes e usam as mesmas variáveis? | Correlação entre as probabilidades dos sete; clientes que todos erram; variáveis mais importantes (SHAP no XGBoost, funções do KAN) | Rafael e Gabriel B |
 
Com as probabilidades salvas, uma média simples dos sete modelos sai sem custo extra como resultado adicional.
 
**Roteiro da apresentação:**
 
1. Problema, base e métrica (KS), com a conferência da função de KS.
2. Pré-processamento e o registro evolutivo (o que cada etapa mudou no MLP e no XGBoost).
3. Protocolo comum: mesmas dobras, mesmo orçamento e a tabela dos 8 hiperparâmetros por modelo.
4. Os sete modelos em um slide, uma linha cada.
5. As seis perguntas da tabela acima.
6. Conclusões e limitações.
Nenhum slide traz o resultado de um modelo sozinho; o detalhe de cada um fica no notebook e num apêndice.
 
## Divisão de tarefas
 
Cada pessoa fica dona de uma frente de modelos e de uma peça comum, para que os sete notebooks avancem em paralelo sobre a mesma avaliação. A divisão é uma sugestão; a equipe pode trocar os nomes.
 
| Pessoa | Modelos | Peça comum | Pergunta na apresentação |
| --- | --- | --- | --- |
| Rafael | XGBoost (referência, com o GB do scikit-learn como linha extra) e Mitra v2 | Coordenação, `comum.py`, formato de resultados e `comparacao.ipynb` | Qual modelo separa melhor; erros e variáveis em comum |
| Jefferson | STab e TabKAN | Montagem do PPT | Entropia cruzada ou MSE |
| Diego | MLP | Modelo de estudo do Optuna (sampler, pruner, trial 0, SQLite) e registro evolutivo do pré-processamento | Quanto a busca ajudou cada modelo |
| Gabriel A | TabPFN v2 | Ambiente com GPU (Kaggle ou Colab) e cronômetro de um trial para fixar N | Modelos pré-treinados valem o custo |
| Gabriel B | KAN 2.0 | Gráficos padronizados (ROC, PR, KS sobrepostos e importâncias) | Quais hiperparâmetros importam; erros e variáveis em comum |
 
Rafael e Jeff ficam com dois modelos cada. O Mitra foi para Rafael porque, usado em contexto, pede pouco código próprio e equilibra a carga de coordenação. STab e TabKAN ficaram com Jeff porque os dois saem da implementação de um artigo, em PyTorch. Para equilibrar, o registro evolutivo do pré-processamento passou de Jeff para Diego, que já treina o MLP usado nele. TabPFN (Gabriel A) e Mitra (Rafael) dividem o mesmo ambiente com GPU, e KAN (Gabriel B) e TabKAN (Jeff) devem combinar a mesma forma de medir a perda.
 
## Cronograma
 
| Etapa | Período | Duração |
| --- | --- | --- |
| Preparação da base comum | 22/10 a 28/10 | 1 semana |
| Criação dos modelos | 29/10 a 04/11 | 1 semana |
| Treinamento e busca de hiperparâmetros | 05/11 a 18/11 | 2 semanas |
| Teste e comparação dos resultados | 19/11 a 25/11 | 1 semana |
| Apresentação e entrega | 26/11 a 01/12 | 6 dias |
 
Os sete modelos são criados e treinados em paralelo. A execução termina em 25/11/2026 e a entrega do projeto é em 01/12/2026.
 
## Entregáveis
 
Em 01/12/2026 a equipe entrega o código no GitHub e uma apresentação em PPT, feita por todos os cinco, com a estrutura experimental e a interpretação dos resultados.
 
- Sete notebooks executados, cada um com o estudo do Optuna salvo (8 hiperparâmetros, N trials) e a melhor configuração com média e dispersão nas cinco dobras
- `comum.py` (dados, dobras, métricas, limiar e gráficos) e `resultados/<modelo>.json` no mesmo formato para os sete
- `comparacao.ipynb` com a tabela comparativa no teste (KS com intervalo, AUROC, F1, MSE, entropia cruzada) e os gráficos sobrepostos
- Registro evolutivo do pré-processamento
- PPT final organizado pelas seis perguntas comuns
## Riscos
 
| Risco | Efeito | Mitigação |
| --- | --- | --- |
| Modelos pré-treinados (TabPFN, Mitra) exigem GPU | Treino lento ou impossível localmente | Rodar no Kaggle ou Colab; Mitra v1 como reserva |
| O orçamento de N trials não cabe no tempo do modelo mais lento | Busca desigual ou atraso | Cronometrar na semana 1 e baixar N para todos |
| Código de STab e TabKAN pouco documentado | Implementação atrasa | Começar na etapa de criação dos modelos, em paralelo com os demais |
| KS calculado errado | Comparação inválida | Conferir contra scipy e os slides antes de usar |
| Diferenças pequenas de KS entre modelos | Conclusão sobre o vencedor baseada em ruído | Intervalo por bootstrap e teste pareado contra o XGBoost |
| CSV fora do repositório | Notebooks não rodam igual para todos | Baixar via kagglehub com caminho fixo |
| Perda de resultados | Retrabalho | Estudos do Optuna em SQLite e commits frequentes |
 
## Perguntas para o professor
 
- [ ] A busca comum (8 hiperparâmetros por modelo e o mesmo N de trials, com N = 100 como proposta) atende ao pedido de busca extensiva igual para todos?
- [ ] A apresentação organizada por perguntas comuns aos sete modelos atende ao pedido de discussão integrada?
- [ ] O STab pedido é o artigo de Hajiramezanali et al. (2022), que não tem atenção, ou outro modelo baseado em Transformer?
- [ ] Como o KAN 2.0 não aceita logística, tangente ou ReLU como função de base, basta variar as bases disponíveis?
- [ ] XGBoost e o gradient boosting do scikit-learn devem ser dois modelos, ou basta um como referência?
## Referências
 
- Vasconcelos, G. C. *Projeto da Disciplina*, IF702, CIn-UFPE (slides, 2025-2)
- [Código-base da disciplina](https://github.com/RomeroBarata/IF702-redes-neurais)
- [Base no Kaggle](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services)
- Hajiramezanali et al. (2022), STab; Eslamian, Afzal Aghaei e Cheng (2025), TabKAN
- [Optuna](https://optuna.readthedocs.io/), framework de busca de hiperparâmetros