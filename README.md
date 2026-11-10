# churn_project

Previsão de churn (cancelamento) de clientes de telecomunicações. Projeto da disciplina IF702.

O pré-processamento base está no notebook `if702-preprocessamento.ipynb`, e cada algoritmo tem uma pasta própria em `experimentos/`. Cada modelo pode evoluir o próprio pré-processamento e registra essa evolução em `experimentos/<modelo>/preprocessamento/`: veja as regras em `docs/estrutura_preprocessamento.md` e a comparação em `docs/PREPROCESSAMENTOS.md`. O que testar nos dados de cada modelo, a partir do que o MLP aprendeu, está em `docs/GUIA_DADOS.md`.

## Estrutura

```
├── comum.py        dados, dobras, métricas, estudo Optuna, avaliação final e placar comum
├── docs/           pre-proposta, estrutura do pré-processamento, guia de dados e comparação dos pré-processamentos
├── resultados/
│   ├── <modelo>.json   entregável: um por modelo, no formato comum (mlp.json, stab.json, tabkan.json, ...)
│   ├── placar.jsonl, PLACAR.md, placar_oof/, escolhidos.json   o placar comum (docs/pre-proposta.md)
│   ├── processamentos_comuns.json   features testadas em outros modelos (docs/GUIA_DADOS.md)
│   └── <modelo>/       arquivos de trabalho de cada modelo (execuções, intermediários, tabelas dos slides)
└── experimentos/
    ├── stab/       stab.ipynb, modelo.py, avaliar.py
    ├── tabkan/     tabkan.ipynb, modelo.py, avaliar.py
    └── kan/, mlp/, mitra/, tabpfn_v2/, gradient_boosting/
```

Em cada pasta, `modelo.py` tem o modelo, o treino e o espaço de busca, e `avaliar.py` tem só o que é específico daquele modelo na avaliação. Para rodar, a partir de qualquer diretório:

```
python experimentos/stab/modelo.py 40    # estudo Optuna com 40 trials
python experimentos/stab/avaliar.py 3    # avaliação final com 3 sementes
```

## Dados

Dataset [Customers churned in telecom services](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services), do Kaggle: 7043 clientes, com `Churn` como alvo (cerca de 26,5% de cancelamentos).

## O que o notebook faz

1. **Dados faltantes:** os 11 valores ausentes de `TotalCharges` são de clientes com `tenure` igual a 0 e são preenchidos com 0.
2. **Inconsistências:** confere se `PhoneService` e `InternetService` batem com as colunas dos serviços opcionais. Nenhuma foi encontrada.
3. **Codificação:** colunas sim/não viram binárias; `InternetService`, `Contract` e `PaymentMethod` recebem one-hot.
4. **Outliers:** nenhum pelo critério IQR em `tenure`, `MonthlyCharges` e `TotalCharges`.
5. **Variáveis criadas:** `total_addons` (número de serviços opcionais assinados) e `PaymentMethod_Automatic`.
6. **Divisão treino/teste:** 75% / 25%, estratificada por `Churn`.
7. **K-fold com normalização e balanceamento:** 5 dobras estratificadas sobre o treino original.

## Sobre o k-fold

O k-fold é feito antes do `StandardScaler` e do `RandomOverSampler`. Em cada dobra, o scaler é ajustado só na parte de treino e o oversampling só duplica essa parte. A validação da dobra continua com um cliente por linha e só recebe o `transform`.

Balancear e padronizar o treino inteiro antes do k-fold espalharia as cópias de um mesmo cliente entre o treino e a validação da dobra, e a métrica subiria sem refletir clientes novos.

A lista `folds` guarda, para cada dobra, `(X_fold_train, y_fold_train, X_fold_val, y_fold_val)`. O teste fica sem escala e sem balanceamento; `preparar(X_train, y_train, X_test)` gera o treino final e o teste padronizado quando for a hora de usá-lo.

## Como rodar

O notebook foi feito no Kaggle e lê o CSV de `/kaggle/input/...`. Para rodar lá, basta anexar o dataset ao notebook.

Para rodar localmente, instale as dependências e troque a variável `path` pelo caminho do CSV na sua máquina:

```
pip install numpy pandas scikit-learn imbalanced-learn kagglehub jupyter
```
