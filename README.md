# churn_project

Previsão de churn (cancelamento) de clientes de telecomunicações. Projeto da disciplina IF702.

Por enquanto o repositório tem só o pré-processamento dos dados, no notebook `if702-preprocessamento.ipynb`.

## Dados

Dataset [Customers churned in telecom services](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services), do Kaggle: 7043 clientes, com `Churn` como alvo (cerca de 26,5% de cancelamentos).

## O que o notebook faz

1. **Dados faltantes:** os 11 valores ausentes de `TotalCharges` são de clientes com `tenure` igual a 0 e são preenchidos com 0.
2. **Inconsistências:** confere se `PhoneService` e `InternetService` batem com as colunas dos serviços opcionais. Nenhuma foi encontrada.
3. **Codificação:** colunas sim/não viram binárias; `InternetService`, `Contract` e `PaymentMethod` recebem one-hot.
4. **Outliers:** nenhum pelo critério IQR em `tenure`, `MonthlyCharges` e `TotalCharges`.
5. **Variáveis criadas:** `total_addons` (número de serviços opcionais assinados) e `PaymentMethod_Automatic`.
6. **Divisão treino/teste:** 75% / 25%, estratificada por `Churn`.
7. **K-fold com normalização e balanceamento (50% treino / 25% validação / 25% teste):** 3 dobras estratificadas sobre o pool de 75%, garantindo que cada dobra possua exatamente 50% de treino e 25% de validação do total de cada classe, com oversampling (`RandomOverSampler`) aplicado exclusivamente na porção de treino da dobra.

## Sobre o k-fold (50% Treino / 25% Validação / 25% Teste)

Com o conjunto de teste isolado com 25% do total (estratificado), o pool de desenvolvimento restante corresponde a 75%. Ao utilizar `StratifiedKFold(n_splits=3)`:
- Em cada dobra, a validação fica com 1/3 de 75% = **25% do total de dados** (e 25% de cada classe);
- O treino da dobra fica com 2/3 de 75% = **50% do total de dados** (e 50% de cada classe);
- O teste permanece intocado com **25% do total de dados** (e 25% de cada classe).

Em cada dobra, o `StandardScaler` é ajustado somente na parte de treino e o `RandomOverSampler` só duplica a classe minoritária dessa parte de treino. A validação da dobra e o teste continuam com a distribuição real dos clientes e recebem apenas o `transform`.

Balancear e padronizar os dados antes do k-fold espalharia cópias de um mesmo cliente entre treino e validação, inflando artificialmente as métricas de avaliação.

A lista `folds` guarda, para cada dobra, `(X_fold_train, y_fold_train, X_fold_val, y_fold_val)`. O teste fica sem escala e sem balanceamento até a avaliação final.

## Como rodar

O notebook foi feito no Kaggle e lê o CSV de `/kaggle/input/...`. Para rodar lá, basta anexar o dataset ao notebook.

Para rodar localmente, instale as dependências e troque a variável `path` pelo caminho do CSV na sua máquina:

```
pip install numpy pandas scikit-learn imbalanced-learn kagglehub jupyter
```
