# churn_project

Previsão de churn (cancelamento) de clientes de telecomunicações. Projeto da disciplina IF702.

O código comum a todos os modelos está em `comum.py`: dados, dobras, métricas, validação cruzada e leitura e gravação de `resultados/<modelo>.json`.
O notebook `if702-preprocessamento.ipynb` continua como documentação do pré-processamento, e `tests/test_comum.py` confere que o `comum.py` gera os mesmos dados.
Os experimentos estão em `experimentos/`.

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

Para rodar o notebook localmente, troque a variável `path` pelo caminho do CSV na sua máquina.

O `comum.py` não precisa de ajuste: ele lê o caminho recebido, depois a variável `CHURN_CSV` e, se nenhum existir, baixa a base pelo `kagglehub`, sem `kaggle.json`.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ruff check .
pytest
```

## Mitra v2

O notebook `experimentos/mitra.ipynb` só roda com GPU e usa um ambiente próprio, `requirements-mitra.txt`, porque o `autogluon.tabular` 1.6.3 exige `pandas<2.4`.
As dobras geradas com pandas 2.3.3 e 3.0.6 são idênticas, byte a byte.

No Colab, escolha um ambiente com GPU e rode, a partir de um notebook vazio:

```
!git clone -b feat/rafael-comum-xgboost-mitra https://github.com/diegoflyra/churn_project.git
%cd churn_project
!pip install -q -r requirements-mitra.txt
!nvidia-smi
```

Depois rode cada etapa em uma célula, sem abrir o `mitra.ipynb` na interface:

```
!cd experimentos && MITRA_ETAPA=medidas jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.timeout=-1 mitra.ipynb
```

O `nbconvert` abre um kernel novo, então não é preciso reiniciar o ambiente depois do `pip install`.
No Kaggle o caminho é o mesmo, com a GPU e a internet ligadas nas configurações do notebook.

| `MITRA_ETAPA` | O que roda | O que guardar |
| --- | --- | --- |
| `medidas` | `bfloat16`, reprodutibilidade e tempo de dois trials | `experimentos/mitra.ipynb` |
| `curto` | Trial 0 e mais 4 trials | `experimentos/mitra.ipynb` |
| `completo`, com `MITRA_N_TRIALS=<N>` | Estudo que conta, retomável | `experimentos/mitra.ipynb` e `resultados/mitra.db` |

O disco do Colab e do Kaggle é apagado ao fim da sessão, então baixe esses arquivos antes de encerrar.
Para retomar o estudo completo, coloque `resultados/mitra.db` de volta no mesmo lugar e rode de novo.
