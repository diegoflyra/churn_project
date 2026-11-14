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

## Experimento KAN

A série em `experimentos/kan/` divide o treinamento em cinco etapas: `kan.ipynb` (baseline e hiperparâmetros), `kan_fase2.ipynb` (dados e diagnóstico), `kan_fase3.ipynb` (regularização e arquitetura), `kan_fase4.ipynb` (validação interna) e `kan_fase5.ipynb` (placar comum e teste final). Instale `requirements.txt` e configure `CHURN_CSV` para indicar o dataset quando ele não estiver no caminho padrão. O teste oficial só é aberto na última fase.

### Executar a KAN no Kaggle ou Colab com GPU

Antes de abrir os notebooks na nuvem, envie para o GitHub as alterações do projeto (ou disponibilize o repositório atualizado por outro meio). O clone público `https://github.com/diegoflyra/churn_project.git` só contém o código que já foi enviado; os arquivos ainda não publicados precisam ser copiados junto com o notebook.

Em cada plataforma, selecione uma GPU **antes** de executar as células:

- **Kaggle:** `Settings → Accelerator → GPU`; habilite Internet para instalar dependências e clonar o repositório. Em `Add Input`, anexe o dataset de churn.
- **Colab:** `Runtime → Change runtime type → T4 GPU`. Monte o Google Drive para guardar os resultados entre sessões.

Na primeira célula de cada notebook, clone o repositório atualizado, entre na pasta e instale as dependências. No Kaggle:

```python
!git clone --depth 1 --branch main https://github.com/diegoflyra/churn_project.git /kaggle/working/churn_project
%cd /kaggle/working/churn_project
%pip install -q -r requirements.txt
```

No Colab, use `/content` no lugar de `/kaggle/working`:

```python
!git clone --depth 1 --branch main https://github.com/diegoflyra/churn_project.git /content/churn_project
%cd /content/churn_project
%pip install -q -r requirements.txt
from google.colab import drive
drive.mount("/content/drive")
```

Antes da célula que importa `comum` e `experimentos.kan.modelo`, defina os caminhos. O Kaggle disponibiliza o CSV em `/kaggle/input/datasets/kapturovalexander/customers-churned-in-telecom-services/customer_churn_telecom_services.csv`. No Colab, substitua `CSV` pelo caminho do arquivo no Drive:

```python
import os
os.environ["CHURN_CSV"] = CSV
os.environ["CHURN_RESULTADOS"] = PASTA_DE_RESULTADOS
```

Use `/kaggle/working/resultados` para `PASTA_DE_RESULTADOS` no Kaggle e, no Colab, um caminho persistente, por exemplo `/content/drive/MyDrive/churn_kan/resultados`. Confira a GPU antes do treino:

```python
import torch
from experimentos.kan import modelo as K
print(torch.cuda.is_available(), K.DEVICE)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

O dispositivo é selecionado automaticamente ao importar `modelo.py`; o esperado com GPU ativa é `cuda`. Execute as fases em ordem, sempre com o mesmo `CHURN_RESULTADOS`, pois cada fase lê o JSON salvo pela anterior. No Colab, o Drive mantém esses arquivos entre sessões. No Kaggle, ao terminar cada fase, salve uma versão com seus outputs e, na sessão seguinte, copie a pasta `resultados` do output anterior para `/kaggle/working/resultados` antes de executar a próxima. A fase 5 revela o teste apenas depois de congelar e registrar a configuração. `MODO_RAPIDO=True` serve apenas para verificar o fluxo; use `False` para as medições completas.
