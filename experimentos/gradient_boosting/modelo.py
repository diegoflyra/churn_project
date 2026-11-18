"""Gradient boosting do scikit-learn (linha extra ao lado do XGBoost): espaco de busca, treino com parada e medida no placar.

Os parametros sao os do slide 28 (loss, learning rate, numero de estimadores, subsample, criterion, min_samples_leaf e
max_depth), com tres ressalvas medidas no scikit-learn 1.9.1:
- a loss "deviance" do slide hoje se chama "log_loss";
- o `criterion` esta obsoleto e nao muda o ajuste (friedman_mse e squared_error deram a mesma decision_function), entao fica fora;
- o numero de estimadores sai da parada, como no XGBoost, e nao conta entre os hiperparametros da busca.

A parada usa so os 20% internos do treino da dobra (`parada`): a perda do proprio modelo em `parada` e acompanhada arvore a
arvore pelo `monitor` do fit, com paciencia PACIENCIA. Depois o modelo retreina no treino inteiro da dobra com o numero de
arvores encontrado, como manda o placar (docs/pre-proposta.md, "Placar comum", item 2). O `n_iter_no_change` do
scikit-learn nao serve aqui: ele sorteia a propria validacao dentro do treino que recebe.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # comum.py fica na raiz
import comum  # noqa: E402

MODELO = "gradient_boosting"
TETO_ARVORES = 10_000  # slides 24 e 27: no maximo 10.000 epocas
PACIENCIA = 20  # a mesma do XGBoost

# Trial 0: os padroes do GradientBoostingClassifier. max_features=None e o mesmo que 1.0. min_impurity_decrease fica em
# 1e-6, o minimo do espaco em escala log (o padrao 0 nao cabe nela): com 300 arvores, nas duas losses, a margem saiu
# identica a do padrao 0, porque o menor ganho por corte medido fica na casa de 1e-5.
PADRAO = dict(loss="log_loss", learning_rate=0.1, max_depth=3, min_samples_leaf=1, subsample=1.0, max_features=1.0,
              min_impurity_decrease=1e-6)


def sugerir(t):
    """Os cinco parametros do slide 28 que mudam o ajuste, nas mesmas faixas do XGBoost, mais os equivalentes de
    colsample_bytree (max_features) e de gamma (min_impurity_decrease). A faixa de min_impurity_decrease vem do ganho por
    corte medido nos dados (log_loss: mediana 1e-4, 99% abaixo de 4e-3; exponential: mediana 5e-4, 99% abaixo de 2e-2)."""
    return dict(
        loss=t.suggest_categorical("loss", ["log_loss", "exponential"]),
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=t.suggest_int("max_depth", 1, 6),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 100, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1),
        max_features=t.suggest_float("max_features", 0.3, 1),
        min_impurity_decrease=t.suggest_float("min_impurity_decrease", 1e-6, 1e-2, log=True),
    )


def _modelo(hp, semente, n_estimators):
    return GradientBoostingClassifier(loss=hp["loss"], learning_rate=hp["learning_rate"], max_depth=hp["max_depth"],
                                      min_samples_leaf=hp["min_samples_leaf"], subsample=hp["subsample"],
                                      max_features=hp["max_features"], min_impurity_decrease=hp["min_impurity_decrease"],
                                      n_estimators=int(n_estimators), random_state=int(semente))


def perda(loss, y, margem):
    """A perda que o GradientBoostingClassifier minimiza, na margem (decision_function)."""
    y = np.asarray(y, dtype=float)
    if loss == "log_loss":
        return float(np.mean(np.logaddexp(0, margem) - y * margem))
    if loss == "exponential":
        return float(np.mean(np.exp(-(2 * y - 1) * margem)))
    raise ValueError(f"loss desconhecida: {loss!r}")


class Parada:
    """monitor do fit: soma a contribuicao de cada arvore na margem de Xp e para depois de PACIENCIA arvores sem melhora."""

    def __init__(self, loss, Xp, yp, paciencia=PACIENCIA):
        self.loss, self.Xp, self.yp, self.paciencia = loss, Xp, yp, paciencia
        self.margem, self.perdas, self.melhor = None, [], 0

    def __call__(self, i, est, _locais):
        if self.margem is None:
            self.margem = est._raw_predict_init(self.Xp)[:, 0].astype(float)
        self.margem = self.margem + est.learning_rate * est.estimators_[i, 0].predict(self.Xp)
        self.perdas.append(perda(self.loss, self.yp, self.margem))
        if self.perdas[-1] < self.perdas[self.melhor]:
            self.melhor = i
        return i - self.melhor >= self.paciencia


def treinar(hp, Xa, ya, semente, Xp=None, yp=None, arvores=None):
    """Com (Xp, yp): para pela perda em Xp, com paciencia PACIENCIA e teto TETO_ARVORES. Sem: treina `arvores` arvores.
    Devolve (modelo, numero de arvores usado)."""
    if (Xp is None) == (arvores is None):
        raise ValueError("passe (Xp, yp) para a parada ou `arvores` para o retreino, nao os dois")
    Xa, ya = np.asarray(Xa, dtype=float), np.asarray(ya)
    if arvores is not None:
        return _modelo(hp, semente, arvores).fit(Xa, ya), int(arvores)
    parada = Parada(hp["loss"], np.asarray(Xp, dtype=float), yp)
    m = _modelo(hp, semente, TETO_ARVORES).fit(Xa, ya, monitor=parada)
    return m, parada.melhor + 1


def prever(m, X, arvores):
    """Probabilidade de churn com as primeiras `arvores` arvores."""
    X = np.asarray(X, dtype=float)
    if int(arvores) == m.n_estimators_:
        return m.predict_proba(X)[:, 1]
    for k, p in enumerate(m.staged_predict_proba(X), 1):
        if k == int(arvores):
            return p[:, 1]
    raise ValueError(f"o modelo tem {m.n_estimators_} arvores, menos que {arvores}")


def ajustar_prever_de(hp, features=(), over=True):
    """ajustar_prever no formato de comum.medir_placar. Anota em `.arvores` quantas arvores cada divisao usou."""
    arvores = []

    def ajustar_prever(X, y, X_val, ajuste, parada, semente):
        Xa, ya, Xp = comum.preparar(X.iloc[ajuste], y.iloc[ajuste], X.iloc[parada], features=features, over=over)
        _, k = treinar(hp, Xa, ya, semente, Xp, y.iloc[parada])
        XA, yA, XV = comum.preparar(X, y, X_val, features=features, over=over)
        m, _ = treinar(hp, XA, yA, semente, arvores=k)
        arvores.append(k)
        return prever(m, XV, k), (np.asarray(yA, dtype=float), prever(m, XA, k))

    ajustar_prever.arvores = arvores
    return ajustar_prever


def final(hp, X_tr, y_tr, X_te, n_sementes=5, features=(), over=True):
    """Treino inteiro, como no placar: 20% internos decidem as arvores, depois retreina em tudo. Uma previsao por semente."""
    y = np.asarray(y_tr)
    XA, yA, XT = comum.preparar(X_tr, y_tr, X_te, features=features, over=over)
    p_sementes, arvores = [], []
    for s in range(n_sementes):
        a1, a2 = train_test_split(np.arange(len(y)), test_size=comum.PLACAR_FRACAO_PARADA, stratify=y,
                                  random_state=comum.SEMENTE + s)
        Xa, ya, Xp = comum.preparar(X_tr.iloc[a1], y_tr.iloc[a1], X_tr.iloc[a2], features=features, over=over)
        _, k = treinar(hp, Xa, ya, comum.SEMENTE + s, Xp, y[a2])
        m, _ = treinar(hp, XA, yA, comum.SEMENTE + s, arvores=k)
        p_sementes.append(prever(m, XT, k))
        arvores.append(k)
    return dict(p_sementes=[p.tolist() for p in p_sementes], arvores=arvores)
