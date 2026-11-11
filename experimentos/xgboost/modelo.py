"""XGBoost, a referencia do grupo: espaco de busca, treino com parada e medida no placar comum.

A parada usa so os 20% internos do treino da dobra (`parada`); depois o modelo retreina no treino inteiro da dobra com o
numero de arvores encontrado, como manda o placar (docs/pre-proposta.md, "Placar comum", item 2).

As duas perdas pedidas nos slides saem da mesma saida, p = sigmoide(margem):
- "ce": entropia cruzada (binary:logistic);
- "mse": erro quadratico sobre p, com hessiana de Gauss-Newton. O reg:squarederror do XGBoost preve fora de [0, 1] e,
  na sondagem de 05/11, perdeu 0,0018 de KS para a entropia cruzada; o MSE sobre p empatou com ela.
"""
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # comum.py fica na raiz
import comum  # noqa: E402

MODELO = "xgboost"
TETO_ARVORES = 10_000  # slides 24 e 27: no maximo 10.000 epocas
PACIENCIA = 20  # slides: 10, subindo para 20 se parar cedo; com arvores rasas para cedo (sondagem de 05/11: +0,0021, 5 de 5)
N_JOBS = int(os.environ.get("CHURN_N_JOBS", "8"))

# Trial 0: os padroes do XGBoost. reg_alpha fica em 1e-3, o minimo do espaco em escala log (o padrao 0 nao cabe nela,
# e o TPE quebraria com log(0)); com 1e-3 o efeito da penalidade L1 e desprezivel.
PADRAO = dict(max_depth=6, min_child_weight=1.0, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
              reg_lambda=1.0, reg_alpha=1e-3, gamma=0.0, perda="ce")


def sugerir(t):
    """Espaco validado em 05/11: o melhor de 60 trials neste espaco ganhou 0,0083 de KS do espaco anterior em 4 divisoes
    que a busca nao viu (4 de 4). Arvores rasas ganham porque o efeito das variaveis e quase aditivo."""
    return dict(
        max_depth=t.suggest_int("max_depth", 1, 6),
        min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1),
        reg_lambda=t.suggest_float("reg_lambda", 1e-2, 100, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        gamma=t.suggest_float("gamma", 0, 5),
        perda=t.suggest_categorical("perda", ["ce", "mse"]),
    )


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def mse_sigmoide(margem, dtrain):
    """Gradiente e hessiana de (sigmoide(margem) - y)^2. A hessiana e a de Gauss-Newton, sempre positiva."""
    y = dtrain.get_label()
    p = sigmoide(margem)
    g = p * (1 - p)
    return 2 * (p - y) * g, np.maximum(2 * g * g, 1e-6)


def _mse_metrica(margem, dmat):
    return "mse", float(np.mean((sigmoide(margem) - dmat.get_label()) ** 2))


def _params(hp, semente):
    p = dict(max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"], eta=hp["learning_rate"],
             subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"], reg_lambda=hp["reg_lambda"],
             reg_alpha=hp["reg_alpha"], gamma=hp["gamma"], seed=int(semente), nthread=N_JOBS, tree_method="hist")
    if hp["perda"] == "ce":
        p.update(objective="binary:logistic", eval_metric="logloss")
    elif hp["perda"] == "mse":
        p.update(base_score=0.0, disable_default_eval_metric=1)  # margem 0 = p 0,5; a saida e sempre a margem
    else:
        raise ValueError(f"perda desconhecida: {hp['perda']!r}")
    return p


def treinar(hp, Xa, ya, semente, Xp=None, yp=None, arvores=None):
    """Com (Xp, yp): para pela perda em Xp, com paciencia PACIENCIA e teto TETO_ARVORES. Sem: treina `arvores` arvores.
    Devolve (booster, numero de arvores usado)."""
    if (Xp is None) == (arvores is None):
        raise ValueError("passe (Xp, yp) para a parada ou `arvores` para o retreino, nao os dois")
    mse = hp["perda"] == "mse"
    extra = dict(obj=mse_sigmoide, custom_metric=_mse_metrica) if mse else {}
    da = xgb.DMatrix(np.asarray(Xa, dtype="float32"), label=np.asarray(ya, dtype="float32"))
    if arvores is not None:
        b = xgb.train(_params(hp, semente), da, int(arvores), verbose_eval=False, **extra)
        return b, int(arvores)
    dp = xgb.DMatrix(np.asarray(Xp, dtype="float32"), label=np.asarray(yp, dtype="float32"))
    b = xgb.train(_params(hp, semente), da, TETO_ARVORES, evals=[(dp, "parada")], early_stopping_rounds=PACIENCIA,
                  verbose_eval=False, **extra)
    return b, b.best_iteration + 1


def prever(b, X, arvores):
    """Probabilidade de churn com as primeiras `arvores` arvores, nas duas perdas."""
    margem = b.predict(xgb.DMatrix(np.asarray(X, dtype="float32")), output_margin=True, iteration_range=(0, int(arvores)))
    return sigmoide(margem)


def ajustar_prever_de(hp, features=(), over=True):
    """ajustar_prever no formato de comum.medir_placar. Anota em `.arvores` quantas arvores cada divisao usou."""
    arvores = []

    def ajustar_prever(X, y, X_val, ajuste, parada, semente):
        Xa, ya, Xp = comum.preparar(X.iloc[ajuste], y.iloc[ajuste], X.iloc[parada], features=features, over=over)
        _, k = treinar(hp, Xa, ya, semente, Xp, y.iloc[parada])
        XA, yA, XV = comum.preparar(X, y, X_val, features=features, over=over)
        b, _ = treinar(hp, XA, yA, semente, arvores=k)
        arvores.append(k)
        return prever(b, XV, k), (np.asarray(yA, dtype=float), prever(b, XA, k))

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
        b, _ = treinar(hp, XA, yA, comum.SEMENTE + s, arvores=k)
        p_sementes.append(prever(b, XT, k))
        arvores.append(k)
    return dict(p_sementes=[p.tolist() for p in p_sementes], arvores=arvores)
