"""Diagnostico do campeao: onde ainda ha sinal e quais combinacoes de variaveis o modelo erra.

So usa as previsoes de validacao das dobras. O teste nao entra aqui. Nao muda o campeao: gera hipoteses para os blocos.
"""
import sys, os
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, blocos as BL

CV = StratifiedKFold(5, shuffle=True, random_state=comum.SEMENTE)


def oof(camp, n_seeds=5):
    """Probabilidade media de validacao do campeao, por cliente, junto das variaveis originais."""
    folds, _ = BL.dobras_de(camp)
    ind_val, _ = comum.indices_dobras()
    indice = sum(ind_val, [])
    p = np.mean([np.concatenate([M.fit(camp["hp"], f, [f[2]], 1000 * s + 42 + i)[0][0] for i, f in enumerate(folds)]) for s in range(n_seeds)], 0)
    df = comum.carregar_df().loc[indice]
    return df.drop(columns=["Churn"]), df["Churn"].astype(int), pd.Series(p, index=indice, name="p")


def _segmentos(X):
    S = {}
    for c in X:
        if X[c].nunique() <= 2:
            S[f"{c}=1"] = X[c] == X[c].max()
        elif c == "total_addons":
            S.update({"addons 0-1": X[c] <= 1, "addons 2-4": X[c].between(2, 4), "addons 5+": X[c] >= 5})
        else:
            q = pd.qcut(X[c], 4, duplicates="drop")
            for b in q.cat.categories: S[f"{c} em {b}"] = q == b
    return S


def _calibrada(y, p):
    """Probabilidade na escala real: o treino foi balanceado, e o residuo precisa comparar com a taxa verdadeira."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).values.reshape(-1, 1)
    return pd.Series(cross_val_predict(LogisticRegression(), z, y, cv=CV, method="predict_proba")[:, 1], index=y.index)


def _tabela_residuos(S, y, q, n_min):
    """Churn observado menos esperado em cada grupo, com correcao de Benjamini-Hochberg."""
    r, linhas = y - q, []
    for nome, m in S.items():
        n = int(m.sum())
        if n < n_min: continue
        z = r[m].sum() / np.sqrt((q[m] * (1 - q[m])).sum())
        linhas.append(dict(grupo=nome, n=n, churn_real=y[m].mean(), churn_previsto=q[m].mean(), residuo=r[m].mean(), z=z))
    t = pd.DataFrame(linhas)
    t["p"] = 2 * norm.sf(t.z.abs())
    t = t.sort_values("p")
    t["q_bh"] = (t.p * len(t) / np.arange(1, len(t) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return t.sort_values("residuo", key=abs, ascending=False)


def residuos(X, y, p, n_min=40):
    """Grupos e pares de grupos em que o modelo erra de forma sistematica. E daqui que saem as variaveis candidatas."""
    q, S = _calibrada(y, p), _segmentos(X)
    pares = {}
    for (a, ma), (b, mb) in combinations(S.items(), 2):
        if a.split(" em ")[0].split("=")[0] != b.split(" em ")[0].split("=")[0]:
            pares[f"{a} & {b}"] = ma & mb
    return _tabela_residuos(S, y, q, n_min), _tabela_residuos(pares, y, q, n_min)


def sinal_restante(X, y, p):
    """Um modelo flexivel com as variaveis e o escore bate o proprio escore? Se nao, o sinal disponivel ja foi usado."""
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=.05, random_state=42)
    Xp = X.assign(p=p)
    so_var = cross_val_predict(gb, X, y, cv=CV, method="predict_proba")[:, 1]
    com_p = cross_val_predict(gb, Xp, y, cv=CV, method="predict_proba")[:, 1]
    return pd.DataFrame([dict(entrada="escore do campeao", ks=comum.ks(y.values, p.values), auroc=roc_auc_score(y, p)),
                         dict(entrada="boosting so com as variaveis", ks=comum.ks(y.values, so_var), auroc=roc_auc_score(y, so_var)),
                         dict(entrada="boosting com variaveis + escore", ks=comum.ks(y.values, com_p), auroc=roc_auc_score(y, com_p))])


def mesmo_lado(X, y, p, profundidade=3):
    """Dentro de quem o modelo acusa (risco) ou absolve (fiel), as variaveis ainda separam os rotulos?"""
    thr, linhas, regras = comum.limiar_ks(y.values, p.values), [], {}
    for lado, m in (("risco", p >= thr), ("fiel", p < thr)):
        Xl, yl = X[m], y[m]
        arv = DecisionTreeClassifier(max_depth=profundidade, min_samples_leaf=60, random_state=42)
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=.05, random_state=42)
        linhas.append(dict(lado=lado, clientes=int(m.sum()), churn=yl.mean(), auroc_do_escore=roc_auc_score(yl, p[m]),
                           auroc_arvore=roc_auc_score(yl, cross_val_predict(arv, Xl, yl, cv=CV, method="predict_proba")[:, 1]),
                           auroc_boosting=roc_auc_score(yl, cross_val_predict(gb, Xl, yl, cv=CV, method="predict_proba")[:, 1])))
        regras[lado] = export_text(arv.fit(Xl, yl), feature_names=list(X.columns), show_weights=True)
    return pd.DataFrame(linhas), regras


def por_grupo(X, y, p):
    """KS dentro de cada grupo: mostra onde o modelo separa bem e onde quase nao separa."""
    linhas = []
    for nome, m in _segmentos(X).items():
        if m.sum() >= 100 and 10 <= y[m].sum() <= m.sum() - 10:
            linhas.append(dict(grupo=nome, n=int(m.sum()), churn=y[m].mean(), ks=comum.ks(y[m].values, p[m].values)))
    return pd.DataFrame(linhas).sort_values("ks")


def curva_aprendizado(camp, fracoes=(.25, .5, .75, 1.0), n_seeds=3):
    """KS de validacao treinando com uma fracao do treino da dobra. Curva plana: mais dados nao devem ajudar."""
    folds, _ = BL.dobras_de(camp)
    linhas = []
    for fr in fracoes:
        kss = []
        for s in range(n_seeds):
            rng = np.random.default_rng(s)
            for i, f in enumerate(folds):
                idx = rng.choice(len(f[0]), int(fr * len(f[0])), replace=False)
                g = (f[0][idx], f[1][idx], f[2], f[3])
                kss.append(comum.ks(f[3], M.fit(camp["hp"], g, [f[2]], 1000 * s + 42 + i)[0][0]))
        linhas.append(dict(fracao_do_treino=fr, ks=np.mean(kss), dp=np.std(kss, ddof=1)))
    return pd.DataFrame(linhas)
