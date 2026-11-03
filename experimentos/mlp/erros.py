"""Erros consistentes: quem varias "lentes" (sementes, configuracoes, outras familias) erram juntas, e qual o perfil.

So usa as previsoes de validacao das dobras. O teste nao entra aqui, para continuar independente.
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M


def top_configs(st, k):
    ok = sorted([t for t in st.trials if t.state.name == "COMPLETE"], key=lambda t: t.value, reverse=True)
    return [dict(t.params) for t in ok[:k]]


def _xgb(s):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=4, subsample=.8, colsample_bytree=.8,
                         random_state=s, n_jobs=4, tree_method="hist")


def lentes(folds, configs, n_seeds=5):
    """Probabilidade de validacao por cliente em cada lente. configs[0] = configuracao campea."""
    ind_val, _ = comum.indices_dobras()
    indice = sum(ind_val, [])
    P = {}
    for s in range(n_seeds):
        P[f"mlp_melhor_s{s}"] = np.concatenate([M.fit(configs[0], f, [f[2]], 1000 * s + 42 + i)[0][0] for i, f in enumerate(folds)])
    for k, hp in enumerate(configs[1:], 2):
        P[f"mlp_top{k}"] = np.concatenate([M.fit(hp, f, [f[2]], 42 + i)[0][0] for i, f in enumerate(folds)])
    P["regressao_logistica"] = np.concatenate([LogisticRegression(max_iter=3000).fit(f[0], f[1]).predict_proba(f[2])[:, 1] for f in folds])
    P["xgboost"] = np.concatenate([_xgb(42).fit(f[0], f[1]).predict_proba(f[2])[:, 1] for f in folds])
    y = np.concatenate([f[3] for f in folds]).astype(int)
    return pd.DataFrame(P, index=pd.Index(indice, name="indice")), pd.Series(y, index=pd.Index(indice, name="indice"), name="y")


def classificar(P, y, corte=.9):
    """Cada lente decide no proprio limiar de KS maximo. Consistente = pelo menos `corte` das lentes concordam."""
    E = pd.DataFrame({c: (P[c] >= comum.limiar_ks(y.values, P[c].values)).astype(int) != y for c in P})
    frac = E.mean(axis=1)
    tipo = pd.Series("instavel", index=P.index)
    tipo[frac <= 1 - corte] = "acerto consistente"
    tipo[(frac >= corte) & (y == 1)] = "FN consistente"
    tipo[(frac >= corte) & (y == 0)] = "FP consistente"
    return pd.DataFrame(dict(y=y, prob_media=P.mean(axis=1), frac_erro=frac, tipo=tipo)), E


def resumo(tab, E):
    """Compara o observado com o esperado se as lentes errassem de forma independente (dentro de cada classe)."""
    linhas = []
    for nome, classe in (("churn (y=1)", 1), ("nao churn (y=0)", 0)):
        m = tab.y == classe
        linhas.append(dict(classe=nome, clientes=int(m.sum()), taxa_erro_media=E[m].mean().mean(),
                           todas_erram=(E[m].mean(axis=1) == 1).mean(), esperado_se_independentes=E[m].mean().prod()))
    return pd.DataFrame(linhas)


def concordancia(E):
    """Jaccard entre os conjuntos de erros de cada par de lentes."""
    return pd.DataFrame({a: {b: (E[a] & E[b]).sum() / max((E[a] | E[b]).sum(), 1) for b in E} for a in E})


def _segmentos(df):
    S = {}
    for c in df:
        v = df[c]
        if v.nunique() <= 2:
            S[f"{c}=1"] = v == v.max()
        elif v.nunique() <= 8:
            for x in sorted(v.unique()): S[f"{c}={x}"] = v == x
        else:
            q = pd.qcut(v, 4, duplicates="drop")
            for b in q.cat.categories: S[f"{c} em {b}"] = q == b
    return S


def perfil(tab, classe):
    """Segmentos super-representados nos erros consistentes de uma classe, contra os acertos consistentes da mesma classe.
    Mostra quais variaveis os modelos usam: os dois grupos estao em lados opostos da fronteira."""
    erro = (tab.y == classe) & tab.tipo.isin(["FN consistente", "FP consistente"])
    return _comparar_grupos(tab, erro, (tab.y == classe) & (tab.tipo == "acerto consistente"))


def mesmo_lado(tab, lado):
    """Clientes que os modelos veem do mesmo lado, com rotulos opostos. Se nada os separa, falta informacao na base.
    lado="risco": fieis acusados x churners pegos. lado="fiel": churners perdidos x fieis reconhecidos."""
    ok = tab.tipo == "acerto consistente"
    if lado == "risco":
        return _comparar_grupos(tab, tab.tipo == "FP consistente", ok & (tab.y == 1))
    return _comparar_grupos(tab, tab.tipo == "FN consistente", ok & (tab.y == 0))


def _comparar_grupos(tab, erro, ok):
    """Fisher exato com correcao de Benjamini-Hochberg, porque muitos segmentos sao testados. Descreve, nao prova causa."""
    df = comum.carregar_df().drop(columns=["Churn"]).loc[tab.index]
    linhas = []
    for nome, m in _segmentos(df).items():
        a, b, c, d = int((m & erro).sum()), int((~m & erro).sum()), int((m & ok).sum()), int((~m & ok).sum())
        if a + c == 0 or not erro.any() or not ok.any(): continue
        linhas.append(dict(segmento=nome, n_erros=a, pct_nos_erros=a / (a + b), pct_nos_acertos=c / (c + d), p=fisher_exact([[a, b], [c, d]])[1]))
    r = pd.DataFrame(linhas).sort_values("p")
    if r.empty: return r
    r["q_bh"] = (r.p * len(r) / np.arange(1, len(r) + 1))[::-1].cummin()[::-1].clip(upper=1)
    r["razao"] = r.pct_nos_erros / r.pct_nos_acertos.replace(0, np.nan)
    return r
