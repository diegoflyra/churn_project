"""Registro evolutivo do pre-processamento e teste de hipoteses vindas dos erros (so treino; o teste fica de fora).

Tudo aqui e exploratorio: as dobras oficiais (comum.carregar_dobras) nao mudam.
"""
import sys, os
import numpy as np, pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M
from erros import _xgb

CRIADAS = ["total_addons", "PaymentMethod_Automatic"]
# Etapas cumulativas. Imputacao e codificacao ficam sempre: sem elas os modelos nao rodam.
ETAPAS = [("codificacao", dict(criadas=False, escala=False, over=False)),
          ("+ variaveis criadas", dict(criadas=True, escala=False, over=False)),
          ("+ StandardScaler", dict(criadas=True, escala=True, over=False)),
          ("+ oversampling (oficial)", dict(criadas=True, escala=True, over=True))]


def _prep(Xa, ya, Xb, criadas=True, escala=True, over=True, extra=None):
    Xa, Xb = Xa.copy(), Xb.copy()
    novas = []
    if extra:
        A, B = extra(Xa), extra(Xb)
        novas = [c for c in A if c not in Xa]
        Xa, Xb = A, B
    if not criadas:
        Xa, Xb = Xa.drop(columns=CRIADAS), Xb.drop(columns=CRIADAS)
    if escala:
        cols = comum.NUMERIC_COLS + [c for c in novas if Xa[c].nunique() > 2]
        sc = StandardScaler().fit(Xa[cols])
        Xa[cols], Xb[cols] = sc.transform(Xa[cols]), sc.transform(Xb[cols])
    if over:
        Xa, ya = RandomOverSampler(random_state=comum.SEMENTE).fit_resample(Xa, ya)
    return Xa.values.astype("float32"), ya.values.astype("float32"), Xb.values.astype("float32")


def _cv(modelo, hp, Xtr, ytr, rep, semente=0, **k):
    """KS medio em 5 dobras na divisao `rep`. rep=0 sao as dobras oficiais; as demais medem a variacao da divisao.
    `semente` muda so a inicializacao do modelo, para medir o ruido com a mesma divisao."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=comum.SEMENTE + rep)
    kss = []
    for i, (a, b) in enumerate(skf.split(Xtr, ytr)):
        Xa, ya, Xb = _prep(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], **k)
        yb = ytr.iloc[b].values.astype("float32")
        if modelo == "mlp":
            p = M.fit(hp, (Xa, ya, Xb, yb), [Xb], 42 + i + 1000 * rep + 100000 * semente)[0][0]
        else:
            p = _xgb(42 + rep + 100000 * semente).fit(Xa, ya).predict_proba(Xb)[:, 1]
        kss.append(comum.ks(yb, p))
    return float(np.mean(kss))


def _comparar(variantes, hp, repeticoes):
    """variantes = [(nome, base, kwargs)]; compara cada uma com `base`, pareado por divisao."""
    Xtr, _, ytr, _ = comum.dividir(comum.carregar_df())
    linhas = []
    for modelo in ("mlp", "xgboost"):
        ks = {n: np.array([_cv(modelo, hp, Xtr, ytr, r, **k) for r in range(repeticoes)]) for n, _, k in variantes}
        for n, base, _ in variantes:
            d = ks[n] - ks[base] if base else np.full(repeticoes, np.nan)
            linhas.append(dict(modelo=modelo, variante=n, ks=ks[n].mean(), dp_ks=ks[n].std(ddof=1) if repeticoes > 1 else np.nan,
                               delta=np.nanmean(d) if base else np.nan, dp_delta=d.std(ddof=1) if base and repeticoes > 1 else np.nan,
                               ganhou=f"{int((d > 0).sum())}/{repeticoes}" if base else ""))
    return pd.DataFrame(linhas)


def registro(hp, repeticoes=5):
    """O que cada etapa do pre-processamento muda no MLP e no XGBoost (cada delta e contra a etapa anterior)."""
    v, ant = [], None
    for nome, k in ETAPAS:
        v.append((nome, ant, k)); ant = nome
    return _comparar(v, hp, repeticoes)


def hipoteses(candidatas, hp, repeticoes=5):
    """candidatas = {nome: funcao(df) -> df com colunas novas}. Exploratorio; muitas comparacoes inflam o melhor delta."""
    return _comparar([("oficial", None, {})] + [(n, "oficial", dict(extra=f)) for n, f in candidatas.items()], hp, repeticoes)
