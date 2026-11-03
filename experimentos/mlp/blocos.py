"""Blocos em sequencia: cada bloco testa uma intervencao sobre o campeao do anterior e so aceita o que supera o ruido.

O campeao e {hp, dados, extras}. Comparacoes em divisoes novas do treino (1..N): as dobras oficiais escolheram a
configuracao, entao reutiliza-las favoreceria quem foi selecionado nelas. O teste so entra em final_campeao.
"""
import sys, os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, evolucao as EV

# Aceita se ganhar em pelo menos esta fracao das divisoes E o ganho medio superar a regua de ruido.
MIN_VITORIAS = 0.8
N_RUIDO = 2
CANDIDATAS = {
    "cliente_novo": lambda d: d.assign(cliente_novo=(d.tenure <= 6).astype(int)),
    "delta_preco": lambda d: d.assign(delta_preco=d.MonthlyCharges - d.TotalCharges / (d.tenure + 1)),
    "mtm_fibra": lambda d: d.assign(mtm_fibra=d["Contract_Month-to-month"] * d["InternetService_Fiber optic"]),
    "preco_por_servico": lambda d: d.assign(preco_por_servico=d.MonthlyCharges / (d.total_addons + 1)),
}
_cfg = dict(pasta=None, repeticoes=5, apos=None)


def configurar(pasta, repeticoes=5, apos=None):
    _cfg.update(pasta=os.path.join(pasta, "blocos"), repeticoes=repeticoes, apos=apos)
    os.makedirs(_cfg["pasta"], exist_ok=True)


def campeao_inicial(hp):
    return dict(hp=dict(hp), dados={}, extras=[])


def _kwargs(c):
    k = dict(c["dados"])
    if c["extras"]:
        fs = [CANDIDATAS[n] for n in c["extras"]]

        def extra(d):
            for f in fs: d = f(d)
            return d
        k["extra"] = extra
    return k


def _aplicar(base, v):
    return dict(hp={**base["hp"], **v.get("hp", {})}, dados={**base["dados"], **v.get("dados", {})},
                extras=base["extras"] + [e for e in v.get("extras", []) if e not in base["extras"]])


def _medir(c, Xtr, ytr, reps, semente=0):
    return np.array([EV._cv("mlp", c["hp"], Xtr, ytr, r, semente=semente, **_kwargs(c)) for r in reps])


def rodar(nome, pergunta, variantes, base):
    """variantes = {nome: {"hp": {...}, "dados": {...}, "extras": [...]}}. Devolve o campeao que segue para o proximo bloco.
    Resultado salvo em blocos/<nome>.json: rodar de novo so reaproveita, o que permite retomar a sequencia."""
    arq = os.path.join(_cfg["pasta"], f"{nome}.json")
    if os.path.isfile(arq):
        r = json.load(open(arq))
        if r["base"] == base:
            print(f"{nome}: ja concluido, reaproveitado.")
            return r["campeao"], pd.DataFrame(r["tabela"])
    t0 = time.time()
    Xtr, _, ytr, _ = comum.dividir(comum.carregar_df())
    reps = range(1, _cfg["repeticoes"] + 1)
    ks_base = _medir(base, Xtr, ytr, reps)
    # Regua: quanto o KS medio do proprio campeao muda so de trocar a semente, nas mesmas divisoes
    regua = max(abs((_medir(base, Xtr, ytr, reps, s) - ks_base).mean()) for s in range(1, N_RUIDO + 1))
    linhas, melhor, campeao = [dict(variante="(campeao atual)", ks=ks_base.mean(), delta=0.0, dp_delta=0.0, ganhou="", decisao="base")], 0.0, base
    for n, v in variantes.items():
        c = _aplicar(base, v)
        ks = _medir(c, Xtr, ytr, reps)
        d = ks - ks_base
        passou = d.mean() > regua and (d > 0).mean() >= MIN_VITORIAS
        linhas.append(dict(variante=n, ks=ks.mean(), delta=d.mean(), dp_delta=d.std(ddof=1) if len(d) > 1 else 0.0,
                           ganhou=f"{int((d > 0).sum())}/{len(d)}", decisao="supera o ruido" if passou else "rejeitada"))
        if passou and d.mean() > melhor:
            melhor, campeao = d.mean(), c
    if campeao is not base:
        for l in linhas:
            if l["delta"] == melhor and l["decisao"] == "supera o ruido": l["decisao"] = "ACEITA"
    r = dict(nome=nome, pergunta=pergunta, base=base, regua=regua, tabela=linhas, campeao=campeao, tempo_s=time.time() - t0)
    json.dump(r, open(arq, "w"), indent=1)
    print(f"{nome}: {pergunta}\nregua de ruido = {regua:.4f} | {'novo campeao' if campeao is not base else 'campeao mantido'} | {time.time() - t0:.0f}s")
    if _cfg["apos"]:
        _cfg["apos"](r)
    return campeao, pd.DataFrame(linhas)


def historico():
    """A evolucao do campeao ao longo dos blocos ja concluidos, na ordem em que rodaram."""
    arqs = [json.load(open(os.path.join(_cfg["pasta"], a))) for a in os.listdir(_cfg["pasta"]) if a.endswith(".json")]
    linhas = []
    for r in sorted(arqs, key=lambda r: r["nome"]):
        aceita = [l for l in r["tabela"] if l["decisao"] == "ACEITA"]
        linhas.append(dict(bloco=r["nome"], pergunta=r["pergunta"], ks_base=r["tabela"][0]["ks"], regua=r["regua"],
                           aceita=aceita[0]["variante"] if aceita else "-", delta=aceita[0]["delta"] if aceita else 0.0))
    return pd.DataFrame(linhas)


def dobras_de(c):
    """As dobras oficiais com os dados do campeao (mesma divisao, mesmos clientes; com c vazio, iguais a carregar_dobras)."""
    X_train, X_test, y_train, y_test = comum.dividir(comum.carregar_df())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=comum.SEMENTE)
    k, folds = _kwargs(c), []
    for a, b in skf.split(X_train, y_train):
        Xa, ya, Xb = EV._prep(X_train.iloc[a], y_train.iloc[a], X_train.iloc[b], **k)
        folds.append((Xa, ya, Xb, y_train.iloc[b].values.astype("float32")))
    Xa, ya, Xb = EV._prep(X_train, y_train, X_test, **k)
    return folds, (Xa, ya, Xb, y_test.values.astype("float32"))


def final_campeao(c, n_seeds=5):
    """Teste do campeao congelado: treino inteiro, 5 sementes, limiar vindo da validacao. Usar uma unica vez, no fim."""
    folds, final = dobras_de(c)
    P, limiares = [], []
    for s in range(n_seeds):
        oof_p, oof_y, eps = [], [], []
        for i, f in enumerate(folds):
            (pv,), _, ep = M.fit(c["hp"], f, [f[2]], 1000 * s + 42 + i)
            oof_p.append(pv); oof_y.append(f[3]); eps.append(ep)
        limiares.append(comum.limiar_ks(np.concatenate(oof_y), np.concatenate(oof_p)))
        P.append(M.fit_final(c["hp"], final, eps, 1000 * s + 42))
    p = np.mean(P, 0); yte = final[3]
    m = comum.metricas(yte, p, float(np.mean(limiares))); m["ks_ic95"] = comum.bootstrap_ks(yte, p)
    return dict(campeao=c, teste=m, ks_teste_por_semente=[comum.ks(yte, x) for x in P], probabilidades_teste=p.tolist())
