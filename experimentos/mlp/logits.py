"""Estudo dos logits: quando, em qual camada e em quais clientes a rede decora.

Compara sempre clientes unicos do treino (sem as copias do oversampling) com clientes que a rede nao viu. O teste aparece
nas curvas por epoca como informacao; a escolha da camada a regularizar usa so treino e validacao.
"""
import sys, os
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, evolucao as EV, blocos as BL, diagnostico as D, fase3 as F3


def _logits(m, X):
    with torch.no_grad():
        z = m(torch.tensor(X, device=M.DEVICE)).cpu().numpy()
    return np.nan_to_num(z.astype("float64"), nan=0.0, posinf=30.0, neginf=-30.0)


def _ocultas(m, X):
    """A saida de cada camada oculta, depois da ativacao e do Dropout (que fica inativo em eval)."""
    saidas, x = [], torch.tensor(X, device=M.DEVICE)
    camadas = list(m.rede.children())
    with torch.no_grad():
        for i, c in enumerate(camadas[:-1]):
            x = c(x)
            if isinstance(camadas[i + 1], torch.nn.Linear):
                saidas.append(x.cpu().numpy())
    return saidas


def _resumo(z, y, prefixo):
    p = 1 / (1 + np.exp(-z))
    return {f"{prefixo}_ce": comum.ce(y, p), f"{prefixo}_ks": comum.ks(y, p), f"{prefixo}_abs_logit": float(np.abs(z).mean()), f"{prefixo}_saturados": float((np.abs(z) > 4).mean()),
            f"{prefixo}_separacao": float(z[y == 1].mean() - z[y == 0].mean())}


def _dobras(c, n_dobras=5):
    """Para cada dobra: treino com oversampling, os mesmos clientes sem copias, validacao e teste, todos na mesma escala."""
    Xtr, Xte, ytr, yte = F3._dados()
    k = BL._kwargs(c)
    k_sem = {**k, "over": False}
    for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=comum.SEMENTE).split(Xtr, ytr)):
        if i >= n_dobras:
            break
        Xa, ya, Xb = EV._prep(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], **k)
        Xu, yu, Xt = EV._prep(Xtr.iloc[a], ytr.iloc[a], Xte, **k_sem)
        yield i, dict(a=a, b=b, Xa=Xa, ya=ya, Xu=Xu, yu=yu, Xb=Xb, yb=ytr.iloc[b].values.astype("float32"), Xt=Xt, yt=yte.values.astype("float32"))


def _curva_dobra(c, i, D_, s, epocas):
    linhas = []

    def obs(ep, m):
        linhas.append(dict(epoca=ep, **_resumo(_logits(m, D_["Xu"]), D_["yu"], "treino"), **_resumo(_logits(m, D_["Xb"]), D_["yb"], "val"), **_resumo(_logits(m, D_["Xt"]), D_["yt"], "teste")))
    M.treinar(D_["Xa"], D_["ya"], c["hp"], 1000 * s + 42 + i, epocas=epocas, ao_fim_epoca=obs)
    return linhas


def curva_epocas(c, epocas=150, n_seeds=2, n_jobs=1):
    """Treina sem parar antes e olha os logits a cada epoca: quando a validacao descola do treino, a rede comecou a decorar."""
    dobras = list(_dobras(c))
    res = Parallel(n_jobs=n_jobs)(delayed(_curva_dobra)(c, i, d, s, epocas) for s in range(n_seeds) for i, d in dobras)
    t = pd.DataFrame([l for r in res for l in r]).groupby("epoca").mean().reset_index()
    t["lacuna_ce"] = t.val_ce - t.treino_ce
    t["lacuna_ks"] = t.treino_ks - t.val_ks
    return t


def resumir_curva(t):
    """Os marcos da curva. A epoca em que a perda de validacao bate no minimo e o inicio pratico da decoreba."""
    v = t.val_ce.rolling(3, min_periods=1, center=True).mean()
    ep_ce = int(t.epoca.iloc[int(v.values.argmin())])
    no = lambda ep, col: float(t.loc[t.epoca == ep, col].iloc[0])
    fim = int(t.epoca.iloc[-1])
    ep_ks, ep_ks_t = int(t.epoca.iloc[int(t.val_ks.values.argmax())]), int(t.epoca.iloc[int(t.teste_ks.values.argmax())])
    return dict(epoca_min_ce_val=ep_ce, epoca_max_ks_val=ep_ks, epoca_max_ks_teste=ep_ks_t, ks_val_no_min_ce=no(ep_ce, "val_ks"), ks_val_max=no(ep_ks, "val_ks"), ks_teste_max=no(ep_ks_t, "teste_ks"),
                ks_teste_no_min_ce=no(ep_ce, "teste_ks"), lacuna_ks_no_min_ce=no(ep_ce, "lacuna_ks"), lacuna_ks_fim=no(fim, "lacuna_ks"), lacuna_ce_no_min_ce=no(ep_ce, "lacuna_ce"), lacuna_ce_fim=no(fim, "lacuna_ce"),
                abs_logit_val_no_min_ce=no(ep_ce, "val_abs_logit"), abs_logit_val_fim=no(fim, "val_abs_logit"), saturados_val_fim=no(fim, "val_saturados"), epocas_observadas=fim)


def _sonda(Za, ya, Zb, yb):
    """Um classificador linear sobre a saida da camada: mede o que ela guardou de geral e o que guardou so do treino."""
    sc = StandardScaler().fit(Za)
    lr = LogisticRegression(max_iter=300).fit(sc.transform(Za), ya)
    return comum.ks(ya, lr.predict_proba(sc.transform(Za))[:, 1]), comum.ks(yb, lr.predict_proba(sc.transform(Zb))[:, 1])


def _camadas_dobra(c, i, d, s, epocas_longas):
    linhas = []
    parada, _ = M.treinar(d["Xa"], d["ya"], c["hp"], 1000 * s + 42 + i, d["Xb"], d["yb"])
    longa, _ = M.treinar(d["Xa"], d["ya"], c["hp"], 1000 * s + 42 + i, epocas=epocas_longas)
    tr, va = _sonda(d["Xu"], d["yu"], d["Xb"], d["yb"])
    for momento, m in (("parada", parada), ("longa", longa)):
        linhas.append(dict(momento=momento, camada=0, ks_treino=tr, ks_val=va))
        for k, (Zu, Zb) in enumerate(zip(_ocultas(m, d["Xu"]), _ocultas(m, d["Xb"])), 1):
            tr_k, va_k = _sonda(Zu, d["yu"], Zb, d["yb"])
            linhas.append(dict(momento=momento, camada=k, ks_treino=tr_k, ks_val=va_k, unidades_mortas=float((np.abs(Zu).max(0) < 1e-6).mean())))
    return linhas


def por_camada(c, epocas_longas=150, n_seeds=2, n_jobs=1):
    """A camada em que a lacuna treino-validacao mais cresce e onde a rede decora; a camada 0 e a entrada."""
    dobras = list(_dobras(c))
    res = Parallel(n_jobs=n_jobs)(delayed(_camadas_dobra)(c, i, d, s, epocas_longas) for s in range(n_seeds) for i, d in dobras)
    t = pd.DataFrame([l for r in res for l in r]).groupby(["momento", "camada"]).mean().reset_index()
    t["lacuna"] = t.ks_treino - t.ks_val
    t["aumento_lacuna"] = t.groupby("momento").lacuna.diff()
    t["perda_sinal_val"] = -t.groupby("momento").ks_val.diff()
    return t


def camada_critica(t, momento="longa"):
    """Numero da camada oculta (a partir de 1) com o maior aumento de lacuna, ou None se nenhuma aumenta."""
    x = t[(t.momento == momento) & (t.camada > 0)]
    if x.empty or not (x.aumento_lacuna > 0).any():
        return None
    return int(x.loc[x.aumento_lacuna.idxmax(), "camada"])


def _cliente_dobra(c, i, d, s, epocas_longas):
    m_p, _ = M.treinar(d["Xa"], d["ya"], c["hp"], 1000 * s + 42 + i, d["Xb"], d["yb"])
    m_l, _ = M.treinar(d["Xa"], d["ya"], c["hp"], 1000 * s + 42 + i, epocas=epocas_longas)
    return dict(a=d["a"], b=d["b"], zu_p=_logits(m_p, d["Xu"]), zb_p=_logits(m_p, d["Xb"]), zu_l=_logits(m_l, d["Xu"]), zb_l=_logits(m_l, d["Xb"]))


def _grupos(X, lacuna, n_min=40):
    S, media, linhas = D._segmentos(X), lacuna.mean(), []
    for nome, m in S.items():
        if int(m.sum()) >= n_min:
            ep = lacuna[m].std(ddof=1) / np.sqrt(m.sum())
            linhas.append(dict(grupo=nome, n=int(m.sum()), lacuna=lacuna[m].mean(), acima_da_media=lacuna[m].mean() - media, z=(lacuna[m].mean() - media) / ep if ep > 0 else 0.0))
    t = pd.DataFrame(linhas)
    t["p"] = 2 * norm.sf(t.z.abs())
    t = t.sort_values("p")
    t["q_bh"] = (t.p * len(t) / np.arange(1, len(t) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return t.sort_values("acima_da_media", ascending=False).reset_index(drop=True)


def por_cliente(c, epocas_longas=150, n_seeds=3, n_jobs=1):
    """Cada cliente aparece no treino de 4 dobras e na validacao de 1. Margem = logit com o sinal do rotulo verdadeiro:
    a diferenca entre a margem de quem esta no treino e a de quem esta fora mede quanto a rede decorou aquele cliente."""
    Xtr, _, ytr, _ = F3._dados()
    dobras = list(_dobras(c))
    res = Parallel(n_jobs=n_jobs)(delayed(_cliente_dobra)(c, i, d, s, epocas_longas) for s in range(n_seeds) for i, d in dobras)
    n = len(Xtr)
    somas = {k: np.zeros(n) for k in ("in_p", "fora_p", "in_l", "fora_l")}
    conta_in, conta_fora = np.zeros(n), np.zeros(n)
    for r in res:
        somas["in_p"][r["a"]] += r["zu_p"]; somas["in_l"][r["a"]] += r["zu_l"]; conta_in[r["a"]] += 1
        somas["fora_p"][r["b"]] += r["zb_p"]; somas["fora_l"][r["b"]] += r["zb_l"]; conta_fora[r["b"]] += 1
    sinal = 2 * ytr.values.astype(float) - 1
    t = pd.DataFrame(dict(y=ytr.values.astype(int), logit_fora=somas["fora_p"] / conta_fora,
                          margem_dentro_parada=sinal * somas["in_p"] / conta_in, margem_fora_parada=sinal * somas["fora_p"] / conta_fora,
                          margem_dentro_longa=sinal * somas["in_l"] / conta_in, margem_fora_longa=sinal * somas["fora_l"] / conta_fora), index=Xtr.index)
    t["lacuna_parada"] = t.margem_dentro_parada - t.margem_fora_parada
    t["lacuna_longa"] = t.margem_dentro_longa - t.margem_fora_longa
    t["decorado_longa"] = (t.margem_dentro_longa > 0) & (t.margem_fora_longa < 0)      # acerta quando viu, erra quando nao viu
    return t, _grupos(Xtr, t.lacuna_longa), _grupos(Xtr, t.lacuna_parada)


def resumir_clientes(t):
    """A decoreba por rotulo e pela distancia da fronteira: cliente ambiguo costuma ser o mais decorado."""
    por_rotulo = t.groupby("y").agg(clientes=("y", "size"), lacuna_parada=("lacuna_parada", "mean"), lacuna_longa=("lacuna_longa", "mean"), decorados=("decorado_longa", "mean")).reset_index()
    faixa = pd.qcut(t.logit_fora.abs(), 5, duplicates="drop")
    por_ambiguidade = t.groupby(faixa, observed=True).agg(clientes=("y", "size"), lacuna_parada=("lacuna_parada", "mean"), lacuna_longa=("lacuna_longa", "mean"), decorados=("decorado_longa", "mean")).reset_index()
    por_ambiguidade["logit_fora"] = por_ambiguidade.logit_fora.astype(str)
    return por_rotulo, por_ambiguidade


def estudar(c, epocas_longas=150, n_seeds=2, n_jobs=1):
    """As tres perguntas de uma vez: quando, em qual camada e em quais clientes."""
    curva = curva_epocas(c, epocas_longas, n_seeds, n_jobs)
    camadas = por_camada(c, epocas_longas, n_seeds, n_jobs)
    clientes, grupos_longa, grupos_parada = por_cliente(c, epocas_longas, n_seeds + 1, n_jobs)
    rotulo, ambig = resumir_clientes(clientes)
    return dict(curva=curva, marcos=resumir_curva(curva), camadas=camadas, camada_critica=camada_critica(camadas), grupos_longa=grupos_longa, grupos_parada=grupos_parada, por_rotulo=rotulo, por_ambiguidade=ambig)


def para_json(e):
    return {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in e.items()}
