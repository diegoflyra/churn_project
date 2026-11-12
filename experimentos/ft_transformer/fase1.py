"""Fase 1 do FT-Transformer: linha de base no placar comum (comum.py, docs/pre-proposta.md "Placar comum").

So os dados do pre-processamento base (comum.preparar, sem features novas). Perguntas:
1. Quanto mede o FT-Transformer padrao do artigo (trial 0)?
2. Uma grade pequena de tamanho (blocos x d_token) muda alguma coisa? Com 5.282 clientes, um Transformer grande pode decorar.
3. Sem oversampling, o KS se mantem e o MSE melhora, como no MLP?

Medida de cada dobra: 80% do treino da dobra ajustam e 20% decidem a parada (perda nessa separacao); depois a rede
retreina no treino inteiro da dobra pelo numero de epocas encontrado na propria dobra. A dobra de validacao so mede.
"""
import sys, os, json, time, itertools
from functools import lru_cache
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M

MODELO = "ft_transformer"
TRIAL0 = dict(hp=dict(M.PADRAO), dados={})                     # padrao do artigo + pre-processamento base do grupo
ABREV = {"n_blocos": "b", "d_token": "d", "n_cabecas": "h", "lr": "lr", "wd": "wd"}


def _dados():
    return comum.dividir(comum.carregar_df())


@lru_cache(maxsize=1)
def divs():
    return comum.divisoes_placar(np.asarray(_dados()[2]))


def assinatura():
    return comum.assinatura_treino(_dados()[0])


def _chave(c):
    return json.dumps([c["hp"], c["dados"]], sort_keys=True)


# ---------------- uma dobra ----------------
def ajustar_um(c, X, y, X_val, ajuste, parada, semente, dispositivo):
    """Contrato do placar: X, y = treino cru da dobra. A parada so ve X.iloc[parada]; X_val so recebe a previsao."""
    hp, dados = c["hp"], c["dados"]
    Xa, ya, Xp = comum.preparar(X.iloc[ajuste], y.iloc[ajuste], X.iloc[parada], **dados)
    yp = y.iloc[parada]
    f32 = lambda a: np.asarray(a, dtype="float32")
    _, ep = M.treinar(f32(Xa), f32(ya), hp, semente, f32(Xp), f32(yp), dispositivo=dispositivo)
    XA, yA, XV = comum.preparar(X, y, X_val, **dados)
    m, _ = M.treinar(f32(XA), f32(yA), hp, semente, epocas=max(1, ep), dispositivo=dispositivo)
    return dict(p_val=M.prever(m, f32(XV)), y_tr=np.asarray(yA), p_tr=M.prever(m, f32(XA)), epoca=int(ep))


def _semente(d, semente):
    return 42 + 1000 * d["semente"] + d["dobra"] + 100000 * semente


def medir(c, dispositivos=None, por_dispositivo=2, semente=0):
    """Mede uma configuracao no placar (15 dobras). As dobras rodam em paralelo: `por_dispositivo` processos em cada
    dispositivo (GPU ou "cpu"). Devolve comum.resumir_placar com as epocas da parada."""
    dispositivos = list(dispositivos or [M.DEVICE])
    Xtr, _, ytr, _ = _dados()
    D, t0 = divs(), time.time()
    tarefas = [(c, Xtr.iloc[d["treino"]], ytr.iloc[d["treino"]], Xtr.iloc[d["validacao"]], d["ajuste"], d["parada"],
                _semente(d, semente), dispositivos[k % len(dispositivos)]) for k, d in enumerate(D)]
    n_jobs = min(len(D), len(dispositivos) * por_dispositivo)
    if n_jobs == 1:
        rs = [ajustar_um(*t) for t in tarefas]
    else:
        rs = Parallel(n_jobs=n_jobs)(delayed(ajustar_um)(*t) for t in tarefas)
    eps = [r["epoca"] for r in rs]
    return comum.resumir_placar(np.asarray(ytr), D, [r["p_val"] for r in rs], [(r["y_tr"], r["p_tr"]) for r in rs],
                                extra=dict(epocas=float(np.median(eps)), epocas_dobras=eps, tempo_min=(time.time() - t0) / 60))


def garantir(pasta, itens, dispositivos=None, por_dispositivo=2, continuar=lambda: True, ao_registrar=None):
    """itens = [(rotulo, config, contra)], contra = {"vs_trial0": ("ft_transformer", "trial0")}.
    Mede e registra uma configuracao por vez, na ordem dada; o que ja esta no placar com a mesma configuracao e o mesmo
    treino e reaproveitado. Com continuar() falso, nao comeca configuracoes novas (limite de tempo do Kaggle)."""
    sig, ytr = assinatura(), np.asarray(_dados()[2])
    existentes = {r["rotulo"]: r for r in comum.ler_placar(pasta) if r["modelo"] == MODELO}
    linhas = []
    for rot, c, contra in itens:
        r = existentes.get(rot)
        if not (r and r["assinatura"] == sig and _chave(r["config"]) == _chave(c)):
            if not continuar():
                print(f"limite de tempo: {rot} fica para a proxima execucao", flush=True)
                continue
            med = medir(c, dispositivos, por_dispositivo)
            r = comum.registrar_placar(pasta, MODELO, rot, c, med, ytr, sig, {k: tuple(v) for k, v in (contra or {}).items()})
            existentes[rot] = r
            print(f"{rot:<22} KS placar {r['ks_kfold']:.4f} (ep {r['ep']:.4f}) | epocas {r['epocas']:.0f} | {r['tempo_min']:.1f} min", flush=True)
            if ao_registrar:
                ao_registrar(r)
        linhas.append(r)
    return linhas


# ---------------- configuracoes ----------------
def grade(base, **eixos):
    """{rotulo: config}: produto dos valores pedidos sobre a base. A celula igual a base e o proprio trial 0 e fica de fora."""
    nomes = list(eixos)
    saida = {}
    for vals in itertools.product(*eixos.values()):
        hp = dict(base["hp"], **dict(zip(nomes, vals)))
        if hp == base["hp"]:
            continue
        rot = "g_" + "_".join(f"{ABREV.get(n, n)}{v:g}" if isinstance(v, float) else f"{ABREV.get(n, n)}{v}" for n, v in zip(nomes, vals))
        saida[rot] = dict(hp=hp, dados=dict(base["dados"]))
    return saida


def tabela(linhas, campos=("vs_trial0",)):
    rows = []
    for r in linhas:
        hp = r["config"]["hp"]
        row = dict(rotulo=r["rotulo"], blocos=hp["n_blocos"], d_token=hp["d_token"], over=r["config"]["dados"].get("over", True),
                   ks_kfold=r["ks_kfold"], ep=r["ep"], ks_oof=r["ks_oof"], auc_oof=r["auc_oof"],
                   mse_treino=r.get("mse_treino"), mse_val=r.get("mse_val"), ce_val=r.get("ce_val"), epocas=r.get("epocas"),
                   minutos=r.get("tempo_min"))
        for k in campos:
            v = r.get(k)
            row[k] = "" if not v else f"{v['veredito']} {v['delta']:+.4f} [{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)
