"""Fase 4 do MLP: medir melhor antes de escolher melhor.

A medida e o k-fold do treino, repetido em varias divisoes. Em cada dobra externa:
  1. o resto do treino e dividido de novo: 80% ajustam a rede e 20% decidem a parada (a dobra externa nao participa);
  2. a rede e retreinada em todo o resto, pela mediana das epocas da parada, como na avaliacao final;
  3. a dobra externa mede as duas redes. `ks` e o retreino (a medida principal); `ks_parada` e a rede que parou.

Na fase 3 a mesma dobra decidia a parada e media o KS, o que infla o numero (a "parada pelo KS" ganhou 20 de 20 na validacao
e nada no teste). Aqui o teste nao entra em nenhuma escolha: aparece uma unica vez, na avaliacao final.
"""
import sys, os, json, zlib
from functools import lru_cache
import numpy as np, pandas as pd, optuna
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, evolucao as EV, blocos as BL, pilha as P
from erros import _xgb

MIN_VITORIAS = BL.MIN_VITORIAS
FRACAO_INTERNA = 0.2
_cache = {"arq": None, "dados": {}}
_cfg = {"dispositivos": None, "banda": None, "banda_id": None}


@lru_cache(maxsize=1)
def _dados():
    return comum.dividir(comum.carregar_df())


def usar_pilha(dispositivos):
    """Com dispositivos, as medidas treinam em pilha (pilha.py). Com None, uma rede por vez, em processos."""
    _cfg["dispositivos"] = dispositivos


def definir_banda(idx):
    """Posicoes (no treino) dos clientes ambiguos. Com banda, cada medida tambem guarda o KS so nesses clientes."""
    _cfg["banda"] = None if idx is None else np.asarray(idx)
    _cfg["banda_id"] = None if idx is None else zlib.crc32(np.asarray(idx, dtype=np.int64).tobytes())


# ---------------- divisoes e preparacao ----------------
def divisoes(rep):
    """[(a1, a2, b)] por dobra externa: a1 ajusta, a2 decide a parada, b mede. Posicoes no treino oficial."""
    Xtr, _, ytr, _ = _dados()
    y, saida = np.asarray(ytr), []
    for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=comum.SEMENTE + rep).split(Xtr, y)):
        a1, a2 = train_test_split(a, test_size=FRACAO_INTERNA, stratify=y[a], random_state=comum.SEMENTE + 1000 * rep + i)
        saida.append((np.sort(a1), np.sort(a2), b))
    return saida


def _preparar(c, partes):
    """Scaler e oversampling ajustados so na parte que treina. As features criadas sao por linha, sem olhar o rotulo."""
    Xtr, _, ytr, _ = _dados()
    a1, a2, b = partes
    k = BL._kwargs(c)
    Xa1, ya1, resto = EV._prep(Xtr.iloc[a1], ytr.iloc[a1], pd.concat([Xtr.iloc[a2], Xtr.iloc[b]]), **k)
    a = np.sort(np.r_[a1, a2])
    XA, yA, XB = EV._prep(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], **k)
    y = np.asarray(ytr).astype("float32")
    return dict(Xa1=Xa1, ya1=ya1, X2=resto[:len(a2)], y2=y[a2], XB1=resto[len(a2):], XA=XA, yA=yA, XB=XB, yB=y[b], b=b)


class _Preparos:
    """Prepara cada (dados, divisao, dobra) uma vez: varias configuracoes com os mesmos dados dividem os mesmos arrays."""

    def __init__(self):
        self.div, self.prep = {}, {}

    def __call__(self, c, rep, i):
        if rep not in self.div:
            self.div[rep] = divisoes(rep)
        ch = (json.dumps([c["dados"], c["extras"]], sort_keys=True), rep, i)
        if ch not in self.prep:
            self.prep[ch] = _preparar(c, self.div[rep][i])
        return self.prep[ch]


def _resumir(Ds, p_parada, p_retreino, eps):
    oof = np.zeros(len(_dados()[0]), "float32")
    for D, p in zip(Ds, p_retreino):
        oof[D["b"]] = p
    return dict(ks=float(np.mean([comum.ks(D["yB"], p) for D, p in zip(Ds, p_retreino)])),
                ks_parada=float(np.mean([comum.ks(D["yB"], p) for D, p in zip(Ds, p_parada)])), epocas=float(np.median(eps)), oof=oof)


def _semente(rep, semente):
    return 42 + 1000 * rep + 100000 * semente


# ---------------- medida ----------------
def medir(c, rep, semente=0):
    """A medida da fase 4 com modelo.py, uma rede por vez. E a referencia da pilha."""
    prep, s0 = _Preparos(), _semente(rep, semente)
    Ds = [prep(c, rep, i) for i in range(5)]
    eps, pp = [], []
    for i, D in enumerate(Ds):
        m, ep = M.treinar(D["Xa1"], D["ya1"], c["hp"], s0 + i, D["X2"], D["y2"])
        eps.append(ep); pp.append(M.prever(m, D["XB1"]))
    e = max(1, int(np.median(eps)))
    pr = [M.prever(M.treinar(D["XA"], D["yA"], c["hp"], s0 + i, epocas=e)[0], D["XB"]) for i, D in enumerate(Ds)]
    return _resumir(Ds, pp, pr, eps)


def _medir_pilha(tarefas):
    prep, redes, Ds = _Preparos(), [], []
    for c, rep, sem in tarefas:
        s0 = _semente(rep, sem)
        for i in range(5):
            D = prep(c, rep, i); Ds.append(D)
            redes.append(dict(hp=c["hp"], Xa=D["Xa1"], ya=D["ya1"], Xb=D["X2"], yb=D["y2"], Xt=D["XB1"], semente=s0 + i))
    r1 = P.executar(redes, _cfg["dispositivos"])
    retreinos = []
    for j, (c, rep, sem) in enumerate(tarefas):
        e = max(1, int(np.median([r1[j * 5 + i]["epoca"] for i in range(5)])))
        for i in range(5):
            D = Ds[j * 5 + i]
            retreinos.append(dict(hp=c["hp"], Xa=D["XA"], ya=D["yA"], Xt=D["XB"], epocas=e, semente=_semente(rep, sem) + i))
    r2 = P.executar(retreinos, _cfg["dispositivos"])
    return [_resumir(Ds[j * 5:(j + 1) * 5], [r["p_t"] for r in r1[j * 5:(j + 1) * 5]], [r["p"] for r in r2[j * 5:(j + 1) * 5]],
                     [r["epoca"] for r in r1[j * 5:(j + 1) * 5]]) for j in range(len(tarefas))]


def medir_varios(tarefas, n_jobs=1):
    if _cfg["dispositivos"]:
        return _medir_pilha(tarefas)
    return Parallel(n_jobs=n_jobs)(delayed(medir)(*t) for t in tarefas)


def _chave_cfg(c):
    return json.dumps([c["hp"], c["dados"], c["extras"]], sort_keys=True)


def _chave(c, rep, semente):
    return json.dumps(["f4", c["hp"], c["dados"], c["extras"], rep, semente, _cfg["banda_id"]], sort_keys=True)


def abrir_cache(pasta):
    _cache.update(arq=os.path.join(pasta, "medidas_kfold.jsonl"), dados={})
    if os.path.isfile(_cache["arq"]):
        with open(_cache["arq"], encoding="utf-8") as f:
            linhas = f.readlines()
        for l in linhas:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:      # ultima linha cortada por uma interrupcao
                continue
            _cache["dados"][r["chave"]] = r


def _ks_banda(oof):
    b = _cfg["banda"]
    if b is None:
        return None
    return comum.ks(np.asarray(_dados()[2])[b], oof[b])


def _gravar(linhas):
    with open(_cache["arq"], "a", encoding="utf-8") as f:
        for l in linhas:
            _cache["dados"][l["chave"]] = l
            f.write(json.dumps(l) + "\n")


def medir_muitos(tarefas, n_jobs=1):
    """tarefas = [(c, rep, semente)]. Devolve ks, ks_parada, epocas e ks_banda na mesma ordem, sem repetir o que esta no cache."""
    faltam = {}
    for t in tarefas:
        ch = _chave(*t)
        if ch not in _cache["dados"]:
            faltam[ch] = t
    if faltam:
        res = medir_varios(list(faltam.values()), n_jobs)
        _gravar([dict(chave=ch, ks=r["ks"], ks_parada=r["ks_parada"], epocas=r["epocas"], ks_banda=_ks_banda(r["oof"])) for ch, r in zip(faltam, res)])
    return [_cache["dados"][_chave(*t)] for t in tarefas]


def medir_xgb(c, rep):
    """XGBoost nas mesmas dobras externas e com os mesmos dados: diz se uma feature ajuda o dado ou so a rede."""
    prep = _Preparos()
    ks = []
    for i in range(5):
        D = prep(c, rep, i)
        ks.append(comum.ks(D["yB"], _xgb(42 + rep).fit(D["XA"], D["yA"]).predict_proba(D["XB"])[:, 1]))
    return float(np.mean(ks))


def medir_xgb_muitos(tarefas, n_jobs=1):
    """tarefas = [(c, rep)]. Com cache."""
    chaves = [json.dumps(["xgb", c["dados"], c["extras"], rep], sort_keys=True) for c, rep in tarefas]
    faltam = {ch: t for ch, t in zip(chaves, tarefas) if ch not in _cache["dados"]}
    if faltam:
        res = Parallel(n_jobs=n_jobs)(delayed(medir_xgb)(*t) for t in faltam.values())
        _gravar([dict(chave=ch, ks=r) for ch, r in zip(faltam, res)])
    return [_cache["dados"][ch]["ks"] for ch in chaves]


# ---------------- conferencias ----------------
def conferir_pilha(configs, reps, n_jobs=1):
    """A pilha mede o mesmo que modelo.py? O sorteio muda, entao o numero muda; o que nao pode aparecer e vies."""
    nomes, reps = list(configs), list(reps)
    tarefas = [(configs[n], r, 0) for n in nomes for r in reps]
    ref = Parallel(n_jobs=n_jobs)(delayed(medir)(*t) for t in tarefas)
    ref2 = Parallel(n_jobs=n_jobs)(delayed(medir)(c, r, 1) for c, r, _ in tarefas)
    pil = _medir_pilha(tarefas)
    col = lambda R, k: np.array([x[k] for x in R]).reshape(len(nomes), len(reps))
    d, ru = col(pil, "ks") - col(ref, "ks"), col(ref2, "ks") - col(ref, "ks")
    tab = pd.DataFrame(dict(config=nomes, ks_ref=col(ref, "ks").mean(1), ks_pilha=col(pil, "ks").mean(1), d=d.mean(1), ruido=ru.mean(1),
                            epocas_ref=col(ref, "epocas").mean(1), epocas_pilha=col(pil, "epocas").mean(1)))
    ep = lambda x: float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return tab, dict(vies=float(d.mean()), ep_vies=ep(d), ruido=ep(ru), n=int(d.size))


def comparar_protocolos(configs, reps, n_jobs=1):
    """Quanto o protocolo antigo (a mesma dobra decide a parada e mede) inflava o KS, nas mesmas divisoes."""
    import fase3 as F3
    nomes, reps = list(configs), list(reps)
    tarefas = [(configs[n], r, 0) for n in nomes for r in reps]
    antigo = P.medir_pilha(tarefas, F3._dados(), _cfg["dispositivos"]) if _cfg["dispositivos"] else Parallel(n_jobs=n_jobs)(delayed(F3.medir)(*t) for t in tarefas)
    novo = medir_muitos(tarefas, n_jobs)
    col = lambda R, k: np.array([x[k] for x in R]).reshape(len(nomes), len(reps)).mean(1)
    tab = pd.DataFrame(dict(config=nomes, val_antigo=col(antigo, "val"), ks_parada=col(novo, "ks_parada"), ks=col(novo, "ks")))
    tab["otimismo_antigo"] = tab.val_antigo - tab.ks
    return tab


# ---------------- blocos ----------------
def aplicar(base, v):
    c = BL._aplicar(base, v)
    if v.get("tirar_extras"):
        c["extras"] = [e for e in c["extras"] if e not in v["tirar_extras"]]
    return c


def bloco(pasta, nome, pergunta, variantes, base, reps, n_ruido=3, n_jobs=1, xgb=False):
    """A regra das fases anteriores na medida nova: ganho medio acima da regua de ruido e vitoria em pelo menos 80% das divisoes.
    `delta_banda` mostra se o ganho aparece nos clientes ambiguos; `xgb` mede as mesmas variantes de dados no XGBoost."""
    os.makedirs(os.path.join(pasta, "blocos"), exist_ok=True)
    arq = os.path.join(pasta, "blocos", f"{nome}.json")
    if os.path.isfile(arq):
        with open(arq, encoding="utf-8") as f:
            r = json.load(f)
        if r["base"] == base:
            print(f"{nome}: ja concluido, reaproveitado.")
            return r["campeao"], pd.DataFrame(r["tabela"])
    reps = list(reps)
    cands = {n: aplicar(base, v) for n, v in variantes.items()}
    tarefas = [(base, r, 0) for r in reps] + [(base, r, s) for s in range(1, n_ruido + 1) for r in reps] + [(c, r, 0) for c in cands.values() for r in reps]
    res, L = medir_muitos(tarefas, n_jobs), len(reps)
    col = lambda i, k: np.array([np.nan if x[k] is None else x[k] for x in res[i:i + L]], dtype=float)
    b, bp, bb = col(0, "ks"), col(0, "ks_parada"), col(0, "ks_banda")
    regua = max(abs((col(L * s, "ks") - b).mean()) for s in range(1, n_ruido + 1))
    if xgb:     # o XGBoost ja usa 4 nucleos por modelo: um de cada vez
        xb = np.array(medir_xgb_muitos([(base, r) for r in reps]))
    linhas = [dict(variante="(campeao atual)", ks=b.mean(), delta=0.0, ganhou="", delta_parada=0.0, delta_banda=0.0, decisao="base")]
    melhor, campeao = 0.0, base
    for i, (n, c) in enumerate(cands.items()):
        ini = L * (n_ruido + 1 + i)
        k, d = col(ini, "ks"), col(ini, "ks") - b
        passou = bool(d.mean() > regua and (d > 0).mean() >= MIN_VITORIAS)
        l = dict(variante=n, ks=k.mean(), delta=d.mean(), ganhou=f"{int((d > 0).sum())}/{L}", delta_parada=(col(ini, "ks_parada") - bp).mean(),
                 delta_banda=float(np.nanmean(col(ini, "ks_banda") - bb)) if not np.isnan(bb).all() else None, decisao="supera o ruido" if passou else "rejeitada")
        if xgb:
            dx = np.array(medir_xgb_muitos([(c, r) for r in reps])) - xb
            l.update(delta_xgb=dx.mean(), ganhou_xgb=f"{int((dx > 0).sum())}/{L}")
        linhas.append(l)
        if passou and d.mean() > melhor:
            melhor, campeao = d.mean(), c
    if campeao is not base:
        for l, n in zip(linhas[1:], cands):
            if cands[n] is campeao:
                l["decisao"] = "ACEITA"
    r = dict(nome=nome, pergunta=pergunta, base=base, regua=regua, tabela=linhas, campeao=campeao, reps=reps)
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, default=float)
    print(f"{nome}: {pergunta}\nregua {regua:.4f} | {'novo campeao' if campeao is not base else 'campeao mantido'}")
    return campeao, pd.DataFrame(linhas)


def comparar(a, b, reps, n_jobs=1):
    """b contra a, pareado por divisao, em divisoes que nenhuma escolha viu: a estimativa honesta do ganho."""
    ra, rb = medir_muitos([(a, r, 0) for r in reps], n_jobs), medir_muitos([(b, r, 0) for r in reps], n_jobs)
    ka, kb = np.array([x["ks"] for x in ra]), np.array([x["ks"] for x in rb])
    d = kb - ka
    ep = d.std(ddof=1) / np.sqrt(len(d))
    banda = [None if x["ks_banda"] is None or y["ks_banda"] is None else y["ks_banda"] - x["ks_banda"] for x, y in zip(ra, rb)]
    return dict(ks_a=float(ka.mean()), ks_b=float(kb.mean()), delta=float(d.mean()), ep=float(ep), ic95=[float(d.mean() - 1.96 * ep), float(d.mean() + 1.96 * ep)],
                vitorias=float((d > 0).mean()), n=len(d), delta_banda=None if None in banda else float(np.mean(banda)))


def _mudam(c, v):
    return {n: x for n, x in v.items() if aplicar(c, x) != c}


def variantes_balanceamento(c):
    """O que ficou em aberto na fase 3: as copias do oversampling contra peso na classe positiva."""
    if c["dados"].get("over", True):
        v = {"sem oversampling": dict(dados=dict(over=False)), "sem oversampling, peso 1,5": dict(dados=dict(over=False), hp=dict(peso_pos=1.5)),
             "sem oversampling, peso 2,77": dict(dados=dict(over=False), hp=dict(peso_pos=2.77))}
    else:
        v = {"com oversampling, sem peso": dict(dados=dict(over=True), hp=dict(peso_pos=0.0)), "sem peso": dict(hp=dict(peso_pos=0.0)),
             "peso 1,5": dict(hp=dict(peso_pos=1.5)), "peso 2,77": dict(hp=dict(peso_pos=2.77))}
    return _mudam(c, v)


def variantes_treino(c):
    """Os padroes da grade da fase 3 (Dropout na 1a camada, GELU) e os ajustes de treino, agora na medida limpa."""
    hp = c["hp"]
    pl = hp.get("plano") or ("D" if hp.get("dropout", 0) > 0 else "N") * hp["camadas"]
    v = {}
    if pl[0] != "D":
        v["Dropout na 1ª camada"] = dict(hp=dict(plano="D" + pl[1:], dropout=hp.get("dropout") or 0.2))
    if hp["ativ"] != "gelu":
        v["GELU"] = dict(hp=dict(ativ="gelu"))
    if "D" in pl:
        v.update({f"dropout {x:g}": dict(hp=dict(dropout=x)) for x in (0.1, 0.3, 0.5)})
    v.update({f"l2 {x:g}": dict(hp=dict(l2=x)) for x in (1e-4, 1e-3)})
    v.update({"lr 0,0003": dict(hp=dict(lr=3e-4)), "lr 0,003": dict(hp=dict(lr=3e-3)), "lote 128": dict(hp=dict(lote=128)), "lote 512": dict(hp=dict(lote=512)),
              "paciencia 20": dict(hp=dict(paciencia=20)), "parada pelo KS (na separacao interna)": dict(hp=dict(parada="ks")), "suavizacao 0,05": dict(hp=dict(suav=0.05))})
    return _mudam(c, v)


def variantes_features(c, nomes):
    """Uma variante por feature candidata que ainda nao entrou. Os nomes `sem_*` retiram uma variavel."""
    return {(n.replace("sem_", "sem ") if n.startswith("sem_") else f"com {n}"): dict(extras=[n]) for n in nomes if n not in c["extras"]}


# ---------------- clientes ambiguos e relacoes ----------------
def banda_ambigua(y, p, fracao=0.3):
    """Os clientes com escore mais perto do limiar do KS: e entre eles que o KS e decidido."""
    limiar = comum.limiar_ks(y, p)
    return np.argsort(np.abs(np.asarray(p) - limiar), kind="stable")[:int(round(fracao * len(p)))], limiar


def _folhas(arvore, nomes):
    """Regra de cada folha de uma arvore rasa: chave (variavel, lado) para comparar arvores, texto para ler."""
    t, regras = arvore.tree_, {}

    def desce(no, conds):
        if t.children_left[no] == -1:
            regras[no] = conds
            return
        v, lim = nomes[t.feature[no]], t.threshold[no]
        desce(t.children_left[no], conds + [(v, "<=", lim)])
        desce(t.children_right[no], conds + [(v, ">", lim)])
    desce(0, [])
    return {no: (tuple((v, s) for v, s, _ in c), " e ".join(f"{v} {s} {lim:.4g}" for v, s, lim in c)) for no, c in regras.items()}


def relacoes(X, y, oof, fracao=0.3, n_dobras=5):
    """O que ainda separa quem cancela entre os ambiguos. So treino e previsoes fora da dobra: pode ir para os outros modelos.
    variaveis: KS de cada variavel dentro e fora da faixa. regras: folhas de arvores rasas ajustadas em parte da faixa e medidas
    no resto; `aparece_em` conta em quantas dobras a mesma regra reaparece, o que separa padrao de acaso."""
    y, oof = np.asarray(y).astype(int), np.asarray(oof)
    idx, limiar = banda_ambigua(y, oof, fracao)
    m = np.zeros(len(y), bool); m[idx] = True
    Xb, yb, taxa = X.iloc[idx].reset_index(drop=True), y[idx], float(y[idx].mean())
    linhas = []
    for c in X:
        x = X[c].values.astype(float)
        if np.unique(x[m]).size < 2:
            continue
        ks = lambda mm: max(comum.ks(y[mm], x[mm]), comum.ks(y[mm], -x[mm])) if np.unique(y[mm]).size == 2 else np.nan
        corte = 0.5 if np.unique(x).size <= 2 else np.median(x[m])
        alto = m & (x > corte)
        linhas.append(dict(variavel=c, ks_na_banda=ks(m), ks_fora_da_banda=ks(~m), churn_na_banda=taxa,
                           churn_alto=float(y[alto].mean()) if alto.any() else np.nan, churn_baixo=float(y[m & ~(x > corte)].mean()) if (m & ~(x > corte)).any() else np.nan))
    regras = {}
    for a, b in StratifiedKFold(n_dobras, shuffle=True, random_state=comum.SEMENTE).split(Xb, yb):
        arv = DecisionTreeClassifier(max_depth=2, min_samples_leaf=max(20, len(yb) // 20), random_state=0).fit(Xb.iloc[a], yb[a])
        folhas, onde = _folhas(arv, list(X.columns)), arv.apply(Xb.iloc[b])
        for no, (chave, texto) in folhas.items():
            sel = onde == no
            r = regras.setdefault(chave, dict(regra=texto, aparece_em=0, clientes=0, churn_soma=0.0))
            r["aparece_em"] += 1; r["clientes"] += int(sel.sum()); r["churn_soma"] += float(yb[b][sel].sum())
    saida = []
    for r in regras.values():
        churn = r["churn_soma"] / r["clientes"] if r["clientes"] else np.nan
        saida.append(dict(regra=r["regra"], aparece_em=r["aparece_em"], clientes=r["clientes"], churn=churn, lift=churn / taxa if taxa else np.nan))
    saida.sort(key=lambda r: (-r["aparece_em"], -abs((r["lift"] if r["lift"] == r["lift"] else 1) - 1)))
    variaveis = sorted(linhas, key=lambda r: -(r["ks_na_banda"] if r["ks_na_banda"] == r["ks_na_banda"] else 0))
    return dict(limiar=float(limiar), clientes_na_banda=int(m.sum()), churn_na_banda=taxa, fracao=fracao, variaveis=variaveis, regras=saida)


# ---------------- Optuna na medida nova ----------------
def espaco(t, base):
    """Sem BN: perdeu em 1.080 redes da fase 3. Balanceamento entra na busca, porque o oversampling ficou em aberto."""
    camadas = t.suggest_int("camadas", 1, 4)
    hp = dict(camadas=camadas, unidades=t.suggest_int("unidades", 8, 64, log=True), funil=t.suggest_categorical("funil", [False, True]),
              plano="".join(t.suggest_categorical(f"p{k + 1}", ["N", "D"]) for k in range(camadas)),
              ativ=t.suggest_categorical("ativ", ["relu", "gelu", "leaky_relu"]), otim=t.suggest_categorical("otim", ["adam", "adamw", "rmsprop"]),
              lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), dropout=t.suggest_float("dropout", 0.05, 0.5), l2=t.suggest_float("l2", 1e-6, 1e-2, log=True),
              suav=t.suggest_float("suav", 0.0, 0.2), perda=t.suggest_categorical("perda", ["ce", "mse"]), lote=t.suggest_categorical("lote", [64, 128, 256, 512]),
              paciencia=t.suggest_categorical("paciencia", [10, 20, 30]), parada=t.suggest_categorical("parada", ["perda", "ks"]),
              peso_pos=t.suggest_categorical("peso_pos", [0.0, 1.5, 2.77]))
    return dict(hp=hp, dados={"over": t.suggest_categorical("over", [True, False])}, extras=list(base["extras"]))


def _parametros(c):
    """O inverso de `espaco`, para o primeiro trial ser o proprio campeao."""
    hp = c["hp"]
    pl = (hp.get("plano") or ("D" if hp.get("dropout", 0) > 0 else "N") * hp["camadas"]).replace("B", "N")
    p = dict(camadas=hp["camadas"], unidades=min(64, max(8, hp["unidades"])), funil=hp.get("funil", False), ativ=hp["ativ"], otim=hp["otim"], lr=hp["lr"],
             dropout=min(.5, max(hp["dropout"], 0.05)), l2=hp["l2"], suav=hp.get("suav", 0.0), perda=hp["perda"], lote=hp.get("lote", 256),
             paciencia=hp["paciencia"] if hp["paciencia"] in (10, 20, 30) else 10, parada=hp.get("parada", "perda"),
             peso_pos=hp.get("peso_pos") or 0.0, over=c["dados"].get("over", True))
    p.update({f"p{k + 1}": x for k, x in enumerate(pl)})
    return p


def estudar(pasta, base, n_trials, reps=(201, 202), n_jobs=1, continuar=lambda: True, ao_salvar=None):
    """Maximiza o KS do retreino no k-fold. Fica na CPU: cada trial tem um formato, e pilhas pequenas perdem da CPU."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name="mlp_fase4", storage=f"sqlite:///{pasta}/mlp_fase4.db", load_if_exists=True,
                             direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if not st.trials:
        st.enqueue_trial(_parametros(base), skip_if_exists=True)
    while continuar():
        feitos = len([t for t in st.trials if t.state.is_finished()])
        if feitos >= n_trials:
            break
        ts = [st.ask() for _ in range(min(max(1, n_jobs), n_trials - feitos))]
        cs = [espaco(t, base) for t in ts]
        res = Parallel(n_jobs=n_jobs)(delayed(medir)(c, r) for c in cs for r in reps)
        for j, t in enumerate(ts):
            parte = res[j * len(reps):(j + 1) * len(reps)]
            t.set_user_attr("ks_parada", float(np.mean([x["ks_parada"] for x in parte])))
            st.tell(t, float(np.mean([x["ks"] for x in parte])))
        print(f"optuna: {feitos + len(ts)}/{n_trials} | melhor {st.best_value:.4f}", flush=True)
        if ao_salvar:
            ao_salvar()
    if not [t for t in st.trials if t.state.name == "COMPLETE"]:
        return st, base
    return st, espaco(optuna.trial.FixedTrial(st.best_params), base)
