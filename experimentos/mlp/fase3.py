"""Fase 3 do MLP: treinar do zero e procurar padroes, com o teste liberado.

Duas linhagens andam lado a lado:
- "val":   escolhe so pela validacao. O KS de teste dela continua comparavel com os outros modelos.
- "teste": escolhe pelo KS de teste. E a busca sem freio; o numero dela e otimista e fica fora da comparacao entre modelos.

O teste nunca entra no ajuste (scaler, oversampling, pesos ou parada): ele so decide qual configuracao segue na linhagem
"teste". Cada medicao sai de divisoes do treino: `rep` muda a divisao em 5 dobras e `semente` muda so a inicializacao.
"""
import sys, os, json, itertools
from functools import lru_cache
import numpy as np, pandas as pd, optuna
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, evolucao as EV, blocos as BL, pilha as P

MIN_VITORIAS = BL.MIN_VITORIAS
COLUNAS = {"val": "val", "teste": "teste"}
PESO_CLASSES = 2.77     # razao nao churn / churn, para o teste sem oversampling
_cache = {"arq": None, "dados": {}}
_pilha = {"dispositivos": None}


def usar_pilha(dispositivos):
    """Com dispositivos, as medicoes treinam em pilha (pilha.py). Com None, uma rede por vez, em processos (medir)."""
    _pilha["dispositivos"] = dispositivos


def _medir_varios(tarefas, n_jobs=1):
    if _pilha["dispositivos"]:
        return P.medir_pilha(tarefas, _dados(), _pilha["dispositivos"])
    return Parallel(n_jobs=n_jobs)(delayed(medir)(*t) for t in tarefas)


def planos(profundidades=(1, 2, 3, 4)):
    """Todas as permutacoes de N, B e D por camada. Uma letra por camada: BN e Dropout nunca ficam juntos."""
    saida = []
    for d in profundidades:
        saida += ["".join(p) for p in itertools.product("NBD", repeat=d)]
    return saida


def grade(base, profundidades=(1, 2, 3, 4), ativacoes=("relu", "gelu", "leaky_relu"), larguras=(16, 32, 64), dropout=0.2):
    """{nome: campeao} com as features de `base` e cada combinacao de profundidade, largura, ativacao e plano."""
    saida = {}
    for pl, at, u in itertools.product(planos(profundidades), ativacoes, larguras):
        hp = {**base["hp"], "camadas": len(pl), "unidades": u, "ativ": at, "plano": pl, "dropout": dropout}
        saida[f"{len(pl)}x{u}_{at}_{pl}"] = dict(hp=hp, dados=base["dados"], extras=base["extras"])
    return saida


@lru_cache(maxsize=1)
def _dados():
    return comum.dividir(comum.carregar_df())


def medir(c, rep, semente=0):
    """KS medio em 5 dobras na divisao `rep` e KS de teste do modelo treinado no treino inteiro, pela mediana das epocas."""
    Xtr, Xte, ytr, yte = _dados()
    k, s0 = BL._kwargs(c), 42 + 1000 * rep + 100000 * semente
    kss, eps = [], []
    for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=comum.SEMENTE + rep).split(Xtr, ytr)):
        Xa, ya, Xb = EV._prep(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], **k)
        yb = ytr.iloc[b].values.astype("float32")
        (p,), _, ep = M.fit(c["hp"], (Xa, ya, Xb, yb), [Xb], s0 + i)
        kss.append(comum.ks(yb, p)); eps.append(ep)
    Xa, ya, Xt = EV._prep(Xtr, ytr, Xte, **k)
    y_t = yte.values.astype("float32")
    pt = M.fit_final(c["hp"], (Xa, ya, Xt, y_t), eps, s0)
    return dict(val=float(np.mean(kss)), teste=comum.ks(y_t, pt), epocas=float(np.median(eps)), prob=pt)


def _chave(c, rep, semente):
    return json.dumps([c["hp"], c["dados"], c["extras"], rep, semente], sort_keys=True)


def abrir_cache(pasta):
    """As medicoes ficam em disco: uma interrupcao perde pouco, e as duas linhagens reaproveitam o que tiverem em comum."""
    _cache.update(arq=os.path.join(pasta, "medidas.jsonl"), dados={})
    if os.path.isfile(_cache["arq"]):
        for l in open(_cache["arq"], encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:      # ultima linha cortada por uma interrupcao
                continue
            _cache["dados"][r["chave"]] = r


def medir_muitos(tarefas, n_jobs=1):
    """tarefas = [(c, rep, semente)]. Devolve as medidas na mesma ordem, sem repetir o que ja esta no cache."""
    faltam = {}
    for t in tarefas:
        ch = _chave(*t)
        if ch not in _cache["dados"]:
            faltam[ch] = t
    if faltam:
        res = _medir_varios(list(faltam.values()), n_jobs)
        with open(_cache["arq"], "a", encoding="utf-8") as f:
            for ch, r in zip(faltam, res):
                linha = dict(chave=ch, val=r["val"], teste=r["teste"], epocas=r["epocas"])
                _cache["dados"][ch] = linha
                f.write(json.dumps(linha) + "\n")
    return [_cache["dados"][_chave(*t)] for t in tarefas]


def conferir_pilha(configs, reps, n_jobs=1, dispositivos=None):
    """A pilha mede o mesmo que a referencia? Mesmas configuracoes e divisoes nos dois treinadores. O sorteio muda, entao o numero
    nao repete: o que importa e a diferenca ficar no tamanho do ruido de trocar a semente da propria referencia, sem vies."""
    nomes, reps = list(configs), list(reps)
    tarefas = [(configs[n], r, 0) for n in nomes for r in reps]
    ref = Parallel(n_jobs=n_jobs)(delayed(medir)(*t) for t in tarefas)
    ref2 = Parallel(n_jobs=n_jobs)(delayed(medir)(c, r, 1) for c, r, _ in tarefas)
    pil = P.medir_pilha(tarefas, _dados(), dispositivos or _pilha["dispositivos"])
    col = lambda R, k: np.array([x[k] for x in R]).reshape(len(nomes), len(reps))
    dv, dt = col(pil, "val") - col(ref, "val"), col(pil, "teste") - col(ref, "teste")
    rv, rt = col(ref2, "val") - col(ref, "val"), col(ref2, "teste") - col(ref, "teste")
    tab = pd.DataFrame(dict(config=nomes, val_ref=col(ref, "val").mean(1), val_pilha=col(pil, "val").mean(1), d_val=dv.mean(1), ruido_val=rv.mean(1),
                            teste_ref=col(ref, "teste").mean(1), teste_pilha=col(pil, "teste").mean(1), d_teste=dt.mean(1), ruido_teste=rt.mean(1),
                            epocas_ref=col(ref, "epocas").mean(1), epocas_pilha=col(pil, "epocas").mean(1)))
    ep = lambda d: float(d.std(ddof=1) / np.sqrt(d.size))
    return tab, dict(vies_val=float(dv.mean()), ep_vies_val=ep(dv), vies_teste=float(dt.mean()), ep_vies_teste=ep(dt), ruido_val=ep(rv), ruido_teste=ep(rt),
                     vies_epocas=float((col(pil, "epocas") - col(ref, "epocas")).mean()), ep_vies_epocas=ep(col(pil, "epocas") - col(ref, "epocas")), n=int(dv.size))


# ---------------- grade: efeitos medios de cada escolha ----------------
def rodar_grade(pasta, configs, reps, n_jobs=1, continuar=lambda: True, ao_salvar=None):
    """Mede cada configuracao em `reps` divisoes. Salva a tabela e a probabilidade media de teste a cada lote; retoma de onde parou."""
    arq, arq_p = os.path.join(pasta, "grade.jsonl"), os.path.join(pasta, "grade_prob_teste.npz")
    linhas, probs = [], {}
    if os.path.isfile(arq) and os.path.isfile(arq_p):
        guardadas = np.load(arq_p)
        probs = {k: guardadas[k] for k in guardadas.files}
        linhas = [json.loads(l) for l in open(arq, encoding="utf-8")]
        linhas = [l for l in linhas if l["nome"] in probs]
    feitas = {l["nome"] for l in linhas}
    faltam = [n for n in configs if n not in feitas]
    if _pilha["dispositivos"]:       # configuracoes de mesmo formato treinam juntas: e dai que vem o ganho da GPU
        por_chave = {}
        for n in faltam:
            por_chave.setdefault(P.chave_hp(configs[n]["hp"]), []).append(n)
        grupos = list(por_chave.values())
    else:
        tam = max(1, n_jobs) * 4
        grupos = [faltam[i:i + tam] for i in range(0, len(faltam), tam)]
    for nomes in grupos:
        if not continuar():
            print("Limite de tempo: a grade parou antes do fim e pode ser retomada.")
            break
        tarefas = [(n, r) for n in nomes for r in reps]
        res = _medir_varios([(configs[n], r, 0) for n, r in tarefas], n_jobs)
        por_nome = {n: [x for (m, _), x in zip(tarefas, res) if m == n] for n in nomes}
        for n in nomes:
            rs, hp = por_nome[n], configs[n]["hp"]
            probs[n] = np.mean([r["prob"] for r in rs], 0).astype("float16")
            linhas.append(dict(nome=n, camadas=hp["camadas"], unidades=hp["unidades"], ativ=hp["ativ"], plano=hp["plano"],
                               val=float(np.mean([r["val"] for r in rs])), teste=float(np.mean([r["teste"] for r in rs])),
                               epocas=float(np.mean([r["epocas"] for r in rs]))))
        np.savez_compressed(arq_p, **probs)
        with open(arq, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(l) + "\n" for l in linhas)
        print(f"grade: {len(linhas)}/{len(configs)} | melhor val {max(l['val'] for l in linhas):.4f} | melhor teste {max(l['teste'] for l in linhas):.4f}", flush=True)
        if ao_salvar:
            ao_salvar()
    return pd.DataFrame(linhas), probs


def efeitos(tab, coluna):
    """Regressao com o efeito medio de cada escolha, mantidas as demais. Medias sobre muitas configuracoes reduzem o ruido
    que nenhuma comparacao isolada conseguiu vencer nas fases anteriores. O nivel de referencia e 1 camada, 16 unidades, relu e nada."""
    camadas = [d for d in sorted(tab.camadas.unique()) if d > 1]
    larguras = sorted(tab.unidades.unique())[1:]
    ativs = [a for a in sorted(tab.ativ.unique()) if a != "relu"]
    prof = int(tab.camadas.max())
    termos, X = ["intercepto"], [np.ones(len(tab))]
    for d in camadas:
        termos.append(f"{d} camadas"); X.append((tab.camadas == d).astype(float).values)
    for u in larguras:
        termos.append(f"{u} unidades"); X.append((tab.unidades == u).astype(float).values)
    for a in ativs:
        termos.append(a); X.append((tab.ativ == a).astype(float).values)
    for k in range(prof):
        for letra, nome in (("B", "BN"), ("D", "Dropout")):
            coluna_x = tab.plano.map(lambda p: float(len(p) > k and p[k] == letra)).values
            if coluna_x.sum() > 0:
                termos.append(f"{nome} na camada {k + 1}"); X.append(coluna_x)
    X, y = np.column_stack(X), tab[coluna].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    gl = max(1, len(y) - np.linalg.matrix_rank(X))
    ep = np.sqrt(np.clip(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r) / gl, 0, None))
    return pd.DataFrame(dict(termo=termos, efeito=b, ep=ep, t=b / np.where(ep > 0, ep, np.nan)))


def padroes(tab):
    """Os efeitos na validacao e no teste, lado a lado. Um padrao confiavel aponta para o mesmo lado nos dois."""
    v, t = efeitos(tab, "val"), efeitos(tab, "teste")
    p = v.merge(t, on="termo", suffixes=("_val", "_teste"))
    p = p[p.termo != "intercepto"]
    p["concorda"] = np.sign(p.efeito_val) == np.sign(p.efeito_teste)
    p["forte_nos_dois"] = p.concorda & (p.t_val.abs() > 2) & (p.t_teste.abs() > 2)
    return p.sort_values("t_val", key=abs, ascending=False).reset_index(drop=True)


def repescagem(configs, nomes, reps, n_jobs=1):
    """Mede de novo as melhores em divisoes que a grade nao viu: ser a melhor entre centenas infla o numero de quem venceu."""
    res = medir_muitos([(configs[n], r, 0) for n in nomes for r in reps], n_jobs)
    linhas, i = [], 0
    for n in nomes:
        parte = res[i:i + len(reps)]; i += len(reps)
        linhas.append(dict(nome=n, val=np.mean([x["val"] for x in parte]), teste=np.mean([x["teste"] for x in parte])))
    return pd.DataFrame(linhas)


def otimismo_teste(tab, probs, y, nome_val, nome_ref=None, n=200, semente=0):
    """Escolher pelo teste rende de verdade ou so acerta aquele conjunto? Sorteia metades do teste, escolhe a melhor na
    metade A e mede na metade B, que nao participou da escolha. Se a queda de A para B for grande, o ganho era do conjunto."""
    rng = np.random.default_rng(semente)
    nomes = list(tab.nome)
    P = np.column_stack([probs[k].astype("float64") for k in nomes])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    linhas = []
    for _ in range(n):
        a = np.r_[rng.permutation(pos)[:len(pos) // 2], rng.permutation(neg)[:len(neg) // 2]]
        b = np.setdiff1d(np.arange(len(y)), a)
        ks_a = [comum.ks(y[a], P[a, j]) for j in range(P.shape[1])]
        j = int(np.argmax(ks_a))
        v = dict(ks_a_escolhida=ks_a[j], ks_b_escolhida=comum.ks(y[b], P[b, j]),
                 ks_b_val=comum.ks(y[b], P[b, nomes.index(nome_val)]), escolhida=nomes[j])
        if nome_ref:
            v["ks_b_ref"] = comum.ks(y[b], P[b, nomes.index(nome_ref)])
        linhas.append(v)
    d = pd.DataFrame(linhas)
    return dict(queda_a_para_b=float((d.ks_a_escolhida - d.ks_b_escolhida).mean()),
                ganho_b_contra_val=float((d.ks_b_escolhida - d.ks_b_val).mean()),
                vitorias_b_contra_val=float((d.ks_b_escolhida > d.ks_b_val).mean()),
                configuracoes_distintas=int(d.escolhida.nunique()), n=n, ks_b_ref=float(d.ks_b_ref.mean()) if nome_ref else None,
                ks_b_escolhida=float(d.ks_b_escolhida.mean()), ks_b_val=float(d.ks_b_val.mean()))


# ---------------- blocos: uma intervencao por vez, na linhagem ----------------
def aplicar(base, v):
    c = BL._aplicar(base, v)
    if v.get("tirar_extras"):
        c["extras"] = [e for e in c["extras"] if e not in v["tirar_extras"]]
    return c


def bloco(pasta, nome, pergunta, variantes, base, linhagem, reps, n_ruido=3, n_jobs=1):
    """Mesma regra de aceite das fases anteriores, aplicada na coluna da linhagem: ganho medio acima da regua de ruido e
    vitoria em pelo menos 80% das divisoes. As duas colunas ficam na tabela, para ver se validacao e teste concordam."""
    os.makedirs(os.path.join(pasta, "blocos"), exist_ok=True)
    arq = os.path.join(pasta, "blocos", f"{nome}_{linhagem}.json")
    if os.path.isfile(arq):
        r = json.load(open(arq, encoding="utf-8"))
        if r["base"] == base:
            print(f"{nome} ({linhagem}): ja concluido, reaproveitado.")
            return r["campeao"], pd.DataFrame(r["tabela"])
    col, reps = COLUNAS[linhagem], list(reps)
    cands = {n: aplicar(base, v) for n, v in variantes.items()}
    tarefas = [(base, r, 0) for r in reps] + [(base, r, s) for s in range(1, n_ruido + 1) for r in reps] + [(c, r, 0) for c in cands.values() for r in reps]
    res, L = medir_muitos(tarefas, n_jobs), len(reps)

    def bloco_de(i, k):
        return np.array([x[k] for x in res[i:i + L]])
    bv, bt = bloco_de(0, "val"), bloco_de(0, "teste")
    regua = {k: max(abs((bloco_de(L * s, k) - b).mean()) for s in range(1, n_ruido + 1)) for k, b in (("val", bv), ("teste", bt))}
    ref = bv if col == "val" else bt
    linhas = [dict(variante="(campeao atual)", val=bv.mean(), delta_val=0.0, ganhou_val="", teste=bt.mean(), delta_teste=0.0, ganhou_teste="", decisao="base")]
    melhor, campeao = 0.0, base
    for i, (n, c) in enumerate(cands.items()):
        ini = L * (n_ruido + 1 + i)
        v, t = bloco_de(ini, "val"), bloco_de(ini, "teste")
        d = (v if col == "val" else t) - ref
        passou = bool(d.mean() > regua[col] and (d > 0).mean() >= MIN_VITORIAS)
        linhas.append(dict(variante=n, val=v.mean(), delta_val=(v - bv).mean(), ganhou_val=f"{int((v > bv).sum())}/{L}", teste=t.mean(),
                           delta_teste=(t - bt).mean(), ganhou_teste=f"{int((t > bt).sum())}/{L}", decisao="supera o ruido" if passou else "rejeitada"))
        if passou and d.mean() > melhor:
            melhor, campeao = d.mean(), c
    if campeao is not base:
        for l, n in zip(linhas[1:], cands):
            if cands[n] is campeao:
                l["decisao"] = "ACEITA"
    r = dict(nome=nome, linhagem=linhagem, pergunta=pergunta, base=base, regua=regua, tabela=linhas, campeao=campeao)
    json.dump(r, open(arq, "w", encoding="utf-8"), indent=1, default=float)
    print(f"{nome} ({linhagem}): {pergunta}\nregua: val {regua['val']:.4f}, teste {regua['teste']:.4f} | {'novo campeao' if campeao is not base else 'campeao mantido'}")
    return campeao, pd.DataFrame(linhas)


def _plano_de(hp):
    return hp.get("plano") or ("D" if hp.get("dropout", 0) > 0 else "N") * hp["camadas"]


def variantes_estrutura(c, camada):
    """Onde regularizar: a camada apontada pelo estudo dos logits vem primeiro; as demais servem de comparacao."""
    pl = _plano_de(c["hp"])
    d, v = len(pl), {"sem regularizacao": dict(hp=dict(plano="N" * len(pl)))}
    if camada:
        v[f"Dropout na camada {camada}"] = dict(hp=dict(plano=pl[:camada - 1] + "D" + pl[camada:]))
        v[f"BN na camada {camada}"] = dict(hp=dict(plano=pl[:camada - 1] + "B" + pl[camada:]))
    v["Dropout em todas"] = dict(hp=dict(plano="D" * d))
    v["BN em todas"] = dict(hp=dict(plano="B" * d))
    if d > 1:
        v["BN e Dropout alternados, BN primeiro"] = dict(hp=dict(plano=("BD" * d)[:d]))
        v["BN e Dropout alternados, Dropout primeiro"] = dict(hp=dict(plano=("DB" * d)[:d]))
    return {n: x for n, x in v.items() if x["hp"]["plano"] != pl}


def variantes_intensidade(c):
    v = {f"l2 {x:g}": dict(hp=dict(l2=x)) for x in (1e-4, 1e-3, 1e-2)}
    if "D" in _plano_de(c["hp"]):
        v.update({f"dropout {x:g}": dict(hp=dict(dropout=x)) for x in (0.1, 0.3, 0.5)})
    return v


def variantes_logit():
    """Atacam a confianca excessiva diretamente no logit."""
    return {**{f"suavizacao {x:g}": dict(hp=dict(suav=x)) for x in (0.05, 0.1, 0.2)},
            **{f"penalidade no logit {x:g}": dict(hp=dict(pen_logit=x)) for x in (1e-3, 1e-2, 1e-1)}}


def variantes_treino():
    return {"lote 64": dict(hp=dict(lote=64)), "lote 128": dict(hp=dict(lote=128)), "lote 512": dict(hp=dict(lote=512)), "AdamW": dict(hp=dict(otim="adamw")),
            "lr 0,0003": dict(hp=dict(lr=3e-4)), "lr 0,003": dict(hp=dict(lr=3e-3)), "paciencia 30": dict(hp=dict(paciencia=30)), "parada pelo KS": dict(hp=dict(parada="ks"))}


def variantes_balanceamento():
    """As copias do oversampling sao exemplos faceis de decorar. Sem elas, o peso na classe positiva mede o trade-off entre os erros."""
    return {"sem oversampling": dict(dados=dict(over=False)),
            "sem oversampling, peso 2,77": dict(dados=dict(over=False), hp=dict(peso_pos=PESO_CLASSES)),
            "sem oversampling, peso 1,5": dict(dados=dict(over=False), hp=dict(peso_pos=1.5)),
            "sem oversampling, peso 4": dict(dados=dict(over=False), hp=dict(peso_pos=4.0))}


def variantes_features(c):
    """Decisoes que ficaram no limite da regua nas fases anteriores, agora com mais divisoes."""
    v = {"com tenure_ate_3": dict(extras=["tenure_ate_3"]), "sem gender": dict(extras=["sem_gender"])}
    for e in ("delta_preco", "cliente_novo"):
        if e in c["extras"]:
            v[f"sem {e}"] = dict(tirar_extras=[e])
    return v


# ---------------- Optuna: refina em volta do que a grade mostrou ----------------
def espaco(t):
    camadas = t.suggest_int("camadas", 1, 4)
    return dict(camadas=camadas, unidades=t.suggest_int("unidades", 8, 128, log=True), funil=t.suggest_categorical("funil", [False, True]),
                plano="".join(t.suggest_categorical(f"p{k + 1}", ["N", "B", "D"]) for k in range(camadas)),
                ativ=t.suggest_categorical("ativ", ["relu", "gelu", "leaky_relu", "tanh"]), otim=t.suggest_categorical("otim", ["adam", "adamw", "rmsprop"]),
                lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), dropout=t.suggest_float("dropout", 0.05, 0.5), l2=t.suggest_float("l2", 1e-6, 1e-2, log=True),
                suav=t.suggest_float("suav", 0.0, 0.2), pen_logit=t.suggest_categorical("pen_logit", [0.0, 1e-3, 1e-2, 1e-1]), perda=t.suggest_categorical("perda", ["ce", "mse"]),
                lote=t.suggest_categorical("lote", [64, 128, 256, 512]), paciencia=t.suggest_categorical("paciencia", [10, 20, 30]), parada=t.suggest_categorical("parada", ["perda", "ks"]))


def _parametros(hp):
    """O inverso de `espaco`, para o primeiro trial ser o proprio campeao."""
    pl = _plano_de(hp)
    p = dict(camadas=hp["camadas"], unidades=hp["unidades"], funil=hp.get("funil", False), ativ=hp["ativ"], otim=hp["otim"], lr=hp["lr"], dropout=max(hp["dropout"], 0.05),
             l2=hp["l2"], suav=hp.get("suav", 0.0), pen_logit=hp.get("pen_logit", 0.0), perda=hp["perda"], lote=hp.get("lote", 256), paciencia=hp["paciencia"], parada=hp.get("parada", "perda"))
    p.update({f"p{k + 1}": x for k, x in enumerate(pl)})
    return p


def estudar(pasta, linhagem, base, n_trials, reps=(0, 1), n_jobs=1, continuar=lambda: True, ao_salvar=None):
    """Busca em lotes paralelos. A linhagem define o que o Optuna maximiza. Retomavel pelo SQLite.
    Fica na CPU mesmo com a pilha ligada: cada trial tem um formato, e pilhas pequenas perdem da CPU."""
    col = COLUNAS[linhagem]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name=f"mlp_fase3_{linhagem}", storage=f"sqlite:///{pasta}/mlp_fase3_{linhagem}.db", load_if_exists=True,
                             direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if not st.trials:
        st.enqueue_trial(_parametros(base["hp"]), skip_if_exists=True)
    while continuar():
        feitos = len([t for t in st.trials if t.state.is_finished()])
        if feitos >= n_trials:
            break
        ts = [st.ask() for _ in range(min(max(1, n_jobs), n_trials - feitos))]
        cs = [dict(hp=espaco(t), dados=base["dados"], extras=base["extras"]) for t in ts]
        medidas = Parallel(n_jobs=n_jobs)(delayed(medir)(c, r) for c in cs for r in reps)
        res = [_resumir_trial(medidas[i * len(reps):(i + 1) * len(reps)]) for i in range(len(cs))]
        for t, r in zip(ts, res):
            t.set_user_attr("val", r["val"]); t.set_user_attr("teste", r["teste"])
            st.tell(t, r[col])
        print(f"optuna {linhagem}: {feitos + len(ts)}/{n_trials} | melhor {st.best_value:.4f}", flush=True)
        if ao_salvar:
            ao_salvar()
    if not [t for t in st.trials if t.state.name == "COMPLETE"]:
        return st, base
    return st, dict(hp=espaco(optuna.trial.FixedTrial(st.best_params)), dados=base["dados"], extras=base["extras"])


def _resumir_trial(rs):
    return dict(val=float(np.mean([r["val"] for r in rs])), teste=float(np.mean([r["teste"] for r in rs])))


# ---------------- final ----------------
def erros_no_teste(p):
    """Em quem o modelo final erra no teste. So mostra: nada aqui alimenta o treino nem o pre-processamento dos outros modelos."""
    import diagnostico as D
    _, Xte, _, yte = _dados()
    y, ps = yte.astype(int), pd.Series(np.asarray(p), index=yte.index, name="p")
    um, pares = D.residuos(Xte, y, ps)
    return D.por_grupo(Xte, y, ps), um, pares


def trade_off_erros(p, limiar_val=None):
    """Falsos positivos e falsos negativos em cada limiar do teste: o custo de trocar um erro pelo outro."""
    _, _, _, yte = _dados()
    y, p, linhas = yte.values.astype(int), np.asarray(p), []
    limiares = sorted(set(np.quantile(p, [.1, .2, .3, .4, .5, .6, .7, .8, .9]).tolist() + ([limiar_val] if limiar_val else [])))
    for t in limiares:
        pred = p >= t
        fp, fn, vp, vn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()), int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        linhas.append(dict(limiar=t, falsos_positivos=fp, falsos_negativos=fn, recall=vp / max(1, vp + fn), especificidade=vn / max(1, vn + fp), ks=vp / max(1, vp + fn) - fp / max(1, vn + fp)))
    return pd.DataFrame(linhas)


def bootstrap_delta(y, a, b, n=1000, semente=0):
    """Diferenca de KS de teste entre dois modelos nos mesmos clientes."""
    rng = np.random.default_rng(semente)
    d = []
    for _ in range(n):
        i = rng.choice(len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(comum.ks(y[i], a[i]) - comum.ks(y[i], b[i]))
    return dict(delta=comum.ks(y, a) - comum.ks(y, b), ic95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
