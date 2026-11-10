"""Fase 5 do MLP: tudo medido no placar comum (comum.py, docs/pre-proposta.md "Placar comum").

1. As versoes de cada fase entram no placar, contra o trial 0 (1x10 padrao) e contra a referencia (campeao da fase 2).
2. Arquiteturas x ritmo de treino: a rede pequena so ganha porque as grandes param cedo? (lr baixo + paciencia alta).
3. Optuna com mais rodadas por trial (divisoes proprias, diferentes das do placar), e o vencedor medido de novo no placar.
4. O teste entra uma vez, so para a configuracao escolhida pelo placar (e para a referencia, como conferencia).

A medida de cada dobra e a da fase 4: 80% do treino da dobra ajustam, 20% decidem a parada, e a rede retreina no treino
inteiro da dobra pela mediana das epocas das 5 dobras daquela semente. A dobra de validacao so mede.
"""
import sys, os, json
from functools import lru_cache
import numpy as np, pandas as pd, optuna
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, evolucao as EV, blocos as BL, pilha as P, fase4 as F4

MODELO = "mlp"
_cfg = {"dispositivos": None}


def usar_pilha(dispositivos):
    _cfg["dispositivos"] = dispositivos


def _dados():
    return F4._dados()


@lru_cache(maxsize=1)
def divs():
    return comum.divisoes_placar(np.asarray(_dados()[2]))


def assinatura():
    return comum.assinatura_treino(_dados()[0])


def _chave_cfg(c):
    return json.dumps([c["hp"], c["dados"], c["extras"]], sort_keys=True)


# ---------------- medida no placar ----------------
class _Preparos:
    def __init__(self):
        self.prep = {}

    def __call__(self, c, k):
        ch = (json.dumps([c["dados"], c["extras"]], sort_keys=True), k)
        if ch not in self.prep:
            d = divs()[k]
            self.prep[ch] = F4._preparar(c, (d["treino"][d["ajuste"]], d["treino"][d["parada"]], d["validacao"]))
        return self.prep[ch]


def _cpu(r):
    """Mesma interface de pilha.executar, uma rede por vez com modelo.py."""
    if "Xb" in r:
        m, ep = M.treinar(r["Xa"], r["ya"], r["hp"], r["semente"], r["Xb"], r["yb"])
        return dict(epoca=ep, p_t=M.prever(m, r["Xt"]))
    m, _ = M.treinar(r["Xa"], r["ya"], r["hp"], r["semente"], epocas=r["epocas"])
    return dict(p=M.prever(m, r["Xt"]))


def _executar(redes, n_jobs):
    if _cfg["dispositivos"]:
        return P.executar(redes, _cfg["dispositivos"])
    return Parallel(n_jobs=n_jobs)(delayed(_cpu)(r) for r in redes)


def _semente(d, semente):
    return 42 + 1000 * d["semente"] + d["dobra"] + 100000 * semente


def medir(configs, semente=0, n_jobs=1):
    """Mede varias configuracoes no placar de uma vez (uma pilha so na GPU). Devolve comum.resumir_placar por config."""
    prep, D = _Preparos(), divs()
    ytr = np.asarray(_dados()[2])
    fase1 = []
    for c in configs:
        for k, d in enumerate(D):
            X = prep(c, k)
            fase1.append(dict(hp=c["hp"], Xa=X["Xa1"], ya=X["ya1"], Xb=X["X2"], yb=X["y2"], Xt=X["XB1"], semente=_semente(d, semente)))
    r1 = _executar(fase1, n_jobs)
    fase2, eps_cfg = [], []
    for j, c in enumerate(configs):
        eps = [r1[j * len(D) + k]["epoca"] for k in range(len(D))]
        eps_cfg.append(eps)
        for k, d in enumerate(D):
            X = prep(c, k)
            mesma = [i for i, e in enumerate(D) if e["semente"] == d["semente"]]
            e = max(1, int(np.median([eps[i] for i in mesma])))
            fase2.append(dict(hp=c["hp"], Xa=X["XA"], ya=X["yA"], Xt=np.concatenate([X["XB"], X["XA"]]), epocas=e, semente=_semente(d, semente)))
    r2 = _executar(fase2, n_jobs)
    saida = []
    for j, c in enumerate(configs):
        p_val, p_tr = [], []
        for k, d in enumerate(D):
            X, p = prep(c, k), np.asarray(r2[j * len(D) + k]["p"])
            n = len(X["XB"])
            p_val.append(p[:n]); p_tr.append((X["yA"], p[n:]))
        saida.append(comum.resumir_placar(ytr, D, p_val, p_tr, extra=dict(epocas=float(np.median(eps_cfg[j])))))
    return saida


def garantir(pasta, itens, semente=0, n_jobs=1):
    """itens = [(rotulo, config, contra)], contra = {"vs_x": ("mlp", rotulo)}. Mede de uma vez o que falta no placar
    (mesma configuracao e assinatura = reaproveita) e registra na ordem dada, para as referencias entrarem antes."""
    sig, existentes = assinatura(), {r["rotulo"]: r for r in comum.ler_placar(pasta) if r["modelo"] == MODELO}
    faltam = [(rot, c) for rot, c, _ in itens
              if not (rot in existentes and existentes[rot]["assinatura"] == sig and _chave_cfg(existentes[rot]["config"]) == _chave_cfg(c))]
    medidas = dict(zip([r for r, _ in faltam], medir([c for _, c in faltam], semente, n_jobs))) if faltam else {}
    ytr = np.asarray(_dados()[2])
    linhas = []
    for rot, c, contra in itens:
        if rot in medidas:
            existentes[rot] = comum.registrar_placar(pasta, MODELO, rot, c, medidas[rot], ytr, sig, {k: tuple(v) for k, v in (contra or {}).items()})
        linhas.append(existentes[rot])
    return linhas


def tabela(linhas, campos=("vs_trial0", "vs_referencia")):
    rows = []
    for r in linhas:
        row = dict(rotulo=r["rotulo"], ks_kfold=r["ks_kfold"], ep=r["ep"], ks_oof=r["ks_oof"], auc_oof=r["auc_oof"],
                   mse_treino=r.get("mse_treino"), mse_val=r.get("mse_val"), epocas=r.get("epocas"))
        for k in campos:
            v = r.get(k)
            row[k] = "" if not v else f"{v['veredito']} {v['delta']:+.4f} [{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------- configuracoes ----------------
TRIAL0 = dict(hp=dict(camadas=1, unidades=10, ativ="relu", otim="adam", lr=1e-3, dropout=0.0, l2=1e-6, perda="ce", paciencia=10, plano="N"),
              dados={}, extras=[])

RITMOS = {"padrao": dict(lr=1e-3, paciencia=10), "lento": dict(lr=3e-4, paciencia=30), "muito_lento": dict(lr=1e-4, paciencia=50)}
ARQUITETURAS = {"1x10": dict(camadas=1, unidades=10, plano="N", dropout=0.0),
                "1x32": dict(camadas=1, unidades=32, plano="D", dropout=0.3),
                "2x9": dict(camadas=2, unidades=9, plano="ND", dropout=0.3),
                "2x16": dict(camadas=2, unidades=16, plano="DN", dropout=0.3),
                "3x16": dict(camadas=3, unidades=16, plano="DDN", dropout=0.3),
                "2x64": dict(camadas=2, unidades=64, plano="DD", dropout=0.5)}


def grade_arquiteturas(base):
    """{rotulo: config}: cada arquitetura em cada ritmo, com as features e o resto do treino da base."""
    saida = {}
    for na, a in ARQUITETURAS.items():
        for nr, r in RITMOS.items():
            saida[f"arq_{na}_{nr}"] = dict(hp={**base["hp"], **a, **r}, dados=dict(base["dados"]), extras=list(base["extras"]))
    return saida


# ---------------- Optuna com mais rodadas por trial ----------------
def espaco(t, base):
    """Mais largo que a fase 4 onde ela ficou em aberto: paciencia ate 50, lr ate 1e-4, logistica e tanh (slides)."""
    camadas = t.suggest_int("camadas", 1, 3)
    hp = dict(camadas=camadas, unidades=t.suggest_int("unidades", 4, 64, log=True), funil=t.suggest_categorical("funil", [False, True]),
              plano="".join(t.suggest_categorical(f"p{k + 1}", ["N", "D"]) for k in range(camadas)),
              ativ=t.suggest_categorical("ativ", ["relu", "gelu", "leaky_relu", "tanh", "logistica"]),
              otim=t.suggest_categorical("otim", ["adam", "adamw", "rmsprop"]),
              lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), dropout=t.suggest_float("dropout", 0.05, 0.5), l2=t.suggest_float("l2", 1e-6, 1e-2, log=True),
              suav=t.suggest_float("suav", 0.0, 0.2), perda=t.suggest_categorical("perda", ["ce", "mse"]), lote=t.suggest_categorical("lote", [64, 128, 256, 512]),
              paciencia=t.suggest_categorical("paciencia", [10, 20, 30, 50]), parada=t.suggest_categorical("parada", ["perda", "ks"]),
              peso_pos=t.suggest_categorical("peso_pos", [0.0, 1.5, 2.77]))
    return dict(hp=hp, dados={"over": t.suggest_categorical("over", [True, False])}, extras=list(base["extras"]))


def _parametros(c):
    hp = c["hp"]
    pl = (hp.get("plano") or ("D" if hp.get("dropout", 0) > 0 else "N") * hp["camadas"]).replace("B", "N")
    p = dict(camadas=min(3, hp["camadas"]), unidades=min(64, max(4, hp["unidades"])), funil=hp.get("funil", False), ativ=hp["ativ"], otim=hp["otim"],
             lr=hp["lr"], dropout=min(.5, max(hp["dropout"], 0.05)), l2=hp["l2"], suav=hp.get("suav", 0.0), perda=hp["perda"], lote=hp.get("lote", 256),
             paciencia=hp["paciencia"] if hp["paciencia"] in (10, 20, 30, 50) else 10, parada=hp.get("parada", "perda"),
             peso_pos=hp.get("peso_pos") or 0.0, over=c["dados"].get("over", True))
    p.update({f"p{k + 1}": x for k, x in enumerate(pl[:3])})
    return p


def estudar(pasta, base, n_trials, reps=(201, 202, 203, 204, 205, 206), n_jobs=1, continuar=lambda: True, ao_salvar=None):
    """Cada trial vale o KS medio do retreino em len(reps) divisoes de 5 dobras (as da fase 4, que nao sao as do placar).
    O trial 0 e a base. Na CPU, como na fase 4."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name="mlp_fase5", storage=f"sqlite:///{pasta}/mlp_fase5.db", load_if_exists=True,
                             direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if not st.trials:
        st.enqueue_trial(_parametros(base), skip_if_exists=True)
    while continuar():
        feitos = len([t for t in st.trials if t.state.is_finished()])
        if feitos >= n_trials:
            break
        ts = [st.ask() for _ in range(min(max(1, n_jobs // len(reps)) or 1, n_trials - feitos))]
        cs = [espaco(t, base) for t in ts]
        res = Parallel(n_jobs=n_jobs)(delayed(F4.medir)(c, r) for c in cs for r in reps)
        for j, t in enumerate(ts):
            parte = res[j * len(reps):(j + 1) * len(reps)]
            t.set_user_attr("ks_parada", float(np.mean([x["ks_parada"] for x in parte])))
            t.set_user_attr("epocas", float(np.median([x["epocas"] for x in parte])))
            st.tell(t, float(np.mean([x["ks"] for x in parte])))
        print(f"optuna: {feitos + len(ts)}/{n_trials} | melhor {st.best_value:.4f}", flush=True)
        if ao_salvar:
            ao_salvar()
    if not [t for t in st.trials if t.state.name == "COMPLETE"]:
        return st, base
    return st, espaco(optuna.trial.FixedTrial(st.best_params), base)


def resumo_optuna(st):
    """Melhor KS por valor de cada categoria e por faixa de unidades/lr/paciencia: a investigacao pedida nos slides."""
    df = st.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    saida = {}
    for col in [c for c in df.columns if c.startswith("params_")]:
        s = df[col]
        if pd.api.types.is_float_dtype(s) and s.nunique() > 6:
            s = pd.qcut(s, 4, duplicates="drop").astype(str)
        g = df.groupby(s.astype(str)).value.agg(["count", "max", "median"])
        saida[col.replace("params_", "")] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in g.to_dict("index").items()}
    return saida


# ---------------- escolha e teste ----------------
def escolher(linhas, campo="vs_referencia"):
    """So entra quem ganhou da referencia no placar; entre esses, o maior KS da OOF. Ninguem ganhou: fica a referencia."""
    ganhou = [r for r in linhas if (r.get(campo) or {}).get("veredito") == "ganhou"]
    return max(ganhou, key=lambda r: r["ks_oof"]) if ganhou else None


def final(c, n_seeds=5, semente_base=0):
    """Treino inteiro: 20% internos decidem as epocas, depois retreina em tudo por essas epocas. Previsoes no teste por semente."""
    Xtr, Xte, ytr, yte = _dados()
    y = np.asarray(ytr)
    k = BL._kwargs(c)
    XA, yA, XT = EV._prep(Xtr, ytr, Xte, **k)
    P_, eps = [], []
    for s in range(n_seeds):
        sem = 42 + 1000 * s + 100000 * semente_base
        a1, a2 = train_test_split(np.arange(len(y)), test_size=comum.PLACAR_FRACAO_PARADA, stratify=y, random_state=comum.SEMENTE + s)
        Xa1, ya1, X2 = EV._prep(Xtr.iloc[a1], ytr.iloc[a1], Xtr.iloc[a2], **k)
        _, ep = M.treinar(Xa1, ya1, c["hp"], sem, X2, y[a2].astype("float32"))
        m, _ = M.treinar(XA, yA, c["hp"], sem, epocas=max(1, ep))
        P_.append(M.prever(m, XT)); eps.append(ep)
    return dict(p_sementes=[p.tolist() for p in P_], epocas=eps, y_teste=[int(v) for v in yte])


def metricas_teste(pasta, rotulo, fin):
    """Metricas completas do teste, com o limiar do KS escolhido na OOF do placar (nunca no teste)."""
    linha = {r["rotulo"]: r for r in comum.ler_placar(pasta) if r["modelo"] == MODELO}[rotulo]
    oof = np.load(os.path.join(pasta, linha["oof"])).mean(0)
    lim = comum.limiar_ks(np.asarray(_dados()[2]), oof)
    y, p = np.asarray(fin["y_teste"]), np.mean(fin["p_sementes"], 0)
    m = comum.metricas(y, p, lim)
    m["ks_ic95"] = comum.bootstrap_ks(y, p)
    m["ks_por_semente"] = [comum.ks(y, np.asarray(x)) for x in fin["p_sementes"]]
    return m


def delta_teste(fa, fb, n=2000, semente=0):
    """b - a no teste, bootstrap pareado: so conferencia."""
    y = np.asarray(fa["y_teste"]); a = np.mean(fa["p_sementes"], 0); b = np.mean(fb["p_sementes"], 0)
    rng, ds = np.random.default_rng(semente), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ds.append(comum.ks(y[i], b[i]) - comum.ks(y[i], a[i]))
    return dict(delta=comum.ks(y, b) - comum.ks(y, a), ic95=[float(x) for x in np.percentile(ds, [2.5, 97.5])])
