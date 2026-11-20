"""Fase 7 do MLP: as features que ajudaram o FT-Transformer e um Optuna com o conjunto de dados como eixo.

Por que: no FT, as features da analise de erro (fase 6 do FT) deram ganho no KS OOF (cobranca_razao +0,0047; o pacote de
cinco foi o maior KS do placar do FT, 0,5573) e o Optuna da fase 7 do FT preferiu cobranca_razao + antigo em todos os
melhores trials. Nada disso foi medido no MLP. O Optuna do MLP (fase 5) e anterior a essas features.

1. Triagem nas divisoes proprias (sementes 201-203, nao as do placar): cada feature e os dois pacotes sobre o MLP adotado.
2. Optuna em volta do MLP adotado (1x10, rmsprop, parada pelo KS), com o conjunto de dados como eixo: a base e os dois
   melhores da triagem. Os primeiros trials sao o MLP adotado com cada conjunto.
3. No placar: o melhor da triagem com os hiperparametros do adotado e os 3 melhores trials, contra o trial 0 e o adotado.
"""
import sys, os, json
import numpy as np, pandas as pd, optuna
from functools import lru_cache
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, fase5 as F5

ROT_BASE = "s6_rmsprop_ks_p50"
BASE = dict(hp=dict(camadas=1, unidades=10, ativ="relu", otim="rmsprop", lr=1e-3, dropout=0.0, l2=1e-6, perda="ce", paciencia=50,
                    plano="N", parada="ks"), dados={}, extras=["cliente_novo", "delta_preco", "tenure_1"])   # o MLP adotado (fase 6)
QUASE = ("cobranca_razao", "antigo", "caro_sem_backup", "fibra_sem_protecao", "atrito_pagamento")   # ganho no OOF do FT
PACOTES = {"p_razao_antigo": ["cobranca_razao", "antigo"], "p_cinco": list(QUASE)}
SEMENTES_OPTUNA = (201, 202, 203)
P_MIN = 0.8


def com(base, extras):
    return dict(hp=dict(base["hp"]), dados=dict(base["dados"]), extras=list(base["extras"]) + [e for e in extras if e not in base["extras"]])


def candidatos(base=BASE):
    c = {f"m_{f}": com(base, [f]) for f in QUASE}
    c.update({n: com(base, f) for n, f in PACOTES.items()})
    return c


@lru_cache(maxsize=1)
def divs_optuna(sementes=SEMENTES_OPTUNA):
    return comum.divisoes_placar(np.asarray(F5._dados()[2]), sementes)


def triagem(cands, base=BASE, D=None, n_jobs=1):
    """Mede a base e os candidatos de uma vez (uma pilha so) e devolve as linhas com o veredito pareado contra a base."""
    D = divs_optuna() if D is None else D
    nomes = list(cands)
    rs = F5.medir([base] + [cands[n] for n in nomes], n_jobs=n_jobs, D=D)
    y, rb = np.asarray(F5._dados()[2]), rs[0]
    linhas = []
    for n, r in zip(nomes, rs[1:]):
        v = comum.comparar_placar(y, rb["oof"], r["oof"])
        linhas.append(dict(candidato=n, ks_kfold=r["ks_kfold"], delta_kfold=r["ks_kfold"] - rb["ks_kfold"], ks_oof=r["ks_oof"],
                           delta_oof=v["delta"], ic95=v["ic95"], p_positivo=v["p_positivo"],
                           passou=bool(v["delta"] > 0 and v["p_positivo"] >= P_MIN)))
    return dict(base=dict(ks_kfold=rb["ks_kfold"], ks_oof=rb["ks_oof"]), linhas=linhas)


def conjuntos_optuna(linhas, cands, n=2, base=BASE):
    """{nome: extras}: a base e os n candidatos de maior ganho no KS OOF (so os com ganho)."""
    bons = sorted([l for l in linhas if l["delta_oof"] > 0], key=lambda l: -l["delta_oof"])[:n]
    saida = {"base": list(base["extras"])}
    saida.update({l["candidato"]: list(cands[l["candidato"]]["extras"]) for l in bons})
    return saida


# ---------------- Optuna ----------------
def espaco(t, conjuntos):
    """Em volta do MLP adotado. Chaves fora do padrao so entram quando mudam algo, para o trial da base sair identico a BASE."""
    camadas = t.suggest_int("camadas", 1, 2)
    plano = "".join(t.suggest_categorical(f"p{k + 1}", ["N", "D"]) for k in range(camadas))
    hp = dict(camadas=camadas, unidades=t.suggest_int("unidades", 4, 32, log=True),
              ativ=t.suggest_categorical("ativ", ["relu", "gelu", "tanh", "logistica"]),
              otim=t.suggest_categorical("otim", ["rmsprop", "adam", "adamw"]),
              lr=t.suggest_float("lr", 2e-4, 5e-3, log=True),
              dropout=t.suggest_float("dropout", 0.05, 0.5) if "D" in plano else 0.0,
              l2=t.suggest_float("l2", 1e-7, 1e-3, log=True), perda="ce",
              paciencia=t.suggest_categorical("paciencia", [10, 30, 50]), plano=plano,
              parada=t.suggest_categorical("parada", ["ks", "perda"]))
    lote = t.suggest_categorical("lote", [64, 128, 256, 512])
    if lote != M.LOTE:
        hp["lote"] = lote
    dados = {} if t.suggest_categorical("over", [True, False]) else {"over": False}
    return dict(hp=hp, dados=dados, extras=list(conjuntos[t.suggest_categorical("conjunto", list(conjuntos))]))


def parametros_base(conjunto):
    hp = BASE["hp"]
    return dict(camadas=hp["camadas"], p1=hp["plano"], unidades=hp["unidades"], ativ=hp["ativ"], otim=hp["otim"], lr=hp["lr"],
                l2=hp["l2"], paciencia=hp["paciencia"], parada=hp["parada"], lote=M.LOTE, over=True, conjunto=conjunto)


def config_do_trial(params, conjuntos):
    return espaco(optuna.trial.FixedTrial(params), conjuntos)


def chave(c):
    return json.dumps([c["hp"], c["dados"], c["extras"]], sort_keys=True)


def estudar(pasta, n_trials, conjuntos, D=None, lote_trials=8, n_jobs=1, continuar=lambda: True, ao_lote=None):
    """Trials em lotes de `lote_trials` (as redes de um lote treinam juntas na pilha). Estudo em sqlite na pasta: retomar a
    execucao retoma o estudo. Cada trial vale o KS medio das 15 dobras das sementes 201-203."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name="mlp_fase7", storage=f"sqlite:///{os.path.join(pasta, 'mlp_fase7_optuna.db')}",
                             load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if not st.trials:
        for nome in conjuntos:
            st.enqueue_trial(parametros_base(nome))
    D = divs_optuna() if D is None else D
    for t in st.trials:                            # trials que ficaram abertos numa execucao que caiu
        if t.state == optuna.trial.TrialState.RUNNING:
            st.tell(t.number, state=optuna.trial.TrialState.FAIL)
    while continuar():
        feitos = len([t for t in st.trials if t.state.is_finished()])
        if feitos >= n_trials:
            break
        ts = [st.ask() for _ in range(min(lote_trials, n_trials - feitos))]
        cs = [espaco(t, conjuntos) for t in ts]
        rs = F5.medir(cs, n_jobs=n_jobs, D=D)
        for t, r in zip(ts, rs):
            t.set_user_attr("epocas", float(r.get("epocas") or 0)); t.set_user_attr("ks_oof", float(r["ks_oof"]))
            st.tell(t, float(r["ks_kfold"]))
        print(f"optuna: {feitos + len(ts)}/{n_trials} | melhor {st.best_value:.4f}", flush=True)
        if ao_lote:
            ao_lote(st)
    return st


def melhores(st, conjuntos, n=3, excluir=()):
    """Os n trials completos de maior KS, sem repetir configuracao nem repetir o que ja esta em `excluir` (chaves)."""
    ok = sorted([t for t in st.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
    saida, vistos = [], set(excluir)
    for t in ok:
        ch = chave(config_do_trial(t.params, conjuntos))
        if ch not in vistos:
            vistos.add(ch); saida.append(t)
        if len(saida) == n:
            break
    return saida
