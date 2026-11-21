"""Fase 5 do FT-Transformer: triagem do que nunca foi testado e Optuna (parte obrigatoria do trabalho).

Ponto de partida: f_trial0_tres (trial 0 + cliente_novo, delta_preco, tenure_1), o unico FT que ganhou do trial 0
(fase 4: KS do placar 0,5506, +0,0155 no veredito).

1. Triagem, uma mudanca por vez sobre a base, no placar (vs_base = contra f_trial0_tres):
   oversampling, taxa de aprendizado, agenda do lr (aquecimento + cosseno), cabecas de atencao, largura da FFN,
   tokens categoricos (um token por variavel), embeddings periodicos nas numericas (PLR), features extras e a media
   de 3 redes por dobra.
2. Optuna sobre esses eixos, com a arquitetura e a regularizacao. Cada trial vale o KS medio em divisoes PROPRIAS
   (sementes 201-203, 15 dobras), nao as do placar: as dobras do placar so medem o vencedor no fim. No MLP, um Optuna
   medido em 2 divisoes escolheu um vencedor que era sorte; aqui sao 15 dobras por trial.
3. Os melhores trials entram no placar, contra o trial 0 e contra a base.
"""
import sys, os, json
from functools import lru_cache
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, fase1 as F1, fase4 as F4

BASE = F4.grade_features()["f_trial0_tres"][1]
ROT_BASE = "f_trial0_tres"
EXTRAS = ("tenure_ate_3", "tenure_ate_12", "log_tenure", "novo_mes_a_mes", "mtm_fibra", "atrito_pagamento",
          "sozinho", "n_protecao", "fibra_sem_protecao")
SEMENTES_OPTUNA = (201, 202, 203)


def _muda(hp=None, **dados):
    return dict(hp=dict(BASE["hp"], **(hp or {})), dados=dict(BASE["dados"], **dados))


def grade_triagem():
    """{rotulo: (eixo, config)}: uma mudanca por vez sobre a base."""
    g = {
        "t5_sem_over": ("dados", _muda(over=False)),
        "t5_lr3e-4": ("lr", _muda(dict(lr=3e-4))),
        "t5_lr1e-3": ("lr", _muda(dict(lr=1e-3))),
        "t5_cos": ("agenda", _muda(dict(agenda="cos", aquec=5, horizonte=60))),
        "t5_cos_lr3e-4": ("agenda", _muda(dict(lr=3e-4, agenda="cos", aquec=5, horizonte=60))),
        "t5_cab2": ("cabecas", _muda(dict(n_cabecas=2))),
        "t5_cab4": ("cabecas", _muda(dict(n_cabecas=4))),
        "t5_ffn2": ("ffn", _muda(dict(ffn_fator=2.0))),
        "t5_tok_cat": ("tokens", _muda(dict(tokens="cat"))),
        "t5_plr_s0.1": ("plr", _muda(dict(num_emb="plr", plr_k=16, plr_sigma=0.1))),
        "t5_plr_s1": ("plr", _muda(dict(num_emb="plr", plr_k=16, plr_sigma=1.0))),
        "t5_3redes": ("sementes", _muda(dict(n_redes=3))),
    }
    for f in EXTRAS:
        g[f"t5_mais_{f}"] = ("features", _muda(features=list(BASE["dados"]["features"]) + [f]))
    return g


@lru_cache(maxsize=1)
def divs_optuna(sementes=SEMENTES_OPTUNA):
    return comum.divisoes_placar(np.asarray(F1._dados()[2]), sementes)


# ---------------- Optuna ----------------
D_TOKEN = (64, 96, 128, 192, 256)
CABECAS = (1, 2, 4, 8)


def espaco(t, extras=()):
    """Arquitetura, regularizacao, treino e as novidades da triagem. As tres features da base ficam sempre; cada extra
    vira um sim/nao. A parada fica a da base (perda, paciencia 16): na fase 3 a parada pelo KS nao ajudou o FT."""
    d = t.suggest_categorical("d_token", list(D_TOKEN))
    h = t.suggest_categorical("n_cabecas", list(CABECAS))
    hp = dict(BASE["hp"],
              n_blocos=t.suggest_int("n_blocos", 1, 4), d_token=d, n_cabecas=h,             # todo d_token e multiplo de 8
              ffn_fator=t.suggest_float("ffn_fator", 2 / 3, 8 / 3),
              drop_atencao=t.suggest_float("drop_atencao", 0.0, 0.5), drop_ffn=t.suggest_float("drop_ffn", 0.0, 0.5),
              drop_residual=t.suggest_float("drop_residual", 0.0, 0.2),
              lr=t.suggest_float("lr", 3e-5, 1e-3, log=True), wd=t.suggest_float("wd", 1e-6, 1e-3, log=True),
              lote=t.suggest_categorical("lote", [128, 256, 512]))
    if t.suggest_categorical("agenda", ["fixa", "cos"]) == "cos":
        hp.update(agenda="cos", aquec=5, horizonte=t.suggest_categorical("horizonte", [30, 60, 120]))
    if t.suggest_categorical("tokens", ["coluna", "cat"]) == "cat":
        hp["tokens"] = "cat"
    if t.suggest_categorical("num_emb", ["linear", "plr"]) == "plr":
        hp.update(num_emb="plr", plr_k=16, plr_sigma=t.suggest_float("plr_sigma", 0.01, 10.0, log=True))
    feats = list(BASE["dados"]["features"]) + [f for f in extras if t.suggest_categorical(f"f_{f}", [False, True])]
    dados = dict(features=feats)
    if not t.suggest_categorical("over", [True, False]):
        dados["over"] = False                     # sem a chave quando True: a base sai identica a BASE
    return dict(hp=hp, dados=dados)


def parametros_base(extras=()):
    """O trial 0 do Optuna e a propria base (enfileirado), para o estudo comecar do ponto conhecido."""
    hp = BASE["hp"]
    p = dict(d_token=hp["d_token"], n_cabecas=hp["n_cabecas"], n_blocos=hp["n_blocos"], ffn_fator=hp["ffn_fator"],
             drop_atencao=hp["drop_atencao"], drop_ffn=hp["drop_ffn"], drop_residual=hp["drop_residual"], lr=hp["lr"],
             wd=hp["wd"], lote=hp["lote"], agenda="fixa", tokens="coluna", num_emb="linear", over=True)
    p.update({f"f_{f}": False for f in extras})
    return p


def config_do_trial(params, extras=()):
    import optuna
    return espaco(optuna.trial.FixedTrial(params), extras)


def estudar(pasta, n_trials, extras=(), medir=None, continuar=lambda: True, ao_trial=None, D=None):
    """Um trial por vez (as 15 dobras de cada trial ja rodam em paralelo nas GPUs). O estudo fica em sqlite na pasta
    da execucao, entao retomar a execucao retoma o estudo. medir(config, D) -> comum.resumir_placar."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name="ft_fase5", storage=f"sqlite:///{os.path.join(pasta, 'ft_fase5_optuna.db')}",
                             load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if not st.trials:
        st.enqueue_trial(parametros_base(extras))
    D = divs_optuna() if D is None else D
    while continuar():
        feitos = [t for t in st.trials if t.state.is_finished()]
        if len(feitos) >= n_trials:
            break
        t = st.ask()
        c = espaco(t, extras)
        try:
            r = medir(c, D)
        except Exception as e:                    # configuracao invalida ou sem memoria: o trial falha, o estudo segue
            print(f"optuna: trial {t.number} falhou: {e}", flush=True)
            st.tell(t, state=optuna.trial.TrialState.FAIL)
            continue
        t.set_user_attr("epocas", float(r.get("epocas") or 0)); t.set_user_attr("minutos", float(r.get("tempo_min") or 0))
        t.set_user_attr("ks_oof", float(r["ks_oof"]))
        st.tell(t, float(r["ks_kfold"]))
        print(f"optuna: {len(feitos) + 1}/{n_trials} | KS {r['ks_kfold']:.4f} | melhor {st.best_value:.4f} | "
              f"{r.get('tempo_min', 0):.1f} min", flush=True)
        if ao_trial:
            ao_trial(st, t, r)
    return st


def melhores(st, n=3):
    """Os n trials completos de maior KS, sem repetir a mesma configuracao."""
    ok = sorted([t for t in st.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
    saida, vistos = [], set()
    for t in ok:
        ch = json.dumps(t.params, sort_keys=True)
        if ch not in vistos:
            vistos.add(ch); saida.append(t)
        if len(saida) == n:
            break
    return saida


def resumo_optuna(st):
    """Por eixo: numero de trials, melhor e mediana do KS por valor (os continuos em quartis)."""
    df = st.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    linhas = []
    for col in [c for c in df.columns if c.startswith("params_")]:
        s = df[col]
        if pd.api.types.is_float_dtype(s) and s.nunique() > 6:
            s = pd.qcut(s, 4, duplicates="drop").astype(str)
        for v, g in df.groupby(s.astype(str)).value:
            linhas.append(dict(eixo=col.replace("params_", ""), valor=v, trials=len(g), melhor=g.max(), mediana=g.median()))
    return pd.DataFrame(linhas)


# ---------------- media com o MLP ----------------
def media_com_mlp(y, oof_ft, oof_mlp):
    """Media das probabilidades das duas redes, por repeticao do placar (as OOF tem a mesma forma: 3 x n)."""
    from scipy.stats import rankdata
    rk = lambda a: np.stack([rankdata(x) / len(x) for x in a])
    return dict(prob=(np.asarray(oof_ft) + np.asarray(oof_mlp)) / 2, rank=(rk(oof_ft) + rk(oof_mlp)) / 2)
