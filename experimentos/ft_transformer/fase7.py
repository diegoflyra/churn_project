"""Fase 7 do FT-Transformer: pacotes de features, target encoding dentro da dobra e Optuna com o conjunto de dados.

Por que (fase 6 e curva de aprendizado):
- na selecao da fase 6 nenhuma feature passou sozinha, mas cinco deram ganho no KS OOF (cobranca_razao +0,0047, p+ 0,75;
  antigo, caro_sem_backup, fibra_sem_protecao e atrito_pagamento entre +0,0024 e +0,0031). Juntas podem somar;
- a curva de aprendizado mostrou o FT com perda de treino ~ perda de validacao (0,409 x 0,413) e o KS plano de 75% para
  100% do treino: o ganho vem de informacao nova na entrada. O target encoding da a rede a taxa de churn do grupo do
  cliente, que hoje ela precisa aprender com poucos exemplos por grupo;
- os hiperparametros do o5_top3 foram escolhidos sem essas entradas: o Optuna escolhe de novo, junto com o conjunto.

1. Triagem nas divisoes do Optuna (sementes 201-203, nao as do placar): pacotes e target encodings sobre o o5_top3.
2. Optuna: o conjunto de dados vira um eixo (a base e os dois melhores da triagem), com a regiao da arquitetura e da
   regularizacao em volta do que a fase 5 achou (1-2 blocos, dropout de atencao alto, sem oversampling).
3. No placar: o melhor da triagem com os hiperparametros do o5_top3, os 3 melhores trials e o melhor com 3 redes.
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, fase1 as F1, fase5 as F5, fase6 as F6

ROT_BASE, BASE = F6.ROT_BASE, F6.BASE
QUASE = ("cobranca_razao", "antigo", "caro_sem_backup", "fibra_sem_protecao", "atrito_pagamento")   # ganho no OOF na fase 6
PACOTES = {
    "p_razao_antigo": ["cobranca_razao", "antigo"],
    "p_cinco": list(QUASE),
}
TES = tuple(comum.TE_GRUPOS)
P_MIN = 0.8


def com(base, features=(), te=()):
    dados = dict(base["dados"], features=list(base["dados"]["features"]) + list(features))
    if te:
        dados["te"] = list(te)
    return dict(hp=dict(base["hp"]), dados=dados)


def candidatos(base=BASE):
    """{nome: config} da triagem: pacotes, cada target encoding sozinho, todos juntos e todos com o pacote de cinco."""
    c = {n: com(base, f) for n, f in PACOTES.items()}
    c.update({t: com(base, te=[t]) for t in TES})
    c["te_todos"] = com(base, te=TES)
    c["p_cinco_te_todos"] = com(base, QUASE, TES)
    return c


def triagem(pasta, medir, y, cands, base=BASE, continuar=lambda: True, ao_medir=None):
    """Mede a base e cada candidato nas divisoes do Optuna (cache em selecao.jsonl, como na fase 6). Devolve as linhas com o
    veredito pareado contra a base."""
    cache = F6.Cache(pasta)
    _, ob = cache.medir(base, medir)
    rb = cache.linhas[F1._chave(base)]
    linhas = []
    for nome, c in cands.items():
        if not continuar():
            print(f"limite de tempo: a triagem parou antes de {nome}", flush=True)
            break
        rc, oc = cache.medir(c, medir)
        v = comum.comparar_placar(y, ob, oc)
        l = dict(candidato=nome, ks_kfold=rc["ks_kfold"], delta_kfold=rc["ks_kfold"] - rb["ks_kfold"], ks_oof=rc["ks_oof"],
                 delta_oof=v["delta"], ic95=v["ic95"], p_positivo=v["p_positivo"], passou=bool(v["delta"] > 0 and v["p_positivo"] >= P_MIN))
        linhas.append(l)
        if ao_medir:
            ao_medir(l)
    return linhas


def conjuntos_optuna(linhas, cands, n=2, base=BASE):
    """{nome: dados} para o eixo do Optuna: a base e os n candidatos de maior ganho no KS OOF (so os com ganho)."""
    bons = sorted([l for l in linhas if l["delta_oof"] > 0], key=lambda l: -l["delta_oof"])[:n]
    saida = {"base": dict(base["dados"])}
    saida.update({l["candidato"]: dict(cands[l["candidato"]]["dados"]) for l in bons})
    return saida


# ---------------- Optuna ----------------
def espaco(t, conjuntos):
    """Regiao em volta do vencedor da fase 5. Tokens categoricos e PLR ficaram de fora (nao ajudaram na fase 5) e o
    oversampling tambem (todos os melhores trials da fase 5 foram sem)."""
    hp = dict(BASE["hp"],
              n_blocos=t.suggest_int("n_blocos", 1, 2),
              d_token=t.suggest_categorical("d_token", [128, 192, 256]),
              n_cabecas=t.suggest_categorical("n_cabecas", [2, 4, 8]),
              ffn_fator=t.suggest_float("ffn_fator", 1.0, 8 / 3),
              drop_atencao=t.suggest_float("drop_atencao", 0.2, 0.6),
              drop_ffn=t.suggest_float("drop_ffn", 0.1, 0.5),
              drop_residual=t.suggest_float("drop_residual", 0.0, 0.1),
              lr=t.suggest_float("lr", 5e-5, 5e-4, log=True),
              wd=t.suggest_float("wd", 1e-5, 3e-3, log=True),
              lote=t.suggest_categorical("lote", [128, 256, 512]))
    if t.suggest_categorical("agenda", ["fixa", "cos"]) == "cos":
        hp.update(agenda="cos", aquec=5, horizonte=t.suggest_categorical("horizonte", [60, 120]))
    nome = t.suggest_categorical("conjunto", list(conjuntos))
    return dict(hp=hp, dados=json.loads(json.dumps(conjuntos[nome])))


def parametros_base(conjunto):
    """Os hiperparametros do o5_top3 com um dos conjuntos: o estudo comeca de pontos conhecidos."""
    hp = BASE["hp"]
    p = {k: hp[k] for k in ("n_blocos", "d_token", "n_cabecas", "ffn_fator", "drop_atencao", "drop_ffn", "drop_residual", "lr", "wd", "lote")}
    p.update(agenda="fixa", conjunto=conjunto)
    return p


def config_do_trial(params, conjuntos):
    import optuna
    return espaco(optuna.trial.FixedTrial(params), conjuntos)


def estudar(pasta, n_trials, conjuntos, medir, continuar=lambda: True, ao_trial=None, D=None):
    """Como fase5.estudar: um trial por vez, estudo em sqlite na pasta (retomar a execucao retoma o estudo).
    Os primeiros trials sao o o5_top3 com cada conjunto."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name="ft_fase7", storage=f"sqlite:///{os.path.join(pasta, 'ft_fase7_optuna.db')}",
                             load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if not st.trials:
        for nome in conjuntos:
            st.enqueue_trial(parametros_base(nome))
    D = F5.divs_optuna() if D is None else D
    while continuar():
        feitos = [t for t in st.trials if t.state.is_finished()]
        if len(feitos) >= n_trials:
            break
        t = st.ask()
        c = espaco(t, conjuntos)
        try:
            r = medir(c, D)
        except Exception as e:                    # configuracao invalida ou sem memoria: o trial falha, o estudo segue
            print(f"optuna: trial {t.number} falhou: {e}", flush=True)
            st.tell(t, state=optuna.trial.TrialState.FAIL)
            continue
        t.set_user_attr("epocas", float(r.get("epocas") or 0)); t.set_user_attr("minutos", float(r.get("tempo_min") or 0))
        t.set_user_attr("ks_oof", float(r["ks_oof"]))
        st.tell(t, float(r["ks_kfold"]))
        print(f"optuna: {len(feitos) + 1}/{n_trials} | {t.params['conjunto']:<20} KS {r['ks_kfold']:.4f} | "
              f"melhor {st.best_value:.4f} | {r.get('tempo_min', 0):.1f} min", flush=True)
        if ao_trial:
            ao_trial(st, t, r)
    return st
