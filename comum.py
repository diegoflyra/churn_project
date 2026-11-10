"""comum.py - dados, dobras, preparar, KS, metricas, estudo Optuna e avaliacao final (IF702, churn).

Reproduz exatamente o pre-processamento de if702-preprocessamento.ipynb.
A unica diferenca e a leitura do CSV: aqui via CAMINHO_CSV (ver abaixo).
"""
import os, json, time, hashlib
import numpy as np
import pandas as pd
import optuna
from scipy.stats import ks_2samp, wilcoxon, mannwhitneyu
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import RandomOverSampler

# Kaggle: "/kaggle/input/datasets/kapturovalexander/customers-churned-in-telecom-services/customer_churn_telecom_services.csv"
CAMINHO_CSV = os.environ.get("CHURN_CSV", os.path.join(os.path.dirname(__file__), "dados", "telco_ibm.csv"))
# Kaggle: CHURN_RESULTADOS=/kaggle/working/resultados
RESULTADOS = os.environ.get("CHURN_RESULTADOS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados"))
SEMENTE = 42
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def carregar_df(path=CAMINHO_CSV):
    df = pd.read_csv(path)
    if "customerID" in df.columns:          # a copia do Kaggle nao tem essa coluna
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
        pass
    binarias = {
        "gender": {"Female": 0, "Male": 1}, "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1}, "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1}, "Churn": {"No": 0, "Yes": 1},
    }
    for col, m in binarias.items():
        df[col] = df[col].map(m)
    df["MultipleLines"] = df["MultipleLines"].map({"Yes": 1, "No": 0, "No phone service": 0})
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in addons:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=False, dtype=int)
    add7 = ["MultipleLines"] + addons
    df["total_addons"] = df[add7].sum(axis=1)
    auto = [c for c in df.columns if "PaymentMethod_" in c and "automatic" in c.lower()]
    df["PaymentMethod_Automatic"] = df[auto].max(axis=1)
    return df


# Features por linha (sem olhar o rotulo nem outras linhas: nao vazam). Vieram do MLP; a evidencia de cada uma,
# no MLP e no XGBoost, esta em docs/GUIA_DADOS.md.
FEATURES = {
    "tenure_1": lambda d: (d["tenure"] <= 1).astype(int),                                    # 61% de churn no treino
    "cliente_novo": lambda d: (d["tenure"] <= 6).astype(int),
    "delta_preco": lambda d: d["MonthlyCharges"] - d["TotalCharges"] / (d["tenure"] + 1),   # preco atual - media ja paga
}


def adicionar_features(X, nomes):
    """Copia de X com as features pedidas de FEATURES."""
    X = X.copy()
    for n in nomes:
        X[n] = FEATURES[n](X)
    return X


def preparar(X_tr, y_tr, X_aval, features=(), over=True):
    """Scaler ajustado so em X_tr; oversampling so em X_tr; X_aval so recebe transform.
    features: nomes de FEATURES a acrescentar (as numericas nao binarias entram no scaler).
    over: True = oversampling 1:1 (padrao do grupo); False = sem; numero (ex.: 0.5) = minoria ate essa fracao da maioria."""
    if features:
        X_tr, X_aval = adicionar_features(X_tr, features), adicionar_features(X_aval, features)
    cols = NUMERIC_COLS + [f for f in features if X_tr[f].nunique() > 2]
    scaler = StandardScaler()
    X_tr_s = X_tr.copy()
    X_tr_s[cols] = scaler.fit_transform(X_tr[cols])
    X_av_s = X_aval.copy()
    X_av_s[cols] = scaler.transform(X_aval[cols])
    if over is False:
        return X_tr_s, y_tr, X_av_s
    X_b, y_b = RandomOverSampler(sampling_strategy="auto" if over is True else float(over), random_state=SEMENTE).fit_resample(X_tr_s, y_tr)
    return X_b, y_b, X_av_s


def dividir(df):
    """Divisao treino/teste oficial (75/25 estratificada)."""
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    return train_test_split(X, y, test_size=0.25, random_state=SEMENTE, stratify=y)


def indices_dobras(path=CAMINHO_CSV):
    """Posicao no CSV dos clientes de validacao de cada dobra e do teste (para cruzar previsoes entre modelos)."""
    X_train, X_test, y_train, _ = dividir(carregar_df(path))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    return [X_train.index[v].tolist() for _, v in skf.split(X_train, y_train)], X_test.index.tolist()


def assinatura_teste(path=CAMINHO_CSV):
    """Identifica os clientes do teste. Precisa ser igual em todos os pre-processamentos, para o teste ser comparavel."""
    _, ind_teste = indices_dobras(path)
    return hashlib.sha256(json.dumps(sorted(ind_teste)).encode()).hexdigest()[:16]


def carregar_dobras(path=CAMINHO_CSV):
    df = carregar_df(path)
    X = df.drop(columns=["Churn"])
    X_train, X_test, y_train, y_test = dividir(df)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    folds = []
    for idx_tr, idx_val in skf.split(X_train, y_train):
        Xv, yv = X_train.iloc[idx_val], y_train.iloc[idx_val]
        Xt, yt, Xv = preparar(X_train.iloc[idx_tr], y_train.iloc[idx_tr], Xv)
        folds.append((Xt.values.astype("float32"), yt.values.astype("float32"),
                      Xv.values.astype("float32"), yv.values.astype("float32")))
    Xtr_f, ytr_f, Xte_f = preparar(X_train, y_train, X_test)
    final = (Xtr_f.values.astype("float32"), ytr_f.values.astype("float32"),
             Xte_f.values.astype("float32"), y_test.values.astype("float32"))
    return folds, final, list(X.columns)


# ---------------- metricas ----------------
def ks(y, p):
    """KS = max_t |TPR(t) - FPR(t)| (equivale a ks_2samp entre os escores dos dois grupos)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def limiar_ks(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def ce(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mse(y, p):
    return float(np.mean((p - y) ** 2))


def metricas(y, p, limiar):
    pred = (p >= limiar).astype(int)
    return dict(ks=ks(y, p), auroc=float(roc_auc_score(y, p)), ce=ce(y, p), mse=mse(y, p),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)),
                confusao=confusion_matrix(y, pred).tolist(), limiar=limiar)


def bootstrap_ks(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if y[b].min() == y[b].max():
            continue
        out.append(ks(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------- estudo Optuna ----------------
def objetivo(hp_fn, fit, folds):
    def f(trial):
        hp = hp_fn(trial)
        t0 = time.time(); kss = []
        for i, f_ in enumerate(folds):
            Xv, yv = f_[2], f_[3]
            (pv,), *_ = fit(hp, f_, [Xv], 42 + i)
            kss.append(ks(yv, pv))
            trial.report(float(np.mean(kss)), i)
            if i == 1 and trial.should_prune():       # depois da segunda dobra
                raise optuna.TrialPruned()
        trial.set_user_attr("ks_dobras", [float(k) for k in kss])
        trial.set_user_attr("tempo_s", time.time() - t0)
        return float(np.mean(kss))
    return f


def estudo(modelo, n_trials, hp_fn, fit, padrao, pasta=RESULTADOS, timeout=None, callbacks=None):
    os.makedirs(pasta, exist_ok=True)
    folds, _, _ = carregar_dobras()
    st = optuna.create_study(study_name=modelo, storage=f"sqlite:///{pasta}/{modelo}.db", load_if_exists=True,
                             direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    if len(st.trials) == 0:
        st.enqueue_trial(padrao)     # trial 0 = padrao
    restantes = n_trials - len([t for t in st.trials if t.state.is_finished()])
    t0 = time.time()
    if restantes > 0:
        st.optimize(objetivo(hp_fn, fit, folds), n_trials=restantes, timeout=timeout, callbacks=callbacks, show_progress_bar=False)
    st.set_user_attr("tempo_parede_s", st.user_attrs.get("tempo_parede_s", 0) + time.time() - t0)
    return st


# ---------------- avaliacao final + Entropia Cruzada x MSE ----------------
PI = 0.2654          # taxa real de churn (o treino das dobras foi balanceado para 50%)


def corrige_prior(p, pi=PI):
    """Desfaz o deslocamento de prior causado pelo oversampling (treino 50/50 -> prior real)."""
    w = pi / (1 - pi); q = p * w / (p * w + (1 - p)); return q


def sumario_estudo(st):
    ok = [t for t in st.trials if t.state.name == "COMPLETE"]
    pr = [t for t in st.trials if t.state.name == "PRUNED"]
    por = {}
    for l in ("ce", "mse"):
        v = sorted([t.value for t in ok if t.params["perda"] == l], reverse=True)
        por[l] = dict(n=len(v), melhor=v[0] if v else None, mediana=float(np.median(v)) if v else None,
                      media_top5=float(np.mean(v[:5])) if v else None)
    a = [t.value for t in ok if t.params["perda"] == "ce"]; b = [t.value for t in ok if t.params["perda"] == "mse"]
    mw = float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else None
    try:
        imp = optuna.importance.get_param_importances(st)
    except Exception:
        imp = {}
    return dict(n_completos=len(ok), n_podados=len(pr), por_perda=por, mannwhitney_p=mw, importancias=imp,
                ks_trial0=st.trials[0].value, melhor_ks=st.best_value,
                tempo_trials_s=float(sum(t.user_attrs.get("tempo_s", 0) for t in ok)),
                convergencia=[float(max([t.value for t in ok[:i + 1]])) for i in range(len(ok))])


def avaliar(modelo, hp_from_params, fit, n_seeds, fit_final=None, salvar_oof=False):
    """fit_final(hp, final, infos, seed) -> prob. do teste: treina no treino inteiro em vez do comite das dobras.
    infos = terceiro valor devolvido por fit em cada dobra (ex.: melhor epoca)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.load_study(study_name=modelo, storage=f"sqlite:///{RESULTADOS}/{modelo}.db")
    folds, final, _ = carregar_dobras()
    Xte, yte = final[2], final[3]
    best = st.best_trial; hp0 = hp_from_params(best.params)
    out = dict(modelo=modelo, melhor_config=hp0, ks_dobras_melhor=best.user_attrs["ks_dobras"], estudo=sumario_estudo(st))
    ctrl = {}
    for perda in ("ce", "mse"):
        hp = dict(hp0, perda=perda)
        R = dict(val=[], tr_ce=[], tr_mse=[], teste_prob=[], oof=[], oof_prob=[])
        for s in range(n_seeds):
            pte_folds, oof_p, oof_y, fold_rows, infos = [], [], [], [], []
            for i, f in enumerate(folds):
                res = fit(hp, f, [f[2], Xte], 1000 * s + 42 + i)
                (pv, pte), ptr = res[:2]
                if len(res) > 2: infos.append(res[2])
                yv = f[3]
                fold_rows.append(dict(seed=s, fold=i, ks=ks(yv, pv), auroc=metricas(yv, pv, .5)["auroc"],
                                      ce_raw=ce(yv, pv), mse_raw=mse(yv, pv),
                                      ce_corr=ce(yv, corrige_prior(pv)), mse_corr=mse(yv, corrige_prior(pv)),
                                      tr_ce=ce(f[1], ptr), tr_mse=mse(f[1], ptr)))
                pte_folds.append(pte); oof_p.append(pv); oof_y.append(yv)
                print(modelo, perda, "semente", s, "dobra", i, "KS %.4f" % fold_rows[-1]["ks"], flush=True)
            R["val"] += fold_rows
            thr = limiar_ks(np.concatenate(oof_y), np.concatenate(oof_p))      # limiar escolhido so na validacao
            R["teste_prob"].append(fit_final(hp, final, infos, 1000 * s + 42) if fit_final else np.mean(pte_folds, 0)); R["oof"].append(thr)
            R["oof_prob"].append(np.concatenate(oof_p))
        ptest = np.mean(R["teste_prob"], 0); thr = float(np.mean(R["oof"]))
        m_teste = metricas(yte, ptest, thr)
        m_teste["ce_corr"] = ce(yte, corrige_prior(ptest)); m_teste["mse_corr"] = mse(yte, corrige_prior(ptest))
        m_teste["ks_ic95"] = bootstrap_ks(yte, ptest)
        per_seed = [ks(yte, p) for p in R["teste_prob"]]
        ctrl[perda] = dict(val=R["val"], teste=m_teste, ks_teste_por_semente=per_seed, prob_teste=ptest.tolist(),
                           oof_prob=np.mean(R["oof_prob"], 0), oof_y=np.concatenate(oof_y))
    # comparacao pareada CE x MSE (mesmas dobras, mesmas sementes)
    ce_v = np.array([r["ks"] for r in ctrl["ce"]["val"]]); ms_v = np.array([r["ks"] for r in ctrl["mse"]["val"]])
    d = ce_v - ms_v
    d_fold = d.reshape(n_seeds, 5).mean(0)
    pc, pm = np.array(ctrl["ce"]["prob_teste"]), np.array(ctrl["mse"]["prob_teste"])
    rng = np.random.default_rng(0); n = len(yte); bs = []
    for _ in range(1000):
        b = rng.choice(n, n)
        if yte[b].min() == yte[b].max(): continue
        bs.append(ks(yte[b], pc[b]) - ks(yte[b], pm[b]))
    out["ce_vs_mse"] = dict(
        delta_ks_val_media=float(d.mean()), delta_ks_val_dp=float(d.std(ddof=1)), delta_por_dobra=[float(x) for x in d_fold],
        dobras_ce_melhor=int((d_fold > 0).sum()),
        wilcoxon_p=float(wilcoxon(d_fold).pvalue) if np.any(d_fold != 0) else 1.0,
        delta_ks_teste=float(ctrl["ce"]["teste"]["ks"] - ctrl["mse"]["teste"]["ks"]),
        delta_ks_teste_ic95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        resumo={k: {kk: (float(np.mean([r[kk] for r in v["val"]])), float(np.std([r[kk] for r in v["val"]])))
                    for kk in ("ks", "auroc", "ce_raw", "mse_raw", "ce_corr", "mse_corr", "tr_ce", "tr_mse")} for k, v in ctrl.items()},
        teste={k: {kk: v["teste"][kk] for kk in ("ks", "auroc", "ce", "mse", "ce_corr", "mse_corr", "f1", "precision", "recall", "limiar", "confusao", "ks_ic95")} for k, v in ctrl.items()},
        ks_teste_por_semente={k: v["ks_teste_por_semente"] for k, v in ctrl.items()})
    # probabilidades e metricas do teste da melhor configuracao (perda escolhida pelo Optuna)
    esc = hp0["perda"]
    out["teste"] = ctrl[esc]["teste"]; out["probabilidades_teste"] = ctrl[esc]["prob_teste"]
    out["y_teste"] = [int(v) for v in yte]
    if salvar_oof:
        ind_val, ind_teste = indices_dobras()
        out["oof"] = dict(indice=sum(ind_val, []), y=[int(v) for v in ctrl[esc]["oof_y"]], prob=ctrl[esc]["oof_prob"].tolist())
        out["indice_teste"] = ind_teste
    os.makedirs(RESULTADOS, exist_ok=True)
    json.dump(out, open(f"{RESULTADOS}/{modelo}.json", "w"), indent=1)
    print("OK", modelo)


# ---------------- placar comum (docs/pre-proposta.md, "Placar comum") ----------------
# Uma regua so para todos os modelos: k-fold estratificado repetido (3 x 5) sobre o treino oficial, separacao interna
# de 20% para qualquer parada, OOF salva por cliente e veredito pareado (ganhou / empatou / perdeu) pelo IC95 de um
# bootstrap de clientes. O teste nao entra aqui.
PLACAR_SEMENTES = (101, 102, 103)
PLACAR_DOBRAS = 5
PLACAR_FRACAO_PARADA = 0.2
PLACAR_BOOT = 2000


def divisoes_placar(y, sementes=PLACAR_SEMENTES):
    """[dict(semente, dobra, treino, validacao, ajuste, parada)] com posicoes no treino oficial.
    `ajuste` e `parada` sao posicoes dentro de `treino` (ajuste + parada = treino); a validacao nao decide nada."""
    y = np.asarray(y)
    saida = []
    for s in sementes:
        for i, (a, b) in enumerate(StratifiedKFold(PLACAR_DOBRAS, shuffle=True, random_state=s).split(np.zeros(len(y)), y)):
            aj, pa = train_test_split(np.arange(len(a)), test_size=PLACAR_FRACAO_PARADA, stratify=y[a], random_state=1000 * s + i)
            saida.append(dict(semente=s, dobra=i, treino=a, validacao=b, ajuste=np.sort(aj), parada=np.sort(pa)))
    return saida


def assinatura_treino(X_train):
    """Identifica os clientes do treino oficial, na ordem: OOFs so se comparam com a mesma assinatura."""
    return hashlib.sha256(json.dumps([int(i) for i in X_train.index]).encode()).hexdigest()[:16]


def resumir_placar(y, divs, p_val, p_treino=None, extra=None):
    """Junta as previsoes de cada divisao (mesma ordem de `divs`) na medida do placar.
    p_treino (opcional): previsoes no treino da dobra, como o modelo treinou nele (MSE/CE de treino, slide 14)."""
    y = np.asarray(y).astype(float)
    sementes = sorted({d["semente"] for d in divs})
    oof = np.full((len(sementes), len(y)), np.nan, dtype="float32")
    ks_d = []
    for d, p in zip(divs, p_val):
        p = np.asarray(p, dtype=float)
        oof[sementes.index(d["semente"]), d["validacao"]] = p
        ks_d.append(ks(y[d["validacao"]], p))
    if np.isnan(oof).any():
        raise ValueError("faltam previsoes de validacao: as divisoes nao cobrem o treino inteiro")
    media = oof.mean(0)
    r = dict(ks_kfold=float(np.mean(ks_d)), ep=float(np.std(ks_d, ddof=1) / np.sqrt(len(ks_d))), ks_dobras=[float(k) for k in ks_d],
             ks_oof=ks(y, media), auc_oof=float(roc_auc_score(y, media)), mse_val=mse(y, media), ce_val=ce(y, media),
             sementes=sementes, oof=oof)
    if p_treino is not None:
        r["mse_treino"] = float(np.mean([mse(np.asarray(yt), np.asarray(p)) for yt, p in p_treino]))
        r["ce_treino"] = float(np.mean([ce(np.asarray(yt), np.asarray(p)) for yt, p in p_treino]))
    if extra:
        r.update(extra)
    return r


def medir_placar(X_train, y_train, ajustar_prever, divs=None):
    """Mede um modelo qualquer no placar.

    ajustar_prever(X_tr, y_tr, X_val, ajuste, parada, semente) -> p_val, ou (p_val, (y_tr_usado, p_tr_usado)).
      X_tr/y_tr chegam crus (sem scaler nem oversampling): o modelo prepara so com eles (ex.: comum.preparar).
      Qualquer parada (epocas, arvores) usa so X_tr.iloc[parada]; depois o modelo pode retreinar em X_tr inteiro."""
    y = np.asarray(y_train)
    divs = divs or divisoes_placar(y)
    p_val, p_tr = [], []
    for d in divs:
        r = ajustar_prever(X_train.iloc[d["treino"]], y_train.iloc[d["treino"]], X_train.iloc[d["validacao"]],
                           d["ajuste"], d["parada"], 1000 * d["semente"] + d["dobra"])
        if isinstance(r, tuple):
            p_val.append(r[0]); p_tr.append(r[1])
        else:
            p_val.append(r)
    return resumir_placar(y, divs, p_val, p_tr if p_tr else None)


def comparar_placar(y, oof_a, oof_b, n_boot=PLACAR_BOOT, semente=0):
    """b contra a, pareado nos mesmos clientes. Delta = KS(OOF b) - KS(OOF a); IC95 por bootstrap de clientes."""
    y = np.asarray(y)
    a = np.asarray(oof_a).reshape(-1, len(y)).mean(0)
    b = np.asarray(oof_b).reshape(-1, len(y)).mean(0)
    rng = np.random.default_rng(semente)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ds.append(ks(y[i], b[i]) - ks(y[i], a[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(delta=ks(y, b) - ks(y, a), ic95=[float(lo), float(hi)], p_positivo=float(np.mean(np.array(ds) > 0)),
                veredito="ganhou" if lo > 0 else "perdeu" if hi < 0 else "empatou")


def _arq_placar(pasta):
    return os.path.join(pasta, "placar.jsonl")


def ler_placar(pasta=RESULTADOS):
    """Linhas do placar; a ultima linha de cada (modelo, rotulo) vale."""
    linhas = {}
    if os.path.isfile(_arq_placar(pasta)):
        with open(_arq_placar(pasta), encoding="utf-8") as f:
            for l in f:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                linhas[(r["modelo"], r["rotulo"])] = r
    return list(linhas.values())


def _oof_placar(pasta, linha):
    return np.load(os.path.join(pasta, linha["oof"]))


def registrar_placar(pasta, modelo, rotulo, config, medida, y_train, assinatura, contra=None):
    """Grava a OOF e uma linha no placar. contra = {"vs_trial0": (modelo, rotulo), ...}: comparacoes pareadas.
    Se ja houver uma linha do xgboost (rotulo 'final', senao a ultima) com a mesma assinatura, entra vs_xgboost."""
    os.makedirs(os.path.join(pasta, "placar_oof"), exist_ok=True)
    nome = f"placar_oof/{modelo}__{rotulo}.npy"
    np.save(os.path.join(pasta, nome), medida["oof"])
    existentes = {(r["modelo"], r["rotulo"]): r for r in ler_placar(pasta)}
    contra = dict(contra or {})
    xg = [r for r in existentes.values() if r["modelo"] == "xgboost" and r["assinatura"] == assinatura]
    if xg and modelo != "xgboost":
        contra.setdefault("vs_xgboost", ("xgboost", next((r["rotulo"] for r in xg if r["rotulo"] == "final"), xg[-1]["rotulo"])))
    linha = dict(modelo=modelo, rotulo=rotulo, data=time.strftime("%Y-%m-%d %H:%M"), assinatura=assinatura, config=config, oof=nome,
                 **{k: v for k, v in medida.items() if k != "oof"})
    for campo, chave in contra.items():
        ref = existentes.get(tuple(chave))
        if ref is None or ref["assinatura"] != assinatura:
            linha[campo] = None
            continue
        linha[campo] = dict(referencia=list(chave), **comparar_placar(y_train, _oof_placar(pasta, ref), medida["oof"]))
    linha.update(ks_teste=None, auc_teste=None, mse_teste=None)
    with open(_arq_placar(pasta), "a", encoding="utf-8") as f:
        f.write(json.dumps(linha) + "\n")
    return linha


def registrar_teste_placar(pasta, modelo, rotulo, y_teste, p_teste_sementes):
    """Preenche o teste de uma linha ja medida (uma vez, so na configuracao final). Media das sementes + IC95."""
    linhas = {(r["modelo"], r["rotulo"]): r for r in ler_placar(pasta)}
    r = dict(linhas[(modelo, rotulo)])
    p = np.mean(np.asarray(p_teste_sementes, dtype=float).reshape(-1, len(y_teste)), 0)
    y = np.asarray(y_teste)
    r.update(ks_teste=ks(y, p), ks_teste_ic95=list(bootstrap_ks(y, p)), auc_teste=float(roc_auc_score(y, p)), mse_teste=mse(y, p), ce_teste=ce(y, p),
             data_teste=time.strftime("%Y-%m-%d %H:%M"))
    with open(_arq_placar(pasta), "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    return r


def tabela_placar(pasta=RESULTADOS):
    """Uma linha por (modelo, rotulo): KS do placar, vereditos e teste."""
    rows = []
    for r in ler_placar(pasta):
        row = dict(modelo=r["modelo"], rotulo=r["rotulo"], ks_kfold=r["ks_kfold"], ep=r["ep"], ks_oof=r["ks_oof"], auc_oof=r["auc_oof"],
                   mse_val=r.get("mse_val"), mse_treino=r.get("mse_treino"))
        for campo in ["vs_trial0", "vs_xgboost"] + sorted(k for k in r if k.startswith("vs_") and k not in ("vs_trial0", "vs_xgboost")):
            v = r.get(campo)
            row[campo] = None if not v else f"{v['veredito']} ({v['delta']:+.4f} [{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}])"
        row.update(ks_teste=r.get("ks_teste"), auc_teste=r.get("auc_teste"))
        rows.append(row)
    return pd.DataFrame(rows)


def adotar_placar(pasta, modelo, rotulo, motivo):
    """Marca a versao adotada de um modelo (a que vai para a comparacao final). Guarda em escolhidos.json."""
    linhas = {(r["modelo"], r["rotulo"]) for r in ler_placar(pasta)}
    if (modelo, rotulo) not in linhas:
        raise KeyError(f"{modelo}/{rotulo} nao esta no placar")
    arq = os.path.join(pasta, "escolhidos.json")
    esc = json.load(open(arq, encoding="utf-8")) if os.path.isfile(arq) else {}
    esc[modelo] = dict(rotulo=rotulo, motivo=motivo, data=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(esc, open(arq, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return esc[modelo]


def relatorio_placar(pasta=RESULTADOS):
    """Markdown do placar: a versao adotada de cada modelo e, abaixo, todas as versoes medidas de cada um."""
    arq = os.path.join(pasta, "escolhidos.json")
    esc = json.load(open(arq, encoding="utf-8")) if os.path.isfile(arq) else {}
    linhas = ler_placar(pasta)
    f = lambda x: "-" if x is None else f"{x:.4f}"
    v = lambda r, k: "-" if not r.get(k) else f"{r[k]['veredito']} ({r[k]['delta']:+.4f})"
    L = ["# Placar", "", "Gerado por `comum.relatorio_placar`. Regra em `docs/pre-proposta.md` (\"Placar comum\"): k-fold estratificado 3 x 5 (sementes "
         f"{', '.join(map(str, PLACAR_SEMENTES))}), parada em 20% internos, veredito pareado pelo IC95 do bootstrap de clientes. "
         "O KS do placar decide; o teste so confere.", "", "## Modelo adotado de cada um", "",
         "| Modelo | Versao | KS placar | ep | KS teste | IC95 teste | AUROC teste | vs trial 0 | vs XGBoost | Motivo |", "|---|---|---|---|---|---|---|---|---|---|"]
    por = {(r["modelo"], r["rotulo"]): r for r in linhas}
    for m, e in sorted(esc.items()):
        r = por.get((m, e["rotulo"]), {})
        ic = r.get("ks_teste_ic95")
        L.append(f"| {m} | {e['rotulo']} | {f(r.get('ks_kfold'))} | {f(r.get('ep'))} | {f(r.get('ks_teste'))} | "
                 f"{'-' if not ic else f'{ic[0]:.3f} a {ic[1]:.3f}'} | {f(r.get('auc_teste'))} | {v(r, 'vs_trial0')} | {v(r, 'vs_xgboost')} | {e['motivo']} |")
    for m in sorted({r["modelo"] for r in linhas}):
        rs = sorted([r for r in linhas if r["modelo"] == m], key=lambda r: -r["ks_kfold"])
        outros = sorted({k for r in rs for k in r if k.startswith("vs_") and k not in ("vs_trial0", "vs_xgboost")})
        L += ["", f"## {m}: todas as versoes (maior KS do placar primeiro)", "",
              "| Versao | KS placar | ep | KS OOF | AUROC OOF | MSE treino | MSE val | vs trial 0 | " + " | ".join(outros) + (" | " if outros else "") + "KS teste |",
              "|---" * (9 + len(outros)) + "|"]
        for r in rs:
            L.append(f"| {r['rotulo']}{' **(adotado)**' if esc.get(m, {}).get('rotulo') == r['rotulo'] else ''} | {f(r['ks_kfold'])} | {f(r['ep'])} | "
                     f"{f(r['ks_oof'])} | {f(r['auc_oof'])} | {f(r.get('mse_treino'))} | {f(r.get('mse_val'))} | {v(r, 'vs_trial0')} | "
                     + "".join(f"{v(r, k)} | " for k in outros) + f"{f(r.get('ks_teste'))} |")
    L += ["", "Diferencas de KS menores que uns 0,01 ficam dentro do ruido do placar: veja o veredito antes de dizer que algo melhorou."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    folds, final, cols = carregar_dobras()
    for i, (a, b, c, d) in enumerate(folds):
        print(f"Dobra {i}: treino {a.shape} (churn {b.mean()*100:.1f}%), validacao {c.shape} (churn {d.mean()*100:.1f}%)")
    print("final:", final[0].shape, final[2].shape, "churn teste %.1f%%" % (final[3].mean() * 100), "n_feat", len(cols))
    # conferencia do KS contra scipy
    rng = np.random.default_rng(1); y = (rng.random(2000) < .265).astype(int)
    p = np.clip(rng.normal(.3 + .25 * y, .2), 0, 1)
    print("KS meu     :", ks(y, p))
    print("KS scipy   :", ks_2samp(p[y == 1], p[y == 0]).statistic)
