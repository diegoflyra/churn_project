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


def preparar(X_tr, y_tr, X_aval):
    """Scaler ajustado so em X_tr; oversampling so em X_tr; X_aval so recebe transform."""
    scaler = StandardScaler()
    X_tr_s = X_tr.copy()
    X_tr_s[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
    X_av_s = X_aval.copy()
    X_av_s[NUMERIC_COLS] = scaler.transform(X_aval[NUMERIC_COLS])
    X_b, y_b = RandomOverSampler(random_state=SEMENTE).fit_resample(X_tr_s, y_tr)
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
