"""comum.py - dados, dobras, preparar, KS e metricas (IF702, churn).

Reproduz exatamente o pre-processamento de if702-preprocessamento.ipynb.
A unica diferenca e a leitura do CSV: aqui via CAMINHO_CSV (ver abaixo).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import RandomOverSampler

# Kaggle: "/kaggle/input/datasets/kapturovalexander/customers-churned-in-telecom-services/customer_churn_telecom_services.csv"
CAMINHO_CSV = os.environ.get("CHURN_CSV", os.path.join(os.path.dirname(__file__), "dados", "telco_ibm.csv"))
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


def carregar_dobras(path=CAMINHO_CSV):
    df = carregar_df(path)
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEMENTE, stratify=y)
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
