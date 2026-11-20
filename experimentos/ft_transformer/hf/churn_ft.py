"""FT-Transformer para churn (Telco IBM): pre-processamento e modelo, sem depender do resto do projeto.

Uso:
    import pandas as pd, churn_ft
    modelo = churn_ft.carregar(".")                      # pasta com config.json e pesos_semente*.pt
    p = modelo.prever(pd.read_csv("clientes.csv"))       # probabilidade de churn, media das sementes
    classe = p >= modelo.limiar                          # limiar do KS escolhido na validacao cruzada

O CSV tem as colunas do dataset Telco Customer Churn (IBM); customerID e Churn sao opcionais.
"""
import json, math, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

BINARIAS = {"gender": {"Female": 0, "Male": 1}, "Partner": {"No": 0, "Yes": 1}, "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1}, "PaperlessBilling": {"No": 0, "Yes": 1}}
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORIAS = {"InternetService": ["DSL", "Fiber optic", "No"], "Contract": ["Month-to-month", "One year", "Two year"],
              "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]}
PROTECAO = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
FEATURES = {
    "tenure_1": lambda d: (d["tenure"] <= 1).astype(int),
    "cliente_novo": lambda d: (d["tenure"] <= 6).astype(int),
    "delta_preco": lambda d: d["MonthlyCharges"] - d["TotalCharges"] / (d["tenure"] + 1),
    "cobranca_razao": lambda d: d["TotalCharges"] / (d["MonthlyCharges"] * d["tenure"].clip(lower=1)),
    "antigo": lambda d: (d["tenure"] > 48).astype(int),
    "caro_sem_backup": lambda d: ((d["MonthlyCharges"] > 98) & (d["OnlineBackup"] == 0)).astype(int),
    "fibra_sem_protecao": lambda d: ((d["InternetService_Fiber optic"] == 1) & (d[PROTECAO].sum(axis=1) == 0)).astype(int),
    "atrito_pagamento": lambda d: ((d["PaymentMethod_Electronic check"] == 1) & (d["PaperlessBilling"] == 1)
                                   & (d["Contract_Month-to-month"] == 1)).astype(int),
}


def codificar(df):
    """CSV cru -> colunas numericas (o pre-processamento do grupo). As dummies tem categorias fixas, entao uma linha so funciona."""
    d = df.drop(columns=[c for c in ("customerID", "Churn") if c in df.columns]).copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    for col, m in BINARIAS.items():
        d[col] = d[col].map(m)
    d["MultipleLines"] = d["MultipleLines"].map({"Yes": 1, "No": 0, "No phone service": 0})
    for col in ADDONS:
        d[col] = d[col].map({"Yes": 1, "No": 0, "No internet service": 0})
    for var, cats in CATEGORIAS.items():
        for c in cats:
            d[f"{var}_{c}"] = (d[var] == c).astype(int)
        d = d.drop(columns=var)
    d["total_addons"] = d[["MultipleLines"] + ADDONS].sum(axis=1)
    d["PaymentMethod_Automatic"] = d[["PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)"]].max(axis=1)
    return d


# ---------------- modelo (mesma arquitetura de experimentos/ft_transformer/modelo.py) ----------------
class Tokenizador(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.peso = nn.Parameter(torch.empty(n_features, d_token))
        self.vies = nn.Parameter(torch.empty(n_features, d_token))

    def forward(self, x):
        return x.unsqueeze(-1) * self.peso + self.vies


class Bloco(nn.Module):
    def __init__(self, d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator, primeiro):
        super().__init__()
        h = int(d_token * ffn_fator)
        self.norm_atencao = None if primeiro else nn.LayerNorm(d_token)
        self.atencao = nn.MultiheadAttention(d_token, n_cabecas, dropout=drop_atencao, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_token)
        self.ffn_entrada = nn.Linear(d_token, 2 * h)
        self.ffn_saida = nn.Linear(h, d_token)
        self.drop_ffn, self.drop_residual = nn.Dropout(drop_ffn), nn.Dropout(drop_residual)

    def forward(self, x):
        z = x if self.norm_atencao is None else self.norm_atencao(x)
        z, _ = self.atencao(z, z, z, need_weights=False)
        x = x + self.drop_residual(z)
        a, b = self.ffn_entrada(self.norm_ffn(x)).chunk(2, dim=-1)
        return x + self.drop_residual(self.ffn_saida(self.drop_ffn(a * F.relu(b))))      # ReGLU


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_blocos, d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator):
        super().__init__()
        self.tokenizador = Tokenizador(n_features, d_token)
        self.cls = nn.Parameter(torch.empty(d_token))
        self.blocos = nn.ModuleList(Bloco(d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator, primeiro=(k == 0))
                                    for k in range(n_blocos))
        self.norm_final = nn.LayerNorm(d_token)
        self.saida = nn.Linear(d_token, 1)

    def forward(self, x):
        t = torch.cat([self.tokenizador(x), self.cls.expand(len(x), 1, -1)], dim=1)     # [CLS] no fim
        for b in self.blocos:
            t = b(t)
        return self.saida(F.relu(self.norm_final(t[:, -1]))).squeeze(-1)


class Modelo:
    def __init__(self, cfg, redes):
        self.cfg, self.redes, self.limiar = cfg, redes, cfg["limiar_ks"]

    def entrada(self, df):
        d = codificar(df)
        for f in self.cfg["features"]:
            d[f] = FEATURES[f](d)
        d = d[self.cfg["colunas"]].astype(float)
        esc = self.cfg["escala"]
        cols = list(esc["colunas"])
        d[cols] = (d[cols] - np.asarray(esc["media"])) / np.asarray(esc["desvio"])
        return np.asarray(d, dtype="float32")

    @torch.no_grad()
    def prever(self, df):
        x = torch.as_tensor(self.entrada(df))
        return np.mean([torch.sigmoid(m(x)).numpy() for m in self.redes], 0)


def carregar(pasta="."):
    cfg = json.load(open(os.path.join(pasta, "config.json"), encoding="utf-8"))
    redes = []
    for arq in cfg["pesos"]:
        m = FTTransformer(len(cfg["colunas"]), **cfg["arquitetura"])
        m.load_state_dict(torch.load(os.path.join(pasta, arq), map_location="cpu", weights_only=True))
        redes.append(m.eval())
    return Modelo(cfg, redes)
