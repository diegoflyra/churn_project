"""Os processamentos de dados que ajudaram o MLP, medidos em outras familias de modelo no placar comum.

Uso (da raiz):  python experimentos/sondar_processamentos.py
Gera: resultados/processamentos_comuns.json (nao entra no placar.jsonl: e uma sonda, nao o modelo de ninguem).

Cada processamento usa so comum.preparar(features=..., over=...), igual para todos os modelos. Os modelos da sonda sao
um XGBoost fixo (sem busca) e uma regressao logistica: dizem se o processamento ajuda o dado ou so o MLP.
Para testar no seu modelo, use a mesma chamada de comum.preparar dentro da sua funcao do placar (README).
"""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import comum

PROCESSAMENTOS = {
    "base do grupo": dict(),
    "+ tenure_1": dict(features=("tenure_1",)),
    "+ cliente_novo": dict(features=("cliente_novo",)),
    "+ delta_preco": dict(features=("delta_preco",)),
    "+ as tres (MLP adotado)": dict(features=("cliente_novo", "delta_preco", "tenure_1")),
    "sem oversampling": dict(over=False),
    "oversampling ate 0,5": dict(over=0.5),
    "as tres, sem oversampling": dict(features=("cliente_novo", "delta_preco", "tenure_1"), over=False),
}
MODELOS = {
    "xgboost": lambda s: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
                                        n_jobs=4, random_state=s, verbosity=0),
    "logistica": lambda s: LogisticRegression(max_iter=2000, C=1.0),
}


def medir(nome_modelo, proc, X, y):
    def ajustar_prever(X_tr, y_tr, X_val, ajuste, parada, semente):
        Xa, ya, Xv = comum.preparar(X_tr, y_tr, X_val, **proc)
        m = MODELOS[nome_modelo](semente).fit(Xa, ya)
        return m.predict_proba(Xv)[:, 1], (np.asarray(ya), m.predict_proba(Xa)[:, 1])
    return comum.medir_placar(X, y, ajustar_prever)


def main():
    X, _, y, _ = comum.dividir(comum.carregar_df())
    yv = np.asarray(y)
    saida = dict(aviso="Placar comum (3x5, sementes 101-103), sem o teste. Veredito pareado contra a 'base do grupo' do mesmo modelo.",
                 assinatura=comum.assinatura_treino(X), resultados={})
    for mod in MODELOS:
        med = {p: medir(mod, kw, X, y) for p, kw in PROCESSAMENTOS.items()}
        base = med["base do grupo"]["oof"]
        saida["resultados"][mod] = {}
        for p, r in med.items():
            v = comum.comparar_placar(yv, base, r["oof"])
            saida["resultados"][mod][p] = dict(ks_kfold=r["ks_kfold"], ep=r["ep"], ks_oof=r["ks_oof"], auc_oof=r["auc_oof"], mse_val=r["mse_val"],
                                               ce_val=r["ce_val"], vs_base=v)
            print(f"{mod:<10} {p:<28} KS {r['ks_kfold']:.4f} | MSE val {r['mse_val']:.4f} | vs base {v['veredito']} {v['delta']:+.4f} "
                  f"[{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}]", flush=True)
    json.dump(saida, open(os.path.join(comum.RESULTADOS, "processamentos_comuns.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
