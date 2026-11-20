"""FT-Transformer final: treino no treino oficial inteiro, teste medido uma vez e exportacao para o Hugging Face.

Como o final do MLP (experimentos/mlp/fase5.final): em cada semente, 20% do treino decidem as epocas pela parada da
configuracao; depois a rede retreina no treino inteiro por essas epocas e preve o teste. A previsao e a media das sementes.
O limiar das metricas de classe e o do KS na OOF do placar (nunca escolhido no teste).
"""
import sys, os, json, shutil, time
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, fase1 as F1

HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
f32 = lambda a: np.asarray(a, dtype="float32")


def final(c, n_seeds=5, dispositivo=None):
    hp, dados = c["hp"], c["dados"]
    if hp.get("tokens") or hp.get("num_emb") or hp.get("n_redes", 1) != 1:
        raise ValueError("o final e a exportacao cobrem o FT de tokens por coluna, sem PLR e com uma rede por semente")
    Xtr, Xte, ytr, yte = F1._dados()
    y = np.asarray(ytr)
    XA, yA, XT = comum.preparar(Xtr, ytr, Xte, **dados)
    p, eps, estados = [], [], []
    for s in range(n_seeds):
        sem = 42 + 1000 * s
        a1, a2 = train_test_split(np.arange(len(y)), test_size=comum.PLACAR_FRACAO_PARADA, stratify=y, random_state=comum.SEMENTE + s)
        Xa, ya, Xp = comum.preparar(Xtr.iloc[a1], ytr.iloc[a1], Xtr.iloc[a2], **dados)
        _, ep = M.treinar(f32(Xa), f32(ya), hp, sem, f32(Xp), f32(y[a2]), dispositivo=dispositivo, colunas=list(Xa.columns))
        m, _ = M.treinar(f32(XA), f32(yA), hp, sem, epocas=max(1, ep), dispositivo=dispositivo, colunas=list(XA.columns))
        p.append(M.prever(m, f32(XT))); eps.append(int(ep))
        estados.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
    return dict(p_sementes=[x.tolist() for x in p], epocas=eps, y_teste=[int(v) for v in yte], colunas=list(XA.columns),
                estados=estados)


def metricas_teste(oof, fin):
    """Metricas do teste com o limiar do KS escolhido na OOF do placar."""
    lim = comum.limiar_ks(np.asarray(F1._dados()[2]), np.asarray(oof).reshape(-1, len(F1._dados()[2])).mean(0))
    y, p = np.asarray(fin["y_teste"]), np.mean(fin["p_sementes"], 0)
    m = comum.metricas(y, p, lim)
    m["ks_ic95"] = [float(v) for v in comum.bootstrap_ks(y, p)]
    m["ks_por_semente"] = [comum.ks(y, np.asarray(x)) for x in fin["p_sementes"]]
    return json.loads(json.dumps(m, default=float))


def escala(c):
    """Media e desvio do scaler de comum.preparar no treino oficial inteiro (as mesmas colunas, na mesma ordem)."""
    Xtr, _, _, _ = F1._dados()
    feats = list(c["dados"].get("features", ()))
    X = comum.adicionar_features(Xtr, feats)
    cols = comum.NUMERIC_COLS + [f for f in feats if X[f].nunique() > 2]
    sc = StandardScaler().fit(X[cols])
    return dict(colunas=cols, media=sc.mean_.tolist(), desvio=sc.scale_.tolist())


def exportar(destino, rotulo, c, fin, metricas, linha_placar):
    """Pasta pronta para o Hugging Face: pesos, config.json, churn_ft.py (sem dependencia do projeto) e README.md."""
    os.makedirs(destino, exist_ok=True)
    pesos = []
    for s, est in enumerate(fin["estados"]):
        arq = f"pesos_semente{s}.pt"
        torch.save(est, os.path.join(destino, arq)); pesos.append(arq)
    cfg = dict(modelo="FT-Transformer", rotulo=rotulo, arquitetura=M.arquitetura(c["hp"]), hp=c["hp"],
               features=list(c["dados"].get("features", ())), colunas=fin["colunas"], escala=escala(c),
               limiar_ks=metricas["limiar"], pesos=pesos, epocas=fin["epocas"],
               placar=dict(ks_kfold=linha_placar["ks_kfold"], ks_oof=linha_placar["ks_oof"], auc_oof=linha_placar["auc_oof"]),
               teste={k: metricas[k] for k in ("ks", "ks_ic95", "auroc", "precision", "recall", "f1")},
               data=time.strftime("%Y-%m-%d"))
    json.dump(cfg, open(os.path.join(destino, "config.json"), "w", encoding="utf-8"), indent=1)
    shutil.copy(os.path.join(HF, "churn_ft.py"), destino)
    modelo_card = open(os.path.join(HF, "README_modelo.md"), encoding="utf-8").read()
    t = cfg["teste"]
    campos = dict(rotulo=rotulo, ks=f"{t['ks']:.4f}", ks_lo=f"{t['ks_ic95'][0]:.3f}", ks_hi=f"{t['ks_ic95'][1]:.3f}",
                  auroc=f"{t['auroc']:.4f}", recall=f"{t['recall']:.3f}", precision=f"{t['precision']:.3f}", f1=f"{t['f1']:.3f}",
                  ks_placar=f"{cfg['placar']['ks_kfold']:.4f}", limiar=f"{cfg['limiar_ks']:.4f}", n_sementes=len(pesos),
                  features=", ".join(f"`{f}`" for f in cfg["features"]), arquitetura=json.dumps(cfg["arquitetura"]),
                  epocas=", ".join(map(str, cfg["epocas"])))
    open(os.path.join(destino, "README.md"), "w", encoding="utf-8").write(modelo_card.format(**campos))
    return cfg
