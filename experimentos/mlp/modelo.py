"""MLP - modelo, treino por dobra e espaco de busca (8 hiperparametros da pre-proposta + paciencia).

Uso:  python experimentos/mlp/modelo.py 100      # estudo com 100 trials
Salva resultados/mlp.db (SQLite do Optuna, retomavel).
"""
import sys, os, copy, math
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")     # no PyTorch 2.14 o produto em lote da pilha pede um compilador que o ambiente pode nao ter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import optuna
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum

torch.set_num_threads(1)
# GPU usada automaticamente se existir. Para forcar CPU: CHURN_DEVICE=cpu
DEVICE = torch.device(os.environ.get("CHURN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
ACT = {"logistica": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": lambda: nn.LeakyReLU(0.1)}
OTIM = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop, "adadelta": torch.optim.Adadelta,
        "sgd": lambda p, **k: torch.optim.SGD(p, momentum=0.9, **k)}
# Fixos: no maximo 10.000 epocas e lote fixo. A paciencia e hiperparametro (padrao 10).
MAX_EPOCAS = int(os.environ.get("CHURN_MAX_EPOCAS", 10000))
PACIENCIA = 10
LOTE = 256


def perda(z, y, tipo):
    """z = logit. CE = entropia cruzada binaria; MSE = erro quadratico da probabilidade (Brier)."""
    if tipo == "ce":
        return F.binary_cross_entropy_with_logits(z, y)
    return F.mse_loss(torch.sigmoid(z), y)


def perda_treino(z, y, hp):
    """Perda do treino, com os opcionais da fase 3: suavizacao do rotulo, peso da classe positiva e penalidade no logit."""
    suav, peso, pen = hp.get("suav", 0.0), hp.get("peso_pos"), hp.get("pen_logit", 0.0)
    if not (suav or peso or pen):
        return perda(z, y, hp["perda"])
    alvo = y * (1 - suav) + 0.5 * suav
    if hp["perda"] == "ce":
        l = F.binary_cross_entropy_with_logits(z, alvo, pos_weight=torch.tensor(peso, device=z.device) if peso else None)
    else:
        w = 1 + (peso - 1) * y if peso else 1.0
        l = (w * (torch.sigmoid(z) - alvo) ** 2).mean()
    return l + pen * (z ** 2).mean()


class MLP(nn.Module):
    """`plano` tem uma letra por camada oculta: N = nada, B = BatchNorm, D = Dropout. Uma letra so por camada garante
    que BN e Dropout nunca ficam juntos. Sem plano, vale o formato antigo: Dropout(dropout) em todas as camadas."""

    def __init__(self, d_in, camadas, unidades, ativ, dropout, plano=None, funil=False):
        super().__init__()
        if plano is not None and (len(plano) != camadas or set(plano) - set("NBD")):
            raise ValueError(f"plano invalido para {camadas} camadas: {plano!r}")
        ls, d = [], d_in
        for k in range(camadas):
            u = max(4, unidades // 2 ** k) if funil else unidades
            tipo = plano[k] if plano is not None else "D"
            ls.append(nn.Linear(d, u))
            if tipo == "B":
                ls.append(nn.BatchNorm1d(u))
            ls.append(ACT[ativ]())
            if tipo == "D":
                ls.append(nn.Dropout(dropout))
            d = u
        self.rede = nn.Sequential(*ls, nn.Linear(d, 1))

    def forward(self, x):
        return self.rede(x).squeeze(-1)


def treinar(Xtr, ytr, hp, seed, Xval=None, yval=None, epocas=None, ao_fim_epoca=None):
    """Com validacao: early stopping na perda dela, ou no KS se hp["parada"] == "ks". Com `epocas`: treina esse numero
    fixo de epocas, sem parar antes. `ao_fim_epoca(ep, m)` permite observar a rede a cada epoca."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1], hp["camadas"], hp["unidades"], hp["ativ"], hp["dropout"], hp.get("plano"), hp.get("funil", False)).to(DEVICE)
    opt = OTIM[hp["otim"]](m.parameters(), lr=hp["lr"], weight_decay=hp["l2"])
    Xt, yt = torch.tensor(Xtr, device=DEVICE), torch.tensor(ytr, device=DEVICE)
    if Xval is not None:
        Xv, yv = torch.tensor(Xval, device=DEVICE), torch.tensor(yval, device=DEVICE)
    lote, tem_bn = hp.get("lote", LOTE), "B" in (hp.get("plano") or "")
    best, best_state, melhor_ep, esp = math.inf, None, 0, 0
    for ep in range(1, (epocas or MAX_EPOCAS) + 1):
        m.train(); perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), lote):
            idx = perm[i:i + lote]
            if tem_bn and len(idx) < 2:        # BatchNorm nao treina com um exemplo so
                continue
            opt.zero_grad()
            perda_treino(m(Xt[idx]), yt[idx], hp).backward(); opt.step()
        if ao_fim_epoca:
            m.eval(); ao_fim_epoca(ep, m)
        if epocas is not None:
            continue
        m.eval()
        with torch.no_grad():
            zv = m(Xv)
            lv = perda(zv, yv, hp["perda"]).item()
        if not math.isfinite(lv): break
        if hp.get("parada") == "ks":
            lv = -comum.ks(yval, torch.sigmoid(zv).cpu().numpy())
        if lv < best - 1e-6: best, best_state, melhor_ep, esp = lv, copy.deepcopy(m.state_dict()), ep, 0
        else:
            esp += 1
            if esp >= hp.get("paciencia", PACIENCIA): break
    if best_state: m.load_state_dict(best_state)
    return m.eval(), melhor_ep


def prever(m, X):
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(X, device=DEVICE))).cpu().numpy()
    return np.nan_to_num(p, nan=0.5)         # configuracao divergente nao derruba o estudo: vira KS ruim


def fit(hp, f, Xlist, seed):
    Xt, yt, Xv, yv = f
    m, melhor_ep = treinar(Xt, yt, hp, seed, Xv, yv)
    return [prever(m, X) for X in Xlist], prever(m, Xt), melhor_ep


def fit_final(hp, final, epocas, seed):
    """Treino inteiro com `preparar`, pela mediana das melhores epocas nas dobras (nao ha validacao para parar)."""
    Xt, yt, Xte, _ = final
    m, _ = treinar(Xt, yt, hp, seed, epocas=max(1, int(np.median(epocas))))
    return prever(m, Xte)


def hp_mlp(t):
    return dict(camadas=t.suggest_int("camadas", 1, 2), unidades=t.suggest_int("unidades", 4, 128, log=True),
                ativ=t.suggest_categorical("ativ", ["logistica", "tanh", "relu"]),
                otim=t.suggest_categorical("otim", ["adam", "rmsprop", "sgd", "adadelta"]),
                lr=t.suggest_float("lr", 1e-4, 1e-1, log=True), dropout=t.suggest_float("dropout", 0.0, 0.5),
                l2=t.suggest_float("l2", 1e-6, 1e-2, log=True), perda=t.suggest_categorical("perda", ["ce", "mse"]),
                paciencia=t.suggest_categorical("paciencia", [5, 10, 20, 30]))


# Trial 0 (pre-proposta): 1 camada com 10 unidades
MLP_PADRAO = dict(camadas=1, unidades=10, ativ="relu", otim="adam", lr=1e-3, dropout=0.0, l2=1e-6, perda="ce", paciencia=PACIENCIA)


def estudo(n_trials, timeout=None, callbacks=None):
    return comum.estudo("mlp", n_trials, hp_mlp, fit, MLP_PADRAO, timeout=timeout, callbacks=callbacks)


if __name__ == "__main__":
    n = int(sys.argv[1])
    optuna.logging.set_verbosity(optuna.logging.INFO)
    st = estudo(n)
    print("melhor KS", st.best_value, st.best_params)
