"""STab (Jefferson) - modelo, treino por dobra e espaco de busca.

Uso:  python experimentos/stab/modelo.py 40      # estudo com 40 trials
Salva resultados/stab.db (SQLite do Optuna, retomavel).
"""
import sys, os, copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import optuna
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum

torch.set_num_threads(1)
# GPU (Colab T4): usada automaticamente se existir. Para forcar CPU: CHURN_DEVICE=cpu
DEVICE = torch.device(os.environ.get("CHURN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
GPU = DEVICE.type == "cuda"
ACT = {"logistica": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu}
MAX_PACIENCIA = 10
# Amostras que compartilham a mesma mascara de DropConnect. CPU: 16 (desvio do artigo, por custo).
# GPU: 1 = mascara por amostra, como no artigo.
GRUPO = int(os.environ.get("CHURN_GRUPO", 1 if GPU else 16))
LARGURAS_STAB = [64, 128, 256] if GPU else [32, 64, 128]      # no artigo o codificador tem largura 256
LARG_PADRAO_STAB = 256 if GPU else 128


def perda(z, y, tipo):
    """z = logit. CE = entropia cruzada binaria; MSE = erro quadratico da probabilidade (Brier)."""
    if tipo == "ce":
        return F.binary_cross_entropy_with_logits(z, y)
    return F.mse_loss(torch.sigmoid(z), y)


# =====================================================================
# STab: dois codificadores MLP com pesos compartilhados + DropConnect,
# cabeca de projecao g, stop-gradient, perda de cosseno simetrica.
# Hajiramezanali, Shen, Scalia, Diamant (2022). Reimplementacao propria.
# =====================================================================
class DCLinear(nn.Module):
    """Linear com DropConnect por amostra: cada conexao sobrevive com prob. p (como em Wan et al. 2013)."""
    def __init__(self, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(o, i)); self.b = nn.Parameter(torch.zeros(o))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        bound = 1 / math.sqrt(i); nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, p=None):
        if p is None:                       # inferencia: sem mascara
            return F.linear(x, self.w, self.b)
        # DESVIO DO ARTIGO (custo em CPU): a mascara e sorteada por GRUPO de GRUPO amostras, nao por amostra.
        B = x.shape[0]; s = min(GRUPO, B); G = -(-B // s); pad = G * s - B
        xp = F.pad(x, (0, 0, 0, pad)).view(G, s, -1)
        m = (torch.rand(G, *self.w.shape, device=x.device) < p).float() / p     # (G,o,i), escala invertida
        y = torch.bmm(xp, (self.w.unsqueeze(0) * m).transpose(1, 2)).reshape(G * s, -1)[:B]
        return y + self.b


class MLPDC(nn.Module):
    def __init__(self, dims, act):
        super().__init__()
        self.layers = nn.ModuleList(DCLinear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = ACT[act]

    def forward(self, x, p=None):
        for k, l in enumerate(self.layers):
            x = l(x, p)
            if k < len(self.layers) - 1:
                x = self.act(x)
        return x


class STab(nn.Module):
    def __init__(self, d_in, prof, larg, act, larg_g=None, prof_g=2):
        super().__init__()
        self.enc = MLPDC([d_in] + [larg] * prof, act)
        # g: 2 camadas [larg, larg] (artigo); a entrada acompanha a saida do codificador
        self.g = MLPDC([larg] * (prof_g + 1), act)

    def perda_cos(self, x, p):
        z1, z2 = self.enc(x, p), self.enc(x, p)          # mesmos pesos, mascaras diferentes
        p1, p2 = self.g(z1), self.g(z2)
        d = lambda a, b: -F.cosine_similarity(a, b.detach(), dim=-1).mean()   # stop-gradient em b
        return 0.5 * d(p1, z2) + 0.5 * d(p2, z1)

    @torch.no_grad()
    def embed(self, x):
        return self.enc(x, None)


def stab_pretreino(Xtr, Xval, hp, seed, max_ep=30, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = STab(Xtr.shape[1], hp["prof"], hp["larg"], hp["ativ"]).to(DEVICE)
    if hp["otim"] == "adam":
        opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    else:
        opt = torch.optim.SGD(m.parameters(), lr=hp["lr"], momentum=0.9, weight_decay=hp["wd"])
    Xt, Xv = torch.tensor(Xtr, device=DEVICE), torch.tensor(Xval, device=DEVICE)
    best, best_state, esp = 1e9, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i + bs]]
            if len(xb) < 8: continue
            opt.zero_grad(); l = m.perda_cos(xb, hp["p"]); l.backward(); opt.step()
        with torch.no_grad():    # perda nao supervisionada na validacao (sem rotulos)
            lv = m.perda_cos(Xv[:512], hp["p"]).item()
        if lv < best - 1e-5: best, best_state, esp = lv, copy.deepcopy(m.state_dict()), 0
        else:
            esp += 1
            if esp >= MAX_PACIENCIA: break
    if best_state: m.load_state_dict(best_state)
    return m.eval()


def cabeca_logistica(Etr, ytr, wd, tipo, passos=100):
    """Regressao logistica sobre o embedding congelado (LBFGS, L2 = wd), perda CE ou MSE."""
    mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
    lin = nn.Linear(Etr.shape[1], 1).to(Etr.device)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=passos, line_search_fn="strong_wolfe")
    E, y = (Etr - mu) / sd, ytr

    def clo():
        opt.zero_grad()
        l = perda(lin(E).squeeze(-1), y, tipo) + wd * (lin.weight ** 2).sum()
        l.backward(); return l
    opt.step(clo)
    return lambda Ex: torch.sigmoid(lin((Ex - mu) / sd).squeeze(-1)).detach().cpu().numpy()


def stab_fit_predict(Xtr, ytr, Xav_list, hp, seed, Xval_unsup=None):
    m = stab_pretreino(Xtr, Xval_unsup if Xval_unsup is not None else Xav_list[0], hp, seed)
    T = lambda a: torch.tensor(a, device=DEVICE)
    Etr = m.embed(T(Xtr))
    head = cabeca_logistica(Etr, T(ytr), hp["wd"], hp["perda"])
    return [head(m.embed(T(X))) for X in Xav_list], head(Etr)


# =====================================================================
# Espaco de busca (8 hiperparametros, conforme a pre-proposta)
# =====================================================================
def hp_stab(t):
    return dict(prof=t.suggest_int("prof", 1, 4), larg=t.suggest_categorical("larg", LARGURAS_STAB),
                ativ=t.suggest_categorical("ativ", ["logistica", "tanh", "relu"]),
                otim=t.suggest_categorical("otim", ["adam", "sgd"]),
                lr=t.suggest_float("lr", 1e-4, 1e-1, log=True),
                p=t.suggest_float("p", 0.5, 0.95), wd=t.suggest_float("wd", 1e-6, 1e-2, log=True),
                perda=t.suggest_categorical("perda", ["ce", "mse"]))


STAB_PADRAO = dict(prof=4, larg=LARG_PADRAO_STAB, ativ="relu", otim="adam", lr=1e-3, p=0.9, wd=1e-5, perda="ce")


def fit(hp, f, Xlist, seed):
    Xt, yt, _, _ = f
    return stab_fit_predict(Xt, yt, Xlist, hp, seed)


def estudo(n_trials):
    return comum.estudo("stab", n_trials, hp_stab, fit, STAB_PADRAO)


if __name__ == "__main__":
    n = int(sys.argv[1])
    optuna.logging.set_verbosity(optuna.logging.INFO)
    st = estudo(n)
    print("melhor KS", st.best_value, st.best_params)
