"""TabKAN (Jefferson) - modelo, treino por dobra e espaco de busca.

Uso:  python experimentos/tabkan/modelo.py 40      # estudo com 40 trials
Salva resultados/tabkan.db (SQLite do Optuna, retomavel).
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
MAX_PACIENCIA = 10


def perda(z, y, tipo):
    """z = logit. CE = entropia cruzada binaria; MSE = erro quadratico da probabilidade (Brier)."""
    if tipo == "ce":
        return F.binary_cross_entropy_with_logits(z, y)
    return F.mse_loss(torch.sigmoid(z), y)


# =====================================================================
# TabKAN: camadas KAN com base Chebyshev / Fourier / B-spline.
# Eslamian, Afzal Aghaei, Cheng (2025). Reimplementacao propria (o codigo do
# artigo nao estava acessivel): mesma ideia, nao o mesmo codigo.
# =====================================================================
class ChebLayer(nn.Module):
    def __init__(self, i, o, grau):
        super().__init__(); self.g = grau
        self.c = nn.Parameter(torch.randn(i, o, grau + 1) / (i * (grau + 1)))

    def forward(self, x):
        x = torch.tanh(x); T = [torch.ones_like(x), x]
        for _ in range(2, self.g + 1): T.append(2 * x * T[-1] - T[-2])
        return torch.einsum("bid,iod->bo", torch.stack(T[: self.g + 1], -1), self.c)


class FourierLayer(nn.Module):
    def __init__(self, i, o, gs):
        super().__init__(); self.gs = gs
        self.c = nn.Parameter(torch.randn(2, o, i, gs) / (math.sqrt(i) * math.sqrt(gs)))
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x):
        k = torch.arange(1, self.gs + 1, dtype=x.dtype, device=x.device).view(1, 1, -1)
        a = x.unsqueeze(-1) * k
        return torch.einsum("bik,oik->bo", torch.cos(a), self.c[0]) + \
               torch.einsum("bik,oik->bo", torch.sin(a), self.c[1]) + self.b


class SplineLayer(nn.Module):
    """B-spline de ordem k em grade uniforme [-1.5,1.5] + ramo residual SiLU (como em efficient-kan)."""
    def __init__(self, i, o, grid, k=3):
        super().__init__(); self.k, self.grid = k, grid
        h = 3.0 / grid
        pts = torch.arange(-k, grid + k + 1, dtype=torch.float32) * h - 1.5
        self.register_buffer("pts", pts.expand(i, -1).contiguous())
        self.base = nn.Parameter(torch.randn(o, i) / math.sqrt(i))
        self.coef = nn.Parameter(torch.randn(o, i, grid + k) * 0.1 / math.sqrt(i))

    def bases(self, x):
        g = self.pts; x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for j in range(1, self.k + 1):
            b = (x - g[:, : -(j + 1)]) / (g[:, j:-1] - g[:, : -(j + 1)]) * b[:, :, :-1] + \
                (g[:, j + 1:] - x) / (g[:, j + 1:] - g[:, 1:-j]) * b[:, :, 1:]
        return b

    def forward(self, x):
        return F.linear(F.silu(x), self.base) + torch.einsum("bik,oik->bo", self.bases(x), self.coef)


def camada(fam, i, o, p):
    if fam == "chebyshev": return ChebLayer(i, o, p)
    if fam == "fourier": return FourierLayer(i, o, p)
    return SplineLayer(i, o, p)


class TabKAN(nn.Module):
    def __init__(self, d_in, fam, prof, larg, p):
        super().__init__()
        dims = [d_in] + [larg] * prof + [1]
        self.ls = nn.ModuleList(camada(fam, a, b, p) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for l in self.ls: x = l(x)
        return x.squeeze(-1)


def l2(model): return sum((p ** 2).sum() for p in model.parameters())


def tabkan_fit_predict(Xtr, ytr, Xav_list, Xval, yval, hp, seed, max_passos=100):
    """Treina com early stopping (paciencia 10) na perda de validacao; devolve probs de cada X em Xav_list e do treino."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = TabKAN(Xtr.shape[1], hp["familia"], hp["prof"], hp["larg"], hp["base"]).to(DEVICE)
    Xt, yt, Xv, yv = [torch.tensor(a, device=DEVICE) for a in (Xtr, ytr, Xval, yval)]
    L = lambda X, y: perda(m(X), y, hp["perda"])
    best, best_state, esp = 1e9, None, 0
    if hp["otim"] == "lbfgs":
        opt = torch.optim.LBFGS(m.parameters(), lr=hp["lr"], max_iter=1, history_size=10, line_search_fn="strong_wolfe")
        def clo():
            opt.zero_grad(); l = L(Xt, yt) + hp["l2"] * l2(m); l.backward(); return l
    else:
        opt = torch.optim.Adam(m.parameters(), lr=hp["lr"]); bs = 512
    for st in range(max_passos):
        m.train()
        if hp["otim"] == "lbfgs":
            try: opt.step(clo)
            except Exception: break
        else:
            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                (L(Xt[idx], yt[idx]) + hp["l2"] * l2(m)).backward(); opt.step()
        with torch.no_grad(): lv = L(Xv, yv).item()
        if not math.isfinite(lv): break
        if lv < best - 1e-6: best, best_state, esp = lv, copy.deepcopy(m.state_dict()), 0
        else:
            esp += 1
            if esp >= MAX_PACIENCIA: break
    if best_state: m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return ([torch.sigmoid(m(torch.tensor(X, device=DEVICE))).cpu().numpy() for X in Xav_list],
                torch.sigmoid(m(Xt)).cpu().numpy())


# =====================================================================
# Espaco de busca (8 hiperparametros, conforme a pre-proposta)
# =====================================================================
def hp_tabkan(t):
    fam = t.suggest_categorical("familia", ["spline", "chebyshev", "fourier"])
    base = {"spline": lambda: t.suggest_categorical("base_spline", [1, 3, 5, 7, 9]),
            "chebyshev": lambda: t.suggest_int("base_cheb", 2, 6),
            "fourier": lambda: t.suggest_int("base_fourier", 1, 6)}[fam]()
    otim = t.suggest_categorical("otim", ["lbfgs", "adam"])
    lr = t.suggest_float("lr_lbfgs", 0.05, 1.0, log=True) if otim == "lbfgs" else t.suggest_float("lr_adam", 1e-4, 1e-1, log=True)
    return dict(familia=fam, prof=t.suggest_int("prof", 1, 4), larg=t.suggest_int("larg", 5, 50, step=5), base=base,
                otim=otim, lr=lr, l2=t.suggest_float("l2", 1e-6, 1e-2, log=True),
                perda=t.suggest_categorical("perda", ["ce", "mse"]))


TABKAN_PADRAO = dict(familia="spline", base_spline=5, otim="lbfgs", lr_lbfgs=1.0, prof=1, larg=10, l2=1e-6, perda="ce")


def fit(hp, f, Xlist, seed):
    Xt, yt, Xv, yv = f
    return tabkan_fit_predict(Xt, yt, Xlist, Xv, yv, hp, seed)


def estudo(n_trials):
    return comum.estudo("tabkan", n_trials, hp_tabkan, fit, TABKAN_PADRAO)


if __name__ == "__main__":
    n = int(sys.argv[1])
    optuna.logging.set_verbosity(optuna.logging.INFO)
    st = estudo(n)
    print("melhor KS", st.best_value, st.best_params)
