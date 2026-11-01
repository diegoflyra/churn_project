"""STab e TabKAN (Jefferson) - modelos, treino por dobra e estudo Optuna.

Uso:  python modelos_jeff.py stab 40      # estudo com 40 trials
      python modelos_jeff.py tabkan 40
Salva resultados/<modelo>.db (SQLite do Optuna, retomavel).
"""
import sys, os, time, copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import optuna
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


def sig(z):
    return torch.sigmoid(z)


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
# Espacos de busca (8 hiperparametros por modelo, conforme a pre-proposta)
# =====================================================================
def hp_stab(t):
    return dict(prof=t.suggest_int("prof", 1, 4), larg=t.suggest_categorical("larg", LARGURAS_STAB),
                ativ=t.suggest_categorical("ativ", ["logistica", "tanh", "relu"]),
                otim=t.suggest_categorical("otim", ["adam", "sgd"]),
                lr=t.suggest_float("lr", 1e-4, 1e-1, log=True),
                p=t.suggest_float("p", 0.5, 0.95), wd=t.suggest_float("wd", 1e-6, 1e-2, log=True),
                perda=t.suggest_categorical("perda", ["ce", "mse"]))


STAB_PADRAO = dict(prof=4, larg=LARG_PADRAO_STAB, ativ="relu", otim="adam", lr=1e-3, p=0.9, wd=1e-5, perda="ce")


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


def objetivo(modelo, folds):
    def f(trial):
        hp = (hp_stab if modelo == "stab" else hp_tabkan)(trial)
        t0 = time.time(); kss = []
        for i, (Xt, yt, Xv, yv) in enumerate(folds):
            if modelo == "stab":
                (pv,), _ = stab_fit_predict(Xt, yt, [Xv], hp, seed=42 + i)
            else:
                (pv,), _ = tabkan_fit_predict(Xt, yt, [Xv], Xv, yv, hp, seed=42 + i)
            kss.append(comum.ks(yv, pv))
            trial.report(float(np.mean(kss)), i)
            if i == 1 and trial.should_prune():       # depois da segunda dobra
                raise optuna.TrialPruned()
        trial.set_user_attr("ks_dobras", [float(k) for k in kss])
        trial.set_user_attr("tempo_s", time.time() - t0)
        return float(np.mean(kss))
    return f


def estudo(modelo, n_trials, pasta="resultados"):
    os.makedirs(pasta, exist_ok=True)
    folds, _, _ = comum.carregar_dobras()
    st = optuna.create_study(study_name=modelo, storage=f"sqlite:///{pasta}/{modelo}.db", load_if_exists=True,
                             direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    if len(st.trials) == 0:
        st.enqueue_trial(STAB_PADRAO if modelo == "stab" else TABKAN_PADRAO)     # trial 0 = padrao
    restantes = n_trials - len([t for t in st.trials if t.state.is_finished()])
    t0 = time.time()
    if restantes > 0:
        st.optimize(objetivo(modelo, folds), n_trials=restantes, show_progress_bar=False)
    st.set_user_attr("tempo_parede_s", st.user_attrs.get("tempo_parede_s", 0) + time.time() - t0)
    return st


if __name__ == "__main__":
    modelo, n = sys.argv[1], int(sys.argv[2])
    optuna.logging.set_verbosity(optuna.logging.INFO)
    st = estudo(modelo, n)
    print("melhor KS", st.best_value, st.best_params)
