"""FT-Transformer (Gorishniy et al., 2021, "Revisiting Deep Learning Models for Tabular Data"): modelo e treino.

Cada coluna vira um token: token_j = x_j * W_j + b_j (Feature Tokenizer). Um token [CLS] aprendido entra no fim da
sequencia, os blocos Transformer (pre-norm, atencao multi-cabeca + FFN ReGLU) misturam os tokens, e a cabeca le so o
[CLS]: LayerNorm -> ReLU -> Linear -> logit.

Todas as colunas de comum.preparar entram como tokens numericos. Para uma coluna binaria (0/1), x*W + b e o mesmo que
uma embedding de duas posicoes; as dummies de Contract, InternetService e PaymentMethod viram um token por categoria.

Padrao (PADRAO) = o "default FT-Transformer" do artigo. Parada e epocas seguem os slides (ate 10.000 epocas).
"""
import sys, os, copy, math
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum

torch.set_num_threads(1)
DEVICE = os.environ.get("CHURN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCAS = int(os.environ.get("CHURN_MAX_EPOCAS", 10000))

PADRAO = dict(
    n_blocos=3, d_token=192, n_cabecas=8,                   # arquitetura
    drop_atencao=0.2, drop_ffn=0.1, drop_residual=0.0,      # regularizacao
    ffn_fator=4 / 3,                                        # FFN oculta = d_token * fator (ReGLU)
    lr=1e-4, wd=1e-5, lote=256,                             # AdamW, sem weight decay no tokenizador, normas e vieses
    paciencia=16, perda="ce",                               # parada na perda da separacao interna (artigo: 16)
)
_ARQ = ("n_blocos", "d_token", "n_cabecas", "drop_atencao", "drop_ffn", "drop_residual", "ffn_fator")


def arquitetura(hp):
    return {k: hp[k] for k in _ARQ}


class Tokenizador(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        s = 1 / math.sqrt(d_token)
        self.peso = nn.Parameter(torch.empty(n_features, d_token).uniform_(-s, s))
        self.vies = nn.Parameter(torch.empty(n_features, d_token).uniform_(-s, s))

    def forward(self, x):                       # (lote, n) -> (lote, n, d)
        return x.unsqueeze(-1) * self.peso + self.vies


class Bloco(nn.Module):
    def __init__(self, d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator, primeiro):
        super().__init__()
        h = int(d_token * ffn_fator)
        self.norm_atencao = None if primeiro else nn.LayerNorm(d_token)    # artigo: o 1o bloco nao normaliza os tokens
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
        z = self.ffn_saida(self.drop_ffn(a * F.relu(b)))                  # ReGLU
        return x + self.drop_residual(z)


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_blocos, d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator):
        super().__init__()
        if d_token % n_cabecas:
            raise ValueError(f"d_token ({d_token}) precisa ser multiplo de n_cabecas ({n_cabecas})")
        self.tokenizador = Tokenizador(n_features, d_token)
        s = 1 / math.sqrt(d_token)
        self.cls = nn.Parameter(torch.empty(d_token).uniform_(-s, s))
        self.blocos = nn.ModuleList(Bloco(d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator, primeiro=(k == 0))
                                    for k in range(n_blocos))
        self.norm_final = nn.LayerNorm(d_token)
        self.saida = nn.Linear(d_token, 1)

    def tokens(self, x):
        t = self.tokenizador(x)
        return torch.cat([t, self.cls.expand(len(x), 1, -1)], dim=1)      # [CLS] no fim, como no artigo

    def forward(self, x):
        t = self.tokens(x)
        for b in self.blocos:
            t = b(t)
        return self.saida(F.relu(self.norm_final(t[:, -1]))).squeeze(-1)


def grupos_parametros(m, wd):
    """Artigo: sem weight decay no tokenizador, no [CLS], nas LayerNorms e nos vieses."""
    sem = lambda n: "tokenizador" in n or "norm" in n or n.endswith("bias") or n == "cls"
    return (dict(params=[p for n, p in m.named_parameters() if not sem(n)], weight_decay=wd),
            dict(params=[p for n, p in m.named_parameters() if sem(n)], weight_decay=0.0))


def perda(z, y, tipo):
    """z = logit. CE = entropia cruzada binaria; MSE = erro quadratico da probabilidade (Brier)."""
    return F.binary_cross_entropy_with_logits(z, y) if tipo == "ce" else F.mse_loss(torch.sigmoid(z), y)


def _logits(m, X, lote=2048):
    return torch.cat([m(X[i:i + lote]) for i in range(0, len(X), lote)])


def treinar(Xtr, ytr, hp, semente, Xval=None, yval=None, epocas=None, ao_fim_epoca=None, dispositivo=None):
    """Com (Xval, yval): para quando a perda nelas nao melhora por `paciencia` epocas e volta ao melhor estado.
    Com `epocas`: treina esse numero fixo de epocas (o retreino no treino inteiro da dobra). Devolve (modelo, melhor epoca)."""
    dev = torch.device(dispositivo or DEVICE)
    torch.manual_seed(semente); np.random.seed(semente)
    m = FTTransformer(Xtr.shape[1], **arquitetura(hp)).to(dev)
    opt = torch.optim.AdamW(grupos_parametros(m, hp["wd"]), lr=hp["lr"])
    Xt, yt = torch.as_tensor(Xtr, device=dev), torch.as_tensor(ytr, device=dev)
    if Xval is not None:
        Xv, yv = torch.as_tensor(Xval, device=dev), torch.as_tensor(yval, device=dev)
    lote = hp.get("lote", 256)
    best, best_state, melhor_ep, esp = math.inf, None, 0, 0
    for ep in range(1, (epocas or MAX_EPOCAS) + 1):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), lote):
            idx = perm[i:i + lote]
            opt.zero_grad(set_to_none=True)
            perda(m(Xt[idx]), yt[idx], hp.get("perda", "ce")).backward()
            opt.step()
        if ao_fim_epoca:
            m.eval(); ao_fim_epoca(ep, m)
        if epocas is not None:
            continue
        m.eval()
        with torch.no_grad():
            lv = perda(_logits(m, Xv), yv, hp.get("perda", "ce")).item()
        if not math.isfinite(lv):
            break
        if lv < best - 1e-6:
            best, best_state, melhor_ep, esp = lv, copy.deepcopy(m.state_dict()), ep, 0
        else:
            esp += 1
            if esp >= hp["paciencia"]:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m.eval(), (melhor_ep if epocas is None else epocas)


def prever(m, X):
    dev = next(m.parameters()).device
    with torch.no_grad():
        p = torch.sigmoid(_logits(m.eval(), torch.as_tensor(X, device=dev))).cpu().numpy()
    return np.nan_to_num(p, nan=0.5)          # configuracao divergente vira KS ruim, nao derruba a rodada
