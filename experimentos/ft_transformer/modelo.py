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
)   # chaves opcionais, fora do PADRAO para o trial 0 manter a chave:
#   parada = "ks" (fase 3): o criterio da parada e o KS da separacao interna
#   tokens = "cat" (fase 5): um token por variavel categorica (as dummies de Contract viram 1 token, nao 3)
#   num_emb = "plr", plr_k, plr_sigma (fase 5): embeddings periodicos nas colunas numericas (Gorishniy et al., 2022)
#   agenda = "cos", aquec, horizonte (fase 5): lr sobe linear por `aquec` epocas e cai em cosseno ate `horizonte`
_ARQ = ("n_blocos", "d_token", "n_cabecas", "drop_atencao", "drop_ffn", "drop_residual", "ffn_fator")

# Variaveis categoricas com mais de duas categorias: em comum.preparar cada uma vira um grupo de dummies (uma ativa por linha)
CATEGORICAS = {
    "InternetService": ("InternetService_DSL", "InternetService_Fiber optic", "InternetService_No"),
    "Contract": ("Contract_Month-to-month", "Contract_One year", "Contract_Two year"),
    "PaymentMethod": ("PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
                      "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"),
}


def arquitetura(hp):
    return {k: hp[k] for k in _ARQ}


def estrutura(hp, colunas=None, X=None):
    """Como as colunas viram tokens: (grupos, numericas). grupos = lista de listas de indices de colunas (um token por
    grupo); numericas = indices que recebem o embedding periodico. Sem as chaves da fase 5: cada coluna e um token."""
    n = X.shape[1]
    grupos = [[j] for j in range(n)]
    if hp.get("tokens") == "cat":
        if colunas is None:
            raise ValueError('tokens="cat" precisa dos nomes das colunas')
        pos = {c: j for j, c in enumerate(colunas)}
        dentro = {pos[c]: nome for nome, cs in CATEGORICAS.items() for c in cs if c in pos}
        grupos, vistos = [], {}
        for j in range(n):
            if j not in dentro:
                grupos.append([j])
            elif dentro[j] in vistos:
                grupos[vistos[dentro[j]]].append(j)
            else:
                vistos[dentro[j]] = len(grupos); grupos.append([j])
    numericas = []
    if hp.get("num_emb") == "plr":
        X = np.asarray(X)
        numericas = [j for j in range(n) if len(np.unique(X[:, j])) > 2]      # binarias continuam com x*W + b
    return grupos, numericas


class Tokenizador(nn.Module):
    """token_g = soma_{j em g} x_j * W_j + b_g. Com cada coluna no seu grupo, e o Feature Tokenizer do artigo. Um grupo
    de dummies (uma ativa por linha) vira uma embedding da categoria. Colunas em `numericas` trocam x_j * W_j pelo
    embedding periodico ReLU(Linear([sin(2 pi c x), cos(2 pi c x)])), com as frequencias c aprendidas."""

    def __init__(self, n_features, d_token, grupos=None, numericas=(), plr_k=16, plr_sigma=1.0):
        super().__init__()
        s = 1 / math.sqrt(d_token)
        grupos = grupos or [[j] for j in range(n_features)]
        self.peso = nn.Parameter(torch.empty(n_features, d_token).uniform_(-s, s))
        self.vies = nn.Parameter(torch.empty(len(grupos), d_token).uniform_(-s, s))
        if any(len(g) > 1 for g in grupos):
            A = torch.zeros(len(grupos), n_features)
            for i, g in enumerate(grupos):
                A[i, g] = 1.0
            self.register_buffer("agrupa", A)
        else:
            self.agrupa = None
        self.numericas = list(numericas)
        if self.numericas:
            k = len(self.numericas)
            self.register_buffer("idx_num", torch.as_tensor(self.numericas))
            self.freq = nn.Parameter(torch.randn(k, plr_k) * plr_sigma)
            self.plr_peso = nn.Parameter(torch.empty(k, 2 * plr_k, d_token).uniform_(-1 / math.sqrt(2 * plr_k), 1 / math.sqrt(2 * plr_k)))

    def forward(self, x):                       # (lote, n) -> (lote, n_tokens, d)
        t = x.unsqueeze(-1) * self.peso
        if self.numericas:
            v = 2 * math.pi * x[:, self.idx_num].unsqueeze(-1) * self.freq                # (lote, k, plr_k)
            e = F.relu(torch.einsum("bkf,kfd->bkd", torch.cat([torch.sin(v), torch.cos(v)], -1), self.plr_peso))
            t = t.index_copy(1, self.idx_num, e)
        if self.agrupa is not None:
            t = torch.einsum("gf,bfd->bgd", self.agrupa, t)
        return t + self.vies


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
    def __init__(self, n_features, n_blocos, d_token, n_cabecas, drop_atencao, drop_ffn, drop_residual, ffn_fator,
                 grupos=None, numericas=(), plr_k=16, plr_sigma=1.0):
        super().__init__()
        if d_token % n_cabecas:
            raise ValueError(f"d_token ({d_token}) precisa ser multiplo de n_cabecas ({n_cabecas})")
        self.tokenizador = Tokenizador(n_features, d_token, grupos, numericas, plr_k, plr_sigma)
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


def penalidade(m, hp):
    """L1 e L2 classicos, somados a perda do treino: l1 * soma|w| + l2 * soma w^2, nos mesmos pesos que recebem weight
    decay (fora: tokenizador, [CLS], normas e vieses). Opcionais: sem `l1`/`l2` no hp, vale zero e o trial 0 nao muda.
    Diferente do weight decay do AdamW, que e desacoplado e encolhe lr * wd por passo (quase nada com lr 1e-4)."""
    l1, l2 = hp.get("l1", 0.0), hp.get("l2", 0.0)
    if not (l1 or l2):
        return torch.zeros(())
    ps = grupos_parametros(m, 0.0)[0]["params"]
    return sum(l1 * p.abs().sum() + l2 * (p ** 2).sum() for p in ps)


def perda(z, y, tipo):
    """z = logit. CE = entropia cruzada binaria; MSE = erro quadratico da probabilidade (Brier)."""
    return F.binary_cross_entropy_with_logits(z, y) if tipo == "ce" else F.mse_loss(torch.sigmoid(z), y)


def _logits(m, X, lote=2048):
    return torch.cat([m(X[i:i + lote]) for i in range(0, len(X), lote)])


def fator_lr(hp, ep):
    """Multiplica o lr na epoca `ep` (1, 2, ...). Sem agenda: 1. Com agenda "cos": sobe linear por `aquec` epocas e cai
    em cosseno ate 5% do lr na epoca `horizonte`, onde fica. O retreino por N epocas repete exatamente a mesma agenda."""
    if hp.get("agenda") != "cos":
        return 1.0
    aq, hz = hp.get("aquec", 5), hp.get("horizonte", 60)
    if ep <= aq:
        return ep / aq
    prog = min(1.0, (ep - aq) / max(1, hz - aq))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))


def treinar(Xtr, ytr, hp, semente, Xval=None, yval=None, epocas=None, ao_fim_epoca=None, dispositivo=None, colunas=None):
    """Com (Xval, yval): para quando a perda nelas nao melhora por `paciencia` epocas e volta ao melhor estado. Com
    hp["parada"] == "ks", o criterio e o KS nelas (o maior), como na parada pelo KS do MLP. Com `epocas`: treina esse
    numero fixo de epocas (o retreino no treino inteiro da dobra). `colunas` (nomes) so e preciso com tokens="cat".
    Devolve (modelo, melhor epoca)."""
    dev = torch.device(dispositivo or DEVICE)
    torch.manual_seed(semente); np.random.seed(semente)
    grupos, numericas = estrutura(hp, colunas, Xtr)
    extra = dict(grupos=grupos, numericas=numericas, plr_k=hp.get("plr_k", 16), plr_sigma=hp.get("plr_sigma", 1.0))
    m = FTTransformer(Xtr.shape[1], **arquitetura(hp), **extra).to(dev)
    opt = torch.optim.AdamW(grupos_parametros(m, hp["wd"]), lr=hp["lr"])
    Xt, yt = torch.as_tensor(Xtr, device=dev), torch.as_tensor(ytr, device=dev)
    if Xval is not None:
        Xv, yv = torch.as_tensor(Xval, device=dev), torch.as_tensor(yval, device=dev)
    lote = hp.get("lote", 256)
    best, best_state, melhor_ep, esp = math.inf, None, 0, 0
    for ep in range(1, (epocas or MAX_EPOCAS) + 1):
        if hp.get("agenda"):
            for g in opt.param_groups:
                g["lr"] = hp["lr"] * fator_lr(hp, ep)
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), lote):
            idx = perm[i:i + lote]
            opt.zero_grad(set_to_none=True)
            (perda(m(Xt[idx]), yt[idx], hp.get("perda", "ce")) + penalidade(m, hp)).backward()   # a parada ve so a perda
            opt.step()
        if ao_fim_epoca:
            m.eval(); ao_fim_epoca(ep, m)
        if epocas is not None:
            continue
        m.eval()
        with torch.no_grad():
            zv = _logits(m, Xv)
            lv = perda(zv, yv, hp.get("perda", "ce")).item()
        if not math.isfinite(lv):
            break
        if hp.get("parada") == "ks":
            lv = -comum.ks(np.asarray(yval), torch.sigmoid(zv).cpu().numpy())
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
