"""Treina muitas MLPs de uma vez na GPU: os pesos de K redes viram um tensor [K, entrada, saida].

Uma rede pequena sozinha e mais lenta na GPU que na CPU, mas a GPU quase nao sente a diferenca entre 1 e centenas de redes.
Cada rede tem os proprios dados, sementes, plano, taxas e parada; so precisam ter o mesmo formato, ativacao, otimizador e lote.
A referencia continua sendo modelo.py: test_pilha.py compara as duas numero a numero.

O teste so aparece em medir_pilha, depois do treino, para prever e medir. Nada do treino le o teste.
"""
import os, sys, copy, json, math, queue, zlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, evolucao as EV, blocos as BL     # modelo vem antes do torch: ele ajusta o ambiente
import numpy as np, torch, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

MAX_REDES = 512         # o ganho por rede estabiliza por volta de 128 redes
ATIVACOES = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh, "logistica": torch.sigmoid, "leaky_relu": lambda h: F.leaky_relu(h, 0.1)}
OTIMIZADORES = ("adam", "adamw", "rmsprop")
BETA1, BETA2, ALFA, EPS = 0.9, 0.999, 0.99, 1e-8        # os padroes de torch.optim
BN_MOMENTO, BN_EPS = 0.1, 1e-5                          # os padroes de nn.BatchNorm1d


def _larguras(hp):
    return tuple(max(4, hp["unidades"] // 2 ** k) if hp.get("funil", False) else hp["unidades"] for k in range(hp["camadas"]))


def chave_hp(hp):
    """O que precisa ser igual para duas redes dividirem a mesma pilha."""
    return (_larguras(hp), hp["ativ"], hp["otim"], hp.get("lote", M.LOTE))


class Pilha:
    """K redes com o mesmo formato. Os lotes tem forma [K, B, d] e a mascara [K, B] diz quais linhas sao exemplos de verdade."""

    def __init__(self, hps, d_in, sementes, disp, gerador=None):
        if len({chave_hp(h) for h in hps}) != 1:
            raise ValueError("redes de formatos diferentes nao dividem a mesma pilha")
        h0, K = hps[0], len(hps)
        if h0["otim"] not in OTIMIZADORES:
            raise ValueError(f"otimizador sem suporte na pilha: {h0['otim']}")
        self.disp, self.K, self.ativ, self.otim = disp, K, ATIVACOES[h0["ativ"]], h0["otim"]
        self.g = gerador or torch.Generator(device=disp).manual_seed(int(sementes[0]))
        larg = _larguras(h0)
        planos = [h.get("plano") or "D" * h["camadas"] for h in hps]      # sem plano, o formato antigo: Dropout em todas
        for pl in planos:
            if len(pl) != len(larg) or set(pl) - set("NBD"):
                raise ValueError(f"plano invalido para {len(larg)} camadas: {pl!r}")
        por_rede = lambda f, tipo=torch.float32: torch.tensor([f(h) for h in hps], dtype=tipo, device=disp)
        # Uma letra por camada: uma rede nunca tem BN e Dropout na mesma camada
        self.tem_bn = [torch.tensor([pl[c] == "B" for pl in planos], device=disp) for c in range(len(larg))]
        self.p_drop = [torch.tensor([h["dropout"] if pl[c] == "D" else 0.0 for h, pl in zip(hps, planos)], dtype=torch.float32, device=disp) for c in range(len(larg))]
        self.rede_tem_bn = torch.stack(self.tem_bn).any(0)
        self.lr, self.wd = por_rede(lambda h: h["lr"]), por_rede(lambda h: h["l2"])
        self.suav, self.pen = por_rede(lambda h: h.get("suav", 0.0)), por_rede(lambda h: h.get("pen_logit", 0.0))
        self.peso = por_rede(lambda h: h.get("peso_pos") or 1.0)
        self.eh_mse = por_rede(lambda h: h["perda"] != "ce", torch.bool)
        self._iniciar(d_in, larg, sementes)
        self.t = torch.zeros(K, device=disp)
        self.m, self.v = torch.zeros_like(self.theta), torch.zeros_like(self.theta)
        self._resumir_flags()

    def _iniciar(self, d_in, larg, sementes):
        """Os mesmos numeros de M.MLP com a mesma semente: Linear sorteia pesos e depois o vies em U(-1/raiz(entrada), +1/raiz(entrada)).
        Todos os parametros de uma rede moram numa linha de theta [K, P]: o otimizador faz cada conta uma vez por passo, nao uma vez por camada."""
        formatos = list(zip((d_in,) + larg, larg + (1,)))
        linhas = []
        bn_c = [c for c in range(len(larg)) if self.tem_bn[c].any()]
        for s in sementes:
            g, partes = torch.Generator().manual_seed(int(s)), []
            for e, u in formatos:
                lim = 1 / math.sqrt(e)
                partes.append(torch.empty(u, e).uniform_(-lim, lim, generator=g).T.ravel())
                partes.append(torch.empty(u).uniform_(-lim, lim, generator=g))
            for c in bn_c:
                partes += [torch.ones(larg[c]), torch.zeros(larg[c])]
            linhas.append(torch.cat(partes))
        self.theta = torch.stack(linhas).to(self.disp).requires_grad_()
        self.fatias, a = [], 0
        for e, u in formatos:
            self.fatias += [(a, a + e * u, (e, u)), (a + e * u, a + e * u + u, (1, u))]
            a += e * u + u
        self.n_pesos = len(self.fatias)
        self.bn = {}
        for c in bn_c:
            u = larg[c]
            self.fatias += [(a, a + u, (1, u)), (a + u, a + 2 * u, (1, u))]
            self.bn[c] = dict(i=len(self.fatias) - 2, media=torch.zeros(self.K, 1, u, device=self.disp), var=torch.ones(self.K, 1, u, device=self.disp))
            a += 2 * u

    def _resumir_flags(self):
        """Respostas guardadas em Python: perguntar a GPU a cada passo obrigaria a esperar por ela."""
        self.usa_drop = [bool(p.any()) for p in self.p_drop]
        self.algum_mse = bool(self.eh_mse.any())

    def _vistas(self):
        return [self.theta[:, a:b].view(self.K, *f) for a, b, f in self.fatias]

    @property
    def pesos(self):
        v = self._vistas()
        return [(v[i], v[i + 1]) for i in range(0, self.n_pesos, 2)]

    def mascara_dropout(self, c, K, B, u):
        p = self.p_drop[c][:, None, None]
        return (torch.rand(K, B, u, device=self.disp, generator=self.g) >= p) / (1 - p)

    def _bn(self, c, h, vistas, treino, mascara, ok):
        b = self.bn[c]
        if treino:
            w = mascara[:, :, None].to(h.dtype)
            n = w.sum(1, keepdim=True).clamp(min=1)
            media = (h * w).sum(1, keepdim=True) / n
            var = (((h - media) ** 2) * w).sum(1, keepdim=True) / n
            with torch.no_grad():
                # Como em nn.BatchNorm1d: normaliza com a variancia do lote, mas acumula a variancia com n - 1
                sel = (ok & self.tem_bn[c])[:, None, None]
                var_corr = var * n / (n - 1).clamp(min=1)
                b["media"] = torch.where(sel, (1 - BN_MOMENTO) * b["media"] + BN_MOMENTO * media, b["media"])
                b["var"] = torch.where(sel, (1 - BN_MOMENTO) * b["var"] + BN_MOMENTO * var_corr, b["var"])
        else:
            media, var = b["media"], b["var"]
        norm = (h - media) / torch.sqrt(var + BN_EPS) * vistas[b["i"]] + vistas[b["i"] + 1]
        return torch.where(self.tem_bn[c][:, None, None], norm, h)

    def frente(self, x, treino=False, mascara=None, ok=None):
        v, h = self._vistas(), x
        for c in range(self.n_pesos // 2 - 1):
            h = h @ v[2 * c] + v[2 * c + 1]
            if c in self.bn:
                h = self._bn(c, h, v, treino, mascara, ok)
            h = self.ativ(h)
            if treino and self.usa_drop[c]:
                h = h * self.mascara_dropout(c, h.shape[0], h.shape[1], h.shape[2])
        return (h @ v[-2 - 2 * len(self.bn)] + v[-1 - 2 * len(self.bn)]).squeeze(-1)

    def _perda_treino(self, z, y, mascara):
        """A perda de M.perda_treino, uma por rede. Somar as perdas nao mistura as redes: cada uma so depende dos proprios pesos."""
        suav, peso, pen = self.suav[:, None], self.peso[:, None], self.pen[:, None]
        alvo = y * (1 - suav) + 0.5 * suav
        e = F.binary_cross_entropy_with_logits(z, alvo, pos_weight=peso, reduction="none")
        if self.algum_mse:
            e = torch.where(self.eh_mse[:, None], (1 + (peso - 1) * y) * (torch.sigmoid(z) - alvo) ** 2, e)
        return _media(e + pen * z ** 2, mascara)

    def passo(self, x, y, mascara):
        """Um passo de otimizacao. Rede sem exemplos nao anda, e rede com BN nao anda com um exemplo so, como em M.treinar."""
        n = mascara.sum(1)
        ok = (n >= 1) & ~(self.rede_tem_bn & (n < 2))
        l = self._perda_treino(self.frente(x, True, mascara, ok), y, mascara)
        g, = torch.autograd.grad(l.sum(), self.theta)
        with torch.no_grad():
            self.t = self.t + ok
            self._atualizar(g, ok)

    def _atualizar(self, g, ok):
        """As contas de torch.optim, com lr e weight decay por rede."""
        p, lr, wd, sel = self.theta, self.lr[:, None], self.wd[:, None], ok[:, None]
        if self.otim == "rmsprop":
            g = g + wd * p
            self.v = torch.where(sel, ALFA * self.v + (1 - ALFA) * g ** 2, self.v)
            novo = p - lr * g / (self.v.sqrt() + EPS)
        else:
            if self.otim == "adam":
                g = g + wd * p
            base = p * (1 - lr * wd) if self.otim == "adamw" else p
            t = self.t.clamp(min=1)[:, None]
            self.m = torch.where(sel, BETA1 * self.m + (1 - BETA1) * g, self.m)
            self.v = torch.where(sel, BETA2 * self.v + (1 - BETA2) * g ** 2, self.v)
            novo = base - lr / (1 - BETA1 ** t) * self.m / (self.v.sqrt() / torch.sqrt(1 - BETA2 ** t) + EPS)
        p.copy_(torch.where(sel, novo, p))

    def manter(self, i):
        """Fica so com as redes em i. Rede que ja parou nao deve custar nada."""
        self.K = len(i)
        for nome in ("rede_tem_bn", "lr", "wd", "suav", "pen", "peso", "eh_mse", "t", "m", "v"):
            setattr(self, nome, getattr(self, nome)[i])
        self.tem_bn = [t[i] for t in self.tem_bn]
        self.p_drop = [t[i] for t in self.p_drop]
        self.theta = self.theta[i].detach().requires_grad_()
        self.bn = {c: {**b, "media": b["media"][i], "var": b["var"][i]} for c, b in self.bn.items()}
        self._resumir_flags()
        return self

    def recorte(self, i):
        return copy.copy(self).manter(i)


def _media(e, mascara):
    w = mascara.to(e.dtype)
    return (e * w).sum(1) / w.sum(1).clamp(min=1)


def ks_lotes(z, y, mascara):
    """comum.ks por rede, na GPU: maior distancia entre os acumulados das duas classes, olhando so o fim de cada empate."""
    z = z.masked_fill(~mascara, -math.inf)
    ordem = z.argsort(1, descending=True)
    zo, yo, mo = z.gather(1, ordem), y.gather(1, ordem).double(), mascara.gather(1, ordem).double()
    pos, neg = (yo * mo).cumsum(1), ((1 - yo) * mo).cumsum(1)
    d = pos / pos[:, -1:].clamp(min=1) - neg / neg[:, -1:].clamp(min=1)
    fim = torch.cat([zo[:, 1:] != zo[:, :-1], torch.ones_like(zo[:, :1], dtype=torch.bool)], 1)
    return d.masked_fill(~fim | (mo == 0), 0.0).max(1).values.clamp(min=0)


class _Banco:
    """Junta num tensor so os conjuntos usados pelas redes. Quando varias redes usam a mesma dobra, ela existe uma vez."""

    def __init__(self, X, y, disp):
        unicos, self.base, self.n, topo = {}, [], [], 0
        for a, b in zip(X, y):
            if id(a) not in unicos:
                unicos[id(a)] = (a, b, topo)
                topo += len(a)
            self.base.append(unicos[id(a)][2]); self.n.append(len(a))
        self.X = torch.cat([torch.as_tensor(a, device=disp) for a, _, _ in unicos.values()])
        self.y = torch.cat([torch.as_tensor(b, device=disp) for _, b, _ in unicos.values()]) if y[0] is not None else None
        self.base, self.n = torch.tensor(self.base, device=disp), torch.tensor(self.n, device=disp)

    def fixo(self, ids):
        """Todas as linhas de cada rede, com preenchimento no fim: [K, m, d], [K, m], [K, m]."""
        n = self.n[ids]
        pos = torch.arange(int(n.max()), device=self.X.device).expand(len(ids), -1)
        mascara = pos < n[:, None]
        i = self.base[ids][:, None] + torch.minimum(pos, n[:, None] - 1)
        return self.X[i], (self.y[i] if self.y is not None else None), mascara


def _lotes(banco, ids, lote, g):
    """Uma epoca de lotes. Cada rede embaralha os proprios exemplos; quem tem base menor fica sem exemplos antes das outras."""
    n = banco.n[ids]
    nmax = int(n.max())
    pos = torch.arange(nmax, device=n.device).expand(len(ids), -1)
    valido = pos < n[:, None]
    ordem = torch.rand(len(ids), nmax, device=n.device, generator=g).masked_fill(~valido, 2.0).argsort(1)
    for i in range(0, nmax, lote):
        p = ordem[:, i:i + lote]
        k = banco.base[ids][:, None] + torch.minimum(p, n[:, None] - 1)
        yield banco.X[k], banco.y[k], p < n[:, None]


def _perda_val(z, y, mascara, eh_mse):
    """A perda de M.perda: sem suavizacao, peso nem penalidade."""
    e = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    if eh_mse.any():
        e = torch.where(eh_mse[:, None], (torch.sigmoid(z) - y) ** 2, e)
    return _media(e, mascara)


def treinar_redes(redes, disp, semente):
    """redes = [dict(hp, Xa, ya, semente, Xb, yb)] com parada pela validacao, ou [dict(hp, Xa, ya, semente, epocas, Xt)] com epocas fixas.
    Devolve, para cada rede, p (em Xb, no melhor momento, ou em Xt), a melhor epoca e quantos passos deu."""
    hps = [r["hp"] for r in redes]
    g = torch.Generator(device=disp).manual_seed(int(semente) % 2 ** 31)
    pilha = Pilha(hps, redes[0]["Xa"].shape[1], [r["semente"] for r in redes], disp, g)
    lote, K = chave_hp(hps[0])[3], len(redes)
    treino = _Banco([r["Xa"] for r in redes], [r["ya"] for r in redes], disp)
    fixas = "epocas" in redes[0]
    return (_treino_fixo if fixas else _treino_com_parada)(redes, pilha, treino, lote, g, disp, K)


def _treino_fixo(redes, pilha, treino, lote, g, disp, K):
    final = _Banco([r["Xt"] for r in redes], [None] * K, disp)
    alvo = torch.tensor([int(r["epocas"]) for r in redes], device=disp)
    ids = torch.arange(K, device=disp)
    saida = [None] * K
    passos = [0] * K
    for ep in range(1, int(alvo.max()) + 1):
        for x, y, m in _lotes(treino, ids, lote, g):
            pilha.passo(x, y, m)
        fim = alvo[ids] == ep
        if fim.any():
            with torch.no_grad():
                parte = pilha.recorte(fim.nonzero().ravel())
                ids_fim = ids[fim]
                xt, _, mt = final.fixo(ids_fim)
                z = parte.frente(xt)
            p = torch.sigmoid(z).cpu().numpy()
            for j, i in enumerate(ids_fim.tolist()):
                saida[i] = np.nan_to_num(p[j][:int(final.n[i])], nan=0.5)
                passos[i] = int(parte.t[j])
            pilha.manter((~fim).nonzero().ravel())
            ids = ids[~fim]
    return [dict(p=saida[i], epoca=int(alvo[i]), passos=passos[i]) for i in range(K)]


def _treino_com_parada(redes, pilha, treino, lote, g, disp, K):
    """A regra de M.treinar, rede a rede: guarda o melhor momento, para depois de `paciencia` epocas sem melhora ou ao divergir."""
    val = _Banco([r["Xb"] for r in redes], [r["yb"] for r in redes], disp)
    hps = [r["hp"] for r in redes]
    paciencia = torch.tensor([h.get("paciencia", M.PACIENCIA) for h in hps], device=disp)
    pelo_ks = torch.tensor([h.get("parada") == "ks" for h in hps], device=disp)
    ids = torch.arange(K, device=disp)
    xv, yv, mv = val.fixo(ids)
    melhor = torch.full((K,), math.inf, dtype=torch.float64, device=disp)
    melhor_ep, esp, passos = torch.zeros(K, dtype=torch.long, device=disp), torch.zeros(K, dtype=torch.long, device=disp), torch.zeros(K, dtype=torch.long, device=disp)
    z_melhor = torch.zeros(K, yv.shape[1], device=disp)     # sem momento bom, logit 0 vira probabilidade 0,5
    vivo = torch.ones(K, dtype=torch.bool, device=disp)      # alinhado com ids: quem parou espera a proxima compactacao
    for ep in range(1, M.MAX_EPOCAS + 1):
        for x, y, m in _lotes(treino, ids, lote, g):
            pilha.passo(x, y, m)
        with torch.no_grad():
            z = pilha.frente(xv)
            l = _perda_val(z, yv, mv, pilha.eh_mse).double()
            finito = l.isfinite()
            crit = l
            if pelo_ks[ids].any():
                crit = torch.where(pelo_ks[ids] & finito, -ks_lotes(torch.sigmoid(z), yv, mv), l)
            melhorou = vivo & finito & (crit < melhor[ids] - 1e-6)
            i_m = ids[melhorou]
            melhor[i_m], melhor_ep[i_m], esp[i_m] = crit[melhorou], ep, 0
            z_melhor[i_m, :z.shape[1]] = z[melhorou]
            parados = vivo & ~melhorou
            esp[ids[parados]] += 1
            para = vivo & (~finito | (esp[ids] >= paciencia[ids]))
            passos[ids[vivo]] = pilha.t[vivo].long()
            vivo = vivo & ~para
        n_vivos = int(vivo.sum())
        if n_vivos == 0:
            break
        if (len(ids) - n_vivos) * 10 >= len(ids):     # compacta sem refazer a validacao a cada rede que para
            pilha.manter(vivo.nonzero().ravel())
            ids = ids[vivo]
            xv, yv, mv = val.fixo(ids)
            vivo = torch.ones(len(ids), dtype=torch.bool, device=disp)
    p = torch.sigmoid(z_melhor).cpu().numpy()
    return [dict(p=np.nan_to_num(p[i][:int(val.n[i])], nan=0.5), epoca=int(melhor_ep[i]), passos=int(passos[i])) for i in range(K)]


def agrupar(redes, max_redes=MAX_REDES):
    """Indices de `redes` divididos em pilhas: mesma chave e no maximo max_redes em cada uma."""
    grupos = {}
    for i, r in enumerate(redes):
        grupos.setdefault((r["Xa"].shape[1],) + chave_hp(r["hp"]), []).append(i)
    return [g[j:j + max_redes] for g in grupos.values() for j in range(0, len(g), max_redes)]


def dispositivos_disponiveis():
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]


def _executar(redes, dispositivos, max_redes):
    """Cada dispositivo pega a proxima pilha da fila. O PyTorch solta o GIL durante as contas, entao threads bastam e os dados ficam compartilhados."""
    fila, saida = queue.Queue(), [None] * len(redes)
    for g in sorted(agrupar(redes, max_redes), key=len, reverse=True):
        fila.put(g)

    def trabalhador(disp):
        while True:
            try:
                g = fila.get_nowait()
            except queue.Empty:
                return
            parte = [redes[i] for i in g]
            # A semente da pilha sai da composicao dela: a mesma pilha repete o mesmo resultado em qualquer ordem de execucao
            s = zlib.crc32(json.dumps([[r["semente"], r["hp"], len(r["Xa"])] for r in parte], sort_keys=True, default=str).encode())
            for i, r in zip(g, treinar_redes(parte, disp, s)):
                saida[i] = r
            if str(disp).startswith("cuda"):
                torch.cuda.empty_cache()

    if len(dispositivos) == 1:
        trabalhador(dispositivos[0])
    else:
        with ThreadPoolExecutor(len(dispositivos)) as ex:
            for f in [ex.submit(trabalhador, d) for d in dispositivos]:
                f.result()
    return saida


def medir_pilha(tarefas, dados, dispositivos=None, max_redes=MAX_REDES):
    """tarefas = [(c, rep, semente)], as mesmas de fase3.medir e com o mesmo significado. As dobras sao preparadas uma vez so:
    na grade, todas as configuracoes usam os mesmos dados. A preparacao ajusta scaler e oversampling so no treino de cada dobra."""
    Xtr, Xte, ytr, yte = dados
    dispositivos = dispositivos or dispositivos_disponiveis()
    y_t = yte.values.astype("float32")
    dobras, preparados = {}, {}

    def divisao(rep):
        if rep not in dobras:
            dobras[rep] = list(StratifiedKFold(5, shuffle=True, random_state=comum.SEMENTE + rep).split(Xtr, ytr))
        return dobras[rep]

    def preparar(c, rep, i):
        ch = (json.dumps([c["dados"], c["extras"]], sort_keys=True), rep, i)
        if ch not in preparados:
            k = BL._kwargs(c)
            if i is None:
                preparados[ch] = EV._prep(Xtr, ytr, Xte, **k)
            else:
                a, b = divisao(rep)[i]
                Xa, ya, Xb = EV._prep(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], **k)
                preparados[ch] = (Xa, ya, Xb, ytr.iloc[b].values.astype("float32"))
        return preparados[ch]

    redes = []
    for c, rep, sem in tarefas:
        s0 = 42 + 1000 * rep + 100000 * sem
        for i in range(5):
            Xa, ya, Xb, yb = preparar(c, rep, i)
            redes.append(dict(hp=c["hp"], Xa=Xa, ya=ya, Xb=Xb, yb=yb, semente=s0 + i))
    res = _executar(redes, dispositivos, max_redes)
    kss, eps = [], []
    for j in range(len(tarefas)):
        kss.append(float(np.mean([comum.ks(redes[j * 5 + i]["yb"], res[j * 5 + i]["p"]) for i in range(5)])))
        eps.append([res[j * 5 + i]["epoca"] for i in range(5)])

    finais = []
    for (c, rep, sem), e in zip(tarefas, eps):
        Xa, ya, Xt = preparar(c, rep, None)[:3]       # o teste entra aqui, so para prever depois do treino
        finais.append(dict(hp=c["hp"], Xa=Xa, ya=ya, Xt=Xt, epocas=max(1, int(np.median(e))), semente=42 + 1000 * rep + 100000 * sem))
    res_f = _executar(finais, dispositivos, max_redes)
    return [dict(val=v, teste=comum.ks(y_t, r["p"]), epocas=float(np.median(e)), prob=r["p"]) for v, e, r in zip(kss, eps, res_f)]
