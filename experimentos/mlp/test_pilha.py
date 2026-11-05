"""O treinador em pilha tem de ser a mesma rede que modelo.py treina, so que varias ao mesmo tempo.

Com lote >= n, cada passo usa todos os exemplos e o sorteio deixa de importar: o treino inteiro, parada incluida, pode ser
comparado numero a numero com M.treinar. Com lote menor, a comparacao exata e feita passo a passo, com os mesmos lotes.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/mlp/test_pilha.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")     # a referencia roda na CPU; a pilha roda nos dois
sys.path.insert(0, os.path.dirname(__file__))
import comum, modelo as M, fase3 as F3, pilha as P     # modelo vem antes do torch: ele ajusta o ambiente
import numpy as np, pandas as pd, pytest, torch

DISPOSITIVOS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
HP = dict(camadas=2, unidades=8, ativ="relu", otim="adam", lr=1e-2, dropout=0.0, l2=1e-4, perda="ce", paciencia=5, plano="NN")


def dados(n=120, d=6, semente=0):
    r = np.random.default_rng(semente)
    X = r.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] + X[:, 1] * X[:, 2] + r.normal(size=n) > 0).astype("float32")
    return X, y


def copiar(pilha, k, m):
    """Poe em pilha a rede k os pesos da rede de referencia m."""
    lineares = [c for c in m.rede.children() if isinstance(c, torch.nn.Linear)]
    with torch.no_grad():
        for (W, b), l in zip(pilha.pesos, lineares):
            W[k] = l.weight.T.to(W.device); b[k] = l.bias[None].to(b.device)


def referencia(hp, d, semente):
    torch.manual_seed(semente)
    return M.MLP(d, hp["camadas"], hp["unidades"], hp["ativ"], hp["dropout"], hp.get("plano"), hp.get("funil", False))


def perto(a, b, tol=2e-4):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    assert np.abs(a - b).max() < tol, f"diferenca maxima {np.abs(a - b).max()}"


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_inicializacao_igual_a_da_referencia(disp):
    hp = {**HP, "funil": True, "unidades": 16, "plano": "BD"}
    pilha = P.Pilha([hp, hp], 6, [7, 8], disp)
    for k, s in enumerate((7, 8)):
        lineares = [c for c in referencia(hp, 6, s).rede.children() if isinstance(c, torch.nn.Linear)]
        for (W, b), l in zip(pilha.pesos, lineares):
            perto(W[k].detach().cpu(), l.weight.detach().T, 1e-6)
            perto(b[k].detach().cpu().ravel(), l.bias.detach(), 1e-6)


VARIACOES = {
    "relu": {}, "gelu": dict(ativ="gelu"), "leaky": dict(ativ="leaky_relu"), "tanh": dict(ativ="tanh"),
    "bn": dict(plano="BN"), "bn_nas_duas_gelu": dict(plano="BB", ativ="gelu"),
    "adamw": dict(otim="adamw", l2=1e-2), "rmsprop": dict(otim="rmsprop"),
    "mse": dict(perda="mse"), "suavizacao": dict(suav=0.1), "peso": dict(peso_pos=2.77), "pen_logit": dict(pen_logit=0.01),
    "mse_com_tudo": dict(perda="mse", suav=0.1, peso_pos=2.0, pen_logit=0.01), "funil": dict(funil=True, unidades=16),
}


@pytest.mark.parametrize("disp", DISPOSITIVOS)
@pytest.mark.parametrize("nome", VARIACOES)
def test_passos_iguais_aos_da_referencia(nome, disp):
    """Mesmos pesos e mesmos lotes: pesos e saidas seguem iguais depois de 30 passos, BN em modo de avaliacao incluso."""
    hp = {**HP, **VARIACOES[nome]}
    X, y = dados()
    m = referencia(hp, 6, 3)
    opt = M.OTIM[hp["otim"]](m.parameters(), lr=hp["lr"], weight_decay=hp["l2"])
    pilha = P.Pilha([hp, hp], 6, [3, 4], disp)        # a segunda rede prova que uma nao interfere na outra
    copiar(pilha, 0, m)
    r = np.random.default_rng(1)
    for _ in range(30):
        i = r.choice(len(X), 32, replace=False)
        m.train(); opt.zero_grad()
        M.perda_treino(m(torch.tensor(X[i])), torch.tensor(y[i]), hp).backward(); opt.step()
        xk = torch.tensor(X[i], device=disp).expand(2, -1, -1)
        yk = torch.tensor(y[i], device=disp).expand(2, -1)
        pilha.passo(xk, yk, torch.ones(2, 32, dtype=torch.bool, device=disp))
    m.eval()
    with torch.no_grad():
        ref = m(torch.tensor(X)).numpy()
        z = pilha.frente(torch.tensor(X, device=disp).expand(2, -1, -1)).cpu().numpy()
    perto(z[0], ref)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_otimizador_desconhecido_e_recusado(disp):
    with pytest.raises(ValueError):
        P.Pilha([{**HP, "otim": "sgd"}], 6, [1], disp)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_rede_sem_lote_nao_se_mexe(disp):
    """Bases de tamanhos diferentes acabam a epoca em passos diferentes: quem ficou sem exemplos nao pode andar pelo momento do Adam."""
    X, y = dados()
    pilha = P.Pilha([HP, HP], 6, [3, 4], disp)
    x, yy = torch.tensor(X[:32], device=disp).expand(2, -1, -1), torch.tensor(y[:32], device=disp).expand(2, -1)
    cheia = torch.ones(2, 32, dtype=torch.bool, device=disp)
    pilha.passo(x, yy, cheia)
    antes = [W.detach().clone() for W, _ in pilha.pesos]
    vazia = cheia.clone(); vazia[1] = False
    pilha.passo(x, yy, vazia)
    for (W, _), a in zip(pilha.pesos, antes):
        assert torch.equal(W[1], a[1])
        assert not torch.equal(W[0], a[0])


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_bn_nao_treina_com_um_exemplo_e_ignora_preenchimento(disp):
    """Como em M.treinar, lote de 1 com BN e pulado. Linhas de preenchimento nao entram nas estatisticas do BN."""
    hp = {**HP, "plano": "BN"}
    X, y = dados()
    m = referencia(hp, 6, 3)
    opt = M.OTIM["adam"](m.parameters(), lr=hp["lr"], weight_decay=hp["l2"])
    m.train(); opt.zero_grad()
    M.perda_treino(m(torch.tensor(X[:10])), torch.tensor(y[:10]), hp).backward(); opt.step()
    pilha = P.Pilha([hp], 6, [3], disp)
    copiar(pilha, 0, m if False else referencia(hp, 6, 3))
    x = torch.tensor(X[:32], device=disp)[None].clone(); x[0, 10:] = 1e6     # lixo no preenchimento
    yk = torch.tensor(y[:32], device=disp)[None]
    masc = torch.zeros(1, 32, dtype=torch.bool, device=disp); masc[0, :10] = True
    pilha.passo(x, yk, masc)
    m.eval()
    with torch.no_grad():
        perto(pilha.frente(torch.tensor(X, device=disp)[None])[0].cpu(), m(torch.tensor(X)))
    antes = pilha.pesos[0][0].detach().clone()
    um = torch.zeros(1, 32, dtype=torch.bool, device=disp); um[0, 0] = True
    pilha.passo(x, yk, um)
    assert torch.equal(pilha.pesos[0][0], antes)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_rede_divergente_nao_contamina_as_outras(disp):
    X, y = dados()
    ruim = {**HP, "lr": 1e30}
    a, b = P.Pilha([HP, HP], 6, [3, 4], disp), P.Pilha([ruim, HP], 6, [3, 4], disp)
    xk, yk = torch.tensor(X[:32], device=disp).expand(2, -1, -1), torch.tensor(y[:32], device=disp).expand(2, -1)
    masc = torch.ones(2, 32, dtype=torch.bool, device=disp)
    for _ in range(20):
        a.passo(xk, yk, masc); b.passo(xk, yk, masc)
    with torch.no_grad():
        perto(a.frente(xk)[1].cpu(), b.frente(xk)[1].cpu(), 1e-6)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_plano_nunca_tem_bn_e_dropout_na_mesma_camada(disp):
    hps = [{**HP, "camadas": len(pl), "plano": pl, "dropout": .2} for pl in F3.planos((3,))]
    pilha = P.Pilha(hps, 6, list(range(len(hps))), disp)
    for k, hp in enumerate(hps):
        for c, letra in enumerate(hp["plano"]):
            assert bool(pilha.tem_bn[c][k]) == (letra == "B")
            assert (float(pilha.p_drop[c][k]) > 0) == (letra == "D")
            assert not (pilha.tem_bn[c][k] and pilha.p_drop[c][k] > 0)
    with pytest.raises(ValueError):
        P.Pilha([{**HP, "plano": "BX"}], 6, [1], disp)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_dropout_zera_na_taxa_certa_e_preserva_a_escala(disp):
    hps = [{**HP, "camadas": 1, "unidades": 400, "plano": "D", "dropout": t, "ativ": "tanh"} for t in (.1, .5)]
    torch.manual_seed(0)
    pilha = P.Pilha(hps, 6, [1, 2], disp)
    g = pilha.mascara_dropout(0, 2, 500, 400)
    perto((g == 0).float().mean(dim=(1, 2)).cpu(), [.1, .5], .01)
    perto(g.mean(dim=(1, 2)).cpu(), [1, 1], .02)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_ks_em_lote_igual_ao_comum(disp):
    """Inclui empates e preenchimento com lixo: o KS de cada rede so pode ver as linhas dela."""
    r = np.random.default_rng(0)
    z, y, n = [], [], [50, 37, 50]
    for ni in n:
        yi = (r.random(50) < .3).astype("float32"); yi[0], yi[1] = 0, 1
        z.append(np.round(r.normal(size=50) + yi, 1).astype("float32")); y.append(yi)
    z, y = np.array(z), np.array(y)
    masc = np.array([[j < ni for j in range(50)] for ni in n])
    z[~masc] = 99.0
    got = P.ks_lotes(torch.tensor(z, device=disp), torch.tensor(y, device=disp), torch.tensor(masc, device=disp)).cpu().numpy()
    perto(got, [comum.ks(y[i, :n[i]], z[i, :n[i]]) for i in range(3)], 1e-6)


def _rede(hp, n=120, m=40, sem=1, **k):
    Xa, ya = dados(n, semente=sem)
    Xb, yb = dados(m, semente=sem + 100)
    return dict(hp=hp, Xa=Xa, ya=ya, Xb=Xb, yb=yb, semente=sem, **k)


TREINOS = {
    "padrao": {}, "paciencia": dict(paciencia=2), "paciencia_longa": dict(paciencia=30), "ks": dict(parada="ks"), "mse": dict(perda="mse"),
    "bn": dict(plano="BN"), "suav": dict(suav=.1), "lr_alto": dict(lr=.1), "lr_baixo": dict(lr=1e-3), "l2": dict(l2=.01),
}


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_treino_inteiro_com_parada_igual_ao_da_referencia(disp):
    """Lote maior que a base: sem sorteio. Redes com paciencia, perda, parada, regularizacao e tamanho de base diferentes, juntas."""
    redes = []
    for j, v in enumerate(TREINOS.values()):
        hp = {**HP, "lote": 4096, **v}
        redes.append(_rede(hp, n=120 + j, m=40 + j, sem=j + 1))
    res = P.treinar_redes(redes, disp, semente=0)
    for r, x, nome in zip(res, redes, TREINOS):
        m, ep = M.treinar(x["Xa"], x["ya"], x["hp"], x["semente"], x["Xb"], x["yb"])
        p = M.prever(m, x["Xb"])
        # Com BN, o gradiente do vies antes da normalizacao e so arredondamento, e o Adam o amplifica ate um exemplo cruzar o
        # zero da ReLU. Com lr baixo, a melhora por epoca fica perto do limiar de 1e-6. Nos outros casos o numero tem de bater.
        folga_ep, folga_ks = dict(bn=3, lr_baixo=6).get(nome, 0), dict(bn=.05, lr_baixo=.01).get(nome, 1e-6)
        assert abs(r["epoca"] - ep) <= folga_ep, nome
        perto(comum.ks(x["yb"], r["p"]), comum.ks(x["yb"], p), folga_ks)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_epocas_fixas_e_previsao_no_conjunto_final(disp):
    """Sem validacao: cada rede para na sua epoca e preve Xt com os pesos daquele momento."""
    Xt, _ = dados(30, semente=99)
    redes = []
    for j, ep in enumerate((1, 3, 7, 20)):
        x = _rede({**HP, "lote": 4096}, sem=j + 1)
        redes.append(dict(hp=x["hp"], Xa=x["Xa"], ya=x["ya"], Xt=Xt, epocas=ep, semente=x["semente"]))
    for r, x in zip(P.treinar_redes(redes, disp, semente=0), redes):
        m, _ = M.treinar(x["Xa"], x["ya"], x["hp"], x["semente"], epocas=x["epocas"])
        perto(r["p"], M.prever(m, Xt), 2e-3)
        assert r["epoca"] == x["epocas"]


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_cada_rede_da_o_numero_certo_de_passos(disp):
    hp = {**HP, "lote": 2}
    redes = [dict(_rede(hp, n=5), epocas=1), dict(_rede(hp, n=9, sem=2), epocas=2)]
    for x in redes:
        x.pop("Xb"); x.pop("yb"); x["Xt"] = x["Xa"]
    a, b = P.treinar_redes(redes, disp, semente=0)
    assert (a["passos"], b["passos"]) == (3, 10)


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_rede_que_diverge_vira_probabilidade_neutra(disp):
    """Como em M.prever: configuracao divergente vira KS ruim, sem derrubar o estudo."""
    res = P.treinar_redes([_rede({**HP, "lr": 1e30, "lote": 4096}), _rede({**HP, "lote": 4096}, sem=2)], disp, semente=0)
    assert np.isfinite(res[0]["p"]).all() and np.isfinite(res[1]["p"]).all()


def test_redes_de_formatos_diferentes_nao_entram_na_mesma_pilha():
    with pytest.raises(ValueError):
        P.treinar_redes([_rede(HP), _rede({**HP, "unidades": 9})], "cpu", semente=0)


def _base(n=600):
    r = np.random.default_rng(0)
    cols = ["tenure", "MonthlyCharges", "TotalCharges"] + [f"f{i}" for i in range(4)]
    def tabela(m):
        return pd.DataFrame(r.normal(size=(m, len(cols))).astype("float32"), columns=cols)
    Xtr, Xte = tabela(n), tabela(200)
    ytr = pd.Series((Xtr.tenure + r.normal(size=n) > .6).astype(int))
    yte = pd.Series((Xte.tenure + r.normal(size=200) > .6).astype(int))
    return Xtr, Xte, ytr, yte


C = dict(hp={**HP, "lote": 64, "paciencia": 3}, dados={}, extras=[])


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_medir_devolve_o_mesmo_formato_e_a_mesma_ordem_de_medir(disp):
    c2 = dict(hp={**C["hp"], "plano": "BN"}, dados={}, extras=[])
    res = P.medir_pilha([(c2, 1, 0), (C, 1, 0), (C, 2, 0)], dados=_base(), dispositivos=[disp])
    assert all(set(r) == {"val", "teste", "epocas", "prob"} for r in res)
    assert all(len(r["prob"]) == 200 for r in res)
    assert all(.3 < r["val"] <= 1 and .1 < r["teste"] <= 1 for r in res)
    uma = P.medir_pilha([(C, 2, 0)], dados=_base(), dispositivos=[disp])[0]
    assert abs(uma["val"] - res[2]["val"]) < .15     # a mesma medida, em outra pilha, diverge so pelo sorteio
    assert abs(uma["epocas"] - res[2]["epocas"]) < 50


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_teste_nao_influencia_o_treino(disp):
    """Trocar o teste inteiro, features e rotulos, nao muda nada do que vem do treino: so o KS de teste muda."""
    Xtr, Xte, ytr, yte = _base()
    a = P.medir_pilha([(C, 1, 0)], dados=(Xtr, Xte, ytr, yte), dispositivos=[disp])[0]
    outro = Xte * 100 + 7
    b = P.medir_pilha([(C, 1, 0)], dados=(Xtr, outro, ytr, 1 - yte), dispositivos=[disp])[0]
    assert a["val"] == b["val"] and a["epocas"] == b["epocas"]
    assert a["teste"] != b["teste"]


@pytest.mark.parametrize("disp", DISPOSITIVOS)
def test_pilha_e_reprodutivel(disp):
    k = dict(dados=_base(), dispositivos=[disp])
    a, b = P.medir_pilha([(C, 1, 0)], **k)[0], P.medir_pilha([(C, 1, 0)], **k)[0]
    assert a["val"] == b["val"] and a["teste"] == b["teste"]


def test_duas_gpus_usam_threads_sem_misturar_resultados():
    """Com dois dispositivos, cada pilha roda em um. O resultado tem de vir na ordem pedida."""
    tarefas = [(dict(hp={**C["hp"], "unidades": u, "plano": "NN"}, dados={}, extras=[]), 1, 0) for u in (4, 8, 12, 16)]
    res = P.medir_pilha(tarefas, dados=_base(), dispositivos=["cpu", "cpu"])
    assert all(len(r["prob"]) == 200 for r in res) and len({r["val"] for r in res}) > 1


def test_divide_em_pilhas_por_formato_e_respeita_o_limite():
    base = dict(hp=HP, Xa=np.zeros((4, 6), "float32"))
    grupos = P.agrupar([base] * 5 + [dict(base, hp={**HP, "ativ": "gelu"})] * 2 + [dict(base, hp={**HP, "unidades": 16})], max_redes=3)
    assert sorted(len(g) for g in grupos) == [1, 2, 2, 3]
    assert sorted(i for g in grupos for i in g) == list(range(8))


def test_rodar_grade_e_medir_muitos_usam_a_pilha_quando_ha_dispositivo(tmp_path, monkeypatch):
    monkeypatch.setattr(F3, "_dados", _base)
    F3.abrir_cache(str(tmp_path))
    F3.usar_pilha(["cpu"])
    try:
        cfgs = {"a": dict(hp={**C["hp"]}, dados={}, extras=[]), "b": dict(hp={**C["hp"], "plano": "BN"}, dados={}, extras=[])}
        tab, probs = F3.rodar_grade(str(tmp_path), cfgs, [1, 2])
        assert len(tab) == 2 and set(probs) == {"a", "b"}
        res = F3.medir_muitos([(cfgs["a"], 3, 0), (cfgs["b"], 3, 0)])
        assert len(res) == 2 and all("val" in r for r in res)
    finally:
        F3.usar_pilha(None)


def test_grade_padrao_tem_todas_as_permutacoes_e_tres_larguras():
    base = dict(hp=HP, dados={}, extras=[])
    g = F3.grade(base)
    assert len(F3.planos((4,))) == 81
    assert {c["hp"]["unidades"] for c in g.values()} == {16, 32, 64}
    assert len(g) == (3 + 9 + 27 + 81) * 3 * 3


def test_conferencia_compara_pilha_com_referencia_e_com_o_ruido(monkeypatch):
    monkeypatch.setattr(F3, "_dados", _base)
    F3.usar_pilha(["cpu"])
    try:
        tab, resumo = F3.conferir_pilha({"a": C, "b": dict(hp={**C["hp"], "plano": "DN", "dropout": .2}, dados={}, extras=[])}, [1, 2])
    finally:
        F3.usar_pilha(None)
    assert list(tab.config) == ["a", "b"]
    assert {"val_ref", "val_pilha", "d_val", "ruido_val", "d_teste", "ruido_teste", "epocas_ref", "epocas_pilha"} <= set(tab.columns)
    assert set(resumo) >= {"vies_val", "ep_vies_val", "vies_teste", "ep_vies_teste", "ruido_val", "ruido_teste", "n"}
    assert resumo["n"] == 4


def test_optuna_fica_na_cpu_mesmo_com_a_pilha_ligada(tmp_path, monkeypatch):
    """Cada trial tem um formato: as pilhas ficam tao pequenas que a GPU perde da CPU (medido: 8,4 s contra 3,9 s por trial)."""
    def nao_deve_rodar(*a, **k):
        raise AssertionError("o Optuna nao deve treinar em pilha")
    monkeypatch.setattr(F3, "_dados", _base)
    monkeypatch.setattr(P, "medir_pilha", nao_deve_rodar)
    monkeypatch.setattr(M, "MAX_EPOCAS", 2)
    F3.usar_pilha(["cpu"])
    try:
        st, _ = F3.estudar(str(tmp_path), "val", dict(C, hp={**C["hp"], "paciencia": 10}), 2, reps=(1,))    # o primeiro trial e o campeao, e o espaco so tem paciencia 10, 20 e 30
    finally:
        F3.usar_pilha(None)
    assert len([t for t in st.trials if t.state.name == "COMPLETE"]) == 2
