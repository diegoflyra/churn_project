"""FT-Transformer: modelo, treino e medida no placar comum (fase 1).

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase1.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")          # testes rapidos: a parada real usa ate 10.000 epocas
sys.path.insert(0, os.path.dirname(__file__))
import modelo as M, fase1 as F1                          # modelo vem antes do torch: ele ajusta o ambiente
import comum
import numpy as np, pandas as pd, pytest, torch

HP = dict(M.PADRAO, n_blocos=1, d_token=16, n_cabecas=2, lr=3e-3, paciencia=2)     # rede minuscula para os testes


def _base(n=500, semente=0):
    """Clientes sinteticos com as colunas que comum.preparar escala; o churn depende de tenure e de Contract."""
    r = np.random.default_rng(semente)
    tenure = r.integers(0, 72, n)
    mensal = r.uniform(20, 110, n)
    mes = r.integers(0, 2, n)
    X = pd.DataFrame(dict(tenure=tenure, MonthlyCharges=mensal, TotalCharges=mensal * tenure,
                          Partner=r.integers(0, 2, n), Contract_mes=mes, Contract_ano=1 - mes))
    z = -1.5 + 2.0 * mes - 0.04 * tenure + 0.01 * mensal + r.normal(0, .7, n)
    y = pd.Series((z > 0).astype(int))
    X.index = r.permutation(np.arange(10_000, 10_000 + n))          # indices do CSV, como no treino oficial
    y.index = X.index
    corte = int(.75 * n)
    return X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch):
    monkeypatch.setattr(F1, "_dados", lambda: _base())
    F1.divs.cache_clear()
    yield
    F1.divs.cache_clear()


# ---------------- modelo ----------------
def test_um_token_por_feature_mais_o_cls():
    m = M.FTTransformer(n_features=6, **M.arquitetura(HP))
    x = torch.randn(4, 6)
    assert m.tokens(x).shape == (4, 7, 16)                    # 6 features + [CLS]
    assert m(x).shape == (4,)


def test_padrao_e_o_do_artigo():
    """Gorishniy et al. (2021), 'default FT-Transformer': 3 blocos, d_token 192, 8 cabecas, AdamW 1e-4 / 1e-5."""
    p = M.PADRAO
    assert (p["n_blocos"], p["d_token"], p["n_cabecas"]) == (3, 192, 8)
    assert (p["drop_atencao"], p["drop_ffn"], p["drop_residual"]) == (0.2, 0.1, 0.0)
    assert (p["lr"], p["wd"]) == (1e-4, 1e-5) and abs(p["ffn_fator"] - 4 / 3) < 1e-9


def test_weight_decay_fica_fora_do_tokenizador_das_normas_e_dos_vieses():
    m = M.FTTransformer(n_features=6, **M.arquitetura(HP))
    com, sem = M.grupos_parametros(m, wd=1e-5)
    nomes_sem = {n for n, p in m.named_parameters() if any(p is q for q in sem["params"])}
    assert com["weight_decay"] == 1e-5 and sem["weight_decay"] == 0.0
    assert all(("tokenizador" in n) or ("norm" in n) or n.endswith("bias") or n == "cls" for n in nomes_sem)
    assert len(com["params"]) + len(sem["params"]) == len(list(m.parameters()))


def test_treino_com_parada_e_com_epocas_fixas():
    Xtr, _, ytr, _ = _base()
    X = Xtr.values.astype("float32"); y = ytr.values.astype("float32")
    m, ep = M.treinar(X[:300], y[:300], HP, 1, X[300:], y[300:])
    assert 1 <= ep <= M.MAX_EPOCAS
    vistas = []
    M.treinar(X, y, HP, 1, epocas=3, ao_fim_epoca=lambda e, _m: vistas.append(e))
    assert vistas == [1, 2, 3]


def test_mesma_semente_mesma_previsao():
    Xtr, _, ytr, _ = _base()
    X = Xtr.values.astype("float32"); y = ytr.values.astype("float32")
    a = M.prever(M.treinar(X, y, HP, 7, epocas=2)[0], X)
    b = M.prever(M.treinar(X, y, HP, 7, epocas=2)[0], X)
    c = M.prever(M.treinar(X, y, HP, 8, epocas=2)[0], X)
    assert np.allclose(a, b) and not np.allclose(a, c)
    assert ((a > 0) & (a < 1)).all()


# ---------------- uma dobra do placar ----------------
def test_a_parada_so_ve_a_separacao_interna(monkeypatch):
    """1a chamada de preparar: ajuste -> parada. 2a: treino inteiro da dobra -> validacao. A validacao nunca decide."""
    Xtr, _, ytr, _ = _base()
    d = F1.divs()[0]
    chamadas, original = [], comum.preparar

    def espia(X_tr, y_tr, X_aval, **k):
        chamadas.append((set(X_tr.index), set(X_aval.index)))
        return original(X_tr, y_tr, X_aval, **k)

    monkeypatch.setattr(comum, "preparar", espia)
    X, y, V = Xtr.iloc[d["treino"]], ytr.iloc[d["treino"]], Xtr.iloc[d["validacao"]]
    r = F1.ajustar_um(dict(hp=HP, dados={}), X, y, V, d["ajuste"], d["parada"], 5, "cpu")
    (aj, pa), (tr, va) = chamadas
    assert aj == set(X.index[d["ajuste"]]) and pa == set(X.index[d["parada"]])
    assert tr == set(X.index) and va == set(V.index)
    assert not (aj | pa | tr) & va
    assert len(r["p_val"]) == len(V) and r["epoca"] >= 1


def test_dados_chegam_ao_preparar(monkeypatch):
    Xtr, _, ytr, _ = _base()
    d = F1.divs()[0]
    ks_, original = [], comum.preparar
    monkeypatch.setattr(comum, "preparar", lambda *a, **k: (ks_.append(k), original(*a, **k))[1])
    F1.ajustar_um(dict(hp=HP, dados={"over": False}), Xtr.iloc[d["treino"]], ytr.iloc[d["treino"]], Xtr.iloc[d["validacao"]],
                  d["ajuste"], d["parada"], 5, "cpu")
    assert ks_ == [{"over": False}, {"over": False}]


# ---------------- placar ----------------
def test_medir_cobre_as_15_dobras_e_aprende_o_sinal():
    med = F1.medir(dict(hp=HP, dados={}), dispositivos=["cpu"], por_dispositivo=1)
    ytr = np.asarray(_base()[2])
    assert med["oof"].shape == (3, len(ytr)) and len(med["ks_dobras"]) == 15
    assert med["ks_kfold"] > .3                                  # o sinal sintetico e forte
    assert "mse_treino" in med and med["epocas"] >= 1


def test_garantir_reaproveita_o_que_ja_foi_medido(tmp_path, monkeypatch):
    pasta = str(tmp_path)
    c0, c1 = dict(hp=HP, dados={}), dict(hp=dict(HP, d_token=8), dados={})
    l = F1.garantir(pasta, [("trial0", c0, None), ("menor", c1, {"vs_trial0": ("ft_transformer", "trial0")})], dispositivos=["cpu"], por_dispositivo=1)
    assert [r["rotulo"] for r in l] == ["trial0", "menor"] and l[1]["vs_trial0"]["veredito"] in ("ganhou", "empatou", "perdeu")
    assert all(r["modelo"] == "ft_transformer" for r in comum.ler_placar(pasta))
    monkeypatch.setattr(F1, "medir", lambda *a, **k: pytest.fail("mediu de novo o que ja estava no placar"))
    l2 = F1.garantir(pasta, [("trial0", c0, None), ("menor", c1, {"vs_trial0": ("ft_transformer", "trial0")})], dispositivos=["cpu"])
    assert [r["ks_kfold"] for r in l2] == [r["ks_kfold"] for r in l]


def test_grade_varia_so_o_que_foi_pedido():
    g = F1.grade(F1.TRIAL0, n_blocos=(1, 3), d_token=(64, 192))
    assert len(g) == 3 and "g_b3_d192" not in g                  # a celula do padrao e o proprio trial 0: nao mede duas vezes
    assert g["g_b1_d64"]["hp"] == dict(F1.TRIAL0["hp"], n_blocos=1, d_token=64)
    assert all(c["dados"] == F1.TRIAL0["dados"] for c in g.values())
