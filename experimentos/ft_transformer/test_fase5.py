"""FT-Transformer, fase 5: tokens categoricos, embeddings periodicos, agenda do lr, media de redes, features extras e
o Optuna.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase5.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import modelo as M, fase1 as F1, fase4 as F4, fase5 as F5
import comum
import numpy as np, pandas as pd, pytest, torch
from test_fase1 import _base

f32 = lambda a: np.asarray(a, dtype="float32")
HP = dict(M.PADRAO, n_blocos=1, d_token=16, n_cabecas=2)


@pytest.fixture(scope="module")
def reais():
    """Uma amostra do treino oficial, preparada com as tres features: as colunas que o FT ve de verdade."""
    X, _, y, _ = comum.dividir(comum.carregar_df())
    X, y = X.iloc[:400], y.iloc[:400]
    Xa, ya, Xv = comum.preparar(X.iloc[:300], y.iloc[:300], X.iloc[300:], features=F4.TRES)
    return Xa, ya, Xv, y.iloc[300:]


def test_tokens_categoricos_juntam_as_dummies(reais):
    Xa = reais[0]
    grupos, _ = M.estrutura(dict(tokens="cat"), list(Xa.columns), f32(Xa))
    assert len(grupos) == Xa.shape[1] - 7                       # 3 + 3 + 4 dummies viram 3 tokens
    nomes = [[Xa.columns[j] for j in g] for g in grupos if len(g) > 1]
    assert sorted(map(sorted, nomes)) == sorted(sorted(v) for v in M.CATEGORICAS.values())
    assert "PaymentMethod_Automatic" not in sum(nomes, [])       # variavel criada, nao e categoria
    m = M.FTTransformer(Xa.shape[1], **M.arquitetura(HP), grupos=grupos)
    assert m.tokens(torch.as_tensor(f32(Xa[:5]))).shape == (5, len(grupos) + 1, 16)


def test_token_categorico_e_uma_embedding_da_categoria(reais):
    """Com uma dummy ativa por linha, o token do grupo so depende de qual categoria esta ativa."""
    Xa = reais[0]
    grupos, _ = M.estrutura(dict(tokens="cat"), list(Xa.columns), f32(Xa))
    tok = M.Tokenizador(Xa.shape[1], 8, grupos)
    t = tok(torch.as_tensor(f32(Xa))).detach().numpy()
    g = next(i for i, gg in enumerate(grupos) if Xa.columns[gg[0]].startswith("Contract_"))
    cat = Xa[[Xa.columns[j] for j in grupos[g]]].values.argmax(1)
    for k in np.unique(cat):
        assert np.allclose(t[cat == k, g], t[cat == k, g][0], atol=1e-6)


def test_plr_so_nas_colunas_numericas_e_treina(reais):
    Xa, ya, Xv, yv = reais
    _, num = M.estrutura(dict(num_emb="plr"), list(Xa.columns), f32(Xa))
    assert [Xa.columns[j] for j in num] == [c for c in Xa.columns if Xa[c].nunique() > 2]
    assert {"tenure", "MonthlyCharges", "TotalCharges", "delta_preco"} <= {Xa.columns[j] for j in num}
    hp = dict(HP, num_emb="plr", plr_k=4, plr_sigma=0.5, tokens="cat", paciencia=2)
    m, ep = M.treinar(f32(Xa), f32(ya), hp, 1, f32(Xv), f32(yv), colunas=list(Xa.columns))
    assert 1 <= ep <= M.MAX_EPOCAS and m.tokenizador.freq.shape == (len(num), 4)
    p = M.prever(m, f32(Xv))
    assert p.shape == (len(Xv),) and np.isfinite(p).all()


def test_tokens_cat_sem_nomes_falha():
    with pytest.raises(ValueError):
        M.estrutura(dict(tokens="cat"), None, np.zeros((3, 4)))


def test_agenda_cosseno():
    hp = dict(agenda="cos", aquec=5, horizonte=60)
    assert M.fator_lr({}, 7) == 1.0
    assert M.fator_lr(hp, 1) == pytest.approx(0.2) and M.fator_lr(hp, 5) == pytest.approx(1.0)
    assert M.fator_lr(hp, 60) == pytest.approx(0.05) and M.fator_lr(hp, 500) == pytest.approx(0.05)
    f = [M.fator_lr(hp, e) for e in range(5, 61)]
    assert all(a >= b for a, b in zip(f, f[1:]))


def test_media_de_redes_e_a_media_das_sementes(monkeypatch):
    monkeypatch.setattr(F1, "_dados", lambda: _base())
    X, _, y, _ = _base()
    tr, va = np.arange(250), np.arange(250, len(X))
    aj, pa = np.arange(200), np.arange(200, 250)
    c1 = dict(hp=dict(HP, paciencia=2), dados={})
    a = F1.ajustar_um(c1, X.iloc[tr], y.iloc[tr], X.iloc[va], aj, pa, 5, "cpu")
    b = F1.ajustar_um(c1, X.iloc[tr], y.iloc[tr], X.iloc[va], aj, pa, 5 + 7919, "cpu")
    c = F1.ajustar_um(dict(c1, hp=dict(c1["hp"], n_redes=2)), X.iloc[tr], y.iloc[tr], X.iloc[va], aj, pa, 5, "cpu")
    assert np.allclose(c["p_val"], (a["p_val"] + b["p_val"]) / 2, atol=1e-6)


def test_features_extras_existem_e_nao_olham_o_rotulo():
    X, _, _, _ = comum.dividir(comum.carregar_df())
    Xf = comum.adicionar_features(X, F5.EXTRAS)
    for f in F5.EXTRAS:
        assert Xf[f].notna().all(), f
    assert set(np.unique(Xf["tenure_ate_3"])) == {0, 1} and Xf["n_protecao"].between(0, 4).all()
    assert (Xf["novo_mes_a_mes"] <= Xf["mtm_fibra"] + Xf["novo_mes_a_mes"]).all()


def test_grade_muda_uma_coisa_por_vez():
    g = F5.grade_triagem()
    assert len(g) == 12 + len(F5.EXTRAS)
    base_hp, base_d = F5.BASE["hp"], F5.BASE["dados"]
    for rot, (eixo, c) in g.items():
        muda_hp = {k for k in set(c["hp"]) | set(base_hp) if c["hp"].get(k) != base_hp.get(k)}
        muda_d = {k for k in set(c["dados"]) | set(base_d) if c["dados"].get(k) != base_d.get(k)}
        assert muda_hp or muda_d, rot
        assert not (muda_hp and muda_d), rot
    assert len({F1._chave(c) for _, c in g.values()} | {F1._chave(F5.BASE)}) == len(g) + 1


def test_trial_da_base_reproduz_a_base():
    c = F5.config_do_trial(F5.parametros_base(F5.EXTRAS[:2]), F5.EXTRAS[:2])
    assert F1._chave(c) == F1._chave(F5.BASE)


def test_estudo_retoma_e_comeca_pela_base(tmp_path):
    rng = np.random.default_rng(0)
    vistos = []

    def medir(c, D):
        vistos.append(c)
        return dict(ks_kfold=float(rng.uniform(.5, .55)), ks_oof=.5, epocas=7.0, tempo_min=.1)
    st = F5.estudar(str(tmp_path), 3, ("sozinho",), medir, D=[])
    assert len(st.trials) == 3 and F1._chave(vistos[0]) == F1._chave(F5.BASE)
    st = F5.estudar(str(tmp_path), 5, ("sozinho",), medir, D=[])
    assert len([t for t in st.trials if t.state.name == "COMPLETE"]) == 5
    assert len(F5.melhores(st, 2)) == 2 and not F5.resumo_optuna(st).empty
    c = F5.config_do_trial(F5.melhores(st, 1)[0].params, ("sozinho",))
    assert c["hp"]["d_token"] % c["hp"]["n_cabecas"] == 0


def test_media_com_mlp():
    y = np.array([0, 1, 0, 1])
    a, b = np.array([[.1, .9, .2, .8]] * 3), np.array([[.3, .7, .4, .6]] * 3)
    m = F5.media_com_mlp(y, a, b)
    assert m["prob"].shape == (3, 4) and np.allclose(m["prob"][0], [.2, .8, .3, .7])
    assert comum.ks(y, m["rank"][0]) == 1.0
