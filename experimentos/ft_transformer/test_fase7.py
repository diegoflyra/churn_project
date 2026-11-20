"""FT-Transformer, fase 7: target encoding dentro da dobra, pacotes e Optuna com o conjunto de dados.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase7.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import fase1 as F1, fase7 as F7
import comum
import numpy as np, pandas as pd, pytest


@pytest.fixture(scope="module")
def dados():
    X, _, y, _ = comum.dividir(comum.carregar_df())
    return X, y


def test_grupos_sem_buracos(dados):
    X, _ = dados
    for n, f in comum.TE_GRUPOS.items():
        g = f(X)
        assert g.notna().all() and not g.str.contains("nan").any(), n
        assert 4 <= g.nunique() <= 60, (n, g.nunique())


def test_codificar_alvo_nao_vaza(dados):
    X, y = dados
    Xa, ya, Xv = X.iloc[:3000], y.iloc[:3000], X.iloc[3000:3500]
    A, V = comum.codificar_alvo(Xa, ya, Xv, F7.TES)
    # trocar o rotulo de um cliente do treino nao muda a codificacao dele mesmo
    y2 = ya.copy(); y2.iloc[7] = 1 - y2.iloc[7]
    A2, V2 = comum.codificar_alvo(Xa, y2, Xv, F7.TES)
    for t in F7.TES:
        assert A[t].iloc[7] == A2[t].iloc[7]
        assert np.isfinite(A[t]).all() and np.isfinite(V[t]).all()
        assert 0 < A[t].min() and A[t].max() < 1
    # a avaliacao so depende do treino: os rotulos da avaliacao nao existem para a funcao
    V3 = comum.codificar_alvo(Xa, ya, Xv.iloc[::-1], F7.TES)[1]
    assert np.allclose(V3[list(F7.TES)].iloc[::-1].to_numpy(), V[list(F7.TES)].to_numpy())


def test_preparar_com_te(dados):
    X, y = dados
    a = comum.preparar(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:2100], features=["tenure_1"], over=False)
    b = comum.preparar(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:2100], features=["tenure_1"], over=False, te=())
    assert a[0].equals(b[0]) and a[2].equals(b[2])                 # sem te: identico ao de antes
    Xa, _, Xv = comum.preparar(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:2100], over=False, te=["te_tenure_contrato"])
    assert list(Xa.columns)[-1] == "te_tenure_contrato" and abs(Xa["te_tenure_contrato"].mean()) < 1e-6   # passou no scaler
    assert Xv.shape[1] == Xa.shape[1]


def test_candidatos_validos():
    c = F7.candidatos()
    assert {"p_cinco", "te_todos", "p_cinco_te_todos"} <= set(c)
    for n, cfg in c.items():
        assert set(cfg["dados"]["features"]) <= set(comum.FEATURES), n
        assert set(cfg["dados"].get("te", [])) <= set(comum.TE_GRUPOS), n
        assert cfg["hp"] == F7.BASE["hp"] and cfg["dados"]["over"] is False
    assert "te" not in F7.BASE["dados"] and F7.BASE["dados"]["features"] == ["cliente_novo", "delta_preco", "tenure_1"]


def test_conjuntos_e_trial_da_base():
    cands = F7.candidatos()
    linhas = [dict(candidato="p_cinco", delta_oof=.004), dict(candidato="te_todos", delta_oof=.006),
              dict(candidato="te_net_protecao", delta_oof=.001), dict(candidato="p_razao_antigo", delta_oof=-.002)]
    conj = F7.conjuntos_optuna(linhas, cands)
    assert list(conj) == ["base", "te_todos", "p_cinco"]
    # o trial enfileirado com a base reproduz exatamente o o5_top3
    assert F1._chave(F7.config_do_trial(F7.parametros_base("base"), conj)) == F1._chave(F7.BASE)
    c = F7.config_do_trial(F7.parametros_base("te_todos"), conj)
    assert c["dados"]["te"] == list(F7.TES) and c["hp"] == F7.BASE["hp"]


def test_triagem_com_cache(tmp_path):
    y = (np.random.default_rng(1).random(1500) < .27).astype(int)
    rng = np.random.default_rng(0)
    base = y + rng.normal(0, 1.2, (3, len(y)))
    chamadas = []

    def medir(c):
        chamadas.append(1)
        oof = base + (0.5 * y if c["dados"].get("te") else 0) + rng.normal(0, .02, base.shape)
        return dict(oof=oof, ks_kfold=comum.ks(y, oof.mean(0)), ks_oof=comum.ks(y, oof.mean(0)))
    cands = {k: v for k, v in F7.candidatos().items() if k in ("p_cinco", "te_todos")}
    linhas = F7.triagem(str(tmp_path), medir, y, cands)
    assert [l["passou"] for l in linhas] == [False, True] and len(chamadas) == 3
    F7.triagem(str(tmp_path), medir, y, cands)
    assert len(chamadas) == 3                                     # retomar le do cache
