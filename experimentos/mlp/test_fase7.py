"""Fase 7 do MLP: features do FT, triagem e Optuna com o conjunto de dados.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/mlp/test_fase7.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import comum, modelo as M, blocos as BL, fase4 as F4, fase5 as F5, fase7 as F7
import numpy as np, pytest
from test_fase4 import _base

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def dados_sinteticos(monkeypatch):
    monkeypatch.setattr(F4, "_dados", _base)
    F5.divs.cache_clear(); F7.divs_optuna.cache_clear()
    F5.usar_pilha(None)
    yield
    F5.divs.cache_clear(); F7.divs_optuna.cache_clear()


def test_base_e_o_mlp_adotado():
    linhas = {r["rotulo"]: r for r in comum.ler_placar(os.path.join(RAIZ, "resultados")) if r["modelo"] == "mlp"}
    assert F7.chave(F7.BASE) == F5._chave_cfg(linhas[F7.ROT_BASE]["config"])


def test_features_novas_iguais_as_do_ft():
    X, _, _, _ = comum.dividir(comum.carregar_df())
    for f in ("cobranca_razao", "antigo", "caro_sem_backup"):
        assert np.allclose(BL.CANDIDATAS[f](X)[f].astype(float), comum.FEATURES[f](X).astype(float)), f
    for c in F7.candidatos().values():
        assert set(c["extras"]) <= set(BL.CANDIDATAS)


def test_trial_da_base_reproduz_o_adotado():
    conj = {"base": F7.BASE["extras"], "p_cinco": F7.candidatos()["p_cinco"]["extras"]}
    assert F7.chave(F7.config_do_trial(F7.parametros_base("base"), conj)) == F7.chave(F7.BASE)
    c = F7.config_do_trial(F7.parametros_base("p_cinco"), conj)
    assert c["hp"] == F7.BASE["hp"] and c["extras"][:3] == F7.BASE["extras"] and "cobranca_razao" in c["extras"]


def test_conjuntos_optuna():
    cands = F7.candidatos()
    linhas = [dict(candidato="p_cinco", delta_oof=.003), dict(candidato="m_antigo", delta_oof=.005),
              dict(candidato="m_cobranca_razao", delta_oof=-.001)]
    assert list(F7.conjuntos_optuna(linhas, cands)) == ["base", "m_antigo", "p_cinco"]


def test_medir_com_divisoes_proprias(dados_sinteticos):
    y = np.asarray(_base()[2])
    D = comum.divisoes_placar(y, (201,))
    C = dict(F7.BASE, hp=dict(F7.BASE["hp"], paciencia=3))
    a = F5.medir([C], D=D)[0]
    b = F5.medir([C])[0]
    assert a["oof"].shape == (1, len(y)) and b["oof"].shape == (3, len(y))
    t = F7.triagem({"m_antigo": F7.com(C, ["antigo"])}, base=C, D=D)
    assert t["base"]["ks_kfold"] == pytest.approx(a["ks_kfold"]) and t["linhas"][0]["candidato"] == "m_antigo"


def test_estudo_retoma(dados_sinteticos, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MAX_EPOCAS", 4)
    y = np.asarray(_base()[2])
    D = comum.divisoes_placar(y, (201,))
    conj = {"base": F7.BASE["extras"]}
    st = F7.estudar(str(tmp_path), 3, conj, D=D, lote_trials=2)
    assert len(st.trials) == 3 and st.trials[0].params["conjunto"] == "base"
    st2 = F7.estudar(str(tmp_path), 4, conj, D=D, lote_trials=2)
    assert len(st2.trials) == 4
    assert len(F7.melhores(st2, conj, 2)) == 2
