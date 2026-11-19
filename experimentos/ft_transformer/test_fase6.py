"""FT-Transformer, fase 6: mineracao de features com selecao para frente.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase6.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import modelo as M, fase1 as F1, fase6 as F6
import comum
import numpy as np, pytest


def test_base_e_candidatas_validas():
    hp = F6.BASE["hp"]
    assert hp["d_token"] % hp["n_cabecas"] == 0 and set(M._ARQ) <= set(hp)
    assert F6.BASE["dados"]["over"] is False
    assert set(F6.BASE["dados"]["features"]) | set(F6.CANDIDATAS) <= set(comum.FEATURES)
    assert len(set(F6.CANDIDATAS)) == len(F6.CANDIDATAS) and not set(F6.CANDIDATAS) & set(F6.BASE["dados"]["features"])


def test_features_novas_sao_finitas_e_por_linha():
    X, _, _, _ = comum.dividir(comum.carregar_df())
    Xf = comum.adicionar_features(X, F6.NOVAS)
    for f in F6.NOVAS:
        assert np.isfinite(Xf[f].astype(float)).all(), f
    # por linha: calcular num pedaco da igual a calcular no todo
    parte = comum.adicionar_features(X.iloc[:50], F6.NOVAS)
    assert np.allclose(parte[list(F6.NOVAS)].astype(float), Xf[list(F6.NOVAS)].iloc[:50].astype(float))
    assert (Xf["mtm_tenure"][X["Contract_Month-to-month"] == 0] == 0).all()


def _falso(y, chamadas):
    """OOF sinteticas: a candidata "boa" acrescenta sinal, as outras so ruido."""
    rng = np.random.default_rng(0)
    base = y + rng.normal(0, 1.2, (3, len(y)))

    def medir(c):
        chamadas.append(tuple(c["dados"]["features"]))
        oof = base.copy()
        if "boa" in c["dados"]["features"]:
            oof = oof + 0.5 * y
        extra = [f for f in c["dados"]["features"] if f not in ("boa",) and f not in F6.BASE["dados"]["features"]]
        oof = oof + rng.normal(0, .05, oof.shape) * len(extra)
        return dict(oof=oof, ks_kfold=comum.ks(y, oof.mean(0)), ks_oof=comum.ks(y, oof.mean(0)), epocas=5.0, tempo_min=.1)
    return medir


def test_selecao_escolhe_a_boa_e_para(tmp_path):
    y = (np.random.default_rng(1).random(1500) < .27).astype(int)
    chamadas = []
    esc, hist = F6.selecao_para_frente(str(tmp_path), _falso(y, chamadas), y, candidatas=("ruim1", "boa", "ruim2"), rodadas=3)
    assert esc == ["boa"]
    assert [h["candidata"] for h in hist if h.get("escolhida")] == ["boa"]
    assert {h["rodada"] for h in hist} == {1, 2}
    n = len(chamadas)
    esc2, _ = F6.selecao_para_frente(str(tmp_path), _falso(y, chamadas), y, candidatas=("ruim1", "boa", "ruim2"), rodadas=3)
    assert esc2 == ["boa"] and len(chamadas) == n            # retomar le do cache, nao mede de novo


def test_com_features_nao_altera_a_base():
    c = F6.com_features(F6.BASE, ["antigo"])
    assert c["dados"]["features"][-1] == "antigo" and "antigo" not in F6.BASE["dados"]["features"]
    assert c["hp"] == F6.BASE["hp"] and c["dados"]["over"] is False
