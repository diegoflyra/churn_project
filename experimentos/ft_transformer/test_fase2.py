"""FT-Transformer, fase 2: regularizacao (L1/L2 na perda, dropout, weight decay) e curvas de sobreajuste.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase2.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import modelo as M, fase1 as F1, fase2 as F2
import numpy as np, pytest, torch
from test_fase1 import _base, HP


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch):
    monkeypatch.setattr(F1, "_dados", lambda: _base())
    F1.divs.cache_clear()
    yield
    F1.divs.cache_clear()


def _xy():
    Xtr, _, ytr, _ = _base()
    return Xtr.values.astype("float32"), ytr.values.astype("float32")


def test_sem_l1_nem_l2_a_penalidade_e_zero():
    m = M.FTTransformer(6, **M.arquitetura(HP))
    assert float(M.penalidade(m, HP)) == 0.0
    assert float(M.penalidade(m, dict(HP, l1=1e-3))) > 0 and float(M.penalidade(m, dict(HP, l2=1e-3))) > 0


def test_penalidade_so_nos_pesos_com_decay():
    """Como o weight decay do artigo: tokenizador, [CLS], normas e vieses ficam de fora."""
    m = M.FTTransformer(6, **M.arquitetura(HP))
    hp = dict(HP, l1=1e-2, l2=1e-2)
    antes = float(M.penalidade(m, hp))
    with torch.no_grad():
        m.tokenizador.peso.mul_(100); m.cls.mul_(100); m.norm_final.weight.mul_(100); m.saida.bias.add_(100)
    assert abs(float(M.penalidade(m, hp)) - antes) < 1e-6
    with torch.no_grad():
        m.saida.weight.mul_(2)
    assert float(M.penalidade(m, hp)) > antes


def test_l2_pesado_encolhe_os_pesos():
    X, y = _xy()
    norma = lambda m: sum(float((p ** 2).sum()) for p in M.grupos_parametros(m, 0)[0]["params"])
    livre = norma(M.treinar(X, y, HP, 3, epocas=5)[0])
    preso = norma(M.treinar(X, y, dict(HP, l2=1e-1), 3, epocas=5)[0])
    assert preso < .8 * livre


def test_trial0_continua_igual_ao_da_fase_1():
    """As chaves novas sao opcionais: o trial 0 ja medido no placar e reaproveitado, sem medir de novo."""
    assert "l1" not in M.PADRAO and "l2" not in M.PADRAO
    assert F1._chave(F1.TRIAL0) == F1._chave(dict(hp=dict(M.PADRAO), dados={}))


def test_regularizacoes_so_mudam_a_regularizacao():
    chaves = {"l1", "l2", "wd", "drop_atencao", "drop_ffn", "drop_residual"}
    assert len(F2.REGULARIZACAO) >= 8
    for rot, c in F2.REGULARIZACAO.items():
        assert c["dados"] == F1.TRIAL0["dados"], rot
        mudou = {k for k in set(c["hp"]) | set(F1.TRIAL0["hp"]) if c["hp"].get(k) != F1.TRIAL0["hp"].get(k)}
        assert mudou and mudou <= chaves, (rot, mudou)


def test_curvas_de_treino_e_parada():
    c = dict(hp=HP, dados={})
    cv = F2.curvas(c, k=0, epocas=4, dispositivo="cpu")
    assert len(cv["ce_treino"]) == len(cv["ce_parada"]) == len(cv["ks_parada"]) == 4
    assert cv["melhor_epoca"] == int(np.argmin(cv["ce_parada"])) + 1
    d = F1.divs()[0]
    assert cv["n_treino"] == len(d["ajuste"]) and cv["n_parada"] == len(d["parada"])     # treino sem as copias do oversampling
