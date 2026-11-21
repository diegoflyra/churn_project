"""FT-Transformer, fase 4: as features que ajudaram o MLP (cliente_novo, delta_preco, tenure_1).

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase4.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import modelo as M, fase1 as F1, fase2 as F2, fase4 as F4
import comum
import numpy as np, pytest
from test_fase1 import _base


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch):
    monkeypatch.setattr(F1, "_dados", lambda: _base())
    F1.divs.cache_clear()
    yield
    F1.divs.cache_clear()


def test_as_tres_features_sao_as_do_mlp_adotado():
    """As mesmas do campeao do MLP (extras de fase4 e de s6_rmsprop_ks_p50 em resultados/placar.jsonl)."""
    linhas = {r["rotulo"]: r for r in comum.ler_placar(os.path.join(os.path.dirname(__file__), "..", "..", "resultados"))}
    for rot in ("fase4", "s6_rmsprop_ks_p50"):
        assert sorted(linhas[rot]["config"]["extras"]) == sorted(F4.TRES)
    assert set(F4.TRES) <= set(comum.FEATURES)


def test_grade_so_muda_os_dados():
    g = F4.grade_features()
    assert set(F4.BASES) <= {b for b, _ in g.values()}
    for rot, (b, c) in g.items():
        base = F4.BASES[b]
        assert c["hp"] == base["hp"], rot
        assert c["dados"]["features"], rot
        assert set(c["dados"]) - {"features", "over"} == set(), rot
        assert set(c["dados"]["features"]) <= set(F4.TRES), rot
    assert g["f_r_combo_moderado_tres_sem_over"][1]["dados"]["over"] is False
    assert len({F1._chave(c) for _, c in g.values()} | {F1._chave(b) for b in F4.BASES.values()}) == len(g) + len(F4.BASES)


def test_features_viram_tokens_e_a_rede_treina():
    """Cada feature nova vira um token; delta_preco (nao binaria) entra no scaler."""
    Xtr, Xte, ytr, _ = _base()
    X0, _, _ = comum.preparar(Xtr, ytr, Xte)
    X3, y3, Xv3 = comum.preparar(Xtr, ytr, Xte, features=F4.TRES)
    assert X3.shape[1] == X0.shape[1] + 3
    assert abs(X3["delta_preco"].mean()) < .2 and set(np.unique(X3["tenure_1"])) <= {0, 1}
    c = F4.grade_features()["f_trial0_tres"][1]
    f = lambda a: np.asarray(a, dtype="float32")
    _, ep = M.treinar(f(X3), f(y3), dict(c["hp"], paciencia=2), 7, f(Xv3), f(_base()[3]))
    assert 1 <= ep <= M.MAX_EPOCAS
