"""Fase 6 do MLP: criterio de parada (perda ou KS) x paciencia, sobre as versoes do topo do placar.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/mlp/test_fase6.py
"""
import os, sys, json
os.environ.setdefault("CHURN_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import comum, modelo as M, fase4 as F4, fase5 as F5, fase6 as F6
import numpy as np, pytest
from test_fase4 import _base

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch):
    monkeypatch.setattr(F4, "_dados", _base)
    F5.divs.cache_clear()
    F5.usar_pilha(None)
    yield
    F5.divs.cache_clear()


def test_bases_sao_as_do_placar():
    """As bases precisam ter a mesma chave das linhas ja medidas, para entrarem no placar sem treinar de novo."""
    linhas = {r["rotulo"]: r for r in comum.ler_placar(os.path.join(RAIZ, "resultados")) if r["modelo"] == "mlp"}
    for nome, c in F6.BASES.items():
        assert F5._chave_cfg(c) == F5._chave_cfg(linhas[nome]["config"]), nome


def test_grade_so_muda_parada_e_paciencia():
    g = F6.grade_parada()
    assert len(g) == len(F6.BASES) * len(F6.PARADAS) * len(F6.PACIENCIAS)
    for rot, (base, parada, pac, c) in g.items():
        b = F6.BASES[base]
        if parada == "perda" and pac == b["hp"]["paciencia"]:
            assert rot == base and c is b
            continue
        assert c["dados"] == b["dados"] and c["extras"] == b["extras"]
        mudou = {k for k in set(c["hp"]) | set(b["hp"]) if c["hp"].get(k) != b["hp"].get(k)}
        assert mudou <= {"parada", "paciencia"}, (rot, mudou)
        assert c["hp"]["paciencia"] == pac and c["hp"].get("parada", "perda") == parada


def test_parada_pelo_ks_escolhe_a_epoca_de_maior_ks(monkeypatch):
    """Paciencia maior que o limite de epocas: a epoca devolvida e a de maior KS na separacao de parada."""
    monkeypatch.setattr(M, "MAX_EPOCAS", 8)
    Xtr, Xte, ytr, yte = _base()
    mu, sd = Xtr.mean(), Xtr.std() + 1e-6
    f = lambda a: np.asarray(a, dtype="float32")
    X, Xv, y, yv = f((Xtr - mu) / sd), f((Xte - mu) / sd), f(ytr), f(yte)
    hp = dict(F6.BASES["fase4"]["hp"], parada="ks", paciencia=100)
    ks = []
    _, ep = M.treinar(X, y, hp, 3, Xv, yv, ao_fim_epoca=lambda e, m: ks.append(comum.ks(yv, M.prever(m, Xv))))
    assert len(ks) == 8 and ep == int(np.argmax(ks)) + 1
