"""FT-Transformer, fase 3: criterio de parada (perda ou KS) x paciencia.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_fase3.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import modelo as M, fase1 as F1, fase2 as F2, fase3 as F3
import comum
import numpy as np, pytest
from test_fase1 import _base, HP


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch):
    monkeypatch.setattr(F1, "_dados", lambda: _base())
    F1.divs.cache_clear()
    yield
    F1.divs.cache_clear()


def _xy():
    Xtr, Xte, ytr, yte = _base()
    f = lambda a: np.asarray(a, dtype="float32")
    return f(Xtr), f(ytr), f(Xte), f(yte)


def test_parada_pelo_ks_escolhe_a_epoca_de_maior_ks():
    """Com paciencia maior que o limite de epocas, a epoca devolvida e a de maior KS na separacao de parada."""
    X, y, Xv, yv = _xy()
    ks = []
    _, ep = M.treinar(X, y, dict(HP, parada="ks", paciencia=100), 7, Xv, yv,
                      ao_fim_epoca=lambda e, m: ks.append(comum.ks(yv, M.prever(m, Xv))))
    assert len(ks) == M.MAX_EPOCAS
    assert ep == int(np.argmax(ks)) + 1


def test_parada_pela_perda_continua_a_padrao():
    X, y, Xv, yv = _xy()
    a = M.treinar(X, y, HP, 7, Xv, yv)[1]
    b = M.treinar(X, y, dict(HP, parada="perda"), 7, Xv, yv)[1]
    assert a == b
    assert "parada" not in M.PADRAO                      # o trial 0 ja medido continua com a mesma chave


def test_grade_de_parada():
    g = F3.grade_parada()
    assert len(g) == len(F3.BASES) * len(F3.PARADAS) * len(F3.PACIENCIAS)
    for rot, (base, parada, pac, c) in g.items():
        b = F3.BASES[base]
        if parada == "perda" and pac == b["hp"]["paciencia"]:
            assert rot == base and c is b                 # a celula da base ja esta no placar: nao mede de novo
            continue
        assert c["dados"] == b["dados"]
        mudou = {k for k in set(c["hp"]) | set(b["hp"]) if c["hp"].get(k) != b["hp"].get(k)}
        assert mudou <= {"parada", "paciencia"}, (rot, mudou)
        assert c["hp"]["paciencia"] == pac and c["hp"].get("parada", "perda") == parada
    assert F3.BASES["trial0"] is F1.TRIAL0 and F3.BASES["r_drop_medio"] is F2.REGULARIZACAO["r_drop_medio"]


def test_onde_para_simula_a_parada_numa_curva():
    ce = [.5, .45, .47, .46, .44, .48, .49, .50]
    assert F3.onde_para(ce, paciencia=2) == 2             # 0,47 e 0,46 nao melhoram 0,45: para com a 2
    assert F3.onde_para(ce, paciencia=3) == 5
    ks = [.40, .50, .45, .52, .51]
    assert F3.onde_para(ks, paciencia=1, maior=True) == 2
    assert F3.onde_para(ks, paciencia=5, maior=True) == 4


def test_figura_ks(tmp_path):
    cv = dict(ce_treino=[.5, .4, .3], ce_parada=[.5, .45, .5], ks_parada=[.4, .5, .45], melhor_epoca=2)
    F3.figura_ks({"a": cv, "b": cv}, str(tmp_path / "ks.png"), paciencias=(1, 2))
    assert (tmp_path / "ks.png").stat().st_size > 0
