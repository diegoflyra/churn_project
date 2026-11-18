"""Gradient boosting (experimentos/gradient_boosting/modelo.py), em dados sinteticos: nao precisa da base.

Uso: python -m pytest experimentos/gradient_boosting
"""
import importlib.util
import os
import sys

import numpy as np
import optuna
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import comum  # noqa: E402


def _carregar(nome, arquivo):
    """Carrega pelo caminho: cada experimento tem um modelo.py, e `import modelo` pegaria o primeiro importado."""
    spec = importlib.util.spec_from_file_location(nome, os.path.join(os.path.dirname(__file__), arquivo))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


M = _carregar("gradient_boosting_modelo", "modelo.py")


def _dados(n=600, semente=0):
    """Mesmas colunas numericas do comum, mais duas binarias; o churn depende de tenure e da fibra."""
    rng = np.random.default_rng(semente)
    tenure = rng.integers(0, 73, n)
    mensal = rng.uniform(18, 120, n)
    fibra = rng.integers(0, 2, n)
    X = pd.DataFrame({"tenure": tenure, "MonthlyCharges": mensal, "TotalCharges": tenure * mensal,
                      "InternetService_Fiber optic": fibra, "Partner": rng.integers(0, 2, n)})
    logit = -1.0 - 0.05 * tenure + 1.2 * fibra + rng.normal(0, 0.5, n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


HP = dict(M.PADRAO, max_depth=2, learning_rate=0.1)


def test_padrao_cabe_no_espaco_de_busca():
    assert M.sugerir(optuna.trial.FixedTrial(M.PADRAO)) == M.PADRAO
    st = optuna.create_study(direction="maximize")
    st.enqueue_trial(M.PADRAO)
    t = st.ask()
    M.sugerir(t)
    for nome, dist in t.distributions.items():
        assert dist._contains(dist.to_internal_repr(M.PADRAO[nome])), nome


@pytest.mark.parametrize("loss", ["log_loss", "exponential"])
def test_a_margem_da_parada_e_a_do_modelo(loss):
    """A Parada usa um metodo interno do scikit-learn (_raw_predict_init): a margem que ela acumula tem de bater com a
    decision_function do modelo com o mesmo numero de arvores, e a perda com a do proprio scikit-learn."""
    X, y = _dados()
    hp = dict(HP, loss=loss, subsample=0.8, max_features=0.6)
    Xa, ya, Xp, yp = X.values[:400], y.values[:400], X.values[400:], y.values[400:]
    par = M.Parada(loss, Xp.astype(float), yp, paciencia=10 ** 6)
    m = M._modelo(hp, 3, 30).fit(Xa, ya, monitor=par)
    assert len(par.perdas) == 30
    assert np.allclose(par.margem, m.decision_function(Xp), atol=1e-10)
    referencia = m._loss(yp.astype(float), m.decision_function(Xp))
    assert np.isclose(par.perdas[-1], referencia)


@pytest.mark.parametrize("loss", ["log_loss", "exponential"])
def test_parada_e_retreino_usam_o_numero_de_arvores_encontrado(loss):
    X, y = _dados()
    hp = dict(HP, loss=loss)
    m, k = M.treinar(hp, X.values[:400], y.values[:400], 0, X.values[400:], y.values[400:])
    assert 1 <= k < M.TETO_ARVORES and m.n_estimators_ == k + M.PACIENCIA
    m2, k2 = M.treinar(hp, X.values, y.values, 0, arvores=k)
    assert k2 == k and m2.n_estimators_ == k
    p = M.prever(m2, X.values, k)
    assert p.shape == (len(X),) and np.all((p > 0) & (p < 1))
    assert comum.ks(y.values, p) > 0.2
    assert np.allclose(M.prever(m, X.values, k), list(m.staged_predict_proba(X.values))[k - 1][:, 1])


def test_parada_escolhe_a_arvore_de_menor_perda():
    X, y = _dados()
    _, k = M.treinar(HP, X.values[:400], y.values[:400], 0, X.values[400:], y.values[400:])
    par = M.Parada(HP["loss"], X.values[400:].astype(float), y.values[400:], paciencia=10 ** 6)
    M._modelo(HP, 0, k + M.PACIENCIA).fit(X.values[:400], y.values[:400], monitor=par)
    assert int(np.argmin(par.perdas)) + 1 == k


def test_treinar_exige_parada_ou_numero_de_arvores():
    X, y = _dados(100)
    with pytest.raises(ValueError):
        M.treinar(HP, X.values, y.values, 0)
    with pytest.raises(ValueError):
        M.treinar(HP, X.values, y.values, 0, X.values, y.values, arvores=5)


def test_a_validacao_da_dobra_nao_muda_o_modelo():
    """Os rotulos da validacao nem chegam ao ajustar_prever; aqui, nem os valores de X_val mudam o que foi treinado."""
    X, y = _dados()
    treino, val = np.arange(450), np.arange(450, 600)
    ajuste, parada = np.arange(360), np.arange(360, 450)
    f = M.ajustar_prever_de(HP)
    pv1, (y1, ptr1) = f(X.iloc[treino], y.iloc[treino], X.iloc[val], ajuste, parada, 7)
    X_outro = X.iloc[val].copy()
    X_outro["tenure"] = X_outro["tenure"][::-1].values
    pv2, (y2, ptr2) = f(X.iloc[treino], y.iloc[treino], X_outro, ajuste, parada, 7)
    assert np.array_equal(ptr1, ptr2) and np.array_equal(y1, y2)
    assert not np.array_equal(pv1, pv2)
    assert len(f.arvores) == 2 and f.arvores[0] == f.arvores[1]


def test_medida_no_placar_com_oversampling_e_features():
    X, y = _dados()
    f = M.ajustar_prever_de(HP, features=("cliente_novo", "delta_preco"), over=False)
    m = comum.medir_placar(X, y, f, threads=4)
    assert m["oof"].shape == (3, len(y)) and len(m["ks_dobras"]) == 15 and len(f.arvores) == 15
    assert 0.2 < m["ks_kfold"] <= 1 and "mse_treino" in m
    sequencial = comum.medir_placar(X, y, M.ajustar_prever_de(HP, features=("cliente_novo", "delta_preco"), over=False))
    assert np.array_equal(m["oof"], sequencial["oof"])                   # threads nao mudam o resultado


def test_final_devolve_uma_previsao_por_semente():
    X, y = _dados()
    r = M.final(HP, X.iloc[:500], y.iloc[:500], X.iloc[500:], n_sementes=2)
    assert len(r["p_sementes"]) == 2 and all(len(p) == 100 for p in r["p_sementes"])
    assert len(r["arvores"]) == 2
