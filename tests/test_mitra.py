"""Testes de experimentos/mitra.ipynb sem o modelo real: nem torch nem autogluon sao necessarios.

Um classificador falso substitui o MitraClassifier, com a assinatura do autogluon.tabular 1.6.3.
O que depende do modelo de verdade (previsoes, bfloat16, tempo, memoria) so se mede na GPU.
"""
import ast
import gc
import json
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import optuna
import pytest

import comum
from tests.test_comum import _dobras_sinteticas

NOTEBOOK = Path(__file__).resolve().parent.parent / "experimentos" / "mitra.ipynb"
ARGUMENTOS_INIT = {
    "model_type", "n_estimators", "device", "fine_tune", "fine_tune_steps", "metric", "hf_model", "patience", "lr",
    "warmup_steps", "shuffle_classes", "shuffle_features", "use_random_transforms", "random_mirror_regression",
    "random_mirror_x", "seed", "verbose",
}


def _fontes():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _fonte(trecho):
    return next(f for f in _fontes() if trecho in f)


class MitraFalso:
    """Registra o que recebeu e devolve probabilidades pela primeira coluna, em (n, 2) como o real."""

    instancias = []
    contexto_final = 8192

    def __init__(self, **kwargs):
        assert set(kwargs) <= ARGUMENTOS_INIT, set(kwargs) - ARGUMENTOS_INIT
        self.kwargs = kwargs
        MitraFalso.instancias.append(self)

    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        self.fit_args = (X, y, X_val, y_val)
        self.trainers = [types.SimpleNamespace(
            cfg=types.SimpleNamespace(hyperparams={"max_samples_support": MitraFalso.contexto_final}))]
        return self

    def predict_proba(self, X):
        assert self.fit_args is not None
        p = (np.asarray(X)[:, 0] + 3) / 6
        return np.column_stack([1 - p, p])


@pytest.fixture
def ns():
    """Namespace com as celulas de parametros de modelo do notebook, sobre o Mitra falso."""
    MitraFalso.instancias = []
    MitraFalso.contexto_final = 8192
    torch = types.SimpleNamespace(
        manual_seed=lambda s: None,
        cuda=types.SimpleNamespace(manual_seed_all=lambda s: None, empty_cache=lambda: None),
    )
    espaco = dict(gc=gc, np=np, comum=comum, torch=torch, MitraClassifier=MitraFalso, HF_MODEL="autogluon/mitra-classifier-2",
                  CONTEXTO_PADRAO=8192)
    exec(_fonte("def ajustador"), espaco)
    espaco["folds_por_oversampling"] = {True: _dobras_sinteticas(), False: _dobras_sinteticas(oversampling=False)}
    return espaco


def test_notebook_nunca_executado_nao_guarda_saidas():
    """Enquanto o modelo real nao rodar, nenhum numero pode aparecer como se fosse medido."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == []


def test_primeira_celula_para_sem_gpu_antes_de_importar_o_autogluon(monkeypatch):
    falso = types.ModuleType("torch")
    falso.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", falso)
    monkeypatch.setattr(sys, "path", list(sys.path))
    fonte = _fonte("is_available")
    assert fonte.index("is_available") < fonte.index("import autogluon")
    with pytest.raises(RuntimeError, match="GPU"):
        exec(fonte, {})


def test_notebook_nao_usa_o_teste_nem_grava_resultado_final():
    nomes = set()
    for fonte in _fontes():
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.Name):
                nomes.add(no.id)
            elif isinstance(no, ast.Attribute):
                nomes.add(no.attr)
    assert not nomes & {"final", "salvar_resultado"}


def test_trial_0_e_a_configuracao_do_plano(ns):
    assert ns["PADRAO"] == dict(n_estimators=1, random_mirror_x=True, shuffle_features=False, shuffle_classes=False,
                                oversampling_contexto=True, fine_tune=False)
    assert ns["FIXOS"] == dict(warmup_steps=10, patience=10, use_random_transforms=False, metric="log_loss")


def test_fit_recebe_validacao_da_dobra_e_rotulos_inteiros(ns):
    dobra = ns["folds_por_oversampling"][True][0]
    ns["ajustador"](ns["PADRAO"])(dobra)
    modelo = MitraFalso.instancias[-1]
    X, y, X_val, y_val = modelo.fit_args
    assert X is dobra[0] and X_val is dobra[2]
    assert y.dtype.kind == y_val.dtype.kind == "i"
    np.testing.assert_array_equal(y, dobra[1])
    assert modelo.kwargs["hf_model"] == "autogluon/mitra-classifier-2"
    assert modelo.kwargs["device"] == "cuda"
    assert modelo.kwargs["seed"] == comum.SEMENTE
    assert "oversampling_contexto" not in modelo.kwargs


def test_prever_devolve_a_probabilidade_da_classe_1(ns):
    dobra = ns["folds_por_oversampling"][True][0]
    prever = ns["ajustador"](ns["PADRAO"])(dobra)
    np.testing.assert_allclose(prever(dobra[2]), (dobra[2][:, 0] + 3) / 6)


def test_trial_completo_passa_por_avaliar_cv_nos_dois_tipos_de_contexto(ns):
    for oversampling, com_correcao in [(True, True), (False, False)]:
        hp = dict(ns["PADRAO"], oversampling_contexto=oversampling)
        r = comum.avaliar_cv(ns["ajustador"](hp), ns["folds_por_oversampling"][oversampling])
        assert len(r["prob_oof"]) == 500
        parte = r["perdas_dobras"][0]["validacao"]
        assert (parte["ce_corr"] != parte["ce"]) == com_correcao


@pytest.mark.filterwarnings("ignore:Target and region quantiles")  # o Mitra falso da o mesmo KS em todo trial
def test_estudo_real_do_optuna_com_trial_0_e_parametros_condicionais(ns):
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    estudo.enqueue_trial(ns["PADRAO"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo.optimize(ns["objetivo"], n_trials=12)
    assert estudo.trials[0].params == ns["PADRAO"]
    com_fine_tune = [t for t in estudo.trials if t.params["fine_tune"]]
    sem_fine_tune = [t for t in estudo.trials if not t.params["fine_tune"]]
    assert com_fine_tune and sem_fine_tune
    for t in com_fine_tune:
        assert 1e-6 <= t.params["lr"] <= 1e-4
        assert t.params["fine_tune_steps"] in range(10, 101, 10)
    assert all("lr" not in t.params and "fine_tune_steps" not in t.params for t in sem_fine_tune)
    assert {m.kwargs["n_estimators"] for m in MitraFalso.instancias} <= {1, 2, 4, 8}
    resumo = comum.resumir_estudo(estudo, 12, 1.0)
    assert resumo["por_perda"] == {}
    assert set(resumo["melhor_por_categoria"]["fine_tune"]) <= {"True", "False"}
    json.dumps(resumo)


def test_contexto_reduzido_por_falta_de_memoria_vira_erro(ns):
    MitraFalso.contexto_final = 4096
    with pytest.raises(RuntimeError, match="4096"):
        ns["ajustador"](ns["PADRAO"])(ns["folds_por_oversampling"][True][0])


def test_modelo_da_dobra_anterior_e_liberado_antes_do_proximo_ajuste():
    """Dois modelos vivos na GPU fariam o AutoGluon cortar o contexto sem avisar."""
    vivos_a_cada_ajuste = []
    referencias = []

    class Modelo:
        pass

    def ajustar(dobra):
        gc.collect()
        vivos_a_cada_ajuste.append(sum(r() is not None for r in referencias))
        modelo = Modelo()
        referencias.append(weakref.ref(modelo))
        return lambda X: (np.asarray(X)[:, 0] + 3) / 6 + 0 * id(modelo)

    comum.avaliar_cv(ajustar, _dobras_sinteticas())
    assert vivos_a_cada_ajuste == [0, 0, 0, 0, 0]
