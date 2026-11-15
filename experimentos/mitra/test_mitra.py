"""Mitra (experimentos/mitra) sem o modelo real: nem torch nem autogluon sao necessarios.

Um classificador falso substitui o MitraClassifier, com a assinatura do autogluon.tabular 1.6.3. O que depende do modelo
de verdade (previsoes, precisao, tempo, memoria) so se mede na GPU, na etapa de medidas do notebook.
Uso: python -m pytest experimentos/mitra
"""
import importlib.util
import os
import sys
import types

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


M = _carregar("mitra_modelo", "modelo.py")
INST = _carregar("mitra_instalar", "instalar.py")

ARGUMENTOS_INIT = {  # MitraBase.__init__ do autogluon.tabular 1.6.3
    "model_type", "n_estimators", "device", "fine_tune", "fine_tune_steps", "metric", "hf_model", "patience", "lr",
    "warmup_steps", "shuffle_classes", "shuffle_features", "use_random_transforms", "random_mirror_regression",
    "random_mirror_x", "seed", "verbose",
}


class MitraFalso:
    """Registra o que recebeu. Preve pela primeira coluna, em (n, 2) como o real; o contexto e o X do fit."""

    instancias = []
    contexto_final = M.CONTEXTO_MAXIMO

    def __init__(self, **kwargs):
        assert set(kwargs) <= ARGUMENTOS_INIT, set(kwargs) - ARGUMENTOS_INIT
        self.kwargs = kwargs
        MitraFalso.instancias.append(self)

    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        assert X.dtype == np.float32 and X_val.dtype == np.float32
        assert set(np.unique(y)) <= {0, 1} and np.asarray(y).dtype.kind == "i"
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
        cfg = types.SimpleNamespace(hyperparams={"max_samples_support": MitraFalso.contexto_final})
        self.trainers = [types.SimpleNamespace(cfg=cfg)]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(np.asarray(X)[:, 0]))  # churn cai com o tenure (primeira coluna)
        return np.column_stack([1 - p, p])


class TorchFalso:
    class cuda:
        @staticmethod
        def is_available():
            return False

    sementes = []

    @staticmethod
    def manual_seed(s):
        TorchFalso.sementes.append(s)


@pytest.fixture(autouse=True)
def falsos(monkeypatch):
    MitraFalso.instancias = []
    MitraFalso.contexto_final = M.CONTEXTO_MAXIMO
    M.usar_classe(MitraFalso)
    monkeypatch.setattr(M, "_torch", lambda: TorchFalso)
    yield
    M.usar_classe(None)


def _dados(n=500, semente=0):
    rng = np.random.default_rng(semente)
    tenure = rng.integers(0, 73, n)
    mensal = rng.uniform(18, 120, n)
    X = pd.DataFrame({"tenure": tenure, "MonthlyCharges": mensal, "TotalCharges": tenure * mensal,
                      "Partner": rng.integers(0, 2, n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(1 + 0.05 * tenure - 0.02 * mensal))).astype(int))
    return X, y


# ---------------- espaco ----------------
def test_padrao_cabe_no_espaco_e_fine_tuning_so_abre_lr_e_passos_quando_ligado():
    assert M.sugerir(optuna.trial.FixedTrial(M.PADRAO)) == M.PADRAO
    ft = dict(M.PADRAO, fine_tune=True, lr=3e-6, fine_tune_steps=50)
    assert M.sugerir(optuna.trial.FixedTrial(ft)) == ft
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(30):
        t = st.ask()
        hp = M.sugerir(t)
        assert ("lr" in hp) == ("fine_tune_steps" in hp) == hp["fine_tune"]
        st.tell(t, 0.0)


# ---------------- precisao ----------------
def test_precisao_automatica_e_pedida():
    icl, ft = M.PADRAO, dict(M.PADRAO, fine_tune=True, lr=1e-5, fine_tune_steps=10)
    assert M.precisao_para(icl, "auto", nativo=True) == "bfloat16"
    assert M.precisao_para(icl, "auto", nativo=False) == "float32"
    assert M.precisao_para(ft, "auto", nativo=False) == "float32"
    assert M.precisao_para(icl, "float16") == "float16"
    assert M.precisao_para(ft, "float16") == "float16"                   # com o escalonamento do gradiente ligado
    assert M.precisao_para(ft, "float32", nativo=True) == "float32"
    with pytest.raises(ValueError):
        M.precisao_para(icl, "fp8")


def test_subclasse_troca_a_precisao_fixada_pelo_autogluon():
    class Base:
        def _create_config(self, task, dim_output, time_limit=None):
            return types.SimpleNamespace(hyperparams={"precision": "bfloat16", "max_epochs": 0}), "Tab2D"

    C = M.com_precisao(Base)
    m = C()
    m.precisao = "float32"
    cfg, modelo = m._create_config("classification", 10)
    assert cfg.hyperparams == {"precision": "float32", "max_epochs": 0} and modelo == "Tab2D"
    assert issubclass(C, Base)


def test_float16_liga_o_escalonamento_do_gradiente_sem_o_piso_do_autogluon():
    class Base:
        def _create_config(self, task, dim_output, time_limit=None):
            hp = {"precision": "bfloat16", "grad_scaler_enabled": False, "grad_scaler_scale_min": 65536.0}
            return types.SimpleNamespace(hyperparams=hp), "Tab2D"

    m = M.com_precisao(Base)()
    m.precisao = "float16"
    cfg, _ = m._create_config("classification", 10)
    assert cfg.hyperparams == {"precision": "float16", "grad_scaler_enabled": True, "grad_scaler_scale_min": 1.0}
    m.precisao = "float32"
    cfg, _ = m._create_config("classification", 10)
    assert cfg.hyperparams["grad_scaler_enabled"] is False


def test_treinador_solta_otimizador_e_gradientes_e_guarda_a_escala():
    eventos = []

    class Escala:
        def __init__(self, ligado):
            self.ligado = ligado

        def is_enabled(self):
            return self.ligado

        def get_scale(self):
            return 1024.0

    class Modelo:
        def zero_grad(self, set_to_none=False):
            eventos.append(("zero_grad", set_to_none))

    class Base:
        def __init__(self, ligado):
            self.scaler, self.model, self.optimizer = Escala(ligado), Modelo(), "adamw"

        def train(self, *args):
            eventos.append(("train", args))
            return "ok"

        def post_fit_optimize(self):
            self.optimizer = self.scaler = None

    T = M.que_libera(Base)
    t = T(True)
    assert t.train(1, 2) == "ok" and eventos == [("train", (1, 2)), ("zero_grad", True)]
    assert t.optimizer is None and t.escala_final == 1024.0
    t2 = T(False)
    t2.train()
    assert t2.escala_final is None and t2.optimizer is None


# ---------------- placar ----------------
def _dobra(X, y):
    treino, val = np.arange(400), np.arange(400, 500)
    ajuste, parada = np.arange(320), np.arange(320, 400)
    return X.iloc[treino], y.iloc[treino], X.iloc[val], ajuste, parada


def test_sem_fine_tuning_o_contexto_e_o_treino_inteiro_e_a_validacao_nao_entra():
    X, y = _dados()
    Xt, yt, Xv, ajuste, parada = _dobra(X, y)
    f = M.ajustar_prever_de(M.PADRAO, precisao="float16")
    p = f(Xt, yt, Xv, ajuste, parada, 7)
    m = MitraFalso.instancias[-1]
    n_over = 2 * int((yt == 0).sum())
    assert len(m.X) == n_over and m.y.mean() == 0.5                    # treino inteiro da dobra, com oversampling
    assert len(m.X_val) == min(M.N_AVALIACAO, len(parada)) and set(np.unique(m.y_val)) == {0, 1}
    cont = {tuple(r) for r in m.X}
    assert all(tuple(r) in cont for r in m.X_val)                       # a avaliacao do passo 0 sai da parada, no contexto
    assert p.shape == (len(Xv),)
    assert m.kwargs["hf_model"] == M.HF_MODEL and m.kwargs["fine_tune"] is False and "over" not in m.kwargs
    assert m.precisao == "float16" and f.precisao == "float16"
    assert {k: m.kwargs[k] for k in M.FIXOS} == M.FIXOS and m.kwargs["seed"] == 7


def test_avaliacao_do_passo_0_e_uma_amostra_estratificada_da_parada(monkeypatch):
    X, y = _dados()
    Xt, yt, Xv, ajuste, parada = _dobra(X, y)
    monkeypatch.setattr(M, "N_AVALIACAO", 40)
    M.ajustar_prever_de(M.PADRAO, precisao="float32")(Xt, yt, Xv, ajuste, parada, 7)
    m = MitraFalso.instancias[-1]
    assert len(m.X_val) == 40 and abs(m.y_val.mean() - yt.iloc[parada].mean()) < 0.05
    sub = M.amostra_estratificada(parada, yt, 40, 7)
    assert set(sub) <= set(parada) and np.array_equal(sub, M.amostra_estratificada(parada, yt, 40, 7))


def test_sem_oversampling_o_contexto_fica_com_um_cliente_por_linha():
    X, y = _dados()
    Xt, yt, Xv, ajuste, parada = _dobra(X, y)
    M.ajustar_prever_de(dict(M.PADRAO, over=False), precisao="float32")(Xt, yt, Xv, ajuste, parada, 7)
    m = MitraFalso.instancias[-1]
    assert len(m.X) == len(Xt) and np.isclose(m.y.mean(), yt.mean())


def test_com_fine_tuning_o_checkpoint_sai_da_parada_e_o_contexto_e_o_ajuste():
    X, y = _dados()
    Xt, yt, Xv, ajuste, parada = _dobra(X, y)
    hp = dict(M.PADRAO, fine_tune=True, lr=1e-5, fine_tune_steps=20)
    M.ajustar_prever_de(hp, precisao="float32")(Xt, yt, Xv, ajuste, parada, 7)
    m = MitraFalso.instancias[-1]
    assert len(m.X) == 2 * int((yt.iloc[ajuste] == 0).sum())          # so o ajuste, com oversampling
    assert len(m.X_val) == len(parada) and np.array_equal(m.y_val, yt.iloc[parada].values)
    cont = {tuple(r) for r in m.X}
    assert not any(tuple(r) in cont for r in m.X_val)                   # a parada fica fora do contexto
    assert m.kwargs["fine_tune"] is True and m.kwargs["lr"] == 1e-5 and m.kwargs["fine_tune_steps"] == 20


def test_contexto_reduzido_por_falta_de_memoria_vira_erro():
    X, y = _dados()
    MitraFalso.contexto_final = M.CONTEXTO_MAXIMO // 2
    with pytest.raises(RuntimeError, match="memoria"):
        M.ajustar_prever_de(M.PADRAO, precisao="float32")(*_dobra(X, y), 7)


def test_medida_completa_no_placar_e_no_estudo(tmp_path):
    X, y = _dados()
    m = comum.medir_placar(X, y, M.ajustar_prever_de(M.PADRAO, precisao="float32"))
    assert m["oof"].shape == (3, len(y)) and len(m["ks_dobras"]) == 15 and m["ks_kfold"] > 0.1
    assert len(MitraFalso.instancias) == 15
    st = comum.estudo_placar(M.MODELO, 2, M.sugerir, lambda hp: M.ajustar_prever_de(hp, precisao="float32"), M.PADRAO,
                             X, y, pasta=str(tmp_path))
    assert st.trials[0].params == M.PADRAO and len(st.trials) == 2


def test_final_usa_uma_separacao_por_semente():
    X, y = _dados()
    r = M.final(M.PADRAO, X.iloc[:400], y.iloc[:400], X.iloc[400:], n_sementes=2, precisao="float32")
    assert len(r["p_sementes"]) == 2 and all(len(p) == 100 for p in r["p_sementes"]) and r["precisao"] == "float32"
    assert [i.kwargs["seed"] for i in MitraFalso.instancias] == [comum.SEMENTE, comum.SEMENTE + 1]


# ---------------- instalacao ----------------
def _relatorio(*pacotes):
    return {"install": [{"metadata": {"name": n, "version": v}} for n, v in pacotes]}


def test_instalacao_recusa_trocar_o_torch_ja_instalado():
    instalado = {"torch": "2.15.0", "pandas": "2.2.3"}.get
    novos, trocas = INST.avaliar_relatorio(_relatorio(("autogluon.tabular", "1.6.3"), ("torch", "2.13.0")), instalado)
    assert novos["torch"] == "2.13.0" and trocas == {"torch": ("2.15.0", "2.13.0")}
    _, trocas = INST.avaliar_relatorio(_relatorio(("autogluon.tabular", "1.6.3"), ("nvidia-cudnn-cu12", "9.1")), instalado)
    assert list(trocas) == ["torch (pacotes nvidia-*)"]


def test_instalacao_aceita_quando_o_torch_fica():
    instalado = {"torch": "2.12.0"}.get
    relatorio = _relatorio(("autogluon.tabular", "1.6.3"), ("einx", "0.3"), ("Pandas", "2.3.3"))
    novos, trocas = INST.avaliar_relatorio(relatorio, instalado)
    assert trocas == {} and novos["pandas"] == "2.3.3"
