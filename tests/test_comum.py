"""Testes do comum.py. Os que precisam da base real pulam quando nao ha rede nem cache."""
import copy
import inspect
import json
import os
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from scipy.stats import ks_2samp

import comum

RAIZ = Path(__file__).resolve().parent.parent

COLUNAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "total_addons", "PaymentMethod_Automatic",
]


@pytest.fixture(scope="session")
def csv():
    if os.environ.get("CHURN_CSV"):
        return os.environ["CHURN_CSV"]  # caminho errado aqui e erro, nao falta de rede
    try:
        return comum._caminho_csv()
    except OSError as e:  # as excecoes de rede de requests e kagglehub derivam de OSError
        pytest.skip(f"Base indisponivel: {e}")


@pytest.fixture(scope="session")
def base(csv):
    return comum.carregar_dobras(csv)


@pytest.fixture(scope="session")
def posicoes(csv):
    return comum.posicoes_dobras(csv)


# ---------------- 1. contagens e colunas ----------------
def test_divisao_treino_e_teste(csv):
    _, _, y_treino, y_teste = comum._dividir(comum.carregar_df(csv))
    assert (len(y_treino), int(y_treino.sum())) == (5282, 1402)
    assert (len(y_teste), int(y_teste.sum())) == (1761, 467)
    assert y_treino.mean() == pytest.approx(comum.TAXA_TREINO)
    assert round(comum.TAXA_TREINO, 5) == 0.26543


def test_tamanhos_das_dobras_e_do_treino_final(base):
    folds, final, _ = base
    assert len(folds) == 5
    for Xt, yt, Xv, yv in folds:
        assert len(Xt) == len(yt) == 6208
        assert yt.mean() == 0.5
        assert len(Xv) == len(yv)
        assert len(Xv) in (1056, 1057)
        assert {Xt.dtype, yt.dtype, Xv.dtype, yv.dtype} == {np.dtype("float32")}
    assert final[0].shape == (7760, 28)
    assert final[2].shape == (1761, 28)
    assert final[3].sum() == 467


def test_lista_de_colunas(base):
    assert base[2] == COLUNAS


def test_hash_do_teste_e_fixo(base, csv):
    assert comum.hash_y_teste(base[1][3]) == comum.HASH_Y_TESTE
    sem_variaveis = comum.carregar_dobras(csv, criar_variaveis=False, padronizar=False, balancear=False)
    assert comum.hash_y_teste(sem_variaveis[1][3]) == comum.HASH_Y_TESTE


# ---------------- 2. igual ao notebook ----------------
@pytest.fixture(scope="session")
def notebook(csv):
    """Executa as proprias celulas de if702-preprocessamento.ipynb, trocando so o caminho do CSV."""
    nb = json.loads((RAIZ / "if702-preprocessamento.ipynb").read_text(encoding="utf-8"))
    ns = {"display": lambda *a, **k: None}
    for celula in nb["cells"]:
        if celula["cell_type"] == "code":
            linha = f"path = {Path(csv).as_posix()!r}"
            fonte = re.sub(r"^path = .*$", lambda _, linha=linha: linha, "".join(celula["source"]), flags=re.M)
            exec(fonte, ns)
    return ns


def test_dobras_iguais_as_do_notebook(base, notebook):
    folds, _, colunas = base
    assert colunas == list(notebook["X"].columns)
    for (Xt, yt, Xv, yv), (nXt, nyt, nXv, nyv) in zip(folds, notebook["folds"], strict=True):
        np.testing.assert_allclose(Xt, nXt.values, atol=1e-6)
        np.testing.assert_allclose(Xv, nXv.values, atol=1e-6)
        np.testing.assert_array_equal(yt, nyt.values)
        np.testing.assert_array_equal(yv, nyv.values)


def test_treino_final_igual_ao_do_notebook(base, notebook):
    _, final, _ = base
    Xb, yb, Xa = notebook["preparar"](notebook["X_train"], notebook["y_train"], notebook["X_test"])
    np.testing.assert_allclose(final[0], Xb.values, atol=1e-6)
    np.testing.assert_allclose(final[2], Xa.values, atol=1e-6)
    np.testing.assert_array_equal(final[1], yb.values)
    np.testing.assert_array_equal(final[3], notebook["y_test"].values)


# ---------------- 3. KS contra scipy ----------------
@pytest.mark.parametrize("seed,casas", [(1, 1), (2, 2), (3, 3)])
def test_ks_bate_com_scipy_em_amostras_com_empates(seed, casas):
    rng = np.random.default_rng(seed)
    y = (rng.random(2000) < 0.265).astype(int)
    p = np.round(np.clip(rng.normal(0.3 + 0.25 * y, 0.2), 0, 1), casas)
    assert len(np.unique(p)) < len(p)
    assert comum.ks(y, p) == pytest.approx(ks_2samp(p[y == 1], p[y == 0]).statistic, abs=1e-12)


# ---------------- 4. dobras sem vazamento ----------------
def test_preparar_sem_balancear_nao_duplica(csv):
    X_tr, X_te, y_tr, _ = comum._dividir(comum.carregar_df(csv))
    Xb, yb, _ = comum.preparar(X_tr, y_tr, X_te, balancear=False)
    assert len(Xb) == len(X_tr)
    assert Xb.index.is_unique
    pd.testing.assert_series_equal(yb, y_tr)


def test_preparar_sem_padronizar_mantem_numericas(csv):
    X_tr, X_te, y_tr, _ = comum._dividir(comum.carregar_df(csv))
    Xb, _, Xa = comum.preparar(X_tr, y_tr, X_te, padronizar=False, balancear=False)
    pd.testing.assert_frame_equal(Xb, X_tr)
    pd.testing.assert_frame_equal(Xa, X_te)


def test_scaler_ajustado_so_no_treino(csv):
    X_tr, X_te, y_tr, _ = comum._dividir(comum.carregar_df(csv))
    Xb, _, Xa = comum.preparar(X_tr, y_tr, X_te, balancear=False)
    np.testing.assert_allclose(Xb[comum.NUMERIC_COLS].mean(), 0, atol=1e-10)
    medias = X_tr[comum.NUMERIC_COLS].mean().values
    desvios = X_tr[comum.NUMERIC_COLS].std(ddof=0).values
    np.testing.assert_allclose(Xa[comum.NUMERIC_COLS].values, (X_te[comum.NUMERIC_COLS].values - medias) / desvios)


def test_posicoes_de_validacao_nao_aparecem_no_treino(posicoes):
    assert len(posicoes) == 5
    for idx_tr, idx_val in posicoes:
        assert not set(idx_tr) & set(idx_val)
        assert len(idx_tr) + len(idx_val) == 5282
    assert sorted(np.concatenate([v for _, v in posicoes]).tolist()) == list(range(5282))


def test_linhas_originais_vem_primeiro_no_treino_balanceado(csv, base, posicoes):
    """E o fato que permite medir as perdas de treino sem copias."""
    sem_copias = comum.carregar_dobras(csv, balancear=False)[0]
    for (Xt, yt, _, _), (Xs, ys, _, _), (idx_tr, _) in zip(base[0], sem_copias, posicoes, strict=True):
        assert len(Xs) == len(idx_tr)
        np.testing.assert_array_equal(Xt[: len(idx_tr)], Xs)
        np.testing.assert_array_equal(yt[: len(idx_tr)], ys)


def test_copias_da_dobra_0(base, posicoes):
    """1.983 linhas vem do oversampling; duplicated() da 1.996 porque 13 clientes originais ja sao iguais a outros."""
    Xt = base[0][0][0]
    n = len(posicoes[0][0])
    assert len(Xt) - n == 1983
    assert int(pd.DataFrame(Xt).duplicated().sum()) == 1996
    assert int(pd.DataFrame(Xt[:n]).duplicated().sum()) == 13


def test_opcoes_de_carregar_dobras(csv):
    folds, final, colunas = comum.carregar_dobras(csv, criar_variaveis=False, balancear=False)
    assert colunas == COLUNAS[:-2]
    assert final[0].shape == (5282, 26)
    assert folds[0][0].shape[1] == 26
    with pytest.raises(TypeError):
        comum.carregar_dobras(csv, balanciar=False)


# ---------------- 5. correcao de prior ----------------
def test_corrige_prior_aceita_previsoes_fora_do_intervalo():
    q = comum.corrige_prior(np.array([-0.42, 0.0, 0.5, 1.0, 1.07]))
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(q, [0, 0, comum.TAXA_TREINO, 1, 1])


@pytest.mark.parametrize("tipo", [np.float32, np.float64])
def test_corrige_prior_nao_muda_o_ks(tipo):
    rng = np.random.default_rng(7)
    y = (rng.random(2000) < 0.265).astype(int)
    p = rng.normal(0.35 + 0.3 * y, 0.28).astype(tipo)
    assert p.min() < 0 and p.max() > 1
    assert comum.ks(y, comum.corrige_prior(p)) == pytest.approx(comum.ks(y, p), abs=1e-12)


def test_corrige_prior_e_a_forma_do_jefferson_coincidem_dentro_do_intervalo():
    p = np.linspace(0.01, 0.99, 50)
    w = 0.2654 / (1 - 0.2654)
    np.testing.assert_allclose(comum.corrige_prior(p, pi=0.2654), p * w / (p * w + (1 - p)))


# ---------------- 6. bootstrap ----------------
def _amostra(seed=11, n=1000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.265).astype(int)
    return y, np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)


def test_bootstrap_pareado_contra_si_mesmo_da_zero():
    y, p = _amostra()
    r = comum.bootstrap_pareado(y, p, p, n=100)
    assert r["diferenca"] == 0
    assert r["ic95"] == (0, 0)
    assert r["p_valor"] == 1


def test_bootstrap_pareado_detecta_diferenca_real_nos_dois_sentidos():
    y, p = _amostra()
    ruido = np.random.default_rng(5).random(len(y))
    a = comum.bootstrap_pareado(y, p, ruido, n=200)
    b = comum.bootstrap_pareado(y, ruido, p, n=200)
    assert a["diferenca"] > 0 and a["ic95"][0] > 0
    assert b["diferenca"] == -a["diferenca"]
    assert a["p_valor"] == pytest.approx(2 / 201)
    assert b["p_valor"] == pytest.approx(2 / 201)


def test_bootstrap_pareado_usa_as_mesmas_reamostras():
    """Com as mesmas reamostras, somar uma constante ao escore nao muda nenhum KS, e a diferenca fica nula."""
    y, p = _amostra()
    r = comum.bootstrap_pareado(y, p, p + 0.01, n=100)
    assert r["ic95"] == pytest.approx((0, 0), abs=1e-9)


def test_bootstrap_ks_e_reprodutivel_com_a_semente_padrao():
    y, p = _amostra()
    assert inspect.signature(comum.bootstrap_ks).parameters["seed"].default == comum.SEMENTE == 42
    baixo, alto = comum.bootstrap_ks(y, p, n=100)
    assert (baixo, alto) == comum.bootstrap_ks(y, p, n=100)
    assert (baixo, alto) == comum.bootstrap_ks(y.tolist(), p.tolist(), n=100)
    assert baixo < comum.ks(y, p) < alto


# ---------------- avaliar_cv ----------------
def _dobras_sinteticas(oversampling=True):
    """Cinco dobras que particionam 500 linhas; com oversampling, as copias vao para o fim do treino."""
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.3).astype("float32")
    X = (rng.normal(y, 1.0, size=(3, 500)).T).astype("float32")
    folds = []
    for val in np.array_split(np.arange(500), 5):
        tr = np.setdiff1d(np.arange(500), val)
        Xt, yt = X[tr], y[tr]
        if oversampling:
            pos = np.flatnonzero(yt == 1)
            extra = pos[rng.integers(0, len(pos), int((yt == 0).sum() - len(pos)))]
            Xt, yt = np.vstack([Xt, Xt[extra]]), np.concatenate([yt, yt[extra]])
        folds.append((Xt, yt, X[val], y[val]))
    return folds


def _ajustar_primeira_coluna(chamadas=None):
    def ajustar(dobra):
        def prever(X):
            if chamadas is not None:
                chamadas.append(len(X))
            return (X[:, 0] + 3) / 6
        return prever
    return ajustar


class TrialFalso:
    def __init__(self, cortar=False):
        self.cortar, self.reportes, self.perguntas, self.attrs = cortar, [], 0, {}

    def report(self, valor, passo):
        self.reportes.append((valor, passo))

    def should_prune(self):
        self.perguntas += 1
        return self.cortar

    def set_user_attr(self, nome, valor):
        self.attrs[nome] = valor


def test_avaliar_cv_mede_treino_sem_copias():
    folds = _dobras_sinteticas()
    chamadas = []
    r = comum.avaliar_cv(_ajustar_primeira_coluna(chamadas), folds)
    assert all(len(f[0]) > 400 for f in folds)
    assert chamadas == [100, 400] * 5
    Xt, yt, _, _ = folds[0]
    p = (Xt[:400, 0] + 3) / 6
    assert r["perdas_dobras"][0]["treino"]["mse"] == pytest.approx(comum.mse(yt[:400], p.astype("float64")))
    assert r["perdas_dobras"][0]["treino"]["ce_corr"] != r["perdas_dobras"][0]["treino"]["ce"]


def test_avaliar_cv_devolve_probabilidades_fora_da_dobra():
    folds = _dobras_sinteticas()
    r = comum.avaliar_cv(_ajustar_primeira_coluna(), folds)
    assert r["ks"] == pytest.approx(np.mean(r["ks_dobras"]))
    assert len(r["prob_oof"]) == len(r["y_oof"]) == 500
    np.testing.assert_array_equal(r["y_oof"], np.concatenate([f[3] for f in folds]))
    assert r["limiar"] == comum.limiar_ks(r["y_oof"], r["prob_oof"])


def test_avaliar_cv_sem_oversampling_nao_corrige_prior():
    r = comum.avaliar_cv(_ajustar_primeira_coluna(), _dobras_sinteticas(oversampling=False))
    for dobra in r["perdas_dobras"]:
        for parte in dobra.values():
            assert parte["ce_corr"] == parte["ce"]
            assert parte["mse_corr"] == parte["mse"]


def test_avaliar_cv_corta_so_depois_da_segunda_dobra():
    chamadas = []
    trial = TrialFalso(cortar=True)
    with pytest.raises(optuna.TrialPruned):
        comum.avaliar_cv(_ajustar_primeira_coluna(chamadas), _dobras_sinteticas(), trial)
    assert [passo for _, passo in trial.reportes] == [0, 1]
    assert trial.perguntas == 1
    assert len(chamadas) == 4


def test_avaliar_cv_reporta_media_acumulada_e_grava_atributos():
    trial = TrialFalso()
    r = comum.avaliar_cv(_ajustar_primeira_coluna(), _dobras_sinteticas(), trial)
    esperado = [np.mean(r["ks_dobras"][: i + 1]) for i in range(5)]
    assert [v for v, _ in trial.reportes] == pytest.approx(esperado)
    assert [passo for _, passo in trial.reportes] == list(range(5))
    assert trial.perguntas == 1
    assert trial.attrs["ks_dobras"] == r["ks_dobras"]
    assert trial.attrs["perdas_dobras"] == r["perdas_dobras"]
    assert trial.attrs["tempo_s"] > 0
    json.dumps(trial.attrs)  # o Optuna guarda os atributos como JSON


def test_avaliar_cv_recusa_subconjunto_de_dobras():
    with pytest.raises(ValueError, match="5 dobras"):
        comum.avaliar_cv(_ajustar_primeira_coluna(), _dobras_sinteticas()[:2])


def test_avaliar_cv_recusa_previsao_com_formato_errado():
    """Um vetor (n, 1) faria mse e ce propagarem por broadcast, sem erro e com valor errado."""
    def ajustar(dobra):
        return lambda X: (X[:, 0] + 3).reshape(-1, 1) / 6
    with pytest.raises(ValueError, match="vetor"):
        comum.avaliar_cv(ajustar, _dobras_sinteticas())


def test_avaliar_cv_nas_dobras_reais_usa_o_tamanho_de_posicoes_dobras(base, posicoes):
    chamadas = []
    comum.avaliar_cv(_ajustar_primeira_coluna(chamadas), base[0])
    assert chamadas[1::2] == [len(idx_tr) for idx_tr, _ in posicoes]


# ---------------- resumir_estudo ----------------
def _estudo_falso():
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    estudo.enqueue_trial({"objective": "binary:logistic", "x": 0.5})

    def objetivo(t):
        obj = t.suggest_categorical("objective", ["binary:logistic", "reg:squarederror"])
        x = t.suggest_float("x", 0, 1)
        return x + (0.1 if obj == "binary:logistic" else 0)
    estudo.optimize(objetivo, n_trials=8)
    return estudo


def test_resumir_estudo_traduz_a_perda_e_acha_melhor_por_categoria():
    estudo = _estudo_falso()
    r = comum.resumir_estudo(estudo, 8, 1.5, param_perda="objective",
                             perdas={"binary:logistic": "ce", "reg:squarederror": "mse"})
    assert r["n_trials"] == 8 and r["n_completos"] == 8 and r["n_podados"] == 0
    assert r["config_trial0"] == {"objective": "binary:logistic", "x": 0.5}
    assert r["ks_trial0"] == pytest.approx(0.6)
    assert r["melhor_ks"] == estudo.best_value == r["convergencia"][-1]
    assert r["convergencia"] == sorted(r["convergencia"])
    assert r["por_perda"]["ce"]["n"] + r["por_perda"]["mse"]["n"] == 8
    assert r["melhor_por_categoria"]["objective"]["binary:logistic"] == r["por_perda"]["ce"]["melhor"]
    assert set(r["melhor_por_categoria"]) == {"objective"}
    assert set(r["importancias"]) == {"objective", "x"}
    assert set(r) == comum._CAMPOS_ESTUDO
    json.dumps(r)


def test_resumir_estudo_sem_escolha_de_perda():
    assert comum.resumir_estudo(_estudo_falso(), 8, 1.5)["por_perda"] == {}


# ---------------- 7. esquema ----------------
@pytest.fixture
def y_falso(monkeypatch):
    """Rotulos sinteticos aceitos como o teste, para os testes de esquema nao dependerem da base."""
    y = [int(v) for v in np.random.default_rng(1).random(comum.N_TESTE) < 0.265]
    monkeypatch.setattr(comum, "HASH_Y_TESTE", comum.hash_y_teste(y))
    return y


@pytest.fixture
def resultado(y_falso, tmp_path, monkeypatch):
    monkeypatch.setattr(comum, "PASTA_RESULTADOS", tmp_path)
    perdas = dict(ce=0.5, mse=0.2, ce_corr=0.4, mse_corr=0.1)
    p = np.random.default_rng(2).random(comum.N_TESTE)
    return dict(
        versao_esquema=comum.VERSAO_ESQUEMA,
        modelo="xgboost",
        avaliacao_final="retreino",
        ambiente=dict(python="3.12.10", bibliotecas={"xgboost": "3.4.1"}, dispositivo="cpu"),
        melhor_config=dict(max_depth=6),
        ks_dobras_melhor=[0.5, 0.51, 0.52, 0.53, 0.54],
        perdas_dobras=[dict(treino=perdas, validacao=perdas) for _ in range(5)],
        estudo=dict(n_trials=5, n_completos=5, n_podados=0, por_perda={}, melhor_por_categoria={},
                    importancias={}, config_trial0={}, ks_trial0=0.5, melhor_ks=0.54,
                    convergencia=[0.5] * 5, tempo_parede_s=10.0),
        teste=dict(ks=0.5, ks_ic95=(0.45, 0.55), ks_por_semente=[0.5, 0.5], auroc=0.8, ce=0.5, mse=0.2,
                   ce_corr=0.4, mse_corr=0.1, precision=0.5, recall=0.8, f1=0.6,
                   confusao=[[1, 2], [3, 4]], limiar=0.5),
        tempos=dict(treino_final_s=1.0, predicao_teste_s=0.1),
        probabilidades_teste=p,
        probabilidades_por_semente=[p, p],
        y_teste=y_falso,
    )


def test_salvar_e_carregar_ida_e_volta(resultado, tmp_path):
    caminho = comum.salvar_resultado("xgboost", resultado)
    assert caminho == tmp_path / "xgboost.json"
    lido = comum.carregar_resultados()
    assert list(lido) == ["xgboost"]
    assert lido["xgboost"]["probabilidades_teste"] == resultado["probabilidades_teste"].tolist()
    assert lido["xgboost"]["teste"]["ks_ic95"] == [0.45, 0.55]
    assert list(tmp_path.iterdir()) == [caminho]


def test_carregar_sem_resultados_devolve_vazio(tmp_path, monkeypatch):
    monkeypatch.setattr(comum, "PASTA_RESULTADOS", tmp_path / "nao_existe")
    assert comum.carregar_resultados() == {}


def test_pasta_de_resultados_nao_depende_do_diretorio_atual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert comum.PASTA_RESULTADOS == RAIZ / "resultados"


@pytest.mark.parametrize("campo", sorted(comum._CAMPOS))
def test_esquema_rejeita_falta_de_campo(resultado, campo, tmp_path):
    del resultado[campo]
    with pytest.raises(ValueError, match=campo):
        comum.salvar_resultado("xgboost", resultado)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bloco,campos", [("estudo", comum._CAMPOS_ESTUDO), ("teste", comum._CAMPOS_TESTE),
                                          ("ambiente", comum._CAMPOS_AMBIENTE), ("tempos", comum._CAMPOS_TEMPOS)])
def test_esquema_rejeita_falta_de_campo_aninhado(resultado, bloco, campos):
    for campo in sorted(campos):
        r = copy.deepcopy(resultado)
        del r[bloco][campo]
        with pytest.raises(ValueError, match=campo):
            comum.salvar_resultado("xgboost", r)


def test_esquema_rejeita_y_teste_desalinhado(resultado):
    y = resultado["y_teste"]
    i, j = y.index(0), y.index(1)
    y[i], y[j] = y[j], y[i]  # mesma contagem de churn, outra ordem
    with pytest.raises(ValueError, match="desalinhado"):
        comum.salvar_resultado("xgboost", resultado)


def test_esquema_rejeita_tamanhos_errados(resultado):
    for campo, valor in [("y_teste", resultado["y_teste"][:-1]),
                         ("probabilidades_teste", resultado["probabilidades_teste"][:-1]),
                         ("probabilidades_por_semente", [resultado["probabilidades_teste"][:-1]]),
                         ("ks_dobras_melhor", [0.5]),
                         ("perdas_dobras", [])]:
        with pytest.raises(ValueError, match=campo):
            comum.salvar_resultado("xgboost", {**resultado, campo: valor})


def test_esquema_rejeita_nao_finitos_e_campos_estranhos(resultado):
    p = resultado["probabilidades_teste"].copy()
    p[0] = np.nan
    with pytest.raises(ValueError):
        comum.salvar_resultado("xgboost", {**resultado, "probabilidades_teste": p})
    with pytest.raises(ValueError, match="fora do esquema"):
        comum.salvar_resultado("xgboost", {**resultado, "campo_novo": 1})


def test_esquema_aceita_ce_vs_mse_opcional_e_previsao_fora_de_zero_a_um(resultado):
    p = resultado["probabilidades_teste"].copy()
    p[0], p[1] = -0.42, 1.07  # reg:squarederror
    comum.salvar_resultado("xgboost", {**resultado, "probabilidades_teste": p, "ce_vs_mse": {}})


def test_esquema_exige_retreino(resultado):
    with pytest.raises(ValueError, match="retreino"):
        comum.salvar_resultado("xgboost", {**resultado, "avaliacao_final": "ensemble_dobras"})


def test_esquema_exige_nome_coerente(resultado):
    with pytest.raises(ValueError, match="modelo"):
        comum.salvar_resultado("stab", resultado)
    with pytest.raises(ValueError, match="invalido"):
        comum.salvar_resultado("../fora", {**resultado, "modelo": "../fora"})


def test_carregar_rejeita_arquivo_invalido(resultado, tmp_path):
    caminho = comum.salvar_resultado("xgboost", resultado)
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    del dados["teste"]
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    with pytest.raises(ValueError, match="teste"):
        comum.carregar_resultados()


# ---------------- compatibilidade com o codigo do Jefferson ----------------
def test_chamadas_de_avaliar_jeff_continuam_valendo():
    y, p = _amostra()
    y, p = y.astype("float32"), p.astype("float32")  # o formato de carregar_dobras
    limiar = comum.limiar_ks(y, p)
    m = comum.metricas(y, p, limiar)
    assert set(m) == {"ks", "auroc", "ce", "mse", "precision", "recall", "f1", "confusao", "limiar"}
    assert np.array(m["confusao"]).shape == (2, 2)
    assert len(comum.bootstrap_ks(y, p, n=20)) == 2
    assert comum.SEMENTE == 42
    assert comum.NUMERIC_COLS == ["tenure", "MonthlyCharges", "TotalCharges"]
