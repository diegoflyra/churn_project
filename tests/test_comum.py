"""Acrescimos ao comum.py: dobras do Optuna e estudo com o procedimento do placar. Dados sinteticos.

Uso: python -m pytest tests
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import comum


def _dados(n=300, semente=0):
    rng = np.random.default_rng(semente)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-2 * X["a"]))).astype(int))
    return X, y


def test_dobras_do_optuna_sao_as_oficiais_e_separam_a_parada():
    _, y = _dados()
    D = comum.divisoes_optuna(y.values)
    oficiais = list(StratifiedKFold(5, shuffle=True, random_state=comum.SEMENTE).split(np.zeros(len(y)), y))
    assert len(D) == 5
    for d, (tr, va) in zip(D, oficiais, strict=True):
        assert np.array_equal(d["treino"], tr) and np.array_equal(d["validacao"], va)
        assert not set(d["ajuste"]) & set(d["parada"])
        assert sorted(np.concatenate([d["ajuste"], d["parada"]])) == list(range(len(tr)))
        assert abs(len(d["parada"]) - 0.2 * len(tr)) <= 1


def _ajustar_prever_de(chamadas):
    def de(hp):
        def f(X, y, X_val, ajuste, parada, semente):
            chamadas.append(hp)
            return X_val["a"].values * hp["sinal"] + 0.0 * semente
        return f
    return de


def _sugerir(t):
    return dict(sinal=t.suggest_categorical("sinal", [1.0, -1.0]))


def test_estudo_comeca_pelo_padrao_grava_em_sqlite_e_retoma(tmp_path):
    X, y = _dados()
    chamadas = []
    st = comum.estudo_placar("falso", 3, _sugerir, _ajustar_prever_de(chamadas), {"sinal": 1.0}, X, y, pasta=str(tmp_path))
    assert (tmp_path / "falso.db").exists()
    assert st.trials[0].params == {"sinal": 1.0}
    assert len(st.trials) == 3 and len(st.trials[0].user_attrs["ks_dobras"]) == 5
    assert st.trials[0].value > 0.5 and st.best_value == st.trials[0].value
    n = len(chamadas)
    st2 = comum.estudo_placar("falso", 3, _sugerir, _ajustar_prever_de(chamadas), {"sinal": 1.0}, X, y, pasta=str(tmp_path))
    assert len(st2.trials) == 3 and len(chamadas) == n          # retomou sem refazer nada
    assert st2.user_attrs["tempo_parede_s"] >= st.user_attrs["tempo_parede_s"]


def test_estudo_para_quando_o_tempo_acaba(tmp_path):
    X, y = _dados()
    chamadas = []
    st = comum.estudo_placar("falso", 5, _sugerir, _ajustar_prever_de(chamadas), {"sinal": 1.0}, X, y,
                             pasta=str(tmp_path), continuar=lambda: False)
    assert len(st.trials) == 1 and chamadas == []               # so o trial 0 enfileirado, sem rodar
    restam = iter([True, True, False])
    st = comum.estudo_placar("falso", 5, _sugerir, _ajustar_prever_de(chamadas), {"sinal": 1.0}, X, y,
                             pasta=str(tmp_path), continuar=lambda: next(restam, False))
    assert len([t for t in st.trials if t.state.is_finished()]) == 2


def test_garantir_placar_reaproveita_a_mesma_config_e_mede_a_nova(tmp_path):
    X, y = _dados()
    ass = comum.assinatura_treino(X)
    vezes = []

    def medir(sinal):
        def m():
            vezes.append(sinal)
            return comum.medir_placar(X, y, lambda X_, y_, Xv, a, p, s: Xv["a"].values * sinal)
        return m

    cfg = dict(hp={"sinal": 1.0}, features=("cliente_novo",), over=True)
    l1, m1 = comum.garantir_placar(str(tmp_path), "falso", "base", cfg, medir(1.0), y, ass)
    l2, m2 = comum.garantir_placar(str(tmp_path), "falso", "base", dict(cfg, features=["cliente_novo"]), medir(1.0), y, ass)
    assert vezes == [1.0] and l2 == l1                                  # tupla e lista sao a mesma config
    assert np.array_equal(m2["oof"], m1["oof"]) and m2["ks_kfold"] == m1["ks_kfold"]
    l3, _ = comum.garantir_placar(str(tmp_path), "falso", "outra", dict(cfg, hp={"sinal": -1.0}), medir(-1.0), y, ass,
                                  contra={"vs_base": ("falso", "base")})
    assert vezes == [1.0, -1.0] and l3["vs_base"]["veredito"] == "perdeu"
    assert len(comum.ler_placar(str(tmp_path))) == 2


def test_importar_placar_leva_linhas_e_oof_sem_duplicar(tmp_path):
    X, y = _dados()
    ass = comum.assinatura_treino(X)
    origem, destino = str(tmp_path / "repo"), str(tmp_path / "kaggle")
    medida = comum.medir_placar(X, y, lambda X_, y_, Xv, a, p, s: Xv["a"].values)
    comum.registrar_placar(origem, "xgboost", "final", {"hp": 1}, medida, y, ass)
    comum.registrar_placar(origem, "mlp", "fase4", {"hp": 2}, medida, y, ass)
    assert comum.importar_placar(origem, destino, ["xgboost"]) == 1
    assert comum.importar_placar(origem, destino, ["xgboost"]) == 0
    linhas = comum.ler_placar(destino)
    assert [(r["modelo"], r["rotulo"]) for r in linhas] == [("xgboost", "final")]
    assert np.array_equal(np.load(f"{destino}/{linhas[0]['oof']}"), medida["oof"])
    outro = comum.registrar_placar(destino, "mitra", "trial0", {"hp": 3}, medida, y, ass)   # entra vs_xgboost sozinho
    assert outro["vs_xgboost"]["referencia"] == ["xgboost", "final"] and outro["vs_xgboost"]["veredito"] == "empatou"


def test_completar_vs_xgboost_preenche_so_o_que_falta(tmp_path):
    """Linhas medidas antes da referencia (ou trazidas de outra pasta) ganham o vs_xgboost depois, uma vez so."""
    X, y = _dados()
    ass, pasta = comum.assinatura_treino(X), str(tmp_path)
    bom = comum.medir_placar(X, y, lambda X_, y_, Xv, a, p, s: Xv["a"].values)
    ruim = comum.medir_placar(X, y, lambda X_, y_, Xv, a, p, s: -Xv["a"].values)
    comum.registrar_placar(pasta, "mitra", "final", {"hp": 1}, bom, y, ass)
    comum.registrar_placar(pasta, "outro", "x", {"hp": 2}, bom, y, "outra_assinatura")
    assert comum.completar_vs_xgboost(pasta, y) == 0                   # sem referencia, nada a fazer
    comum.registrar_placar(pasta, "xgboost", "trial0", {"hp": 0}, bom, y, ass)
    comum.registrar_placar(pasta, "xgboost", "final", {"hp": 3}, ruim, y, ass)
    assert comum.completar_vs_xgboost(pasta, y) == 1
    assert comum.completar_vs_xgboost(pasta, y) == 0
    linhas = {(r["modelo"], r["rotulo"]): r for r in comum.ler_placar(pasta)}
    vs = linhas[("mitra", "final")]["vs_xgboost"]
    esperado = comum.comparar_placar(y, ruim["oof"], bom["oof"])
    assert vs["referencia"] == ["xgboost", "final"] and vs["veredito"] == "ganhou" and vs["delta"] == esperado["delta"]
    assert "vs_xgboost" not in linhas[("outro", "x")] and "vs_xgboost" not in linhas[("xgboost", "trial0")]
    assert linhas[("mitra", "final")]["ks_kfold"] == bom["ks_kfold"]                      # a medida nao muda


def test_estudo_corta_depois_da_segunda_dobra(tmp_path):
    """Cinco trials bons completos e um ruim: o ruim para na segunda dobra (passo 1), nao antes."""
    X, y = _dados()
    chamadas = []
    st = optuna.create_study(study_name="falso", storage=f"sqlite:///{tmp_path}/falso.db", direction="maximize")
    for sinal in [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]:
        st.enqueue_trial({"sinal": sinal})
    st = comum.estudo_placar("falso", 6, _sugerir, _ajustar_prever_de(chamadas), {"sinal": 1.0}, X, y, pasta=str(tmp_path))
    ruim = st.trials[-1]
    assert ruim.params == {"sinal": -1.0} and ruim.state == optuna.trial.TrialState.PRUNED
    assert sorted(ruim.intermediate_values) == [0, 1]
    assert len(chamadas) == 5 * 5 + 2
