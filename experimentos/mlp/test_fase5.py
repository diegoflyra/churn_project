"""Placar comum (comum.py) e fase 5 do MLP.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/mlp/test_fase5.py
"""
import os, sys, json
os.environ.setdefault("CHURN_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import comum, modelo as M, fase4 as F4, fase5 as F5, pilha as P     # modelo vem antes do torch: ele ajusta o ambiente
import numpy as np, pandas as pd, pytest
from test_fase4 import _base, C, HP


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch, tmp_path):
    monkeypatch.setattr(F4, "_dados", _base)
    F5.divs.cache_clear()
    F5.usar_pilha(None)
    yield
    F5.divs.cache_clear()
    F5.usar_pilha(None)


# ---------------- comum: placar ----------------
def test_divisoes_do_placar_cobrem_o_treino_e_isolam_a_validacao():
    y = np.asarray(_base()[2])
    D = comum.divisoes_placar(y)
    assert len(D) == 15 and sorted({d["semente"] for d in D}) == [101, 102, 103]
    for s in (101, 102, 103):
        vistos = np.concatenate([d["validacao"] for d in D if d["semente"] == s])
        assert sorted(vistos) == list(range(len(y)))                       # cada cliente validado uma vez por semente
    for d in D:
        assert not set(d["treino"]) & set(d["validacao"])
        assert not set(d["ajuste"]) & set(d["parada"])
        assert sorted(np.r_[d["ajuste"], d["parada"]]) == list(range(len(d["treino"])))
        assert abs(len(d["parada"]) / len(d["treino"]) - .2) < .02
    assert [list(d["validacao"]) for d in D] == [list(d["validacao"]) for d in comum.divisoes_placar(y)]   # fixas


def test_comparar_placar_da_os_tres_vereditos():
    r = np.random.default_rng(0)
    y = (r.random(3000) < .3).astype(int)
    fraco = y * .5 + r.normal(0, .5, 3000)
    forte = y * 1.5 + r.normal(0, .5, 3000)
    assert comum.comparar_placar(y, fraco, forte, n_boot=300)["veredito"] == "ganhou"
    assert comum.comparar_placar(y, forte, fraco, n_boot=300)["veredito"] == "perdeu"
    quase = fraco + r.normal(0, .01, 3000)
    v = comum.comparar_placar(y, fraco, quase, n_boot=300)
    assert v["veredito"] == "empatou" and v["ic95"][0] < 0 < v["ic95"][1]


def test_medir_placar_generico_nao_passa_a_validacao_para_o_modelo(tmp_path):
    """O modelo recebe so o treino da dobra; a parada so ve X_tr.iloc[parada]. Uma logistica serve de modelo qualquer."""
    from sklearn.linear_model import LogisticRegression
    Xtr, _, ytr, _ = _base()
    vistos = []

    def ajustar_prever(X_tr, y_tr, X_val, ajuste, parada, semente):
        assert not set(X_tr.index) & set(X_val.index)
        vistos.append(len(X_val))
        Xa, ya, Xv = comum.preparar(X_tr, y_tr, X_val)
        m = LogisticRegression(max_iter=500).fit(Xa, ya)
        return m.predict_proba(Xv)[:, 1], (np.asarray(ya), m.predict_proba(Xa)[:, 1])

    med = comum.medir_placar(Xtr, ytr, ajustar_prever)
    assert len(vistos) == 15 and med["oof"].shape == (3, len(Xtr))
    assert .2 < med["ks_kfold"] <= 1 and med["ep"] > 0 and "mse_treino" in med and "ce_val" in med

    sig = comum.assinatura_treino(Xtr)
    l0 = comum.registrar_placar(str(tmp_path), "logistica", "trial0", {"C": 1}, med, ytr, sig)
    l1 = comum.registrar_placar(str(tmp_path), "logistica", "igual", {"C": 1}, med, ytr, sig, {"vs_trial0": ("logistica", "trial0")})
    assert l0["ks_teste"] is None and l1["vs_trial0"]["veredito"] == "empatou" and l1["vs_trial0"]["delta"] == 0
    # a linha do xgboost, quando existe, entra sozinha como vs_xgboost
    comum.registrar_placar(str(tmp_path), "xgboost", "final", {}, med, ytr, sig)
    l2 = comum.registrar_placar(str(tmp_path), "logistica", "outra", {"C": 2}, med, ytr, sig)
    assert l2["vs_xgboost"]["referencia"] == ["xgboost", "final"]
    # assinatura diferente nao compara
    l3 = comum.registrar_placar(str(tmp_path), "logistica", "outro_treino", {}, med, ytr, "outra", {"vs_trial0": ("logistica", "trial0")})
    assert l3["vs_trial0"] is None
    # teste preenchido depois, e a ultima linha vale
    Xte_y = np.asarray(_base()[3])
    comum.registrar_teste_placar(str(tmp_path), "logistica", "trial0", Xte_y, [np.random.default_rng(1).random(len(Xte_y))])
    t = comum.tabela_placar(str(tmp_path))
    assert t.set_index("rotulo").loc["trial0", "ks_teste"] is not None and len(t) == 5


# ---------------- fase 5 ----------------
def test_medir_da_mesma_medida_em_cpu_e_na_pilha():
    a = F5.medir([C])[0]
    F5.usar_pilha(["cpu"])
    b = F5.medir([C])[0]
    assert a["oof"].shape == b["oof"].shape == (3, 500)
    assert abs(a["ks_kfold"] - b["ks_kfold"]) < .03
    assert {"mse_treino", "ce_treino", "mse_val", "epocas"} <= set(a)


def test_medida_nao_usa_o_teste(monkeypatch):
    Xtr, Xte, ytr, yte = _base()
    a = F5.medir([C])[0]
    monkeypatch.setattr(F4, "_dados", lambda: (Xtr, Xte * 0 + 99, ytr, 1 - yte))
    F5.divs.cache_clear()
    b = F5.medir([C])[0]
    assert a["ks_kfold"] == b["ks_kfold"]


def test_garantir_reaproveita_e_compara_com_as_referencias(tmp_path):
    pasta = str(tmp_path)
    outra = dict(C, hp=dict(HP, unidades=4))
    itens = [("trial0", C, None), ("outra", outra, {"vs_trial0": ("mlp", "trial0")})]
    l = F5.garantir(pasta, itens)
    assert l[1]["vs_trial0"]["veredito"] in ("ganhou", "empatou", "perdeu")
    n = len(open(os.path.join(pasta, "placar.jsonl")).readlines())
    F5.garantir(pasta, itens)                                   # nada a medir de novo
    assert len(open(os.path.join(pasta, "placar.jsonl")).readlines()) == n
    t = F5.tabela(l)
    assert list(t.rotulo) == ["trial0", "outra"]


def test_grade_e_espaco():
    g = F5.grade_arquiteturas(C)
    assert len(g) == len(F5.ARQUITETURAS) * len(F5.RITMOS)
    assert g["arq_3x16_muito_lento"]["hp"]["lr"] == 1e-4 and g["arq_3x16_muito_lento"]["hp"]["paciencia"] == 50
    assert all("B" not in c["hp"]["plano"] and len(c["hp"]["plano"]) == c["hp"]["camadas"] for c in g.values())
    import optuna
    c = F5.espaco(optuna.trial.FixedTrial(F5._parametros(F5.TRIAL0)), F5.TRIAL0)
    assert c["hp"]["camadas"] == 1 and c["hp"]["unidades"] == 10 and c["hp"]["lr"] == 1e-3


def test_estudar_e_resumo(tmp_path):
    st, melhor = F5.estudar(str(tmp_path), C, 3, reps=(201, 202), n_jobs=2)
    assert len(st.trials) == 3 and st.trials[0].params["unidades"] == 8
    assert "lr" in F5.resumo_optuna(st) and "ativ" in F5.resumo_optuna(st)


def test_escolher_so_aceita_quem_ganhou():
    ls = [dict(rotulo="a", ks_oof=.6, vs_referencia=dict(veredito="empatou")),
          dict(rotulo="b", ks_oof=.55, vs_referencia=dict(veredito="ganhou")),
          dict(rotulo="c", ks_oof=.58, vs_referencia=dict(veredito="ganhou"))]
    assert F5.escolher(ls)["rotulo"] == "c"
    assert F5.escolher(ls[:1]) is None


def test_final_e_metricas_do_teste_com_limiar_da_oof(tmp_path):
    pasta = str(tmp_path)
    F5.garantir(pasta, [("trial0", C, None)])
    fin = F5.final(C, n_seeds=2)
    assert len(fin["p_sementes"]) == 2 and len(fin["p_sementes"][0]) == 150
    m = F5.metricas_teste(pasta, "trial0", fin)
    assert {"ks", "auroc", "f1", "ks_ic95", "ks_por_semente"} <= set(m)
    d = F5.delta_teste(fin, fin, n=50)
    assert d["delta"] == 0


def test_oversampling_com_tamanho_e_relatorio(tmp_path):
    import evolucao as EV
    Xtr, _, ytr, _ = _base()
    y = np.asarray(ytr)
    n0, n1 = min((y == 0).sum(), (y == 1).sum()), max((y == 0).sum(), (y == 1).sum())
    meio = (n0 / n1 + 1) / 2                     # entre a proporcao natural e 1:1
    for over, esperado in ((True, n1), (meio, int(n1 * meio))):
        _, ya, _ = EV._prep(Xtr, ytr, Xtr, over=over)
        assert abs(min((ya == 1).sum(), (ya == 0).sum()) - esperado) <= 1
    pasta = str(tmp_path)
    F5.garantir(pasta, [("trial0", C, None)])
    comum.adotar_placar(pasta, "mlp", "trial0", "teste")
    md = comum.relatorio_placar(pasta)
    assert "trial0 **(adotado)**" in md and "| mlp | trial0 |" in md
    with pytest.raises(KeyError):
        comum.adotar_placar(pasta, "mlp", "nao_existe", "x")


def test_preparar_comum_com_features_e_oversampling():
    """O padrao de comum.preparar nao muda; features e tamanho do oversampling sao opcionais e nao olham a avaliacao."""
    Xtr, Xte, ytr, _ = _base()
    a = comum.preparar(Xtr, ytr, Xte)
    b = comum.preparar(Xtr, ytr, Xte, features=(), over=True)
    assert np.allclose(np.asarray(a[0]), np.asarray(b[0])) and (np.asarray(a[1]) == 1).mean() == .5
    Xa, ya, Xv = comum.preparar(Xtr, ytr, Xte, features=("tenure_1", "cliente_novo", "delta_preco"), over=False)
    assert {"tenure_1", "cliente_novo", "delta_preco"} <= set(Xa.columns) and len(Xa) == len(Xtr)
    assert set(Xa.tenure_1.unique()) <= {0, 1} and abs(Xa.delta_preco.mean()) < 1e-6        # delta_preco entra no scaler
    assert (Xv.tenure_1 == (Xte.tenure <= 1).astype(int)).all()                            # por linha: nao depende do treino
    Xa, ya, _ = comum.preparar(Xtr, ytr, Xte, over=(np.asarray(ytr).mean() / (1 - np.asarray(ytr).mean()) + 1) / 2)
    assert len(Xtr) < len(Xa) < len(comum.preparar(Xtr, ytr, Xte)[0])
