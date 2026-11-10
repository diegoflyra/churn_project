"""Fase 4: a medida e o k-fold do treino, com parada numa separacao interna, e o teste so aparece no fim.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/mlp/test_fase4.py
"""
import os, sys
os.environ.setdefault("CHURN_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import comum, modelo as M, blocos as BL, fase4 as F4, pilha as P     # modelo vem antes do torch: ele ajusta o ambiente
import numpy as np, pandas as pd, pytest

HP = dict(camadas=1, unidades=8, ativ="relu", otim="adam", lr=1e-2, dropout=0.0, l2=1e-4, perda="ce", paciencia=3, plano="N", lote=64)
C = dict(hp=HP, dados={}, extras=[])


def _base(n=500, semente=0):
    """Treino e teste sinteticos com as colunas que as features da fase 4 usam."""
    r = np.random.default_rng(semente)
    def tabela(m):
        d = pd.DataFrame(dict(tenure=r.integers(0, 72, m), MonthlyCharges=r.uniform(18, 120, m)))
        d["TotalCharges"] = d.tenure * d.MonthlyCharges
        for c in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Partner", "Dependents", "PaperlessBilling",
                  "Contract_Month-to-month", "InternetService_Fiber optic", "PaymentMethod_Electronic check"):
            d[c] = r.integers(0, 2, m)
        d["total_addons"] = d[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]].sum(axis=1)
        return d.astype("float32")
    Xtr, Xte = tabela(n), tabela(150)
    f = lambda d: (-.05 * d.tenure + 1.5 * d["Contract_Month-to-month"] + .01 * d.MonthlyCharges + r.normal(size=len(d)) > 0).astype(int)
    return Xtr, Xte, f(Xtr), f(Xte)


@pytest.fixture(autouse=True)
def dados_sinteticos(monkeypatch, tmp_path):
    monkeypatch.setattr(F4, "_dados", _base)
    F4.abrir_cache(str(tmp_path))
    F4.definir_banda(None)
    F4.usar_pilha(None)
    yield
    F4.usar_pilha(None)


def test_divisoes_separam_ajuste_parada_e_avaliacao():
    """A dobra externa nunca entra no ajuste nem na parada; a parte interna de parada nunca entra no ajuste da primeira rede."""
    Xtr, _, ytr, _ = _base()
    vistos = []
    for a1, a2, b in F4.divisoes(3):
        assert not set(a1) & set(a2) and not set(a1) & set(b) and not set(a2) & set(b)
        assert len(a1) + len(a2) + len(b) == len(Xtr)
        assert abs(len(a2) / (len(a1) + len(a2)) - .2) < .02
        vistos += list(b)
    assert sorted(vistos) == list(range(len(Xtr)))       # cada cliente e avaliado uma vez por divisao
    assert [list(b) for *_, b in F4.divisoes(3)] == [list(b) for *_, b in F4.divisoes(3)]
    assert [list(b) for *_, b in F4.divisoes(3)] != [list(b) for *_, b in F4.divisoes(4)]


def test_medir_devolve_ks_do_retreino_ks_da_parada_e_previsao_fora_da_dobra():
    r = F4.medir(C, 1)
    assert set(r) >= {"ks", "ks_parada", "epocas", "oof"}
    assert len(r["oof"]) == 500 and np.isfinite(r["oof"]).all()
    assert .2 < r["ks"] <= 1 and .2 < r["ks_parada"] <= 1


def test_medir_nao_le_o_teste(monkeypatch):
    """O teste pode ser qualquer coisa: a medida da fase 4 nao muda."""
    a = F4.medir(C, 1)
    Xtr, Xte, ytr, yte = _base()
    monkeypatch.setattr(F4, "_dados", lambda: (Xtr, Xte * np.nan, ytr, yte * 0 + 7))
    b = F4.medir(C, 1)
    assert a["ks"] == b["ks"] and a["ks_parada"] == b["ks_parada"]


def test_pilha_mede_o_mesmo_formato_e_sem_ler_o_teste(monkeypatch):
    F4.usar_pilha(["cpu"])
    a = F4.medir_varios([(C, 1, 0), (dict(C, hp={**HP, "plano": "D", "dropout": .2}), 1, 0)])
    assert all(set(r) >= {"ks", "ks_parada", "epocas", "oof"} for r in a)
    Xtr, Xte, ytr, yte = _base()
    monkeypatch.setattr(F4, "_dados", lambda: (Xtr, Xte * np.nan, ytr, yte * 0 + 7))
    b = F4.medir_varios([(C, 1, 0), (dict(C, hp={**HP, "plano": "D", "dropout": .2}), 1, 0)])
    assert [x["ks"] for x in a] == [x["ks"] for x in b]


def test_conferencia_entre_pilha_e_referencia():
    F4.usar_pilha(["cpu"])
    tab, resumo = F4.conferir_pilha({"a": C}, [1, 2])
    assert set(resumo) >= {"vies", "ep_vies", "ruido", "n"} and resumo["n"] == 2
    assert len(tab) == 1


def test_cache_guarda_e_reaproveita(monkeypatch):
    r1 = F4.medir_muitos([(C, 1, 0)])
    monkeypatch.setattr(F4, "medir_varios", lambda *a, **k: pytest.fail("deveria vir do cache"))
    assert F4.medir_muitos([(C, 1, 0)])[0]["ks"] == r1[0]["ks"]


def test_banda_pega_os_clientes_perto_do_limiar_do_ks():
    r = np.random.default_rng(0)
    y = (r.random(1000) < .3).astype(int)
    p = np.clip(r.normal(.3 + .3 * y, .15), 0, 1)
    idx, limiar = F4.banda_ambigua(y, p, fracao=.3)
    assert len(idx) == 300
    assert abs(limiar - comum.limiar_ks(y, p)) < 1e-12
    dentro, fora = np.abs(p[idx] - limiar), np.abs(np.delete(p, idx) - limiar)
    assert dentro.max() <= np.quantile(fora, .05) + 1e-9


def test_ks_na_banda_entra_na_medida_quando_ha_banda():
    F4.definir_banda(np.arange(0, 500, 2))
    r = F4.medir_muitos([(C, 1, 0)])[0]
    assert 0 <= r["ks_banda"] <= 1


def test_bloco_aceita_so_quem_supera_a_regua_e_vence_80_por_cento(monkeypatch, tmp_path):
    base, bom, fraco = C, dict(C, hp={**HP, "lr": 1e-3}), dict(C, hp={**HP, "lr": 3e-3})
    ganho = {json_hp: g for json_hp, g in ((F4._chave_cfg(bom), .01), (F4._chave_cfg(fraco), .0002))}    # a regua aqui e 0,0003
    def falsa(tarefas, n_jobs=1):
        return [dict(ks=.5 + ganho.get(F4._chave_cfg(c), 0) + (.0001 * s) + .001 * (r % 3), ks_parada=.5, epocas=10, ks_banda=None) for c, r, s in tarefas]
    monkeypatch.setattr(F4, "medir_muitos", falsa)
    camp, tab = F4.bloco(str(tmp_path), "b_teste", "pergunta?", {"bom": dict(hp=dict(lr=1e-3)), "fraco": dict(hp=dict(lr=3e-3))}, base, range(1, 11))
    assert camp == bom
    assert dict(zip(tab.variante, tab.decisao)) == {"(campeao atual)": "base", "bom": "ACEITA", "fraco": "rejeitada"}
    camp2, _ = F4.bloco(str(tmp_path), "b_teste", "pergunta?", {}, base, range(1, 11))
    assert camp2 == bom          # retomada: reaproveita o arquivo


def test_features_da_fase4_estao_nas_candidatas_e_sao_calculadas_sem_rotulo():
    Xtr, *_ = _base()
    novas = ["tenure_1", "novo_mes_a_mes", "n_protecao", "n_streaming", "fibra_sem_protecao", "atrito_pagamento", "sozinho"]
    for n in novas:
        d = BL.CANDIDATAS[n](Xtr.copy())
        assert n in d and len(d) == len(Xtr)
    d = BL.CANDIDATAS["n_protecao"](Xtr.copy())
    assert (d.n_protecao == Xtr[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]].sum(axis=1)).all()
    d = BL.CANDIDATAS["fibra_sem_protecao"](Xtr.copy())
    assert ((d.fibra_sem_protecao == 1) == ((Xtr["InternetService_Fiber optic"] == 1) & (Xtr[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]].sum(axis=1) == 0))).all()


def test_xgboost_mede_nas_mesmas_dobras_externas():
    a, b = F4.medir_xgb(C, 1), F4.medir_xgb(dict(C, extras=["n_protecao"]), 1)
    assert .2 < a <= 1 and .2 < b <= 1


def test_relacoes_mostram_variaveis_e_regras_estaveis_na_banda():
    Xtr, _, ytr, _ = _base()
    oof = F4.medir(C, 1)["oof"]
    rel = F4.relacoes(Xtr, ytr.values, oof, fracao=.4)
    assert {"variaveis", "regras", "limiar", "clientes_na_banda"} <= set(rel)
    v = pd.DataFrame(rel["variaveis"])
    assert {"variavel", "ks_na_banda", "ks_fora_da_banda", "churn_na_banda"} <= set(v.columns)
    for r in rel["regras"]:
        assert {"regra", "clientes", "churn", "lift", "aparece_em"} <= set(r)


def test_espaco_do_optuna_nunca_sorteia_bn():
    import optuna
    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(200):
        c = F4.espaco(st.ask(), C)
        assert "B" not in c["hp"]["plano"] and len(c["hp"]["plano"]) == c["hp"]["camadas"]
        assert set(c["dados"]) <= {"over"} and c["extras"] == C["extras"]


def test_optuna_fica_na_cpu_e_usa_a_medida_k_fold(monkeypatch, tmp_path):
    F4.usar_pilha(["cpu"])
    monkeypatch.setattr(P, "medir_pilha", lambda *a, **k: pytest.fail("Optuna nao usa a pilha"))
    monkeypatch.setattr(F4, "_medir_pilha", lambda *a, **k: pytest.fail("Optuna nao usa a pilha"))
    monkeypatch.setattr(M, "MAX_EPOCAS", 2)
    st, melhor = F4.estudar(str(tmp_path), dict(C, hp={**HP, "paciencia": 10}), 2, reps=(1,))
    assert len([t for t in st.trials if t.state.name == "COMPLETE"]) == 2
    assert set(melhor) == {"hp", "dados", "extras"}


def test_comparar_protocolos_mostra_o_antigo_e_o_novo(monkeypatch):
    import fase3 as F3
    monkeypatch.setattr(F3, "_dados", _base)
    tab = F4.comparar_protocolos({"a": C}, [1, 2])
    assert {"config", "val_antigo", "ks_parada", "ks", "otimismo_antigo"} <= set(tab.columns)


def _muda(c, variantes):
    return all(F4.aplicar(c, v) != c for v in variantes.values())


def test_variantes_de_balanceamento_dependem_do_campeao_e_sempre_mudam_algo():
    com = F4.variantes_balanceamento(C)
    assert _muda(C, com) and all(v.get("dados", {}).get("over") is False for v in com.values())
    sem = dict(C, dados={"over": False}, hp={**HP, "peso_pos": 1.5})
    v = F4.variantes_balanceamento(sem)
    assert _muda(sem, v) and any(x.get("dados", {}).get("over") is True for x in v.values())


def test_variantes_de_treino_nunca_poe_bn_e_sempre_mudam_algo():
    for c in (C, dict(C, hp={**HP, "camadas": 2, "plano": "DN", "dropout": .2, "ativ": "gelu"})):
        v = F4.variantes_treino(c)
        assert v and _muda(c, v)
        assert all("B" not in F4.aplicar(c, x)["hp"].get("plano", "") for x in v.values())
    assert "Dropout na 1ª camada" in F4.variantes_treino(C)


def test_variantes_de_features_so_testam_o_que_ainda_nao_entrou():
    c = dict(C, extras=["n_protecao"])
    v = F4.variantes_features(c, ["n_protecao", "tenure_1", "sem_gender"])
    assert set(v) == {"com tenure_1", "sem gender"}
    assert all(x["extras"][0] in BL.CANDIDATAS for x in v.values())


def test_comparar_traz_o_ganho_nos_ambiguos_quando_ha_banda():
    F4.definir_banda(np.arange(0, 500, 2))
    r = F4.comparar(C, dict(C, hp={**HP, "lr": 3e-3}), [1, 2])
    assert {"delta", "ep", "ic95", "vitorias", "n", "delta_banda"} <= set(r) and r["n"] == 2
