"""FT final e exportacao para o Hugging Face: o pacote exportado preve o mesmo que o codigo do projeto.

Uso: CHURN_DEVICE=cpu python -m pytest experimentos/ft_transformer/test_final.py
"""
import os, sys, importlib
os.environ.setdefault("CHURN_DEVICE", "cpu")
os.environ.setdefault("CHURN_MAX_EPOCAS", "6")
sys.path.insert(0, os.path.dirname(__file__))
import comum, modelo as M, fase1 as F1, fase7 as F7, final as FIN
import numpy as np, pandas as pd, pytest

sys.path.insert(0, FIN.HF)
import churn_ft


def test_codificar_igual_ao_do_grupo():
    bruto = pd.read_csv(comum.CAMINHO_CSV)
    ref = comum.carregar_df().drop(columns="Churn")
    d = churn_ft.codificar(bruto)
    assert set(ref.columns) <= set(d.columns)
    assert np.allclose(d[ref.columns].astype(float), ref.astype(float))
    um = churn_ft.codificar(bruto.iloc[[3]])               # uma linha so: as dummies continuam todas la
    assert np.allclose(um[ref.columns].astype(float), ref.iloc[[3]].astype(float))


def test_features_iguais_as_do_projeto():
    X = comum.carregar_df()
    for f, fn in churn_ft.FEATURES.items():
        assert np.allclose(fn(X).astype(float), comum.FEATURES[f](X).astype(float)), f


def test_exportado_preve_igual(tmp_path):
    c = F7.candidatos()["p_cinco"]
    c = dict(hp=dict(c["hp"], n_blocos=1, d_token=16, n_cabecas=2, paciencia=2), dados=c["dados"])
    fin = FIN.final(c, n_seeds=2)
    assert len(fin["p_sementes"]) == 2 and len(fin["y_teste"]) == len(F1._dados()[1])
    oof = np.random.default_rng(0).random((3, len(F1._dados()[2])))
    met = FIN.metricas_teste(oof, fin)
    linha = dict(ks_kfold=.55, ks_oof=.54, auc_oof=.84)
    cfg = FIN.exportar(str(tmp_path), "teste", c, fin, met, linha)
    assert sorted(os.listdir(tmp_path)) == ["README.md", "churn_ft.py", "config.json", "pesos_semente0.pt", "pesos_semente1.pt"]
    assert "{" not in open(tmp_path / "README.md", encoding="utf-8").read().split("```python")[0].split("---")[2]
    sys.path.insert(0, str(tmp_path))
    pacote = importlib.import_module("churn_ft")
    mod = pacote.carregar(str(tmp_path))
    _, Xte, _, _ = F1._dados()
    bruto = pd.read_csv(comum.CAMINHO_CSV).loc[Xte.index]
    p = mod.prever(bruto)
    assert np.allclose(p, np.mean(fin["p_sementes"], 0), atol=1e-5)
    assert mod.limiar == met["limiar"] and cfg["features"] == c["dados"]["features"]


def test_final_recusa_o_que_o_pacote_nao_cobre():
    with pytest.raises(ValueError):
        FIN.final(dict(hp=dict(F7.BASE["hp"], tokens="cat"), dados=F7.BASE["dados"]), n_seeds=1)
