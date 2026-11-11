"""comum.py - dados, dobras, preparar, KS, metricas e resultados (IF702, churn).

Reproduz exatamente o pre-processamento de if702-preprocessamento.ipynb.
A unica diferenca e a leitura do CSV: `path`, a variavel CHURN_CSV ou o kagglehub, nessa ordem.
"""
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import ks_2samp
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Kaggle: "/kaggle/input/datasets/kapturovalexander/customers-churned-in-telecom-services/customer_churn_telecom_services.csv"
CAMINHO_CSV = os.environ.get("CHURN_CSV")
DATASET_KAGGLE = "kapturovalexander/customers-churned-in-telecom-services/versions/3"
ARQUIVO_KAGGLE = "customer_churn_telecom_services.csv"
SEMENTE = 42
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
N_DOBRAS = 5
TAXA_TREINO = 1402 / 5282  # 0,26543: churn no treino de 5.282 clientes, antes do oversampling
PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"


def _caminho_csv(path=None):
    """Argumento, depois CHURN_CSV, depois download pelo kagglehub na versao fixada."""
    caminho = path or os.environ.get("CHURN_CSV")
    if caminho:
        return caminho
    import kagglehub  # so quem baixa a base precisa do pacote

    return os.path.join(kagglehub.dataset_download(DATASET_KAGGLE), ARQUIVO_KAGGLE)


def carregar_df(path=None, criar_variaveis=True):
    df = pd.read_csv(_caminho_csv(path))
    if "customerID" in df.columns:  # a copia do Kaggle nao tem essa coluna
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    binarias = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "Churn": {"No": 0, "Yes": 1},
    }
    for col, m in binarias.items():
        df[col] = df[col].map(m)
    df["MultipleLines"] = df["MultipleLines"].map({"Yes": 1, "No": 0, "No phone service": 0})
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in addons:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=False, dtype=int)
    if criar_variaveis:
        add7 = ["MultipleLines"] + addons
        df["total_addons"] = df[add7].sum(axis=1)
        auto = [c for c in df.columns if "PaymentMethod_" in c and "automatic" in c.lower()]
        df["PaymentMethod_Automatic"] = df[auto].max(axis=1)
    return df


def preparar(X_tr, y_tr, X_aval, padronizar=True, balancear=True):
    """Scaler ajustado so em X_tr; oversampling so em X_tr; X_aval so recebe transform.

    Com balanceamento, as primeiras len(X_tr) linhas devolvidas sao as originais, na ordem de entrada.
    """
    X_tr_s = X_tr.copy()
    X_av_s = X_aval.copy()
    if padronizar:
        scaler = StandardScaler()
        X_tr_s[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
        X_av_s[NUMERIC_COLS] = scaler.transform(X_aval[NUMERIC_COLS])
    if not balancear:
        return X_tr_s, y_tr.copy(), X_av_s
    X_b, y_b = RandomOverSampler(random_state=SEMENTE).fit_resample(X_tr_s, y_tr)
    return X_b, y_b, X_av_s


def _dividir(df):
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    return train_test_split(X, y, test_size=0.25, random_state=SEMENTE, stratify=y)


def _kfold():
    return StratifiedKFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEMENTE)


def carregar_dobras(path=None, **opcoes):
    """Devolve (folds, final, colunas) em float32.

    Opcoes: criar_variaveis (de carregar_df), padronizar e balancear (de preparar).
    """
    df = carregar_df(path, criar_variaveis=opcoes.pop("criar_variaveis", True))
    X_train, X_test, y_train, y_test = _dividir(df)
    folds = []
    for idx_tr, idx_val in _kfold().split(X_train, y_train):
        Xv, yv = X_train.iloc[idx_val], y_train.iloc[idx_val]
        Xt, yt, Xv = preparar(X_train.iloc[idx_tr], y_train.iloc[idx_tr], Xv, **opcoes)
        folds.append((Xt.values.astype("float32"), yt.values.astype("float32"),
                      Xv.values.astype("float32"), yv.values.astype("float32")))
    Xtr_f, ytr_f, Xte_f = preparar(X_train, y_train, X_test, **opcoes)
    final = (Xtr_f.values.astype("float32"), ytr_f.values.astype("float32"),
             Xte_f.values.astype("float32"), y_test.values.astype("float32"))
    return folds, final, list(X_train.columns)


def posicoes_dobras(path=None):
    """Lista de (idx_tr, idx_val) de cada dobra, em posicoes do treino sem copias."""
    X_train, _, y_train, _ = _dividir(carregar_df(path))
    return list(_kfold().split(X_train, y_train))


# ---------------- metricas ----------------
def ks(y, p):
    """KS = max_t (TPR(t) - FPR(t)), igual a ks_2samp entre os grupos quando o escore separa no sentido certo."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def limiar_ks(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def ce(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mse(y, p):
    return float(np.mean((p - y) ** 2))


def metricas(y, p, limiar):
    pred = (p >= limiar).astype(int)
    return dict(ks=ks(y, p), auroc=float(roc_auc_score(y, p)), ce=ce(y, p), mse=mse(y, p),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)),
                confusao=confusion_matrix(y, pred).tolist(), limiar=limiar)


def corrige_prior(p, pi=TAXA_TREINO):
    """Desfaz o deslocamento de prior do oversampling (treino 50/50 -> prior real).

    Use so em modelos treinados com oversampling. Recorta p em [0, 1] antes, porque
    reg:squarederror preve fora desse intervalo. Em float64, para a transformacao
    continuar estritamente crescente e nao criar empates que mudem o KS.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    w = pi / (1 - pi)
    return p * w / (p * w + (1 - p))


def _reamostras(y, n, seed):
    """Indices de bootstrap, sem as reamostras que ficam com uma classe so."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() != y[b].max():
            yield b


def bootstrap_ks(y, p, n=1000, seed=SEMENTE):
    """Intervalo de 95% do KS, por percentil."""
    y, p = np.asarray(y), np.asarray(p)
    out = [ks(y[b], p[b]) for b in _reamostras(y, n, seed)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bootstrap_pareado(y, p_modelo, p_ref, n=1000, seed=SEMENTE):
    """Diferenca de KS (modelo - referencia), com as mesmas reamostras para os dois.

    Devolve `diferenca`, `ic95` por percentil e `p_valor` bilateral: o dobro da menor
    cauda da distribuicao das diferencas em torno de zero, com correcao (r + 1) / (n + 1).
    """
    y, p_modelo, p_ref = np.asarray(y), np.asarray(p_modelo), np.asarray(p_ref)
    d = np.array([ks(y[b], p_modelo[b]) - ks(y[b], p_ref[b]) for b in _reamostras(y, n, seed)])
    cauda = min(np.sum(d <= 0), np.sum(d >= 0))
    return dict(diferenca=ks(y, p_modelo) - ks(y, p_ref),
                ic95=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                p_valor=float(min(1.0, 2 * (cauda + 1) / (len(d) + 1))))


# ---------------- validacao cruzada ----------------
def _perdas(y, p, com_oversampling):
    q = corrige_prior(p) if com_oversampling else p
    return dict(ce=ce(y, p), mse=mse(y, p), ce_corr=ce(y, q), mse_corr=mse(y, q))


def _prever(prever, X):
    p = np.asarray(prever(X), dtype=np.float64)
    if p.shape != (len(X),):
        raise ValueError(f"prever(X) deve devolver um vetor de {len(X)} valores, veio o formato {p.shape}")
    return p


def avaliar_cv(ajustar, folds, trial=None):
    """Roda as dobras e devolve KS, perdas e probabilidades fora da dobra.

    `ajustar(dobra)` recebe (X_tr, y_tr, X_val, y_val) e devolve `prever(X)`.
    Com `trial`, reporta a media acumulada a cada dobra, corta so depois da segunda dobra e
    grava `ks_dobras`, `perdas_dobras` e `tempo_s` nos atributos do trial.

    As validacoes particionam o treino, entao o tamanho do treino sem copias de cada dobra e
    o total das validacoes menos a validacao dela. As linhas originais vem primeiro no treino
    (ver `preparar`). A correcao de prior so vale para dobras com oversampling; sem ele,
    `ce_corr` e `mse_corr` repetem as perdas brutas.
    """
    if len(folds) != N_DOBRAS:
        raise ValueError(f"avaliar_cv precisa das {N_DOBRAS} dobras, recebeu {len(folds)}")
    t0 = time.perf_counter()
    total = sum(len(dobra[2]) for dobra in folds)
    kss, perdas, probs = [], [], []
    for i, dobra in enumerate(folds):
        Xt, yt, Xv, yv = dobra
        n_orig = total - len(Xv)
        if not 0 < n_orig <= len(Xt):
            raise ValueError(f"Dobra {i}: {n_orig} linhas originais para um treino de {len(Xt)}")
        com_oversampling = len(Xt) > n_orig
        prever = ajustar(dobra)
        pv = _prever(prever, Xv)
        ptr = _prever(prever, Xt[:n_orig])
        del prever  # libera o modelo antes do ajuste da proxima dobra, o que importa na GPU
        kss.append(ks(yv, pv))
        perdas.append(dict(treino=_perdas(yt[:n_orig], ptr, com_oversampling),
                           validacao=_perdas(yv, pv, com_oversampling)))
        probs.append(pv)
        if trial is not None:
            trial.report(float(np.mean(kss)), i)
            if i == 1 and trial.should_prune():  # depois da segunda dobra
                raise optuna.TrialPruned()
    prob_oof = np.concatenate(probs)
    y_oof = np.concatenate([dobra[3] for dobra in folds])
    if trial is not None:
        trial.set_user_attr("ks_dobras", kss)
        trial.set_user_attr("perdas_dobras", perdas)
        trial.set_user_attr("tempo_s", time.perf_counter() - t0)
    return dict(ks=float(np.mean(kss)), ks_dobras=kss, perdas_dobras=perdas,
                prob_oof=prob_oof, y_oof=y_oof, limiar=limiar_ks(y_oof, prob_oof))


def resumir_estudo(estudo, n_trials, tempo_parede_s, param_perda=None, perdas=None):
    """Monta o bloco `estudo` do JSON a partir de um estudo do Optuna.

    `param_perda` e o hiperparametro da funcao de perda e `perdas` traduz seus valores para
    "ce" ou "mse". Modelo sem essa escolha fica com `por_perda` vazio.
    """
    trials = estudo.trials
    ok = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    podados = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    por_perda = {}
    if param_perda is not None:
        for valor, nome in perdas.items():
            v = sorted((t.value for t in ok if t.params[param_perda] == valor), reverse=True)
            por_perda[nome] = dict(n=len(v), melhor=v[0] if v else None,
                                   mediana=float(np.median(v)) if v else None,
                                   media_top5=float(np.mean(v[:5])) if v else None)
    melhor_por_categoria = {}
    for t in ok:
        for nome, dist in t.distributions.items():
            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                por_valor = melhor_por_categoria.setdefault(nome, {})
                chave = str(t.params[nome])
                por_valor[chave] = max(por_valor.get(chave, -math.inf), t.value)
    return dict(
        n_trials=n_trials,
        n_completos=len(ok),
        n_podados=len(podados),
        por_perda=por_perda,
        melhor_por_categoria=melhor_por_categoria,
        importancias=dict(optuna.importance.get_param_importances(estudo)),
        config_trial0=trials[0].params,
        ks_trial0=trials[0].value,
        melhor_ks=estudo.best_value,
        convergencia=np.maximum.accumulate([t.value for t in ok]).tolist(),
        tempo_parede_s=tempo_parede_s,
    )


# ---------------- resultados/<modelo>.json ----------------
VERSAO_ESQUEMA = 1
AVALIACAO_FINAL = "retreino"
N_TESTE = 1761
# SHA-256 dos rotulos do teste como uint8, na ordem fixa da divisao. Confere o alinhamento sem precisar da base.
HASH_Y_TESTE = "92b86bbe72b178594c77be630c20bec3e99b75331d4718cd18b9912812a53e9f"

_CAMPOS = {
    "versao_esquema", "modelo", "avaliacao_final", "ambiente", "melhor_config", "ks_dobras_melhor",
    "perdas_dobras", "estudo", "teste", "tempos", "probabilidades_teste", "probabilidades_por_semente", "y_teste",
}
_CAMPOS_OPCIONAIS = {"ce_vs_mse"}
_CAMPOS_ESTUDO = {
    "n_trials", "n_completos", "n_podados", "por_perda", "melhor_por_categoria", "importancias",
    "config_trial0", "ks_trial0", "melhor_ks", "convergencia", "tempo_parede_s",
}
_CAMPOS_TESTE = {
    "ks", "ks_ic95", "ks_por_semente", "auroc", "ce", "mse", "ce_corr", "mse_corr",
    "precision", "recall", "f1", "confusao", "limiar",
}
_CAMPOS_AMBIENTE = {"python", "bibliotecas", "dispositivo"}
_CAMPOS_TEMPOS = {"treino_final_s", "predicao_teste_s"}
_CAMPOS_PERDAS = {"ce", "mse", "ce_corr", "mse_corr"}


def hash_y_teste(y):
    return hashlib.sha256(np.asarray(y).astype(np.uint8).tobytes()).hexdigest()


def _erro(nome, mensagem):
    raise ValueError(f"Resultado de {nome!r}: {mensagem}")


def _exigir(nome, onde, valor, campos):
    if not isinstance(valor, dict):
        _erro(nome, f"{onde} deve ser um objeto")
    faltando = campos - valor.keys()
    if faltando:
        _erro(nome, f"{onde} sem os campos {sorted(faltando)}")


def _exigir_numeros(nome, onde, valores):
    """Probabilidades podem sair de [0, 1] (reg:squarederror), mas precisam ser finitas."""
    if not isinstance(valores, list) or not valores:
        _erro(nome, f"{onde} deve ser uma lista nao vazia")
    for v in valores:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            _erro(nome, f"{onde} tem um valor que nao e numero finito: {v!r}")


def validar_resultado(modelo, resultado):
    """Levanta ValueError se o resultado nao segue o esquema ou se y_teste esta desalinhado."""
    _exigir(modelo, "o resultado", resultado, _CAMPOS)
    extras = resultado.keys() - _CAMPOS - _CAMPOS_OPCIONAIS
    if extras:
        _erro(modelo, f"campos fora do esquema: {sorted(extras)}")
    if resultado["versao_esquema"] != VERSAO_ESQUEMA:
        _erro(modelo, f"versao_esquema {resultado['versao_esquema']!r}, esperada {VERSAO_ESQUEMA}")
    if resultado["modelo"] != modelo:
        _erro(modelo, f"o campo modelo diz {resultado['modelo']!r}")
    if resultado["avaliacao_final"] != AVALIACAO_FINAL:
        _erro(modelo, f"avaliacao_final {resultado['avaliacao_final']!r}, esperada {AVALIACAO_FINAL!r}")
    if not isinstance(resultado["melhor_config"], dict):
        _erro(modelo, "melhor_config deve ser um objeto")

    _exigir(modelo, "ambiente", resultado["ambiente"], _CAMPOS_AMBIENTE)
    _exigir(modelo, "tempos", resultado["tempos"], _CAMPOS_TEMPOS)
    _exigir(modelo, "estudo", resultado["estudo"], _CAMPOS_ESTUDO)
    _exigir(modelo, "teste", resultado["teste"], _CAMPOS_TESTE)

    _exigir_numeros(modelo, "ks_dobras_melhor", resultado["ks_dobras_melhor"])
    if len(resultado["ks_dobras_melhor"]) != N_DOBRAS:
        _erro(modelo, f"ks_dobras_melhor precisa de {N_DOBRAS} valores")
    perdas = resultado["perdas_dobras"]
    if not isinstance(perdas, list) or len(perdas) != N_DOBRAS:
        _erro(modelo, f"perdas_dobras precisa de {N_DOBRAS} dobras")
    for i, dobra in enumerate(perdas):
        _exigir(modelo, f"perdas_dobras[{i}]", dobra, {"treino", "validacao"})
        _exigir(modelo, f"perdas_dobras[{i}].treino", dobra["treino"], _CAMPOS_PERDAS)
        _exigir(modelo, f"perdas_dobras[{i}].validacao", dobra["validacao"], _CAMPOS_PERDAS)

    y = resultado["y_teste"]
    if not isinstance(y, list) or len(y) != N_TESTE or any(v not in (0, 1) for v in y):
        _erro(modelo, f"y_teste precisa de {N_TESTE} rotulos 0 ou 1")
    if hash_y_teste(y) != HASH_Y_TESTE:
        _erro(modelo, "y_teste desalinhado com a ordem fixa do teste")
    _exigir_numeros(modelo, "probabilidades_teste", resultado["probabilidades_teste"])
    if len(resultado["probabilidades_teste"]) != N_TESTE:
        _erro(modelo, f"probabilidades_teste precisa de {N_TESTE} valores")
    por_semente = resultado["probabilidades_por_semente"]
    if not isinstance(por_semente, list) or not por_semente:
        _erro(modelo, "probabilidades_por_semente deve ser uma lista nao vazia")
    for i, p in enumerate(por_semente):
        _exigir_numeros(modelo, f"probabilidades_por_semente[{i}]", p)
        if len(p) != N_TESTE:
            _erro(modelo, f"probabilidades_por_semente[{i}] precisa de {N_TESTE} valores")
    _exigir_numeros(modelo, "teste.ks_por_semente", resultado["teste"]["ks_por_semente"])
    if len(resultado["teste"]["ks_por_semente"]) != len(por_semente):
        _erro(modelo, "teste.ks_por_semente e probabilidades_por_semente tem tamanhos diferentes")


def _para_json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"Valor sem representacao em JSON: {type(v).__name__}")


def salvar_resultado(modelo, resultado):
    """Valida e grava resultados/<modelo>.json, sem deixar arquivo parcial. Devolve o caminho."""
    if not re.fullmatch(r"[a-z0-9_]+", modelo):
        raise ValueError(f"Nome de modelo invalido: {modelo!r}")
    texto = json.dumps(resultado, indent=1, allow_nan=False, default=_para_json)
    validar_resultado(modelo, json.loads(texto))  # valida o que de fato sera lido depois
    pasta = Path(PASTA_RESULTADOS)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{modelo}.json"
    temporario = destino.with_suffix(".json.tmp")
    temporario.write_text(texto, encoding="utf-8")
    os.replace(temporario, destino)
    return destino


def carregar_resultados():
    """Le e valida todos os resultados/*.json. Devolve {modelo: resultado}."""
    resultados = {}
    for arquivo in sorted(Path(PASTA_RESULTADOS).glob("*.json")):
        resultado = json.loads(arquivo.read_text(encoding="utf-8"))
        validar_resultado(arquivo.stem, resultado)
        resultados[arquivo.stem] = resultado
    return resultados


if __name__ == "__main__":
    folds, final, cols = carregar_dobras()
    for i, (a, b, c, d) in enumerate(folds):
        print(f"Dobra {i}: treino {a.shape} (churn {b.mean()*100:.1f}%), validacao {c.shape} (churn {d.mean()*100:.1f}%)")
    print("final:", final[0].shape, final[2].shape, "churn teste %.1f%%" % (final[3].mean() * 100), "n_feat", len(cols))
    # conferencia do KS contra scipy
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < .265).astype(int)
    p = np.clip(rng.normal(.3 + .25 * y, .2), 0, 1)
    print("KS meu     :", ks(y, p))
    print("KS scipy   :", ks_2samp(p[y == 1], p[y == 0]).statistic)
