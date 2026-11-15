"""Mitra v2 (autogluon/mitra-classifier-2): espaco de busca, precisao na GPU e medida no placar comum.

O que vem da leitura do autogluon.tabular 1.6.3 (models/mitra):
- `predict_proba` usa como contexto tudo o que foi passado ao `fit` (`self.X`, `self.y`).
- `fit` avalia `X_val` uma vez no passo 0, mesmo sem fine-tuning; com fine-tuning, o checkpoint guarda o menor log loss
  em `X_val`, incluindo o passo 0, e nao registra qual passo venceu.
- A precisao e fixada em "bfloat16" em `_create_config` (sklearn_interface.py:147) e usada em todo `torch.autocast`
  (trainer_finetune.py:124, 240 e 299). A T4 do Kaggle (capacidade 7.5) nao tem bfloat16 nativo.
- Se faltar memoria na GPU, o `fit` corta o contexto pela metade e so imprime um aviso. Aqui isso vira erro.
- Com fine-tuning, cada passo treina com 80% do contexto como suporte e ate 1024 linhas como consulta
  (dataset_finetune.py:110). O contexto do Mitra e o mesmo `max_samples_support` no treino e na previsao, entao o corte
  por falta de memoria no treino tambem encolhe o contexto da previsao.
- Depois de treinar, cada estimador guarda na GPU o otimizador AdamW (2 copias dos pesos) e os gradientes (1 copia):
  `post_fit_optimize` existe para liberar isso, mas o MitraClassifier nunca o chama. Com 8 estimadores, o oitavo treina
  com o lixo dos sete anteriores ocupando a GPU.
- O escalonamento do gradiente vem desligado e, ligado, nunca desce de 65536 (grad_scaler_scale_min): em float16, um
  estouro faria todos os passos seguintes serem descartados.

Placar (docs/pre-proposta.md, item 2), em cada dobra:
- Sem fine-tuning: o contexto e o treino inteiro da dobra. O `X_val` do `fit` e uma amostra da parada; com zero passos
  ele so e avaliado, nao decide nada.
- Com fine-tuning: os pesos sao ajustados com os 80% de ajuste como contexto e o checkpoint e escolhido nos 20% de
  parada. A previsao usa os 80% de ajuste como contexto. O placar pede retreino no treino inteiro pelo numero de passos
  encontrado, mas o Mitra nao expoe esse numero; e por a parada no contexto e escolher o checkpoint nela vazaria.
"""
import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # comum.py fica na raiz
import comum  # noqa: E402

MODELO = "mitra"
HF_MODEL = "autogluon/mitra-classifier-2"
CONTEXTO_MAXIMO = 8192  # max_samples_support do autogluon.tabular 1.6.3; o treino inteiro com oversampling tem 7.760 linhas
N_AVALIACAO = 128  # linhas da parada que o fit avalia no passo 0 quando nao ha fine-tuning
PRECISOES = ("bfloat16", "float16", "float32")

# Fixos (pre-proposta, Frente 5): com warmup_steps=1000 (padrao) a taxa nem chega ao valor nominal em 100 passos
FIXOS = dict(warmup_steps=10, patience=10, use_random_transforms=False, metric="log_loss")

# Trial 0: contexto puro, um estimador, os padroes do AutoGluon e o oversampling do grupo
PADRAO = dict(n_estimators=1, random_mirror_x=True, shuffle_features=False, shuffle_classes=False, over=True, fine_tune=False)


def sugerir(t):
    """So contexto. O fine-tuning ficou fora da busca pelas medidas na T4 do Kaggle (teste-mitra-20261115-1719, branch
    resultados): em float32 ele nao cabe com oversampling; em float16 custa cerca de 19 s por passo e por estimador, contra
    41 s por estimador para uma dobra inteira sem ele. Um trial com 2 estimadores e 20 passos levou 102 min e empatou com o
    trial 0 (0,5483 contra 0,5479), e o espaco com ate 8 estimadores e 100 passos passaria de centenas de horas de GPU.
    O fine-tuning continua em ajustar_e_prever, para as medidas."""
    return dict(
        n_estimators=t.suggest_categorical("n_estimators", [1, 2, 4, 8]),
        random_mirror_x=t.suggest_categorical("random_mirror_x", [True, False]),
        shuffle_features=t.suggest_categorical("shuffle_features", [True, False]),
        shuffle_classes=t.suggest_categorical("shuffle_classes", [True, False]),
        over=t.suggest_categorical("over", [True, False]),  # oversampling do contexto
        fine_tune=False,
    )


# ---------------- GPU e precisao ----------------
def _torch():
    import torch
    return torch


def bf16_nativo():
    torch = _torch()
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def precisao_para(hp, pedida="auto", nativo=None):
    """auto: bfloat16 se a GPU tem suporte nativo; senao float32. As outras valem como pedidas: em float16, o fine-tuning
    roda com o escalonamento do gradiente ligado (com_precisao), a receita padrao de precisao mista do torch."""
    if pedida not in ("auto", *PRECISOES):
        raise ValueError(f"precisao desconhecida: {pedida!r}")
    if pedida == "auto":
        return "bfloat16" if (bf16_nativo() if nativo is None else nativo) else "float32"
    return pedida


def com_precisao(base):
    """Subclasse de `base` (o MitraClassifier) que troca a precisao fixada em _create_config pela do atributo `precisao`.
    Em float16, liga o escalonamento do gradiente e deixa a escala descer ate 1: sem ele, gradientes pequenos viram zero;
    com o piso de 65536 do AutoGluon, um estouro descartaria todos os passos seguintes."""

    class MitraComPrecisao(base):
        precisao = "bfloat16"

        def _create_config(self, *args, **kwargs):
            cfg, modelo = super()._create_config(*args, **kwargs)
            cfg.hyperparams["precision"] = self.precisao
            if self.precisao == "float16":
                cfg.hyperparams.update(grad_scaler_enabled=True, grad_scaler_scale_min=1.0)
            return cfg, modelo

    return MitraComPrecisao


def que_libera(base):
    """Subclasse de `base` (o TrainerFinetune) que, ao fim do treino, guarda a escala do gradiente e solta da GPU o
    otimizador e os gradientes. A previsao so usa cfg, preprocessor, model e device (trainer_finetune.py:281)."""

    class TrainerQueLibera(base):
        def train(self, *args, **kwargs):
            out = super().train(*args, **kwargs)
            self.escala_final = float(self.scaler.get_scale()) if self.scaler.is_enabled() else None
            self.model.zero_grad(set_to_none=True)
            self.post_fit_optimize()
            liberar_gpu()
            return out

    return TrainerQueLibera


_classe = {}


def classe():
    if "c" not in _classe:
        from autogluon.tabular.models.mitra import sklearn_interface as si
        if not getattr(si.TrainerFinetune, "libera", False):  # _train_ensemble cria os treinadores por este nome
            si.TrainerFinetune = que_libera(si.TrainerFinetune)
            si.TrainerFinetune.libera = True
        _classe["c"] = com_precisao(si.MitraClassifier)
        # autocast em float32 nao existe na GPU: o torch avisa e roda sem autocast, que e o que se quer
        warnings.filterwarnings("ignore", message="In CUDA autocast, but the target dtype is not supported")
    return _classe["c"]


def usar_classe(c):
    """Troca a classe do Mitra (os testes usam um classificador falso). None volta ao MitraClassifier."""
    _classe.clear()
    if c is not None:
        _classe["c"] = c


def semear(semente):
    """O `seed` do Mitra so controla a amostragem do contexto; o espelho e o embaralhamento vem de np.random e os pesos
    do fine-tuning, do torch."""
    np.random.seed(semente % 2**32)
    torch = _torch()
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def liberar_gpu():
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def criar(hp, semente, precisao):
    params = {k: v for k, v in hp.items() if k != "over"}
    m = classe()(hf_model=HF_MODEL, device="cuda", seed=int(semente), verbose=False, **FIXOS, **params)
    m.precisao = precisao
    return m


def conferir_contexto(m):
    contexto = m.trainers[0].cfg.hyperparams["max_samples_support"]
    if contexto != CONTEXTO_MAXIMO:
        raise RuntimeError(f"Faltou memoria na GPU: o AutoGluon reduziu o contexto para {contexto} linhas, o que mudaria "
                           "o modelo. Com fine-tuning, use PRECISAO = \"float16\" (etapa de medidas) ou uma GPU maior.")


# ---------------- placar ----------------
def amostra_estratificada(posicoes, y, n, semente):
    """Ate n das `posicoes`, estratificadas por y[posicoes]."""
    posicoes = np.asarray(posicoes)
    if len(posicoes) <= n:
        return posicoes
    sub, _ = train_test_split(posicoes, train_size=n, stratify=np.asarray(y)[posicoes], random_state=semente % 2**32)
    return np.sort(sub)


def _f32(X):
    return np.asarray(X, dtype="float32")


def _int(y):
    return np.asarray(y).astype(int)


ULTIMO_AJUSTE = {}  # "escalas": escala final do gradiente de cada estimador no ultimo ajuste (None sem escalonamento)


def ajustar_e_prever(hp, X, y, X_aval, parada, semente, precisao, features=()):
    """O ajuste de uma dobra (ou do treino inteiro, na avaliacao final). Devolve a probabilidade de churn em X_aval.
    parada: posicoes em X que ficam fora do ajuste com fine-tuning e de onde sai a avaliacao do passo 0 sem ele."""
    liberar_gpu()
    semear(semente)
    m = criar(hp, semente, precisao)
    parada = np.asarray(parada)
    ULTIMO_AJUSTE.clear()
    try:
        if hp["fine_tune"]:
            ajuste = np.setdiff1d(np.arange(len(X)), parada)
            Xa, ya, Xr = comum.preparar(X.iloc[ajuste], y.iloc[ajuste], pd.concat([X.iloc[parada], X_aval]),
                                        features=features, over=hp["over"])
            m.fit(_f32(Xa), _int(ya), _f32(Xr.iloc[:len(parada)]), _int(y.iloc[parada]))
            Xv = Xr.iloc[len(parada):]
        else:
            sub = amostra_estratificada(parada, y, N_AVALIACAO, semente)
            XA, yA, Xr = comum.preparar(X, y, pd.concat([X_aval, X.iloc[sub]]), features=features, over=hp["over"])
            m.fit(_f32(XA), _int(yA), _f32(Xr.iloc[len(X_aval):]), _int(y.iloc[sub]))
            Xv = Xr.iloc[:len(X_aval)]
        ULTIMO_AJUSTE["escalas"] = [getattr(t, "escala_final", None) for t in m.trainers]
        conferir_contexto(m)
        return np.asarray(m.predict_proba(_f32(Xv)))[:, 1]
    finally:  # tambem quando o contexto encolheu: a proxima medida comeca com a GPU vazia
        del m
        liberar_gpu()


def ajustar_prever_de(hp, features=(), precisao="auto"):
    """ajustar_prever no formato de comum.medir_placar. Sem MSE de treino: prever o proprio contexto e decorar."""
    prec = precisao_para(hp, precisao)

    def ajustar_prever(X, y, X_val, ajuste, parada, semente):
        return ajustar_e_prever(hp, X, y, X_val, parada, semente, prec, features)

    ajustar_prever.precisao = prec
    return ajustar_prever


def medir_divisao(hp, X_tr, y_tr, d, precisao, features=()):
    """Uma divisao do placar, com tempo e pico de memoria da GPU: a base das medidas que fixam o N de trials."""
    import time
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    p = ajustar_e_prever(hp, X_tr.iloc[d["treino"]], y_tr.iloc[d["treino"]], X_tr.iloc[d["validacao"]], d["parada"],
                         1000 * d["semente"] + d["dobra"], precisao, features)
    tempo = time.perf_counter() - t0
    pico = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return dict(p=p, ks=comum.ks(np.asarray(y_tr)[d["validacao"]], p), tempo_s=tempo, pico_gb=pico,
                escalas=list(ULTIMO_AJUSTE["escalas"]))


def final(hp, X_tr, y_tr, X_te, n_sementes=5, features=(), precisao="auto"):
    """Treino inteiro: a mesma separacao de 20% do placar, uma por semente. Uma previsao do teste por semente."""
    prec = precisao_para(hp, precisao)
    y = np.asarray(y_tr)
    p_sementes = []
    for s in range(n_sementes):
        _, parada = train_test_split(np.arange(len(y)), test_size=comum.PLACAR_FRACAO_PARADA, stratify=y,
                                     random_state=comum.SEMENTE + s)
        p_sementes.append(ajustar_e_prever(hp, X_tr, y_tr, X_te, np.sort(parada), comum.SEMENTE + s, prec, features))
    return dict(p_sementes=[p.tolist() for p in p_sementes], precisao=prec)
