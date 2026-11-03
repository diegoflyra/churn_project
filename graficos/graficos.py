# Gráficos padronizados, lidos de resultados/<modelo>.json:
#   - ROC sobreposto
#   - Precision-Recall sobreposto
#   - KS sobreposto (TPR - FPR por limiar)
#   - Importâncias do Optuna agrupadas pelas quatro frentes
#   - Painel de importâncias (os sete modelos lado a lado)
#   - Matrizes de confusão em grade (para "erros em comum")
#   - Correlação das probabilidades (para "erros em comum")
#
# não importa comum.py
# os gráficos passam a funcionar assim que cada
# modelo salvar seu resultados/<modelo>.json.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ----------------------------- Estilo comum ----------------------------------
ORDEM_MODELOS = ["mlp", "stab", "tabpfn", "kan", "tabkan", "mitra", "gradient_boosting"]
NOMES_BONITOS = {
    "mlp": "MLP",
    "stab": "STab",
    "tabpfn": "TabPFN v2",
    "kan": "KAN 2.0",
    "tabkan": "TabKAN",
    "mitra": "Mitra v2",
    "gradient_boosting": "Gradient Boosting",
}

# Paleta consistente: cada modelo tem a mesma cor em todos os gráficos.
_tab = plt.get_cmap("tab10")
CORES_MODELO = {m: _tab(i % 10) for i, m in enumerate(ORDEM_MODELOS)}

# Quatro frentes de hiperparâmetros (a mesma segmentação em todos os modelos).
CORES_TRILHA = {
    "arquitetura":   "#4c72b0",
    "otimizacao":    "#dd8452",
    "regularizacao": "#55a868",
    "perda_dados":   "#c44e52",
}

# Se um modelo novo aparecer, ele cai num fallback: tudo vira uma cor só. Basta acrescentar aqui.
TRILHAS_POR_MODELO = {
    "mlp": {
        "arquitetura":   ["camadas", "unidades", "ativacao"],
        "otimizacao":    ["otimizador", "lr"],
        "regularizacao": ["dropout", "l2"],
        "perda_dados":   ["loss"],
    },
    "stab": {
        "arquitetura":   ["profundidade", "largura", "ativacao"],
        "otimizacao":    ["otimizador", "lr"],
        "regularizacao": ["p_mask", "weight_decay"],
        "perda_dados":   ["loss"],
    },
    "tabpfn": {
        "arquitetura":   ["n_estimators", "temperatura", "media_logits"],
        "otimizacao":    ["fine_tuning", "lr"],
        "regularizacao": ["balanceamento", "contexto"],
        "perda_dados":   ["oversampling"],
    },
    "kan": {
        "arquitetura":   ["width", "depth", "grid", "k"],
        "otimizacao":    ["opt", "lr"],
        "regularizacao": ["lamb"],
        "perda_dados":   ["loss"],
    },
    "tabkan": {
        "arquitetura":   ["familia", "profundidade", "largura"],
        "otimizacao":    ["otimizador", "lr"],
        "regularizacao": ["base_param", "l2"],
        "perda_dados":   ["loss"],
    },
    "mitra": {
        "arquitetura":   ["n_estimators", "espelho"],
        "otimizacao":    ["fine_tuning", "lr", "passos"],
        "regularizacao": ["embaralhar_cols", "embaralhar_classes"],
        "perda_dados":   ["oversampling"],
    },
    "gradient_boosting": {
        "arquitetura":   ["max_depth", "min_child_weight"],
        "otimizacao":    ["learning_rate", "subsample"],
        "regularizacao": ["colsample_bytree", "reg_lambda", "gamma"],
        "perda_dados":   ["loss"],
    },
}


# ----------------------------- Carregamento ----------------------------------
def carregar_resultados(pasta="resultados", modelos=None):
    """Lê resultados/<modelo>.json. Devolve {nome: payload}."""
    modelos = modelos or ORDEM_MODELOS
    out = {}
    for m in modelos:
        caminho = os.path.join(pasta, f"{m}.json")
        if os.path.exists(caminho):
            with open(caminho, encoding="utf-8") as f:
                out[m] = json.load(f)
    return out


def _verificar_y_teste(resultados):
    """Todos os modelos têm que usar o mesmo y_teste. Aborta se não."""
    ys = {m: tuple(r["y_teste"]) for m, r in resultados.items() if "y_teste" in r}
    if not ys:
        raise ValueError("Nenhum resultado tem 'y_teste'. Contrato violado.")
    primeiro_nome, primeiro_y = next(iter(ys.items()))
    for m, y in ys.items():
        if y != primeiro_y:
            raise ValueError(
                f"y_teste diverge entre '{m}' e '{primeiro_nome}'. "
                f"Os sete modelos precisam do mesmo teste estratificado."
            )
    return np.asarray(primeiro_y, dtype=int)


def _curvas_roc(resultados):
    y = _verificar_y_teste(resultados)
    curvas = {}
    for m, r in resultados.items():
        if m not in resultados or "probabilidades_teste_media" not in r:
            continue
        prob = np.asarray(r["probabilidades_teste_media"], dtype=float)
        fpr, tpr, thr = roc_curve(y, prob)
        curvas[m] = {"fpr": fpr, "tpr": tpr, "thr": thr,
                     "ks": tpr - fpr, "prob": prob}
    return y, curvas


# ----------------------------- ROC / PR / KS ---------------------------------
def plot_roc_overlay(resultados, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    _, curvas = _curvas_roc(resultados)
    for m, c in curvas.items():
        ax.plot(c["fpr"], c["tpr"], color=CORES_MODELO.get(m),
                label=f"{NOMES_BONITOS.get(m, m)} (KS={c['ks'].max():.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — sete modelos sobrepostos")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    return ax.figure, ax


def plot_pr_overlay(resultados, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    y, curvas = _curvas_roc(resultados)
    for m, c in curvas.items():
        p, r, _ = precision_recall_curve(y, c["prob"])
        ap = auc(r, p)
        ax.plot(r, p, color=CORES_MODELO.get(m),
                label=f"{NOMES_BONITOS.get(m, m)} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — sete modelos sobrepostos")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)
    return ax.figure, ax


def plot_ks_overlay(resultados, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    _, curvas = _curvas_roc(resultados)
    for m, c in curvas.items():
        ax.plot(c["thr"], c["ks"], color=CORES_MODELO.get(m),
                label=f"{NOMES_BONITOS.get(m, m)} (KS={c['ks'].max():.3f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Limiar"); ax.set_ylabel("TPR − FPR")
    ax.set_title("Curva KS — sete modelos sobrepostos")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    return ax.figure, ax


def plot_barras_ks_com_ic(resultados, ax=None):
    """KS no teste com IC 95% por bootstrap, ordenado."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    itens = []
    for m, r in resultados.items():
        res = r.get("teste_resumo_5_sementes", {}).get("ks")
        if not res: continue
        itens.append((m, res["media"], res.get("ic95_bootstrap", [res["media"]]*2)))
    itens.sort(key=lambda x: x[1])
    ys = np.arange(len(itens))
    medias = [x[1] for x in itens]
    los = [x[1] - x[2][0] for x in itens]
    his = [x[2][1] - x[1] for x in itens]
    cores = [CORES_MODELO.get(x[0], "gray") for x in itens]
    ax.barh(ys, medias, xerr=[los, his], capsize=3, color=cores)
    ax.set_yticks(ys)
    ax.set_yticklabels([NOMES_BONITOS.get(x[0], x[0]) for x in itens])
    ax.set_xlabel("KS no teste (média e IC 95% por bootstrap)")
    ax.set_title("KS no teste com intervalo de 95%")
    ax.grid(alpha=0.2, axis="x")
    return ax.figure, ax


# ----------------------------- Importâncias ----------------------------------
def _importancias_por_trilha(modelo, imp):
    """Devolve [(parametro, importancia, trilha), ...] ordenado por importancia."""
    trilhas = TRILHAS_POR_MODELO.get(modelo)
    if trilhas:
        pares = []
        for trilha, params in trilhas.items():
            for p in params:
                if p in imp:
                    pares.append((p, float(imp[p]), trilha))
        # parâmetros presentes em imp mas não no mapeamento: entram como 'outros'
        for p, v in imp.items():
            if p not in [q[0] for q in pares]:
                pares.append((p, float(v), "outros"))
    else:
        pares = [(p, float(v), "outros") for p, v in imp.items()]
    pares.sort(key=lambda t: t[1])
    return pares


def plot_importancias(resultados, modelo, ax=None):
    """Importâncias do Optuna de UM modelo, coloridas pelas quatro frentes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    r = resultados.get(modelo)
    if r is None or "importancias_optuna" not in r:
        ax.text(0.5, 0.5, f"sem importâncias para {modelo}",
                ha="center", va="center")
        ax.set_axis_off()
        return ax.figure, ax
    pares = _importancias_por_trilha(modelo, r["importancias_optuna"])
    labels = [f"{p}" for p, _, _ in pares]
    vals = [v for _, v, _ in pares]
    cores = [CORES_TRILHA.get(t, "#7f7f7f") for _, _, t in pares]
    ax.barh(np.arange(len(pares)), vals, color=cores)
    ax.set_yticks(np.arange(len(pares)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Importância (Optuna)")
    ax.set_title(f"Importâncias por frente — {NOMES_BONITOS.get(modelo, modelo)}")
    ax.grid(alpha=0.2, axis="x")
    # legenda das trilhas presentes
    presentes = sorted({t for _, _, t in pares})
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORES_TRILHA.get(t, "#7f7f7f"))
               for t in presentes]
    ax.legend(handles, presentes, fontsize=8, loc="lower right")
    return ax.figure, ax


def plot_importancias_painel(resultados):
    """Grade 2×4 com as importâncias dos sete modelos lado a lado."""
    modelos = [m for m in ORDEM_MODELOS if m in resultados
               and "importancias_optuna" in resultados[m]]
    if not modelos:
        raise ValueError("Nenhum modelo tem 'importancias_optuna'.")
    n = len(modelos)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, m in enumerate(modelos):
        plot_importancias(resultados, m, ax=axes[i])
    for j in range(len(modelos), len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Importâncias agrupadas pelas quatro frentes — sete modelos",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    return fig, axes


# ----------------------------- Erros em comum --------------------------------
def plot_matrizes_confusao(resultados):
    """Grade 2×4 com as matrizes de confusão dos sete modelos (limiar de KS)."""
    modelos = [m for m in ORDEM_MODELOS if m in resultados]
    if not modelos:
        raise ValueError("Nenhum resultado carregado.")
    n = len(modelos)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, m in enumerate(modelos):
        r = resultados[m]
        mt = r.get("metricas_teste_prob_media", {})
        cm = np.array([[mt.get("tn", 0), mt.get("fp", 0)],
                       [mt.get("fn", 0), mt.get("tp", 0)]])
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Previsto 0", "Previsto 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0", "Real 1"])
        ax.set_title(NOMES_BONITOS.get(m, m))
    for j in range(len(modelos), len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Matrizes de confusão no teste (limiar de KS)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_correlacao_prob(resultados, ax=None):
    """Correlação de Pearson entre as probabilidades dos sete modelos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    nomes, probs = [], []
    for m in ORDEM_MODELOS:
        if m in resultados and "probabilidades_teste_media" in resultados[m]:
            nomes.append(m)
            probs.append(np.asarray(resultados[m]["probabilidades_teste_media"]))
    if len(probs) < 2:
        ax.text(0.5, 0.5, "menos de dois modelos", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure, ax
    C = np.corrcoef(np.vstack(probs))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [NOMES_BONITOS.get(m, m) for m in nomes]
    ax.set_xticks(range(len(nomes))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(nomes))); ax.set_yticklabels(labels)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(C[i, j]) > 0.6 else "black")
    ax.set_title("Correlação entre probabilidades no teste")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax.figure, ax