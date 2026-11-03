# _exemplo_graficos.py — gera mocks e salva todos os PNGs em resultados/demo/.
# Só para o checkpoint de 3/11. Descartável após os modelos reais existirem.

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import graficos
from graficos import ORDEM_MODELOS, NOMES_BONITOS

SAIDA = "resultados/demo"

# skill sintético por modelo, só para ter curvas com separação visível
SKILL = {"mlp": 0.68, "stab": 0.72, "tabpfn": 0.78, "kan": 0.76,
         "tabkan": 0.74, "mitra": 0.80, "gradient_boosting": 0.84}
TEMPO = {"mlp": 120.0, "stab": 240.0, "tabpfn": 600.0, "kan": 900.0,
         "tabkan": 300.0, "mitra": 1800.0, "gradient_boosting": 60.0}


def _mock_modelo(nome, y_teste, rng):
    skill = SKILL[nome]
    ruido = rng.beta(2, 2, len(y_teste))
    prob = np.clip(skill * y_teste + (1 - skill) * ruido, 0.001, 0.999)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_teste, prob)
    ks_full = float((tpr - fpr).max())
    lo, hi = sorted([ks_full - 0.03, ks_full + 0.03])
    limiar = 0.35
    y_pred = (prob >= limiar).astype(int)
    tn = int(((y_teste == 0) & (y_pred == 0)).sum())
    fp = int(((y_teste == 0) & (y_pred == 1)).sum())
    fn = int(((y_teste == 1) & (y_pred == 0)).sum())
    tp = int(((y_teste == 1) & (y_pred == 1)).sum())
    ks_folds = list(np.clip(rng.normal(ks_full, 0.015, 5), 0, 1))
    importancias = {p: float(rng.random())
                    for trilha in graficos.TRILHAS_POR_MODELO[nome].values()
                    for p in trilha}
    return {
        "modelo": nome,
        "best_params": {},
        "n_trials": 100,
        "best_value_cv": float(np.mean(ks_folds)),
        "ks_trial0_default": float(np.mean(ks_folds) - 0.02),
        "ganho_busca_vs_default": 0.02,
        "ks_folds": [float(x) for x in ks_folds],
        "limiar_ks": limiar,
        "teste_resumo_5_sementes": {
            "ks": {"media": ks_full, "desvio": 0.008, "ic95_bootstrap": [lo, hi]},
        },
        "metricas_teste_prob_media": {
            "ks": ks_full, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        },
        "ks_por_semente": list(np.clip(rng.normal(ks_full, 0.008, 5), 0, 1)),
        "comparacao_gb": None,
        "tempo_busca_s": TEMPO[nome],
        "tempo_medio_trial_s": TEMPO[nome] / 100,
        "tempo_max_trial_s": TEMPO[nome] / 30,
        "probabilidades_teste_media": [float(x) for x in prob],
        "y_teste": [int(x) for x in y_teste],
        "features": [f"f{i}" for i in range(30)],
        "importancias_optuna": importancias,
    }


def _salvar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, f"{nome}.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"  {caminho}")
    plt.close(fig)


def main():
    rng = np.random.default_rng(42)
    n_teste = 1760
    y_teste = (rng.random(n_teste) < 0.265).astype(int)
    os.makedirs("resultados", exist_ok=True)
    resultados = {}
    for m in ORDEM_MODELOS:
        payload = _mock_modelo(m, y_teste, rng)
        with open(f"resultados/{m}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        resultados[m] = payload

    print(f"Mock gerado em resultados/*.json ({len(resultados)} modelos).\n")
    print("Salvando PNGs:")

    _salvar(*graficos.plot_roc_overlay(resultados)[:1], "roc")
    _salvar(*graficos.plot_pr_overlay(resultados)[:1], "pr")
    _salvar(*graficos.plot_ks_overlay(resultados)[:1], "ks")
    _salvar(*graficos.plot_barras_ks_com_ic(resultados)[:1], "ks_barras_ic")
    _salvar(*graficos.plot_importancias_painel(resultados)[:1], "importancias_painel")
    _salvar(*graficos.plot_matrizes_confusao(resultados)[:1], "matrizes_confusao")
    _salvar(*graficos.plot_correlacao_prob(resultados)[:1], "correlacao")

    # painel único para a apresentação
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    graficos.plot_roc_overlay(resultados, ax=axes[0, 0])
    graficos.plot_pr_overlay(resultados, ax=axes[0, 1])
    graficos.plot_ks_overlay(resultados, ax=axes[0, 2])
    graficos.plot_barras_ks_com_ic(resultados, ax=axes[1, 0])
    graficos.plot_correlacao_prob(resultados, ax=axes[1, 1])
    graficos.plot_importancias(resultados, "kan", ax=axes[1, 2])
    fig.suptitle("Painel resumo — sete modelos (dados sintéticos, só demonstração)")
    _salvar(fig, "painel_resumo")


if __name__ == "__main__":
    main()