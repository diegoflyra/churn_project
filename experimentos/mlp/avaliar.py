"""Avaliacao final do MLP (treino inteiro, como na pre-proposta) + Entropia Cruzada x MSE.

Uso: python experimentos/mlp/avaliar.py 5      (5 sementes)
Gera: mlp.json na pasta CHURN_RESULTADOS (nos notebooks, a pasta da execucao).
Avaliacao da fase 1, mantida para os notebooks das fases 1 e 2. O entregavel resultados/mlp.json vem de relatorio_final.py;
sem CHURN_RESULTADOS, este script grava em resultados/mlp/avaliar_fase1 para nao sobrescreve-lo.
"""
import sys, os
_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _RAIZ)   # comum.py fica na raiz
os.environ.setdefault("CHURN_RESULTADOS", os.path.join(_RAIZ, "resultados", "mlp", "avaliar_fase1"))
import comum, modelo as M


def hp_from_params(prm):
    return dict(prm)


def main(n_seeds):
    comum.avaliar("mlp", hp_from_params, M.fit, n_seeds, fit_final=M.fit_final, salvar_oof=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
