"""Avaliacao final do MLP (treino inteiro, como na pre-proposta) + Entropia Cruzada x MSE.

Uso: python experimentos/mlp/avaliar.py 5      (5 sementes)
Gera: resultados/mlp.json, com as previsoes de validacao por cliente
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M


def hp_from_params(prm):
    return dict(prm)


def main(n_seeds):
    comum.avaliar("mlp", hp_from_params, M.fit, n_seeds, fit_final=M.fit_final, salvar_oof=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
