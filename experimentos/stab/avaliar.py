"""Avaliacao final do STab + experimento controlado Entropia Cruzada x MSE.

Uso: python experimentos/stab/avaliar.py 3      (3 sementes)
Gera: resultados/stab.json
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M


def hp_from_params(prm):
    return dict(prm)


if __name__ == "__main__":
    comum.avaliar("stab", hp_from_params, M.fit, int(sys.argv[1]))
