"""Avaliacao final do TabKAN + experimento controlado Entropia Cruzada x MSE.

Uso: python experimentos/tabkan/avaliar.py 3      (3 sementes)
Gera: resultados/tabkan.json
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M


def hp_from_params(prm):
    base = prm.get("base_spline", prm.get("base_cheb", prm.get("base_fourier")))
    lr = prm.get("lr_lbfgs", prm.get("lr_adam"))
    return dict(familia=prm["familia"], prof=prm["prof"], larg=prm["larg"], base=base, otim=prm["otim"],
                lr=lr, l2=prm["l2"], perda=prm["perda"])


if __name__ == "__main__":
    comum.avaliar("tabkan", hp_from_params, M.fit, int(sys.argv[1]))
