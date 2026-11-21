"""Fase 6 do MLP: criterio de parada (perda ou KS) x paciencia, medidos no placar comum.

A parada pelo KS ja existiu, mas nunca foi medida limpa: na fase 3 a mesma dobra decidia a parada e media o KS (inflou
+0,009), e no Optuna da fase 5 o TPE so a sorteou 4 vezes em 100. A paciencia longa (30 e 50) so foi medida com a parada
pela perda. Aqui as versoes do topo do placar mudam so a parada e a paciencia; todo o resto fica igual.

Sem vazamento: o KS que decide a parada e o dos 20% internos da dobra; a dobra externa so mede.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import fase5 as F5

# As versoes do topo do placar, com o rotulo e a configuracao exatos de resultados/placar.jsonl (entram sem medir de novo)
_EXTRAS = ["cliente_novo", "delta_preco", "tenure_1"]
BASES = {
    "fase4": dict(hp=dict(camadas=1, unidades=10, ativ="relu", otim="adam", lr=1e-3, dropout=0.0, l2=1e-6, perda="ce", paciencia=10, plano="N"),
                  dados={}, extras=list(_EXTRAS)),
    "slides_otim_rmsprop": dict(hp=dict(camadas=1, unidades=10, ativ="relu", otim="rmsprop", lr=1e-3, dropout=0.0, l2=1e-6, perda="ce",
                                        paciencia=10, plano="N"), dados={}, extras=list(_EXTRAS)),
    "optuna": dict(hp=dict(camadas=1, unidades=7, funil=False, plano="N", ativ="logistica", otim="adam", lr=0.0005502494560509775,
                           dropout=0.16591432414606982, l2=0.008182215231749446, suav=0.05135642537399436, perda="ce", lote=512,
                           paciencia=10, parada="perda", peso_pos=0.0), dados={"over": False}, extras=["cliente_novo", "delta_preco"]),
}
PARADAS = ("perda", "ks")
PACIENCIAS = (10, 30, 50)
ABREV = {"fase4": "fase4", "slides_otim_rmsprop": "rmsprop", "optuna": "optuna"}


def grade_parada(bases=BASES, paradas=PARADAS, paciencias=PACIENCIAS):
    """{rotulo: (base, parada, paciencia, config)}. A celula (perda, paciencia da base) e a propria base, ja medida."""
    saida = {}
    for nb, b in bases.items():
        for pa in paradas:
            for pc in paciencias:
                if pa == "perda" and pc == b["hp"]["paciencia"]:
                    saida[nb] = (nb, pa, pc, b)
                    continue
                hp = dict(b["hp"], paciencia=pc)
                if pa == "ks":
                    hp["parada"] = "ks"
                saida[f"s6_{ABREV.get(nb, nb)}_{pa}_p{pc}"] = (nb, pa, pc, dict(hp=hp, dados=dict(b["dados"]), extras=list(b["extras"])))
    return saida
