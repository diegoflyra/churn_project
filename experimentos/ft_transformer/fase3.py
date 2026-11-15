"""Fase 3 do FT-Transformer: criterio de parada (perda ou KS) x paciencia, medidos no placar comum.

O que a fase 2 mostrou: o dropout acabou com o sobreajuste (a curva da parada deixou de subir), mas o KS nao saiu do
ruido. A parada olhava so a entropia cruzada, e a CE da parada oscila uns 0,05 de uma epoca para outra: com paciencia 16,
dobras paravam na epoca 1. Nas curvas, o maior KS da parada caiu longe da epoca de menor CE nas versoes com dropout.

Pergunta: parar pelo KS, e com mais paciencia, sobe o KS do placar? Sem vazamento: o KS que decide a parada e o dos 20%
internos da dobra; a dobra externa so mede (o erro da fase 3 do MLP foi usar a mesma dobra para as duas coisas).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, fase1 as F1, fase2 as F2

BASES = {"trial0": F1.TRIAL0, "r_drop_medio": F2.REGULARIZACAO["r_drop_medio"]}
PARADAS = ("perda", "ks")
PACIENCIAS = (16, 32, 64)


def grade_parada(bases=BASES, paradas=PARADAS, paciencias=PACIENCIAS):
    """{rotulo: (base, parada, paciencia, config)}. A celula (perda, paciencia da base) e a propria base, ja medida."""
    saida = {}
    for nb, b in bases.items():
        for pa in paradas:
            for pc in paciencias:
                if pa == "perda" and pc == b["hp"]["paciencia"]:
                    saida[nb] = (nb, pa, pc, b)
                    continue
                hp = dict(b["hp"], paciencia=pc, **({"parada": "ks"} if pa == "ks" else {}))
                saida[f"s_{nb}_{pa}_p{pc}"] = (nb, pa, pc, dict(hp=hp, dados=dict(b["dados"])))
    return saida


def onde_para(serie, paciencia, maior=False):
    """Simula a parada numa curva por epoca: devolve a epoca (1, 2, ...) que a parada escolheria."""
    melhor, ep, esp = None, 0, 0
    for i, v in enumerate(serie, 1):
        v = -v if maior else v
        if melhor is None or v < melhor - 1e-6:
            melhor, ep, esp = v, i, 0
        else:
            esp += 1
            if esp >= paciencia:
                break
    return ep


def paradas_na_curva(cv, paciencias=PACIENCIAS):
    """Para cada criterio e paciencia: a epoca escolhida e o KS da parada nela."""
    linhas = []
    for pa, serie, maior in (("perda", cv["ce_parada"], False), ("ks", cv["ks_parada"], True)):
        for pc in paciencias:
            ep = onde_para(serie, pc, maior)
            linhas.append(dict(parada=pa, paciencia=pc, epoca=ep, ks_parada=cv["ks_parada"][ep - 1], ce_parada=cv["ce_parada"][ep - 1]))
    return linhas


AZUL, LARANJA, FUNDO, TINTA, SUAVE, GRADE = F2.AZUL, F2.LARANJA, F2.FUNDO, F2.TINTA, F2.SUAVE, F2.GRADE


def figura_ks(cvs, arquivo=None, paciencias=(16, 64),
              titulo="KS na separação de parada por época, e onde cada regra para (dobra 0)"):
    """Um painel por configuracao. Pontos: a epoca escolhida pela parada pela perda (azul) e pelo KS (laranja)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    n = len(cvs)
    nc = min(n, 2)
    nl = -(-n // nc)
    fig, axs = plt.subplots(nl, nc, figsize=(max(5.2 * nc, 9), 3.4 * nl + .6), sharey=True, facecolor=FUNDO, squeeze=False)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    for ax, (rot, v) in zip(axs.flat, cvs.items()):
        ks = v["ks_parada"]
        ep = np.arange(1, len(ks) + 1)
        ax.set_facecolor(FUNDO)
        ax.plot(ep, ks, color=SUAVE, lw=1.5)
        for pa, serie, maior, cor in (("perda", v["ce_parada"], False, AZUL), ("ks", ks, True, LARANJA)):
            por_epoca = {}
            for pc in paciencias:                                  # paciencias que param na mesma epoca dividem o ponto
                por_epoca.setdefault(onde_para(serie, pc, maior), []).append(pc)
            for j, (e, pcs) in enumerate(por_epoca.items()):
                ax.plot([e], [ks[e - 1]], "o", ms=8, color=cor, mec=FUNDO, mew=2, zorder=3,
                        label=("para pela perda" if pa == "perda" else "para pelo KS") if j == 0 else None)
                ax.annotate(", ".join(f"p{pc}" for pc in pcs), (e, ks[e - 1]), xytext=(0, 9 if pa == "ks" else -13),
                            textcoords="offset points", ha="center", fontsize=8, color=TINTA)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(rot, color=TINTA, fontsize=10, loc="left")
        ax.set_xlabel("época", color=SUAVE, fontsize=9)
        ax.grid(axis="y", color=GRADE, lw=.8)
        ax.tick_params(colors=SUAVE, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRADE)
    for linha in axs:
        linha[0].set_ylabel("KS na parada", color=SUAVE, fontsize=9)
    fig.suptitle(titulo, color=TINTA, fontsize=11, x=.01, y=.995, ha="left")
    fig.legend(*axs[0][0].get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=TINTA, ncol=2,
               loc="upper left", bbox_to_anchor=(.01, .975 - .02 * (nl == 1)))
    fig.tight_layout(rect=(0, 0, 1, .9 if nl == 1 else .94))
    if arquivo:
        fig.savefig(arquivo, dpi=150, facecolor=FUNDO)
    return fig


def logar_curvas(run, cvs):
    """Curvas por epoca no W&B: um grafico de KS e um de entropia cruzada por configuracao."""
    if run is None:
        return
    import wandb
    for rot, v in cvs.items():
        ep = list(range(1, len(v["ks_parada"]) + 1))
        run.log({f"curvas/{rot}/ks": wandb.plot.line_series(xs=ep, ys=[v["ks_parada"]], keys=["KS na parada"],
                                                             title=f"{rot}: KS na parada por época", xname="época"),
                 f"curvas/{rot}/ce": wandb.plot.line_series(xs=ep, ys=[v["ce_treino"], v["ce_parada"]], keys=["treino", "parada"],
                                                             title=f"{rot}: entropia cruzada por época", xname="época")})
