"""Fase 2 do FT-Transformer: regularizacao, medida no placar comum.

O que a fase 1 mostrou: tudo empatou com o trial 0, e o trial 0 para na epoca mediana 6 (dobras param ja na epoca 1).
A rede aprende o que consegue em poucas epocas e logo a perda na separacao interna piora: a parada corta o treino antes
de o sobreajuste aparecer no MSE. Hipotese: regularizando, a rede treina por mais tempo sem decorar e aprende mais.

Cada regularizador sozinho, em duas intensidades, sobre o trial 0, e depois duas combinacoes:
- dropout: atencao, FFN e residual (o artigo usa 0,2 / 0,1 / 0);
- L2 e L1 classicos, somados a perda (modelo.penalidade);
- weight decay do AdamW: com lr 1e-4, o 1e-5 do artigo nao encolhe nada; so 1,0 pesa (~4% em 20 epocas).
LayerNorm ja esta em todo bloco. BatchNorm fica de fora: normaliza pelo lote, nao combina com os tokens de um
Transformer e piorou o MLP em todas as camadas.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, modelo as M, fase1 as F1

DROP = {"medio": dict(drop_atencao=0.3, drop_ffn=0.3, drop_residual=0.1),
        "pesado": dict(drop_atencao=0.5, drop_ffn=0.5, drop_residual=0.2)}
_MUDA = {"r_drop_medio": DROP["medio"], "r_drop_pesado": DROP["pesado"],
         "r_l2_1e-4": dict(l2=1e-4), "r_l2_1e-3": dict(l2=1e-3),
         "r_l1_1e-6": dict(l1=1e-6), "r_l1_1e-5": dict(l1=1e-5),
         "r_wd_1": dict(wd=1.0),
         "r_combo_moderado": dict(DROP["medio"], l2=1e-4),
         "r_combo_pesado": dict(DROP["pesado"], l2=1e-3)}
REGULARIZACAO = {r: dict(hp=dict(F1.TRIAL0["hp"], **m), dados=dict(F1.TRIAL0["dados"])) for r, m in _MUDA.items()}


def curvas(c, k=0, epocas=100, dispositivo=None, semente=0):
    """Treina na parte de ajuste da dobra k por `epocas` fixas, sem parar, e anota por epoca a entropia cruzada no
    ajuste (clientes unicos, sem as copias do oversampling) e na separacao de parada, mais o KS da parada.
    Mostra o sobreajuste: a CE do treino segue caindo enquanto a da parada volta a subir."""
    Xtr, _, ytr, _ = F1._dados()
    d = F1.divs()[k]
    X, y = Xtr.iloc[d["treino"]], ytr.iloc[d["treino"]]
    Xaj, yaj, Xpa, ypa = X.iloc[d["ajuste"]], y.iloc[d["ajuste"]], X.iloc[d["parada"]], y.iloc[d["parada"]]
    Xa, ya, Xav = comum.preparar(Xaj, yaj, pd.concat([Xpa, Xaj]), **c["dados"])
    f32 = lambda a: np.asarray(a, dtype="float32")
    Xp_, Xaj_ = f32(Xav)[:len(Xpa)], f32(Xav)[len(Xpa):]
    yp_, yaj_ = np.asarray(ypa), np.asarray(yaj)
    ce_t, ce_p, ks_p = [], [], []

    def anota(ep, m):
        pt, pp = M.prever(m, Xaj_), M.prever(m, Xp_)
        ce_t.append(comum.ce(yaj_, pt)); ce_p.append(comum.ce(yp_, pp)); ks_p.append(comum.ks(yp_, pp))

    M.treinar(f32(Xa), f32(ya), c["hp"], F1._semente(d, semente), epocas=epocas, ao_fim_epoca=anota, dispositivo=dispositivo)
    return dict(ce_treino=ce_t, ce_parada=ce_p, ks_parada=ks_p, melhor_epoca=int(np.argmin(ce_p)) + 1,
                n_treino=len(Xaj_), n_parada=len(Xp_))


# Paleta do projeto (a mesma das figuras do MLP), validada para daltonismo sobre o fundo
AZUL, LARANJA, FUNDO, TINTA, SUAVE, GRADE = "#2a78d6", "#eb6834", "#fcfcfb", "#1f2328", "#6b7280", "#e5e7eb"


def figura_curvas(cvs, arquivo=None, titulo="Entropia cruzada por época: treino contra a separação de parada (dobra 0)"):
    """Um painel por configuracao, mesmo eixo y. Linha pontilhada = epoca de menor CE na parada (onde a parada pararia)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    n = len(cvs)
    nc = min(n, 2)
    nl = -(-n // nc)
    fig, axs = plt.subplots(nl, nc, figsize=(max(5.2 * nc, 9), 3.4 * nl + .6), sharey=True, facecolor=FUNDO, squeeze=False)
    topo = max(max(max(v["ce_treino"]), max(v["ce_parada"])) for v in cvs.values())
    base = min(min(min(v["ce_treino"]), min(v["ce_parada"])) for v in cvs.values())
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    for ax, (rot, v) in zip(axs.flat, cvs.items()):
        ep = np.arange(1, len(v["ce_treino"]) + 1)
        ax.set_facecolor(FUNDO)
        ax.plot(ep, v["ce_treino"], color=AZUL, lw=2, label="treino (ajuste, sem cópias)")
        ax.plot(ep, v["ce_parada"], color=LARANJA, lw=2, label="parada (20% internos)")
        ax.axvline(v["melhor_epoca"], color=SUAVE, lw=1, ls=":")
        direita = v["melhor_epoca"] > len(ep) / 2
        ax.text(v["melhor_epoca"], topo + .02, f"menor CE: época {v['melhor_epoca']} ", color=SUAVE, fontsize=8, va="top",
                ha="right" if direita else "left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for serie, cor, nome in (("ce_treino", AZUL, "treino"), ("ce_parada", LARANJA, "parada")):
            ax.annotate(nome, (ep[-1], v[serie][-1]), xytext=(4, 0), textcoords="offset points", color=TINTA, fontsize=8, va="center")
        ax.set_title(rot, color=TINTA, fontsize=10, loc="left")
        ax.set_xlabel("época", color=SUAVE, fontsize=9)
        ax.set_ylim(base - .02, topo + .03)
        ax.grid(axis="y", color=GRADE, lw=.8)
        ax.tick_params(colors=SUAVE, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRADE)
    for linha in axs:
        linha[0].set_ylabel("entropia cruzada", color=SUAVE, fontsize=9)
    fig.suptitle(titulo, color=TINTA, fontsize=11, x=.01, y=.995, ha="left")
    fig.legend(*axs[0][0].get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=TINTA, ncol=2,
               loc="upper left", bbox_to_anchor=(.01, .975 - .02 * (nl == 1)))
    fig.tight_layout(rect=(0, 0, 1, .9 if nl == 1 else .94))
    if arquivo:
        fig.savefig(arquivo, dpi=150, facecolor=FUNDO)
    return fig
