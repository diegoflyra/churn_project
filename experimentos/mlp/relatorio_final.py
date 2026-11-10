"""Fecha o MLP: resultados/mlp.json no formato comum e as tabelas e figuras pedidas nos slides.

Uso (da raiz):  CHURN_DEVICE=cpu python experimentos/mlp/relatorio_final.py
Le:   resultados/placar.jsonl (versao adotada em resultados/escolhidos.json), resultados/mlp/mlp_final.json,
      o estudo do Optuna da fase 5.
Gera: resultados/mlp.json (o entregavel, formato comum), resultados/mlp/slides/TABELAS.md e resultados/mlp/slides/*.png.
      As variacoes dos slides (perda, ativacao, otimizador, camadas) sao medidas no placar, pareadas com a adotada;
      o que ja estiver no placar e reaproveitado.
"""
import os, sys, json
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))]
import numpy as np, pandas as pd, optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import comum, fase5 as F5

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PASTA = os.path.join(RAIZ, "resultados")              # entregaveis e placar comum
TRABALHO = os.path.join(PASTA, "mlp")                   # arquivos de trabalho do MLP
SAIDA = os.path.join(TRABALHO, "slides")
ESTUDO = os.path.join(TRABALHO, "mlp-fase5-20261110-0001", "mlp_fase5.db")

# Cores (paleta de referencia do skill de visualizacao, validada para 2 series no fundo claro)
SUPERFICIE, TEXTO, TEXTO2, MUDO, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
AZUL, LARANJA = "#2a78d6", "#eb6834"


def variantes(c):
    """Uma mudanca por vez sobre a adotada, como pedem os slides 7, 14 e 27. Cada otimizador com a sua taxa usual."""
    hp = c["hp"]
    v = {"perda_mse": dict(perda="mse"),
         **{f"ativ_{a}": dict(ativ=a) for a in ("logistica", "tanh", "gelu", "leaky_relu") if a != hp["ativ"]},
         "otim_rmsprop": dict(otim="rmsprop"), "otim_sgd": dict(otim="sgd", lr=1e-2), "otim_adadelta": dict(otim="adadelta", lr=1.0),
         "camadas_2": dict(camadas=2, plano="NN")}
    return {f"slides_{k}": dict(c, hp={**hp, **m}) for k, m in v.items()}


def _eixos(ax, titulo, xl, yl):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12)
    ax.set_xlabel(xl, color=TEXTO2); ax.set_ylabel(yl, color=TEXTO2)
    ax.tick_params(colors=MUDO, labelcolor=TEXTO2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRADE)
    ax.grid(True, color=GRADE, lw=.6)
    ax.set_axisbelow(True)


def _fig():
    f, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    f.patch.set_facecolor(SUPERFICIE)
    return f, ax


def figuras(y, p, lim, nome):
    os.makedirs(SAIDA, exist_ok=True)
    fpr, tpr, thr = roc_curve(y, p)
    f, ax = _fig()
    ax.plot([0, 1], [0, 1], color=MUDO, lw=1, ls="--")
    ax.plot(fpr, tpr, color=AZUL, lw=2)
    ax.text(.55, .2, f"AUROC = {roc_auc_score(y, p):.3f}", color=TEXTO, fontsize=11)
    _eixos(ax, f"Curva ROC no teste — {nome}", "taxa de falsos positivos", "taxa de verdadeiros positivos")
    f.tight_layout(); f.savefig(os.path.join(SAIDA, "roc.png")); plt.close(f)

    pr, rc, _ = precision_recall_curve(y, p)
    f, ax = _fig()
    ax.axhline(y.mean(), color=MUDO, lw=1, ls="--")
    ax.text(.02, y.mean() + .02, f"churn no teste ({y.mean():.1%})", color=TEXTO2, fontsize=9)
    ax.plot(rc, pr, color=AZUL, lw=2)
    _eixos(ax, f"Precision-recall no teste — {nome}", "recall", "precision")
    ax.set_ylim(0, 1.02)
    f.tight_layout(); f.savefig(os.path.join(SAIDA, "precision_recall.png")); plt.close(f)

    # KS: distribuicao acumulada do escore de quem cancela e de quem fica; o KS e a maior distancia vertical
    xs = np.r_[0, np.sort(np.unique(p)), 1]
    c1 = np.searchsorted(np.sort(p[y == 1]), xs, side="right") / (y == 1).sum()
    c0 = np.searchsorted(np.sort(p[y == 0]), xs, side="right") / (y == 0).sum()
    i = int(np.argmax(c0 - c1))
    f, ax = _fig()
    ax.plot(xs, c0, color=AZUL, lw=2, label="ficam"); ax.plot(xs, c1, color=LARANJA, lw=2, label="cancelam")
    ax.vlines(xs[i], c1[i], c0[i], color=TEXTO, lw=1.5)
    ax.text(xs[i] + .02, (c0[i] + c1[i]) / 2, f"KS = {c0[i] - c1[i]:.4f}", color=TEXTO, fontsize=11, va="center")
    leg = ax.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXTO2)
    _eixos(ax, f"KS no teste — {nome}", "probabilidade prevista de churn", "fração acumulada")
    f.tight_layout(); f.savefig(os.path.join(SAIDA, "ks.png")); plt.close(f)


def figura_evolucao(linhas):
    """Ganho de KS de cada fase sobre o MLP padrao, pareado nos mesmos clientes (o veredito do placar)."""
    ordem = [("referencia", "fase 1–2 (referência)"), ("fase3_val", "fase 3 (validação)"), ("fase3_teste", "fase 3 (teste)"),
             ("fase4", "fase 4 (adotado)"), ("optuna", "fase 5 (Optuna)")]
    por = {r["rotulo"]: r for r in linhas}
    v = [por[r]["vs_trial0"] for r, _ in ordem]
    f, ax = _fig()
    yv = np.arange(len(ordem))[::-1]
    ax.axvline(0, color=MUDO, lw=1)
    for yi, (r, _), x in zip(yv, ordem, v):
        ax.hlines(yi, x["ic95"][0], x["ic95"][1], color=MUDO, lw=2)
        ax.scatter([x["delta"]], [yi], s=60, color=LARANJA if r == "fase4" else AZUL, zorder=3, edgecolor=SUPERFICIE, linewidth=2)
        ax.text(x["ic95"][1] + .001, yi, f"{x['delta']:+.4f}  {x['veredito']}", va="center", color=TEXTO, fontsize=9)
    ax.set_yticks(yv, [n for _, n in ordem], fontsize=9)
    ax.set_xlim(min(x["ic95"][0] for x in v) - .003, max(x["ic95"][1] for x in v) + .013)
    _eixos(ax, "Ganho de KS sobre o MLP padrão (trial 0)", "diferença de KS, pareada nos mesmos clientes (IC95)", "")
    ax.grid(False, axis="y")
    f.tight_layout(); f.savefig(os.path.join(SAIDA, "evolucao_fases.png")); plt.close(f)


def _resumo(r):
    """Os nomes do resumo do comum.avaliar, como [media, dp]. O placar nao guarda o dp de AUROC, MSE e CE (ficam null),
    nem a validacao com prior corrigido (null)."""
    return dict(ks=[r["ks_kfold"], float(np.std(r["ks_dobras"]))], auroc=[r["auc_oof"], None],
                ce_raw=[r.get("ce_val"), None], mse_raw=[r.get("mse_val"), None], ce_corr=[None, None], mse_corr=[None, None],
                tr_ce=[r.get("ce_treino"), None], tr_mse=[r.get("mse_treino"), None])


def _md(df):
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else (str(int(v)) if isinstance(v, float) and v.is_integer() and v > 1 else f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(L)


def main():
    esc = json.load(open(os.path.join(PASTA, "escolhidos.json"), encoding="utf-8"))["mlp"]["rotulo"]
    por = {r["rotulo"]: r for r in comum.ler_placar(PASTA) if r["modelo"] == "mlp"}
    adot = por[esc]
    fin = json.load(open(os.path.join(TRABALHO, "mlp_final.json")))
    _, _, ytr, yte = F5._dados()
    yte = np.asarray(yte)

    # 1. variacoes dos slides, no placar (pareadas com a adotada)
    contra = {"vs_trial0": ("mlp", "trial0"), "vs_adotado": ("mlp", esc)}
    V = variantes(adot["config"])
    linhas_v = F5.garantir(PASTA, [(r, c, contra) for r, c in V.items()], n_jobs=os.cpu_count() or 1)
    por.update({l["rotulo"]: l for l in linhas_v})

    # 2. CE x MSE tambem no teste (so comparacao, como no formato comum; nao muda a escolha)
    arq_mse = os.path.join(SAIDA, "final_mse.json")
    os.makedirs(SAIDA, exist_ok=True)
    if os.path.isfile(arq_mse):
        fin_mse = json.load(open(arq_mse))
    else:
        fin_mse = F5.final(V["slides_perda_mse"], 5)
        json.dump(fin_mse, open(arq_mse, "w"))
    p_ce, p_mse = np.asarray(fin["p_teste_media"]), np.mean(fin_mse["p_sementes"], 0)
    m_mse = F5.metricas_teste(PASTA, "slides_perda_mse", fin_mse)
    dt = F5.delta_teste(dict(y_teste=yte.tolist(), p_sementes=fin_mse["p_sementes"]), dict(y_teste=yte.tolist(), p_sementes=[p_ce.tolist()]))

    # 3. resultados/mlp.json no formato comum (o mesmo de stab.json e tabkan.json), com o placar junto
    st = optuna.load_study(study_name="mlp_fase5", storage=f"sqlite:///{ESTUDO}")
    est = comum.sumario_estudo(st)
    m = dict(fin["teste"])
    m["ce_corr"] = comum.ce(yte, comum.corrige_prior(p_ce)); m["mse_corr"] = comum.mse(yte, comum.corrige_prior(p_ce))
    v_mse = por["slides_perda_mse"]["vs_adotado"]
    # mesmos campos do comum.avaliar (stab.json, tabkan.json): diferenca CE - MSE por dobra; as 15 dobras sao as mesmas nos dois
    d_fold = np.asarray(adot["ks_dobras"]) - np.asarray(por["slides_perda_mse"]["ks_dobras"])
    saida = dict(
        modelo="mlp", melhor_config=adot["config"], ks_dobras_melhor=adot["ks_dobras"], estudo=est,
        ce_vs_mse=dict(delta_ks_val_media=float(d_fold.mean()), delta_ks_val_dp=float(d_fold.std(ddof=1)), delta_por_dobra=d_fold.tolist(),
                       dobras_ce_melhor=int((d_fold > 0).sum()), wilcoxon_p=float(wilcoxon(d_fold).pvalue) if np.any(d_fold != 0) else 1.0,
                       delta_ks_oof_ic95=[-v_mse["ic95"][1], -v_mse["ic95"][0]], veredito_mse_vs_ce=v_mse["veredito"],
                       ks_teste_por_semente=dict(ce=m.get("ks_por_semente"), mse=m_mse.get("ks_por_semente")),
                       delta_ks_teste=float(comum.ks(yte, p_ce) - comum.ks(yte, p_mse)), delta_ks_teste_ic95=dt["ic95"],
                       resumo={"ce": _resumo(adot), "mse": _resumo(por["slides_perda_mse"])},
                       teste={"ce": {k: m[k] for k in ("ks", "auroc", "ce", "mse", "f1")}, "mse": {k: m_mse[k] for k in ("ks", "auroc", "ce", "mse", "f1")}}),
        mse_ce_por_conjunto=dict(treino=dict(mse=adot.get("mse_treino"), ce=adot.get("ce_treino")),       # slide 14
                                 validacao=dict(mse=adot.get("mse_val"), ce=adot.get("ce_val")),
                                 teste=dict(mse=m["mse"], ce=m["ce"])),
        teste=m, ks_teste_por_semente=m.get("ks_por_semente"), probabilidades_teste=p_ce.tolist(), y_teste=[int(v) for v in yte],
        placar=dict(rotulo=esc, ks_kfold=adot["ks_kfold"], ep=adot["ep"], ks_oof=adot["ks_oof"], vs_trial0=adot.get("vs_trial0"),
                    vs_referencia=adot.get("vs_referencia"), arquivo="resultados/placar.jsonl"),
        aviso="ks_dobras_melhor tem as 15 dobras do placar (3 x 5). O estudo e o Optuna da fase 5 (100 trials, 6 divisoes por trial). "
              "mse_ce_por_conjunto: treino e validacao vem das dobras do placar (treino com oversampling 1:1), teste do teste unico.")
    json.dump(saida, open(os.path.join(PASTA, "mlp.json"), "w"), indent=1, default=float)

    # 4. figuras
    figuras(yte, p_ce, m["limiar"], "MLP adotado")
    figura_evolucao(list(por.values()))

    # 5. tabelas
    oof_adot = np.load(os.path.join(PASTA, adot["oof"]))

    def linha(rot, nome):
        r = por[rot]
        v = r.get("vs_adotado") or comum.comparar_placar(np.asarray(ytr), oof_adot, np.load(os.path.join(PASTA, r["oof"])))
        return dict(versao=nome, ks_placar=r["ks_kfold"], auroc_oof=r["auc_oof"], mse_val=r.get("mse_val"),
                    vs_adotado="(adotado)" if rot == esc else f"{v['veredito']} {v['delta']:+.4f}", epocas=r.get("epocas"))
    grupos = {
        "Função de perda (slide 14)": [(esc, "entropia cruzada")] + [("slides_perda_mse", "MSE")],
        "Função de ativação (slides 7 e 27)": [(esc, adot["config"]["hp"]["ativ"])] + [(k, k.split("_", 2)[2]) for k in V if k.startswith("slides_ativ_")],
        "Otimizador (slide 27)": [(esc, "adam, lr 1e-3"), ("slides_otim_rmsprop", "rmsprop, lr 1e-3"), ("slides_otim_sgd", "SGD com momento, lr 1e-2"), ("slides_otim_adadelta", "adadelta, lr 1,0")],
        "Número de camadas (slide 27)": [(esc, "1 camada de 10"), ("slides_camadas_2", "2 camadas de 10")],
        "Unidades e ritmo de treino (fase 5)": [(k, k[4:]) for k in sorted(por) if k.startswith("arq_")],
        "Pré-processamento": [("trial0", "base do grupo"), ("dados_sem_criadas", "sem as variáveis criadas"), ("dados_sem_escala", "sem escala"),
                              ("dados_cliente_novo", "+ cliente_novo"), ("dados_delta_preco", "+ delta_preco"), ("dados_tenure_1", "+ tenure_1"),
                              ("referencia", "+ cliente_novo + delta_preco"), (esc, "+ as três (adotado)")],
        "Oversampling (slide 7)": [(esc, "1:1 (adotado)"), ("fase4_over_0.75", "até 0,75"), ("fase4_over_0.5", "até 0,5"), ("fase4_sem_over", "sem")]}
    L = ["# MLP — tabelas para os slides", "", "Gerado por `experimentos/mlp/relatorio_final.py`. Todas as linhas sao do placar comum (k-fold 3x5, mesmos clientes), "
         "comparadas com a versao adotada pelo IC95 do bootstrap de clientes. Diferencas abaixo de ~0,01 sao ruido.", "",
         "`ks_placar` e a media dos 15 KS das dobras; `vs_adotado` compara o KS da previsao fora da dobra (OOF) de cada cliente, "
         "por isso o sinal dos dois pode divergir quando a diferenca e minima.", "",
         f"Versao adotada: **{esc}**. KS no placar {adot['ks_kfold']:.4f}; **KS no teste {m['ks']:.4f}** (IC95 {m['ks_ic95'][0]:.3f} a {m['ks_ic95'][1]:.3f}).", ""]
    for g, itens in grupos.items():
        L += [f"## {g}", "", _md(pd.DataFrame([linha(r, n) for r, n in itens if r in por])), ""]
    L += ["## MSE e entropia cruzada no treino, na validacao e no teste (slide 14)", "",
          _md(pd.DataFrame([dict(conjunto="treino (dobras do placar)", mse=adot.get("mse_treino"), ce=adot.get("ce_treino")),
                            dict(conjunto="validacao (OOF do placar)", mse=adot.get("mse_val"), ce=adot.get("ce_val")),
                            dict(conjunto="teste", mse=m["mse"], ce=m["ce"]),
                            dict(conjunto="teste, prior corrigido", mse=m["mse_corr"], ce=m["ce_corr"])])),
          "", "O treino tem oversampling 1:1, entao as probabilidades saem infladas; 'prior corrigido' desfaz isso (comum.corrige_prior).", ""]
    cm = m["confusao"]
    L += ["## Teste: matriz de confusao e metricas (slides 11 e 15)", "",
          f"Limiar do KS escolhido na OOF do placar: {m['limiar']:.3f}.", "",
          "| | previsto: fica | previsto: cancela |", "|---|---|---|",
          f"| **real: fica** | {cm[0][0]} | {cm[0][1]} |", f"| **real: cancela** | {cm[1][0]} | {cm[1][1]} |", "",
          _md(pd.DataFrame([dict(KS=m["ks"], AUROC=m["auroc"], precision=m["precision"], recall=m["recall"], F1=m["f1"], MSE=m["mse"], CE=m["ce"])])), "",
          f"KS por semente: {', '.join(f'{x:.4f}' for x in m['ks_por_semente'])}.", "",
          "## Optuna da fase 5 (100 trials, 6 divisoes por trial)", "",
          f"Trial 0 (a base): {est['ks_trial0']:.4f} | melhor: {est['melhor_ks']:.4f}. Importancia dos hiperparametros:", "",
          _md(pd.DataFrame([dict(hiperparametro=k, importancia=float(v)) for k, v in list(est["importancias"].items())[:10]])), "",
          "## Figuras", "", "- `evolucao_fases.png`: ganho de KS de cada fase sobre o MLP padrao, pareado (o veredito do placar)", "- `roc.png`, `precision_recall.png`, `ks.png`: curvas do modelo adotado no teste", ""]
    open(os.path.join(SAIDA, "TABELAS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
    open(os.path.join(PASTA, "PLACAR.md"), "w", encoding="utf-8", newline="\n").write(comum.relatorio_placar(PASTA))
    print("ok:", os.path.join(PASTA, "mlp.json"), SAIDA)


if __name__ == "__main__":
    main()
