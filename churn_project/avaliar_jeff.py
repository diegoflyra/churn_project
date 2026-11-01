"""Avaliacao final de STab / TabKAN + experimento controlado Entropia Cruzada x MSE.

Uso: python avaliar_jeff.py stab 3      (3 sementes)
Gera: resultados/<modelo>.json  e  figuras/<modelo>_*.png
"""
import sys, json, os
import numpy as np, optuna
from scipy.stats import wilcoxon, mannwhitneyu
import comum, modelos_jeff as M

PI = 0.2654          # taxa real de churn (o treino das dobras foi balanceado para 50%)


def corrige_prior(p, pi=PI):
    """Desfaz o deslocamento de prior causado pelo oversampling (treino 50/50 -> prior real)."""
    w = pi / (1 - pi); q = p * w / (p * w + (1 - p)); return q


def hp_from_params(modelo, prm):
    if modelo == "stab":
        return dict(prm)
    base = prm.get("base_spline", prm.get("base_cheb", prm.get("base_fourier")))
    lr = prm.get("lr_lbfgs", prm.get("lr_adam"))
    return dict(familia=prm["familia"], prof=prm["prof"], larg=prm["larg"], base=base, otim=prm["otim"],
                lr=lr, l2=prm["l2"], perda=prm["perda"])


def fit(modelo, hp, f, Xlist, seed):
    Xt, yt, Xv, yv = f
    if modelo == "stab":
        return M.stab_fit_predict(Xt, yt, Xlist, hp, seed)
    return M.tabkan_fit_predict(Xt, yt, Xlist, Xv, yv, hp, seed)


def sumario_estudo(st):
    ok = [t for t in st.trials if t.state.name == "COMPLETE"]
    pr = [t for t in st.trials if t.state.name == "PRUNED"]
    por = {}
    for l in ("ce", "mse"):
        v = sorted([t.value for t in ok if t.params["perda"] == l], reverse=True)
        por[l] = dict(n=len(v), melhor=v[0] if v else None, mediana=float(np.median(v)) if v else None,
                      media_top5=float(np.mean(v[:5])) if v else None)
    a = [t.value for t in ok if t.params["perda"] == "ce"]; b = [t.value for t in ok if t.params["perda"] == "mse"]
    mw = float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else None
    try:
        imp = optuna.importance.get_param_importances(st)
    except Exception:
        imp = {}
    return dict(n_completos=len(ok), n_podados=len(pr), por_perda=por, mannwhitney_p=mw, importancias=imp,
                ks_trial0=st.trials[0].value, melhor_ks=st.best_value,
                tempo_trials_s=float(sum(t.user_attrs.get("tempo_s", 0) for t in ok)),
                convergencia=[float(max([t.value for t in ok[:i + 1]])) for i in range(len(ok))])


def main(modelo, n_seeds):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.load_study(study_name=modelo, storage=f"sqlite:///resultados/{modelo}.db")
    folds, final, _ = comum.carregar_dobras()
    Xte, yte = final[2], final[3]
    best = st.best_trial; hp0 = hp_from_params(modelo, best.params)
    out = dict(modelo=modelo, melhor_config=hp0, ks_dobras_melhor=best.user_attrs["ks_dobras"], estudo=sumario_estudo(st))
    ctrl = {}
    for perda in ("ce", "mse"):
        hp = dict(hp0, perda=perda)
        R = dict(val=[], tr_ce=[], tr_mse=[], teste_prob=[], oof=[])
        for s in range(n_seeds):
            pte_folds, oof_p, oof_y, fold_rows = [], [], [], []
            for i, f in enumerate(folds):
                (pv, pte), ptr = fit(modelo, hp, f, [f[2], Xte], 1000 * s + 42 + i)
                yv = f[3]
                fold_rows.append(dict(seed=s, fold=i, ks=comum.ks(yv, pv), auroc=comum.metricas(yv, pv, .5)["auroc"],
                                      ce_raw=comum.ce(yv, pv), mse_raw=comum.mse(yv, pv),
                                      ce_corr=comum.ce(yv, corrige_prior(pv)), mse_corr=comum.mse(yv, corrige_prior(pv)),
                                      tr_ce=comum.ce(f[1], ptr), tr_mse=comum.mse(f[1], ptr)))
                pte_folds.append(pte); oof_p.append(pv); oof_y.append(yv)
                print(modelo, perda, "semente", s, "dobra", i, "KS %.4f" % fold_rows[-1]["ks"], flush=True)
            R["val"] += fold_rows
            thr = comum.limiar_ks(np.concatenate(oof_y), np.concatenate(oof_p))      # limiar escolhido so na validacao
            R["teste_prob"].append(np.mean(pte_folds, 0)); R["oof"].append(thr)
        ptest = np.mean(R["teste_prob"], 0); thr = float(np.mean(R["oof"]))
        m_teste = comum.metricas(yte, ptest, thr)
        m_teste["ce_corr"] = comum.ce(yte, corrige_prior(ptest)); m_teste["mse_corr"] = comum.mse(yte, corrige_prior(ptest))
        m_teste["ks_ic95"] = comum.bootstrap_ks(yte, ptest)
        per_seed = [comum.ks(yte, p) for p in R["teste_prob"]]
        ctrl[perda] = dict(val=R["val"], teste=m_teste, ks_teste_por_semente=per_seed, prob_teste=ptest.tolist())
    # comparacao pareada CE x MSE (mesmas dobras, mesmas sementes)
    ce_v = np.array([r["ks"] for r in ctrl["ce"]["val"]]); ms_v = np.array([r["ks"] for r in ctrl["mse"]["val"]])
    d = ce_v - ms_v
    d_fold = d.reshape(n_seeds, 5).mean(0)
    pc, pm = np.array(ctrl["ce"]["prob_teste"]), np.array(ctrl["mse"]["prob_teste"])
    rng = np.random.default_rng(0); n = len(yte); bs = []
    for _ in range(1000):
        b = rng.choice(n, n)
        if yte[b].min() == yte[b].max(): continue
        bs.append(comum.ks(yte[b], pc[b]) - comum.ks(yte[b], pm[b]))
    out["ce_vs_mse"] = dict(
        delta_ks_val_media=float(d.mean()), delta_ks_val_dp=float(d.std(ddof=1)), delta_por_dobra=[float(x) for x in d_fold],
        dobras_ce_melhor=int((d_fold > 0).sum()),
        wilcoxon_p=float(wilcoxon(d_fold).pvalue) if np.any(d_fold != 0) else 1.0,
        delta_ks_teste=float(ctrl["ce"]["teste"]["ks"] - ctrl["mse"]["teste"]["ks"]),
        delta_ks_teste_ic95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        resumo={k: {kk: (float(np.mean([r[kk] for r in v["val"]])), float(np.std([r[kk] for r in v["val"]])))
                    for kk in ("ks", "auroc", "ce_raw", "mse_raw", "ce_corr", "mse_corr", "tr_ce", "tr_mse")} for k, v in ctrl.items()},
        teste={k: {kk: v["teste"][kk] for kk in ("ks", "auroc", "ce", "mse", "ce_corr", "mse_corr", "f1", "precision", "recall", "limiar", "confusao", "ks_ic95")} for k, v in ctrl.items()},
        ks_teste_por_semente={k: v["ks_teste_por_semente"] for k, v in ctrl.items()})
    # probabilidades e metricas do teste da melhor configuracao (perda escolhida pelo Optuna)
    esc = hp0["perda"]
    out["teste"] = ctrl[esc]["teste"]; out["probabilidades_teste"] = ctrl[esc]["prob_teste"]
    out["y_teste"] = [int(v) for v in yte]
    json.dump(out, open(f"resultados/{modelo}.json", "w"), indent=1)
    print("OK", modelo)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
