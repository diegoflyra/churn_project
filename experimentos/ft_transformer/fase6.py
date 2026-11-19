"""Fase 6 do FT-Transformer: mineracao de features sobre o melhor FT da fase 5 (o5_top3).

Por que: features foram o que mais subiu o KS nas duas redes (tenure_1 no MLP; as tres features do MLP no FT, fase 4).
Na fase 5, as 9 candidatas do MLP foram testadas sobre a base antiga (oversampling, dropout baixo, 7 epocas); o Optuna
achou outra rede (1 bloco, dropout de atencao ~0,49, sem oversampling, ~55 epocas), e cada candidata precisa ser medida
de novo nela.

Candidatas novas, da analise de erro nas OOF do o5_top3:
- a arvore rasa no residuo (y - p) separou pela razao TotalCharges / (MonthlyCharges * tenure) e por clientes caros sem
  suporte tecnico ou sem backup (residuo +0,16 e +0,08: a rede subestima esses grupos);
- o KS dentro do contrato mes a mes (0,395, 43% de churn) e o pior entre os segmentos: features que separam clientes
  dentro desse contrato.

Selecao para frente, sem gastar as dobras do placar: cada rodada mede a base atual e cada candidata somada a ela nas
divisoes do Optuna (sementes 201-203, 15 dobras); entra a de maior ganho, se o bootstrap pareado der ganho em pelo menos
P_MIN das reamostragens. So o conjunto final vai para o placar.
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, fase1 as F1, fase5 as F5

ROT_BASE = "o5_top3"
BASE = dict(hp=dict(n_blocos=1, d_token=192, n_cabecas=4, drop_atencao=0.49174646579703485, drop_ffn=0.33792357369236903,
                    drop_residual=0.002815031260296183, ffn_fator=1.6205510930929652, lr=0.00013048115078183745,
                    wd=0.0005454677212888819, lote=256, paciencia=16, perda="ce"),
            dados=dict(features=["cliente_novo", "delta_preco", "tenure_1"], over=False))   # o5_top3, fase 5

NOVAS = ("cobranca_razao", "preco_subiu", "caro_sem_suporte", "caro_sem_backup", "mtm_tenure", "mtm_sem_suporte",
         "mtm_eletronico", "antigo_mtm", "antigo", "muitos_servicos")
CANDIDATAS = NOVAS + F5.EXTRAS                      # as 9 do MLP de novo, agora na rede do Optuna
P_MIN = 0.8


def com_features(base, feats):
    return dict(hp=dict(base["hp"]), dados=dict(base["dados"], features=list(base["dados"]["features"]) + list(feats)))


class Cache:
    """Medidas nas divisoes do Optuna, gravadas na pasta (retomar a execucao nao mede de novo)."""

    def __init__(self, pasta):
        self.pasta = pasta
        self.arq = os.path.join(pasta, "selecao.jsonl")
        os.makedirs(os.path.join(pasta, "selecao_oof"), exist_ok=True)
        self.linhas = {}
        if os.path.isfile(self.arq):
            for l in open(self.arq, encoding="utf-8"):
                r = json.loads(l)
                self.linhas[r["chave"]] = r

    def medir(self, c, medir):
        ch = F1._chave(c)
        if ch not in self.linhas:
            r = medir(c)
            arq = os.path.join("selecao_oof", f"{len(self.linhas):03d}.npy")
            np.save(os.path.join(self.pasta, arq), r["oof"])
            linha = dict(chave=ch, features=c["dados"]["features"], ks_kfold=r["ks_kfold"], ks_oof=r["ks_oof"],
                         epocas=r.get("epocas"), minutos=r.get("tempo_min"), oof=arq)
            with open(self.arq, "a", encoding="utf-8") as f:
                f.write(json.dumps(linha) + "\n")
            self.linhas[ch] = linha
        return self.linhas[ch], np.load(os.path.join(self.pasta, self.linhas[ch]["oof"]))


def selecao_para_frente(pasta, medir, y, base=BASE, candidatas=CANDIDATAS, rodadas=3, p_min=P_MIN,
                        continuar=lambda: True, ao_medir=None):
    """medir(config) -> comum.resumir_placar nas divisoes do Optuna. Devolve (features escolhidas, historico)."""
    cache, atual, escolhidas, hist = Cache(pasta), base, [], []
    for rodada in range(1, rodadas + 1):
        rb, ob = cache.medir(atual, medir)
        resto = [f for f in candidatas if f not in escolhidas]
        res = []
        for f in resto:
            if not continuar():
                print(f"limite de tempo: a rodada {rodada} parou antes de {f}", flush=True)
                return escolhidas, hist
            rc, oc = cache.medir(com_features(atual, [f]), medir)
            v = comum.comparar_placar(y, ob, oc)
            linha = dict(rodada=rodada, candidata=f, ks_kfold=rc["ks_kfold"], ks_kfold_base=rb["ks_kfold"],
                         delta_kfold=rc["ks_kfold"] - rb["ks_kfold"], delta_oof=v["delta"], ic95=v["ic95"], p_positivo=v["p_positivo"])
            res.append(linha); hist.append(linha)
            if ao_medir:
                ao_medir(linha)
        melhor = max(res, key=lambda r: r["delta_oof"]) if res else None
        if not melhor or melhor["delta_oof"] <= 0 or melhor["p_positivo"] < p_min:
            print(f"rodada {rodada}: nenhuma candidata passou (melhor: {melhor and melhor['candidata']}, "
                  f"p+ {melhor and round(melhor['p_positivo'], 2)})", flush=True)
            break
        escolhidas.append(melhor["candidata"])
        for h in hist:
            if h["rodada"] == rodada:
                h["escolhida"] = h["candidata"] == melhor["candidata"]
        atual = com_features(atual, [melhor["candidata"]])
        print(f"rodada {rodada}: entra {melhor['candidata']} (delta OOF {melhor['delta_oof']:+.4f}, p+ {melhor['p_positivo']:.2f})", flush=True)
    return escolhidas, hist
