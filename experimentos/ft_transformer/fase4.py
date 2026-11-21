"""Fase 4 do FT-Transformer: as features que ajudaram o MLP, medidas no placar comum.

No MLP, marcar os clientes novos foi a unica mudanca nos dados que ganhou do padrao: tenure_1 sozinha deu +0,014, e o
campeao (fase4, depois s6_rmsprop_ks_p50) usa as tres: cliente_novo (tenure <= 6), delta_preco (MonthlyCharges -
TotalCharges/(tenure+1)) e tenure_1 (tenure <= 1). Ate a fase 3 o FT-Transformer so viu o pre-processamento base.

Pergunta: as mesmas features sobem o KS do FT-Transformer? O tokenizador da a cada coluna um token linear (x*W + b);
tenure entra como uma reta, e um corte em tenure <= 1 ou <= 6 vira um token proprio que a atencao pode usar direto.
As features sao calculadas por linha (sem olhar o rotulo nem outras linhas) e o scaler e ajustado so no treino da dobra.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # comum.py fica na raiz
import comum, fase1 as F1, fase2 as F2

TRES = ("cliente_novo", "delta_preco", "tenure_1")                  # a ordem do campeao do MLP
BASES = {"trial0": F1.TRIAL0,
         "r_drop_medio": F2.REGULARIZACAO["r_drop_medio"],
         "r_combo_moderado": F2.REGULARIZACAO["r_combo_moderado"]}  # o maior KS do placar do FT ate a fase 3
DETALHE = "r_combo_moderado"                                        # base em que cada feature tambem entra sozinha


def _com(base, features, **dados):
    return dict(hp=dict(base["hp"]), dados=dict(base["dados"], features=list(features), **dados))


def grade_features(bases=BASES, detalhe=DETALHE):
    """{rotulo: (base, config)}. As tres features em cada base; na base de detalhe, cada uma sozinha e as tres sem
    oversampling (no FT, trial0_sem_over foi o segundo maior KS do placar)."""
    saida = {f"f_{nb}_tres": (nb, _com(b, TRES)) for nb, b in bases.items()}
    if detalhe in bases:
        for f in TRES:
            saida[f"f_{detalhe}_{f}"] = (detalhe, _com(bases[detalhe], [f]))
        saida[f"f_{detalhe}_tres_sem_over"] = (detalhe, _com(bases[detalhe], TRES, over=False))
    return saida
