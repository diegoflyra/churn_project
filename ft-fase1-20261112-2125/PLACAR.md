# Placar

Gerado por `comum.relatorio_placar`. Regra em `docs/pre-proposta.md` ("Placar comum"): k-fold estratificado 3 x 5 (sementes 101, 102, 103), parada em 20% internos, veredito pareado pelo IC95 do bootstrap de clientes. O KS do placar decide; o teste so confere.

## Modelo adotado de cada um

| Modelo | Versao | KS placar | ep | KS teste | IC95 teste | AUROC teste | vs trial 0 | vs XGBoost | Motivo |
|---|---|---|---|---|---|---|---|---|---|

## ft_transformer: todas as versoes (maior KS do placar primeiro)

| Versao | KS placar | ep | KS OOF | AUROC OOF | MSE treino | MSE val | vs trial 0 | KS teste |
|---|---|---|---|---|---|---|---|---|
| trial0_sem_over | 0.5440 | 0.0066 | 0.5314 | 0.8461 | 0.1322 | 0.1348 | empatou (+0.0024) | - |
| g_b3_d128 | 0.5416 | 0.0070 | 0.5372 | 0.8458 | 0.1546 | 0.1635 | empatou (+0.0082) | - |
| g_b2_d64 | 0.5412 | 0.0066 | 0.5324 | 0.8452 | 0.1558 | 0.1684 | empatou (+0.0034) | - |
| g_b1_d192 | 0.5412 | 0.0067 | 0.5249 | 0.8428 | 0.1594 | 0.1651 | empatou (-0.0042) | - |
| g_b2_d128 | 0.5410 | 0.0061 | 0.5317 | 0.8458 | 0.1554 | 0.1662 | empatou (+0.0026) | - |
| g_b1_d128 | 0.5388 | 0.0071 | 0.5266 | 0.8420 | 0.1600 | 0.1658 | empatou (-0.0024) | - |
| trial0 | 0.5378 | 0.0069 | 0.5291 | 0.8449 | 0.1560 | 0.1651 | - | - |
| g_b3_d64 | 0.5375 | 0.0062 | 0.5254 | 0.8446 | 0.1551 | 0.1648 | empatou (-0.0036) | - |
| g_b1_d64 | 0.5368 | 0.0068 | 0.5297 | 0.8404 | 0.1603 | 0.1700 | empatou (+0.0006) | - |
| g_b2_d192 | 0.5368 | 0.0068 | 0.5301 | 0.8458 | 0.1546 | 0.1690 | empatou (+0.0010) | - |

Diferencas de KS menores que uns 0,01 ficam dentro do ruido do placar: veja o veredito antes de dizer que algo melhorou.
