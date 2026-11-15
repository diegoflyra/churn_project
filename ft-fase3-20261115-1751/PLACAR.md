# Placar

Gerado por `comum.relatorio_placar`. Regra em `docs/pre-proposta.md` ("Placar comum"): k-fold estratificado 3 x 5 (sementes 101, 102, 103), parada em 20% internos, veredito pareado pelo IC95 do bootstrap de clientes. O KS do placar decide; o teste so confere.

## Modelo adotado de cada um

| Modelo | Versao | KS placar | ep | KS teste | IC95 teste | AUROC teste | vs trial 0 | vs XGBoost | Motivo |
|---|---|---|---|---|---|---|---|---|---|

## ft_transformer: todas as versoes (maior KS do placar primeiro)

| Versao | KS placar | ep | KS OOF | AUROC OOF | MSE treino | MSE val | vs trial 0 | vs_base | KS teste |
|---|---|---|---|---|---|---|---|---|---|
| r_combo_moderado | 0.5444 | 0.0059 | 0.5323 | 0.8448 | 0.1584 | 0.1715 | empatou (+0.0033) | - | - |
| trial0_sem_over | 0.5440 | 0.0066 | 0.5314 | 0.8461 | 0.1322 | 0.1348 | empatou (+0.0024) | - | - |
| r_drop_medio | 0.5437 | 0.0063 | 0.5306 | 0.8444 | 0.1583 | 0.1717 | empatou (+0.0016) | - | - |
| s_trial0_ks_p16 | 0.5429 | 0.0063 | 0.5324 | 0.8442 | 0.1575 | 0.1674 | empatou (+0.0033) | empatou (+0.0033) | - |
| s_trial0_ks_p32 | 0.5429 | 0.0063 | 0.5324 | 0.8442 | 0.1575 | 0.1674 | empatou (+0.0033) | empatou (+0.0033) | - |
| s_trial0_ks_p64 | 0.5429 | 0.0063 | 0.5324 | 0.8442 | 0.1575 | 0.1674 | empatou (+0.0033) | empatou (+0.0033) | - |
| s_r_drop_medio_perda_p32 | 0.5422 | 0.0063 | 0.5327 | 0.8443 | 0.1568 | 0.1684 | empatou (+0.0037) | empatou (+0.0021) | - |
| s_r_drop_medio_perda_p64 | 0.5422 | 0.0063 | 0.5327 | 0.8443 | 0.1568 | 0.1684 | empatou (+0.0037) | empatou (+0.0021) | - |
| r_combo_pesado | 0.5419 | 0.0061 | 0.5337 | 0.8450 | 0.1580 | 0.1666 | empatou (+0.0046) | - | - |
| r_drop_pesado | 0.5417 | 0.0061 | 0.5333 | 0.8454 | 0.1586 | 0.1707 | empatou (+0.0043) | - | - |
| g_b3_d128 | 0.5416 | 0.0070 | 0.5372 | 0.8458 | 0.1546 | 0.1635 | empatou (+0.0082) | - | - |
| g_b2_d64 | 0.5412 | 0.0066 | 0.5324 | 0.8452 | 0.1558 | 0.1684 | empatou (+0.0034) | - | - |
| g_b1_d192 | 0.5412 | 0.0067 | 0.5249 | 0.8428 | 0.1594 | 0.1651 | empatou (-0.0042) | - | - |
| g_b2_d128 | 0.5410 | 0.0061 | 0.5317 | 0.8458 | 0.1554 | 0.1662 | empatou (+0.0026) | - | - |
| r_l1_1e-5 | 0.5390 | 0.0065 | 0.5308 | 0.8449 | 0.1565 | 0.1649 | empatou (+0.0017) | - | - |
| g_b1_d128 | 0.5388 | 0.0071 | 0.5266 | 0.8420 | 0.1600 | 0.1658 | empatou (-0.0024) | - | - |
| s_r_drop_medio_ks_p16 | 0.5386 | 0.0069 | 0.5302 | 0.8451 | 0.1593 | 0.1715 | empatou (+0.0011) | empatou (-0.0005) | - |
| r_wd_1 | 0.5386 | 0.0066 | 0.5264 | 0.8451 | 0.1533 | 0.1633 | empatou (-0.0027) | - | - |
| r_l2_1e-3 | 0.5385 | 0.0070 | 0.5269 | 0.8446 | 0.1544 | 0.1659 | empatou (-0.0022) | - | - |
| r_l2_1e-4 | 0.5378 | 0.0066 | 0.5265 | 0.8451 | 0.1544 | 0.1655 | empatou (-0.0026) | - | - |
| trial0 | 0.5378 | 0.0069 | 0.5291 | 0.8449 | 0.1560 | 0.1651 | - | - | - |
| g_b3_d64 | 0.5375 | 0.0062 | 0.5254 | 0.8446 | 0.1551 | 0.1648 | empatou (-0.0036) | - | - |
| g_b1_d64 | 0.5368 | 0.0068 | 0.5297 | 0.8404 | 0.1603 | 0.1700 | empatou (+0.0006) | - | - |
| g_b2_d192 | 0.5368 | 0.0068 | 0.5301 | 0.8458 | 0.1546 | 0.1690 | empatou (+0.0010) | - | - |
| r_l1_1e-6 | 0.5365 | 0.0068 | 0.5255 | 0.8448 | 0.1528 | 0.1652 | empatou (-0.0035) | - | - |
| s_r_drop_medio_ks_p32 | 0.5358 | 0.0065 | 0.5269 | 0.8453 | 0.1577 | 0.1700 | empatou (-0.0022) | empatou (-0.0037) | - |
| s_trial0_perda_p32 | 0.5351 | 0.0072 | 0.5250 | 0.8444 | 0.1528 | 0.1653 | empatou (-0.0041) | empatou (-0.0041) | - |
| s_trial0_perda_p64 | 0.5351 | 0.0072 | 0.5250 | 0.8444 | 0.1528 | 0.1653 | empatou (-0.0041) | empatou (-0.0041) | - |
| s_r_drop_medio_ks_p64 | 0.5317 | 0.0062 | 0.5253 | 0.8445 | 0.1541 | 0.1693 | empatou (-0.0037) | empatou (-0.0053) | - |

Diferencas de KS menores que uns 0,01 ficam dentro do ruido do placar: veja o veredito antes de dizer que algo melhorou.
