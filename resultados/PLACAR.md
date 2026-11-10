# Placar

Gerado por `comum.relatorio_placar`. Regra em `docs/pre-proposta.md` ("Placar comum"): k-fold estratificado 3 x 5 (sementes 101, 102, 103), parada em 20% internos, veredito pareado pelo IC95 do bootstrap de clientes. O KS do placar decide; o teste so confere.

## Modelo adotado de cada um

| Modelo | Versao | KS placar | ep | KS teste | IC95 teste | AUROC teste | vs trial 0 | vs XGBoost | Motivo |
|---|---|---|---|---|---|---|---|---|---|
| mlp | fase4 | 0.5520 | 0.0047 | 0.5447 | 0.503 a 0.586 | 0.8502 | ganhou (+0.0131) | - | Maior KS do placar entre as 33 versoes das fases 1-5; ganha do trial 0, empata com a referencia (fase 2). As 9 variacoes dos slides medidas depois (rmsprop 0.5522, adadelta 0.5521) empatam com ela e nao a substituem |

## mlp: todas as versoes (maior KS do placar primeiro)

| Versao | KS placar | ep | KS OOF | AUROC OOF | MSE treino | MSE val | vs trial 0 | vs_adotado | vs_referencia | KS teste |
|---|---|---|---|---|---|---|---|---|---|---|
| slides_otim_rmsprop | 0.5522 | 0.0059 | 0.5442 | 0.8464 | 0.1568 | 0.1616 | ganhou (+0.0173) | empatou (+0.0041) | - | - |
| slides_otim_adadelta | 0.5521 | 0.0053 | 0.5397 | 0.8476 | 0.1552 | 0.1625 | ganhou (+0.0128) | empatou (-0.0004) | - | - |
| slides_ativ_logistica | 0.5520 | 0.0050 | 0.5412 | 0.8477 | 0.1554 | 0.1621 | ganhou (+0.0143) | empatou (+0.0012) | - | - |
| fase4 **(adotado)** | 0.5520 | 0.0047 | 0.5401 | 0.8479 | 0.1543 | 0.1631 | ganhou (+0.0131) | - | empatou (+0.0063) | 0.5447 |
| slides_perda_mse | 0.5517 | 0.0055 | 0.5398 | 0.8474 | 0.1546 | 0.1620 | ganhou (+0.0129) | empatou (-0.0003) | - | - |
| fase4_over_0.75 | 0.5513 | 0.0054 | 0.5427 | 0.8479 | 0.1536 | 0.1481 | ganhou (+0.0158) | - | empatou (+0.0090) | - |
| slides_ativ_gelu | 0.5511 | 0.0051 | 0.5382 | 0.8480 | 0.1548 | 0.1622 | ganhou (+0.0113) | empatou (-0.0018) | - | - |
| slides_ativ_tanh | 0.5510 | 0.0052 | 0.5422 | 0.8483 | 0.1543 | 0.1617 | ganhou (+0.0153) | empatou (+0.0021) | - | - |
| slides_ativ_leaky_relu | 0.5510 | 0.0050 | 0.5394 | 0.8481 | 0.1545 | 0.1623 | ganhou (+0.0124) | empatou (-0.0007) | - | - |
| fase3_teste | 0.5510 | 0.0059 | 0.5430 | 0.8444 | 0.1384 | 0.1403 | ganhou (+0.0161) | - | empatou (+0.0092) | - |
| fase4_sem_over | 0.5505 | 0.0057 | 0.5386 | 0.8481 | 0.1305 | 0.1335 | empatou (+0.0116) | - | empatou (+0.0048) | - |
| optuna | 0.5503 | 0.0066 | 0.5443 | 0.8438 | 0.1372 | 0.1379 | ganhou (+0.0173) | - | empatou (+0.0105) | - |
| dados_tenure_1 | 0.5495 | 0.0062 | 0.5401 | 0.8475 | 0.1552 | 0.1623 | ganhou (+0.0131) | - | empatou (+0.0063) | - |
| slides_otim_sgd | 0.5491 | 0.0055 | 0.5388 | 0.8474 | 0.1555 | 0.1618 | ganhou (+0.0119) | empatou (-0.0013) | - | - |
| fase3_val | 0.5488 | 0.0057 | 0.5407 | 0.8454 | 0.1570 | 0.1628 | ganhou (+0.0138) | - | empatou (+0.0069) | - |
| fase4_over_0.5 | 0.5485 | 0.0056 | 0.5379 | 0.8478 | 0.1439 | 0.1366 | empatou (+0.0110) | - | empatou (+0.0041) | - |
| arq_2x16_padrao | 0.5484 | 0.0057 | 0.5408 | 0.8456 | 0.1551 | 0.1635 | empatou (+0.0138) | - | empatou (+0.0070) | - |
| arq_2x16_muito_lento | 0.5482 | 0.0059 | 0.5364 | 0.8459 | 0.1538 | 0.1630 | empatou (+0.0094) | - | empatou (+0.0026) | - |
| arq_1x32_padrao | 0.5480 | 0.0051 | 0.5345 | 0.8465 | 0.1536 | 0.1623 | empatou (+0.0075) | - | empatou (+0.0007) | - |
| arq_2x16_lento | 0.5477 | 0.0058 | 0.5368 | 0.8459 | 0.1536 | 0.1629 | empatou (+0.0098) | - | empatou (+0.0030) | - |
| arq_2x64_muito_lento | 0.5476 | 0.0062 | 0.5336 | 0.8460 | 0.1516 | 0.1623 | empatou (+0.0067) | - | empatou (-0.0002) | - |
| arq_1x32_lento | 0.5475 | 0.0054 | 0.5349 | 0.8465 | 0.1534 | 0.1623 | empatou (+0.0079) | - | empatou (+0.0011) | - |
| slides_camadas_2 | 0.5472 | 0.0059 | 0.5360 | 0.8474 | 0.1543 | 0.1612 | empatou (+0.0091) | empatou (-0.0041) | - | - |
| arq_2x64_padrao | 0.5471 | 0.0060 | 0.5373 | 0.8459 | 0.1531 | 0.1629 | empatou (+0.0104) | - | empatou (+0.0036) | - |
| arq_1x32_muito_lento | 0.5469 | 0.0054 | 0.5339 | 0.8466 | 0.1532 | 0.1622 | empatou (+0.0070) | - | empatou (+0.0001) | - |
| arq_1x10_muito_lento | 0.5463 | 0.0059 | 0.5343 | 0.8466 | 0.1555 | 0.1626 | empatou (+0.0073) | - | empatou (+0.0005) | - |
| arq_1x10_lento | 0.5461 | 0.0052 | 0.5337 | 0.8469 | 0.1549 | 0.1624 | empatou (+0.0067) | - | empatou (-0.0001) | - |
| arq_2x9_muito_lento | 0.5460 | 0.0057 | 0.5349 | 0.8455 | 0.1555 | 0.1625 | empatou (+0.0080) | - | empatou (+0.0011) | - |
| arq_3x16_muito_lento | 0.5455 | 0.0057 | 0.5336 | 0.8451 | 0.1537 | 0.1628 | empatou (+0.0067) | - | empatou (-0.0002) | - |
| referencia_sem_over | 0.5455 | 0.0054 | 0.5355 | 0.8472 | 0.1314 | 0.1340 | empatou (+0.0086) | - | empatou (+0.0017) | - |
| dados_cliente_novo | 0.5454 | 0.0060 | 0.5395 | 0.8461 | 0.1558 | 0.1625 | empatou (+0.0125) | - | empatou (+0.0057) | - |
| arq_1x10_padrao | 0.5454 | 0.0051 | 0.5346 | 0.8464 | 0.1553 | 0.1623 | empatou (+0.0076) | - | empatou (+0.0008) | - |
| arq_2x64_lento | 0.5451 | 0.0063 | 0.5322 | 0.8459 | 0.1500 | 0.1619 | empatou (+0.0053) | - | empatou (-0.0016) | - |
| arq_2x9_padrao | 0.5446 | 0.0054 | 0.5342 | 0.8456 | 0.1547 | 0.1621 | empatou (+0.0073) | - | empatou (+0.0004) | - |
| referencia | 0.5445 | 0.0056 | 0.5338 | 0.8466 | 0.1556 | 0.1630 | empatou (+0.0069) | - | - | 0.5454 |
| arq_3x16_padrao | 0.5442 | 0.0050 | 0.5329 | 0.8448 | 0.1534 | 0.1631 | empatou (+0.0060) | - | empatou (-0.0009) | - |
| dados_delta_preco | 0.5441 | 0.0056 | 0.5337 | 0.8458 | 0.1556 | 0.1633 | empatou (+0.0067) | - | empatou (-0.0001) | - |
| arq_2x9_lento | 0.5436 | 0.0056 | 0.5319 | 0.8454 | 0.1545 | 0.1623 | empatou (+0.0050) | - | empatou (-0.0019) | - |
| arq_3x16_lento | 0.5428 | 0.0054 | 0.5321 | 0.8446 | 0.1523 | 0.1625 | empatou (+0.0051) | - | empatou (-0.0017) | - |
| trial0 | 0.5391 | 0.0066 | 0.5269 | 0.8417 | 0.1588 | 0.1662 | - | - | - | - |
| dados_sem_criadas | 0.5360 | 0.0065 | 0.5282 | 0.8432 | 0.1566 | 0.1649 | empatou (+0.0013) | - | empatou (-0.0056) | - |
| dados_sem_escala | 0.5254 | 0.0079 | 0.5295 | 0.8359 | 0.1688 | 0.1764 | empatou (+0.0026) | - | empatou (-0.0043) | - |

Diferencas de KS menores que uns 0,01 ficam dentro do ruido do placar: veja o veredito antes de dizer que algo melhorou.
