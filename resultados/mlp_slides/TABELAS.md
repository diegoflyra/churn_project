# MLP — tabelas para os slides

Gerado por `experimentos/mlp/relatorio_final.py`. Todas as linhas sao do placar comum (k-fold 3x5, mesmos clientes), comparadas com a versao adotada pelo IC95 do bootstrap de clientes. Diferencas abaixo de ~0,01 sao ruido.

`ks_placar` e a media dos 15 KS das dobras; `vs_adotado` compara o KS da previsao fora da dobra (OOF) de cada cliente, por isso o sinal dos dois pode divergir quando a diferenca e minima.

Versao adotada: **fase4**. KS no placar 0.5520; **KS no teste 0.5447** (IC95 0.503 a 0.586).

## Função de perda (slide 14)

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| entropia cruzada | 0.5520 | 0.8479 | 0.1631 | (adotado) | 33 |
| MSE | 0.5517 | 0.8474 | 0.1620 | empatou -0.0003 | 28 |

## Função de ativação (slides 7 e 27)

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| relu | 0.5520 | 0.8479 | 0.1631 | (adotado) | 33 |
| logistica | 0.5520 | 0.8477 | 0.1621 | empatou +0.0012 | 62 |
| tanh | 0.5510 | 0.8483 | 0.1617 | empatou +0.0021 | 44 |
| gelu | 0.5511 | 0.8480 | 0.1622 | empatou -0.0018 | 29 |
| leaky_relu | 0.5510 | 0.8481 | 0.1623 | empatou -0.0007 | 26 |

## Otimizador (slide 27)

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| adam, lr 1e-3 | 0.5520 | 0.8479 | 0.1631 | (adotado) | 33 |
| rmsprop, lr 1e-3 | 0.5522 | 0.8464 | 0.1616 | empatou +0.0041 | 11 |
| SGD com momento, lr 1e-2 | 0.5491 | 0.8474 | 0.1618 | empatou -0.0013 | 25 |
| adadelta, lr 1,0 | 0.5521 | 0.8476 | 0.1625 | empatou -0.0004 | 11 |

## Número de camadas (slide 27)

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| 1 camada de 10 | 0.5520 | 0.8479 | 0.1631 | (adotado) | 33 |
| 2 camadas de 10 | 0.5472 | 0.8474 | 0.1612 | empatou -0.0041 | 24 |

## Unidades e ritmo de treino (fase 5)

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| 1x10_lento | 0.5461 | 0.8469 | 0.1624 | empatou -0.0064 | 100 |
| 1x10_muito_lento | 0.5463 | 0.8466 | 0.1626 | empatou -0.0058 | 285 |
| 1x10_padrao | 0.5454 | 0.8464 | 0.1623 | empatou -0.0055 | 32 |
| 1x32_lento | 0.5475 | 0.8465 | 0.1623 | empatou -0.0052 | 73 |
| 1x32_muito_lento | 0.5469 | 0.8466 | 0.1622 | empatou -0.0062 | 254 |
| 1x32_padrao | 0.5480 | 0.8465 | 0.1623 | empatou -0.0056 | 22 |
| 2x16_lento | 0.5477 | 0.8459 | 0.1629 | empatou -0.0033 | 68 |
| 2x16_muito_lento | 0.5482 | 0.8459 | 0.1630 | empatou -0.0037 | 212 |
| 2x16_padrao | 0.5484 | 0.8456 | 0.1635 | empatou +0.0007 | 18 |
| 2x64_lento | 0.5451 | 0.8459 | 0.1619 | empatou -0.0079 | 52 |
| 2x64_muito_lento | 0.5476 | 0.8460 | 0.1623 | empatou -0.0065 | 127 |
| 2x64_padrao | 0.5471 | 0.8459 | 0.1629 | empatou -0.0027 | 14 |
| 2x9_lento | 0.5436 | 0.8454 | 0.1623 | empatou -0.0081 | 94 |
| 2x9_muito_lento | 0.5460 | 0.8455 | 0.1625 | empatou -0.0051 | 231 |
| 2x9_padrao | 0.5446 | 0.8456 | 0.1621 | empatou -0.0059 | 29 |
| 3x16_lento | 0.5428 | 0.8446 | 0.1625 | empatou -0.0080 | 89 |
| 3x16_muito_lento | 0.5455 | 0.8451 | 0.1628 | empatou -0.0064 | 199 |
| 3x16_padrao | 0.5442 | 0.8448 | 0.1631 | empatou -0.0072 | 23 |

## Pré-processamento

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| base do grupo | 0.5391 | 0.8417 | 0.1662 | perdeu -0.0131 | 31 |
| sem as variáveis criadas | 0.5360 | 0.8432 | 0.1649 | perdeu -0.0119 | 32 |
| sem escala | 0.5254 | 0.8359 | 0.1764 | empatou -0.0105 | 27 |
| + cliente_novo | 0.5454 | 0.8461 | 0.1625 | empatou -0.0006 | 30 |
| + delta_preco | 0.5441 | 0.8458 | 0.1633 | empatou -0.0064 | 37 |
| + tenure_1 | 0.5495 | 0.8475 | 0.1623 | empatou -0.0000 | 36 |
| + cliente_novo + delta_preco | 0.5445 | 0.8466 | 0.1630 | empatou -0.0063 | 26 |
| + as três (adotado) | 0.5520 | 0.8479 | 0.1631 | (adotado) | 33 |

## Oversampling (slide 7)

| versao | ks_placar | auroc_oof | mse_val | vs_adotado | epocas |
|---|---|---|---|---|---|
| 1:1 (adotado) | 0.5520 | 0.8479 | 0.1631 | (adotado) | 33 |
| até 0,75 | 0.5513 | 0.8479 | 0.1481 | empatou +0.0027 | 39 |
| até 0,5 | 0.5485 | 0.8478 | 0.1366 | empatou -0.0021 | 50 |
| sem | 0.5505 | 0.8481 | 0.1335 | empatou -0.0015 | 72 |

## MSE e entropia cruzada no treino, na validacao e no teste (slide 14)

| conjunto | mse | ce |
|---|---|---|
| treino (dobras do placar) | 0.1543 | 0.4715 |
| validacao (OOF do placar) | 0.1631 | 0.4879 |
| teste | 0.1622 | 0.4821 |
| teste, prior corrigido | 0.1340 | 0.4104 |

O treino tem oversampling 1:1, entao as probabilidades saem infladas; 'prior corrigido' desfaz isso (comum.corrige_prior).

## Teste: matriz de confusao e metricas (slides 11 e 15)

Limiar do KS escolhido na OOF do placar: 0.451.

| | previsto: fica | previsto: cancela |
|---|---|---|
| **real: fica** | 904 | 390 |
| **real: cancela** | 75 | 392 |

| KS | AUROC | precision | recall | F1 | MSE | CE |
|---|---|---|---|---|---|---|
| 0.5447 | 0.8502 | 0.5013 | 0.8394 | 0.6277 | 0.1622 | 0.4821 |

KS por semente: 0.5532, 0.5422, 0.5398, 0.5410, 0.5445.

## Optuna da fase 5 (100 trials, 6 divisoes por trial)

Trial 0 (a base): 0.5475 | melhor: 0.5510. Importancia dos hiperparametros:

| hiperparametro | importancia |
|---|---|
| l2 | 0.3515 |
| dropout | 0.2656 |
| unidades | 0.2095 |
| lr | 0.0655 |
| ativ | 0.0233 |
| suav | 0.0191 |
| paciencia | 0.0149 |
| over | 0.0106 |
| peso_pos | 0.0090 |
| lote | 0.0080 |

## Figuras

- `evolucao_fases.png`: ganho de KS de cada fase sobre o MLP padrao, pareado (o veredito do placar)
- `roc.png`, `precision_recall.png`, `ks.png`: curvas do modelo adotado no teste
