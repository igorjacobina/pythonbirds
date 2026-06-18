# Placar de Estratégias

> O registro honesto de cinco famílias de estratégia testadas com rigor no [[INNOVA EA - MOC|INNOVA EA]]. Todos os números são reais. Resultados sempre líquidos de custo. Ver [[Metodologia]].

## Resumo

| # | Família | Veredito | Números-chave |
|---|---------|----------|---------------|
| 1 | [[Estrategia - Direcao (ML)]] | ❌ sem edge | M15: Sharpe OOS -2,75, DSR 0%. H1: Sharpe -0,58, DSR 0%. Acurácia ~0,48 (acaso) |
| 2 | [[Estrategia - Meta-labeling]] (rompimento Donchian) | ❌ sem edge | Ganho de precisão ~0; threshold 0,50 → Sharpe -1,39; DSR 0% |
| 3 | [[Estrategia - Padroes de Vela]] (reversão topo/fundo) | ❌ sem edge | 37.394 eventos; ganho de precisão +0,015 (≈ruído); Sharpe OOS -0,17; só 15% dos folds positivos |
| 4 | [[Estrategia - Volatilidade (Straddle)]] | ✅ **edge real, fino** | H4, custo realista (40 pts), thr 0,50, 15 ativos: **Sharpe OOS +0,59**; 12/15 ativos positivos; 6/6 folds positivos (t-stat ≈ 4,9, p < 0,001); 21.770 trades; DSR conservador ~31% |
| 5 | [[Estrategia - Pares (Relative-Value)]] | ❌ sem edge | 10 pares no H4: Sharpe agregado -0,17; só 3/10 positivos; ouro/prata o pior (-0,69); DSR 34% |

## O que isso significa

De cinco famílias testadas, **quatro falharam de forma honesta e apenas uma funcionou** (volatilidade/straddle). Isso é normal e institucional: fundos quantitativos testam centenas de ideias para encontrar poucas com edge real. O valor do sistema não está em achar muitas estratégias vencedoras, e sim em **reprovar de forma confiável o que não merece** — incluindo ideias famosas e queridas pelo mercado, como pares (statistical arbitrage) e padrões de vela. O fato de o sistema rejeitar essas ideias populares, e aprovar apenas o straddle de volatilidade — uma estratégia **economicamente coerente** e validada de forma independente por [[Estrategia - Larry Williams|Larry Williams]] — é justamente o que dá confiança nos resultados.

## Detalhe importante sobre a vencedora

O straddle aposta no **tamanho** do movimento, não na **direção**. Ele sobreviveu a custos realistas e a um volume maior de trades, e tem núcleo forte em FX majors e prata. A evidência mais limpa é a consistência de **6/6 folds positivos** (t-stat ≈ 4,9). Ver [[Estrategia - Volatilidade (Straddle)]].

## Relacionado

- [[INNOVA EA - MOC]]
- [[Metodologia]]
- [[Estrategia - Volatilidade (Straddle)]]
- [[Estrategia - Larry Williams]]
- [[Glossario]]
