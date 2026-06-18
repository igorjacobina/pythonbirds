# Estratégia - Direção (ML)

> Família 1 do [[Placar de Estrategias|placar]]. **Veredito: ❌ sem edge.**

## A ideia

Usar **Machine Learning** sobre preço/OHLC para prever a **direção** do próximo movimento (alta ou baixa) em majors líquidos. É a tentativa mais intuitiva e mais popular — e também a que o mercado mais arbitrou.

## Resultados (líquidos de custo)

| Timeframe | Sharpe OOS | DSR | Acurácia |
|-----------|-----------|-----|----------|
| M15 | **-2,75** | 0% | — |
| H1 | **-0,58** | 0% | — |

- Acurácia geral **~0,48**, ou seja, **igual ao acaso** (≈ 50%).
- DSR de **0%** em ambos os timeframes: nenhuma evidência de skill após penalizar tentativas.

## Por que falhou

Prever a direção de **majors líquidos** a partir de OHLC é **praticamente impossível**: esses mercados são extremamente eficientes. Se a direção de curto prazo fosse previsível a partir de candles, o sinal já teria sido arbitrado. A acurácia ~0,48 confirma: o modelo não sabe nada que o acaso não saiba. E mesmo uma acurácia ligeiramente acima de 50% no bruto não sobrevive aos custos.

## Lição

Direção é a aposta mais difícil. O contraste com a vencedora é instrutivo: o [[Estrategia - Volatilidade (Straddle)|straddle]] **não tenta adivinhar a direção** — aposta no tamanho do movimento — e por isso encontra edge onde a aposta direcional falha.

## Relacionado

- [[Placar de Estrategias]]
- [[Estrategia - Volatilidade (Straddle)]]
- [[Metodologia]]
- [[Glossario]]
