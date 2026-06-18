# Estratégia - Pares (Relative-Value)

> Família 5 do [[Placar de Estrategias|placar]]. **Veredito: ❌ sem edge.**

## A ideia

**Relative-value / Pairs trading** (statistical arbitrage): em vez de apostar na direção de um ativo, aposta-se na **relação** entre dois ativos historicamente ligados. Quando o spread entre eles se abre além do normal, vende-se o "caro" e compra-se o "barato", esperando que voltem a convergir (reversão à média do spread).

Ver [[Glossario]] para *cointegração*.

## Resultados (10 pares econômicos, H4, custo realista)

- **Sharpe agregado -0,17**
- Apenas **3/10 pares positivos** — concentrado e fraco
- **Ouro/prata foi o PIOR (-0,69)** — o par se desacoplou na última década
- **DSR 34%**

## Por que falhou

Pares é uma estratégia **"lotada"**: foi popularizada e **arbitrada desde ~2010**. Quando todo mundo negocia a mesma relação, o edge desaparece. O caso ouro/prata é emblemático: a relação histórica entre os dois metais **se desacoplou** na última década, e o par que parecia "óbvio" foi o que mais perdeu. Com só 3/10 pares positivos e Sharpe agregado negativo, não há sinal explorável líquido de custo.

## Lição

Uma relação que já foi boa pode morrer. Estratégias populares e antigas tendem a ter o edge já consumido pelo mercado. Comparar com a vencedora é útil: o [[Estrategia - Volatilidade (Straddle)|straddle]] tem amplitude (12/15 ativos) e consistência (6/6 folds); pares tem o oposto (3/10 pares, folds fracos).

## Relacionado

- [[Placar de Estrategias]]
- [[Estrategia - Volatilidade (Straddle)]]
- [[Metodologia]]
- [[Glossario]]
