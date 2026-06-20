# Estratégia - Padrões de Vela

> Família 3 do [[Placar de Estrategias|placar]]. **Veredito: ❌ sem edge.**

## A ideia

Usar **padrões de candlestick** (velas) em regiões de **topo/fundo** como sinal de **reversão**. É um dos pilares da análise técnica clássica: a promessa de que certas formações de velas (martelos, estrelas, engolfos, etc.) antecipam viradas de tendência.

## Resultados (líquidos de custo)

- **37.394 eventos** de padrões analisados.
- **Ganho de precisão +0,015** — praticamente zero, **dentro do ruído**.
- **Sharpe OOS -0,17**.
- Apenas **15% dos folds positivos** — esmagadora maioria dos períodos negativa.

## Por que falhou

O ganho de precisão de +0,015 é estatisticamente indistinguível de zero. Com tão poucos folds positivos (15%), não há consistência alguma. O resultado **confirma a literatura acadêmica**: padrões de vela **não preveem o futuro** de forma líquida de custo. O que parece estrutura no gráfico é, na prática, ruído reorganizado pela mente humana à procura de padrões.

## Lição

Popularidade não é evidência. Um sinal precisa de **consistência entre folds** e ganho acima do ruído — exatamente o que o [[Estrategia - Volatilidade (Straddle)|straddle]] tem (6/6 folds) e os padrões de vela não têm (15% dos folds).

## Relacionado

- [[Placar de Estrategias]]
- [[Estrategia - Volatilidade (Straddle)]]
- [[Metodologia]]
- [[Glossario]]
