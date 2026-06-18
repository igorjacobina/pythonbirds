# Estratégia - Meta-labeling

> Família 2 do [[Placar de Estrategias|placar]]. **Veredito: ❌ sem edge.**

## A ideia

**Meta-labeling** (López de Prado): você não usa o ML para decidir *o que* negociar, e sim para decidir *se* deve seguir um sinal primário. Aqui, a regra primária é um **rompimento de Donchian** (compra/vende quando o preço rompe a máxima/mínima de N períodos), e o modelo de ML atua como **filtro**: aprova ou veta cada rompimento, tentando separar os vencedores dos perdedores.

Ver [[Glossario]] para *meta-labeling*.

## Resultados (líquidos de custo)

- **Ganho de precisão ~0**: a precisão dos rompimentos aprovados pelo modelo (≈ 50%) é praticamente igual à **base win-rate** (~50%) da regra sem filtro. O filtro não adiciona informação.
- Com **threshold 0,50**: Sharpe **-1,39**.
- **DSR 0%**.

## Por que falhou

O ML **não consegue separar** rompimentos vencedores dos perdedores. Se o conjunto de rompimentos já é ~50/50 e o modelo aprova um subconjunto que continua ~50/50, ele não está filtrando nada útil — está apenas reduzindo o número de trades sem melhorar a qualidade. Subtraindo custos, o resultado fica negativo.

## Lição

Filtrar uma regra ruim com ML não cria edge do nada. O problema não era a falta de filtro, era a ausência de sinal previsível na regra primária de rompimento direcional — o mesmo limite que afundou a [[Estrategia - Direcao (ML)|estratégia de direção]].

## Relacionado

- [[Placar de Estrategias]]
- [[Estrategia - Direcao (ML)]]
- [[Metodologia]]
- [[Glossario]]
