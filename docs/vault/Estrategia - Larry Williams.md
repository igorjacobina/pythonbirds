# Estratégia - Larry Williams

> Validação **independente** do edge encontrado pelo [[INNOVA EA - MOC|INNOVA EA]]. O método de Larry Williams é da **mesma família** (rompimento de volatilidade) que o [[Estrategia - Volatilidade (Straddle)|straddle]] que nosso sistema identificou, sozinho, como a única com edge real.

## Quem é

Larry Williams é um dos traders mais conhecidos do mundo. Em **1987**, venceu o **Campeonato Mundial de Futuros** (Robbins World Cup), transformando **US$ 10.000 em US$ 1.137.600** — um retorno de **11.376% em 12 meses**. É um **recorde que se mantém até hoje**.

## O método principal: Volatility Breakout

A peça central da abordagem de Williams é o **rompimento de volatilidade**:

- **Compra** com ordem **stop** em `abertura + k × (range do dia anterior)`
- **Vende** com ordem **stop** em `abertura − k × (range do dia anterior)`
- com **k ≈ 0,25 a 0,6**
- **Stop** no ponto médio entre a mínima anterior e o preço de entrada
- **Saída** na **"primeira abertura lucrativa"**

A ideia: se o preço se move o suficiente além da abertura (uma fração do range do dia anterior), isso sinaliza expansão de volatilidade com continuação — entra-se na direção do rompimento.

## Outras ferramentas que usava

- **Sazonalidade** (padrões de calendário)
- **COT** (Commitment of Traders) — posicionamento dos grandes players
- **Padrão "Oops!"** — reversão de gap de abertura

## A conexão com o nosso straddle — o ponto crucial

O método de Larry Williams é **da mesma família** que o INNOVA EA identificou, de forma **totalmente independente**, como a **única estratégia com edge real**: **rompimento de volatilidade**.

Isso é uma **validação externa poderosa**. Nosso sistema não partiu de Larry Williams — chegou ao rompimento de volatilidade por pesquisa quantitativa rigorosa (custos reais, walk-forward purgado, DSR), reprovando direção, meta-labeling, padrões de vela e pares. Que essa mesma família seja a base de um recorde mundial de trading, descoberta décadas antes por outro caminho, é uma convergência difícil de explicar por acaso. Tanto o straddle do INNOVA EA quanto o breakout de Williams apostam no **tamanho** do movimento após expansão de volatilidade, e não na direção prevista — exatamente o que diferencia o vencedor dos perdedores no nosso [[Placar de Estrategias|placar]].

## Relacionado

- [[Estrategia - Volatilidade (Straddle)]]
- [[Placar de Estrategias]]
- [[INNOVA EA - MOC]]
- [[Glossario]]
