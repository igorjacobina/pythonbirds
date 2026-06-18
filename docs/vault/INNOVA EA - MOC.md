# INNOVA EA - MOC (Map of Content)

> Mapa central da base de conhecimento do projeto **INNOVA EA**. Use esta nota como ponto de partida.

## O que é o projeto

O INNOVA EA é uma plataforma de pesquisa e trading algorítmico para Forex/CFDs, construída com rigor institucional. O objetivo não é "achar a estratégia mágica", e sim testar muitas ideias de forma honesta — aplicando sempre custos reais, evitando look-ahead e penalizando o número de tentativas — para separar **skill** (vantagem real) de **sorte**. De cinco famílias de estratégia testadas, quatro falharam de forma honesta e apenas uma sobreviveu: **rompimento de volatilidade (straddle)**. Esse é o resultado esperado em pesquisa quantitativa séria: fundos testam centenas de ideias para encontrar poucas com edge real.

## Estado atual

Estratégia aprovada (**Straddle de rompimento de volatilidade**) está em **forward test em conta DEMO**, etapa obrigatória antes de qualquer alocação de capital real. Ver [[Estrategia - Volatilidade (Straddle)]].

## Índice de notas

### Fundamentos
- [[Metodologia]] — os pilares anti-autoengano (sem look-ahead, custos, DSR, etc.)
- [[Arquitetura do Sistema]] — as camadas e o fluxo dado → execução
- [[Glossario]] — definições simples dos termos técnicos
- [[Operacao - Como Rodar]] — scripts e comandos principais

### O placar
- [[Placar de Estrategias]] — as 5 famílias, vereditos e números-chave

### Estratégias testadas
- [[Estrategia - Direcao (ML)]] — ❌ sem edge
- [[Estrategia - Meta-labeling]] — ❌ sem edge
- [[Estrategia - Padroes de Vela]] — ❌ sem edge
- [[Estrategia - Volatilidade (Straddle)]] — ✅ **edge real, fino**
- [[Estrategia - Pares (Relative-Value)]] — ❌ sem edge

### Validação externa
- [[Estrategia - Larry Williams]] — validação independente do rompimento de volatilidade

## Ficha técnica rápida

- **Stack**: Python + Polars + NumPy + Numba + LightGBM + PyTorch (opcional)
- **Dados**: MetaTrader5 / Dukascopy — 15 ativos, M1 desde 2015 (~48 milhões de barras)
- **Testes**: 152 testes automatizados, todos passando
- **Código**: raiz do repositório (`src/`, `scripts/`, `tests/`, `docs/`)

## Relacionado

- [[Metodologia]]
- [[Placar de Estrategias]]
- [[Arquitetura do Sistema]]
- [[Operacao - Como Rodar]]
