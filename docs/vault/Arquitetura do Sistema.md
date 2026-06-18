# Arquitetura do Sistema

> As camadas do [[INNOVA EA - MOC|INNOVA EA]] e o fluxo de ponta a ponta: do dado real até o forward test. Ver [[Metodologia]] para os princípios que guiam cada camada.

## O fluxo geral

```
dado real → features → modelo → backtest com risco → execução → forward test (demo)
```

Cada etapa preserva a causalidade (sem look-ahead) e carrega os custos reais adiante. Nada avança para a próxima etapa sem passar pelo crivo da [[Metodologia]].

## As camadas

### core — tipos fundamentais
Define os tipos base do domínio: `Instrument`, `Bars`, `Timeframe`. É o vocabulário comum sobre o qual tudo é construído.

### data — ingestão e dados
Ingestão de várias fontes (**MT5 / Dukascopy / HistData / CSV**), limpeza, detecção de **gaps**, **resample** entre timeframes e armazenamento em **ParquetStore**. Princípios: conversão de fuso do servidor → **UTC** com rigor (tratando DST), calendário de mercado **DST-aware** (17:00 Nova York), e **limpeza defensiva** — nunca fabrica preço; buracos são **detectados e reportados**.

**Dados**: 15 ativos, M1 desde 2015 (~48 milhões de barras) — Forex majors (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD), metais (XAUUSD, XAGUSD), índices US (US30, US100, US500) e globais (DE40, JP225, UK100, HK50). Baixados via Dukascopy (tick→M1), em Parquet particionado.

### features — engenharia de features
Construção das features causais:
- **Volatilidade OHLC**: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
- **Direção**: efficiency ratio, close location value
- **Sessões / horários de liquidez**

### backtest — engine realista
Simulação fiel à realidade: **custos** (spread, slippage, comissão, swap), **margem / stop-out** estilo MT5, **walk-forward** purgado e **métricas institucionais**: Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR, Profit Factor.

### strategy — geração de sinais
Geração de sinais de trading **sem look-ahead**: o sinal da barra `i` só executa na abertura de `i+1`.

### research — pesquisa
O laboratório: **rotulagem** (triple-barrier), **mineração de padrões**, **dataset universal**, **modelos** (LightGBM + "Super Cérebro" Transformer com Asset Embeddings), análise de **overfitting** (PSR/DSR), **meta-labeling**, **straddle** e **pares**. É aqui que o [[Placar de Estrategias|placar]] foi produzido.

### execution — execução ao vivo
Execução **event-driven** ao vivo: **conciliação de posições**, **kill-switch** e execução de **straddle** (ordens **OCO** — one-cancels-the-other). É a camada que leva a estratégia aprovada ao [[Glossario|forward test]] demo.

## Como o fluxo conecta as estratégias

A estratégia [[Estrategia - Volatilidade (Straddle)|straddle]] passou por todas as camadas: dados Dukascopy (data) → features de volatilidade OHLC (features) → modelo de research → backtest com custos e risco → execução event-driven com ordens OCO → forward test demo. As estratégias reprovadas pararam no backtest/research.

## Relacionado

- [[INNOVA EA - MOC]]
- [[Metodologia]]
- [[Operacao - Como Rodar]]
- [[Placar de Estrategias]]
- [[Glossario]]
