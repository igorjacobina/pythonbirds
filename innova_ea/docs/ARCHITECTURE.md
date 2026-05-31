# INNOVA EA — Arquitetura de Alta Performance

> Plataforma modular para pesquisa, backtest e (futuramente) execução de
> estratégias algorítmicas no mercado Forex, com foco em rigor estatístico,
> custos reais e robustez institucional.

## 1. Princípios de projeto

1. **Separation of concerns**: dados, sinais, execução simulada e métricas
   são módulos independentes e testáveis isoladamente.
2. **Realismo acima de tudo**: nenhum backtest é confiável sem spread,
   slippage, comissão e swap modelados. O default é *pessimista*.
3. **Sem look-ahead**: a API impede, por construção, que uma decisão use
   informação da própria vela ou do futuro.
4. **Performance**: pipeline colunar (Polars/Arrow), kernels numéricos em
   NumPy vetorizado e Numba (JIT) nos loops quentes do backtest.
5. **Reprodutibilidade**: dados versionados em Parquet particionado,
   configs declarativas, seeds fixas.

## 2. Visão de camadas

```
┌──────────────────────────────────────────────────────────────┐
│  L5  Research / IA      (feature mining, modelos, otimização)  │
├──────────────────────────────────────────────────────────────┤
│  L4  Strategy           (Strategy ABC → sinais)                │
├──────────────────────────────────────────────────────────────┤
│  L3  Backtest           (costs · execution · portfolio ·       │
│                          engine · metrics)                     │
├──────────────────────────────────────────────────────────────┤
│  L2  Data Pipeline      (sources · resample · storage)         │
├──────────────────────────────────────────────────────────────┤
│  L1  Core               (instruments · bars · tipos)           │
└──────────────────────────────────────────────────────────────┘
```

Cada camada só depende das de baixo. A IA (L5) consome o mesmo backtest (L3)
que valida estratégias manuais — sem caminho privilegiado, sem viés.

## 3. Fluxo de dados (ingestão → pesquisa)

```
MT5 / Dukascopy / CSV
        │  (DataSource.fetch)
        ▼
   Bars (schema canônico OHLCV, UTC, Polars)
        │  (validação: gaps, monotonicidade, NaN)
        ▼
   ParquetStore  (particionado por symbol/timeframe/ano)
        │  (resample: M1 → M5/M15/H1/H4/D1)
        ▼
   Feature engineering  →  Strategy  →  Backtest  →  Metrics
```

Regra de ouro: **armazena-se o timeframe mais granular disponível (idealmente
M1 ou tick)** e todos os superiores são derivados por agregação determinística.
Isso elimina inconsistências entre timeframes e economiza storage.

## 4. Módulos

### L1 — `core`
- `instruments.py`: `Instrument` com especificação real do ativo — pip size,
  dígitos, contract size, moeda de cotação, comissão por lote, swap long/short,
  spread típico. É a fonte de verdade para converter preço → P&L em dinheiro.
- `bars.py`: schema canônico de OHLCV e validações (sem look-ahead, sem gaps
  silenciosos, timestamps em UTC e monotônicos).
- `enums.py`: `Timeframe`, `Side`, `OrderType`.

### L2 — `data`
- `sources/base.py`: `DataSource` (ABC) — contrato `fetch(symbol, tf, start, end)`.
- `sources/mt5_source.py`: conector MetaTrader 5 — init/login com retry, fuso do
  servidor → UTC, histórico profundo em chunks; módulo MT5 injetável para testes.
- `sources/csv_source.py`: loader de histórico (Dukascopy/CSV) para pesquisa.
- `sources/synthetic.py`: gerador GBM com microestrutura, para testes/CI sem rede.
- `timeutils.py`: conversão impecável do epoch do servidor (EET/DST) para UTC.
- `calendar.py`: `ForexCalendar` — janela semanal de mercado e feriados; gera a
  grade de timestamps esperados (referência para detecção de buracos).
- `cleaning.py`: limpeza defensiva (`clean_bars`) e análise de gaps
  (`analyze_gaps`) contra o calendário — detecta buracos reais e fuso errado.
- `resample.py`: agregação de timeframe 100% Polars (right-closed, label correto).
- `storage.py`: `ParquetStore` particionado, idempotente, com upsert por range.
- `pipeline.py`: orquestra fetch → validate → store, com checkpoint/retomada.

#### Ingestão MT5 (princípios)

- **Fuso**: o `time` do MT5 é o relógio do servidor (EET, +2/+3 com DST). Tratá-lo
  como UTC é o erro silencioso mais comum; convertemos via fuso IANA ou offset fixo.
- **Buracos**: nunca se fabrica preço (forward-fill cria candle falso). Buracos são
  detectados contra a grade de mercado e **reportados**; fim de semana/feriado não
  contam como buraco. Barras fora do horário disparam alerta de fuso.
- **Histórico profundo**: M1 desde 2015 (~3,7M barras/par) é baixado em chunks
  mensais; o `ParquetStore` deduplica na gravação (idempotente/retomável).
- Script pronto: `scripts/ingest_mt5.py` (EUR/USD primeiro, depois as majors).

### L3 — `backtest`
- `costs.py`: modelos de `Spread`, `Slippage`, `Commission`, `Swap`.
- `account.py`: `AccountConfig` — alavancagem e níveis de margin call/stop-out
  no estilo MT5.
- `engine.py`: núcleo do backtest. Loop vetorizado por vela; hot path em Numba.
  Aplica margem exigida, **bloqueio de ordens** por margem livre insuficiente e
  **stop-out** (liquidação forçada) da corretora.
- `walkforward.py`: avaliação out-of-sample com janelas deslizantes/ancoradas.
- `metrics.py`: métricas institucionais — CAGR, Sharpe, Sortino, Calmar,
  max drawdown e duração, Profit Factor, expectancy, VaR/CVaR, exposição — e
  `RiskReport` (nível de margem mínimo, stop-outs, ordens rejeitadas).

#### Modelo de margem (MT5)

`Margem = ContractSize × Lots / Leverage`, calculada na **moeda base** do par e
convertida para a **moeda da conta**. Quando a base já é a moeda da conta
(ex. USDJPY/conta USD) a margem independe do preço; caso contrário
(ex. EURUSD/conta USD) multiplica-se pelo preço corrente. O engine bloqueia
aberturas cujo nível de margem resultante fique abaixo do *margin call* e
liquida à força quando o nível atinge o *stop-out*.

### L4 — `strategy`
- `base.py`: `Strategy` (ABC). Recebe um DataFrame de barras *até* t-1 e devolve
  o alvo de posição para t. A interface garante ausência de look-ahead.
- `model_strategy.py`: `ModelStrategy` — converte previsões de um modelo
  universal em alvo de posição, fechando o ciclo com o engine de risco (Fase 3).

### L4b — `features`
- `base.py`: `Feature`/`FeatureSet` — composição de features causais em matriz.
- `volatility.py`: estimadores OHLC de alta eficiência (Parkinson, Garman-Klass,
  Rogers-Satchell, Yang-Zhang) + regime de volatilidade.
- `directional.py`: efficiency ratio (Kaufman), close location value, corpo/pavios.
- `returns.py`: log-retornos, retorno acumulado e z-score de retorno.
- `sessions.py`: features de abertura de sessão (thrust, opening range/breakout,
  expansão de volatilidade) para Tóquio/Londres/NY e killzones (UTC).
- `time_features.py`: encoding cíclico (sin/cos) e perfil de volatilidade
  intradiário (fit/transform, sem vazamento).

### L5 — `research`
- `labeling.py`: forward returns e **triple-barrier** (López de Prado).
- `patterns.py`: mineração por estatística condicional de retorno futuro
  (média, hit rate, expectancy, t-stat vs. baseline).
- `dataset.py`: `UniversalPanel` — empilha todos os ativos num único painel
  (features + `asset_id`/classe + rótulo) com barreiras adaptativas por
  volatilidade; `PerAssetScaler` padroniza por ativo (ajuste só no treino).
- `splitting.py`: `PurgedWalkForward` — walk-forward com **purga + embargo** para
  rótulos sobrepostos (sem vazamento via horizonte).
- `models/`: `UniversalModel` (contrato), `LightGBMUniversal` (baseline com
  `asset_id` categórico), `SuperBrain` (Transformer causal com Asset Embeddings;
  torch lazy).
- `overfitting.py`: Probabilistic & Deflated Sharpe Ratio (Bailey & López de
  Prado) — desconta o número de tentativas para separar skill de sorte.

#### Super Cérebro (Universal Model)

Um único modelo devora o painel de TODOS os ativos. A identidade do ativo entra
por **embeddings** (ativo + classe) no Transformer, ou como **categórico** no
LightGBM — o backbone compartilhado aprende a física universal do preço e
transfere conhecimento entre ativos (*cross-asset learning*). O Transformer usa
**causal mask** (sem look-ahead). Estratégia campeão-desafiante: LightGBM como
baseline robusto, Super Cérebro como modelo-alvo. As previsões são sempre
validadas pelo MESMO backtest com custos/risco (Fase 3) e Deflated Sharpe.

## 5. Decisões técnicas

| Tema            | Escolha                    | Porquê                                            |
|-----------------|----------------------------|---------------------------------------------------|
| DataFrame       | **Polars**                 | Multi-thread, lazy, Arrow zero-copy, > pandas      |
| Storage         | **Parquet particionado**   | Colunar, comprimido, predicate pushdown            |
| Loop quente     | **Numba @njit**            | Velocidade de C mantendo Python                    |
| Tempo           | **UTC sempre**             | Evita bugs de DST e sessão                         |
| Preço           | **int points opcional**    | Evita erro de ponto flutuante em P&L               |
| Config          | **dataclasses + TOML**     | Declarativo, versionável, sem mágica               |

## 6. Anti-overfitting (não negociável)

- Split temporal rígido: **train / validation / out-of-sample intocado**.
- **Walk-forward** com janelas deslizantes como padrão de avaliação.
- Penalização por número de tentativas (deflated Sharpe ratio).
- Forward test obrigatório em conta demo antes de capital real.
- Todo resultado reportado **sempre líquido de custos**.

## 7. Roadmap de implementação

- **Fase 0 (concluída)**: Core + Data Pipeline + Backtest Engine + Metrics,
  com demo rodável em dados sintéticos e suíte de testes.
- **Fase 3 (concluída)**: Margem/stop-out/bloqueio de ordens (estilo MT5) +
  walk-forward + controle de overfitting (PSR/DSR). Risco blindado.
- **Fase 2 (concluída)**: Biblioteca de features avançadas (volatilidade OHLC,
  direção, sessões/horários de liquidez) + rotulagem (triple-barrier) + motor de
  mineração por estatística condicional.
- **Fase 1 (concluída)**: Conector MT5 real (fuso, limpeza defensiva, gaps,
  histórico profundo M1 desde 2015) + ingestão de universo macro global (FX,
  metais, índices) para Parquet.
- **Fase 4 (concluída)**: Camada de IA — modelo universal único (LightGBM
  baseline + Super Cérebro Transformer com Asset Embeddings), painel multi-ativo,
  walk-forward purgado, validado pelo mesmo backtest com risco + Deflated Sharpe.
- **Fase 5 (pacote de implantação pronto)**: script de treino (`train_models.py`),
  persistência de modelos (save/load), conjunto de features de produção
  (`features.presets`), universo centralizado (`innova_ea.universe`), manual de
  deploy Windows (`docs/DEPLOYMENT.md`) e arquitetura cloud
  (`docs/CLOUD_ARCHITECTURE.md`). Falta: forward test em demo → infra cloud
  (AWS/GCP) → execução live.

> A Fase 3 foi priorizada antes da 2: gestão de risco precisa estar perfeita
> antes de minerar padrões, para que toda métrica de estratégia já nasça sob as
> mesmas restrições de margem que a conta real enfrentará.

## 8. Infra cloud (alvo)

- Ingestão e backtests pesados em instâncias compute-optimized (spot) com
  storage em S3/GCS (Parquet). Orquestração por jobs idempotentes.
- Execução live em VM low-latency próxima ao broker (colocation quando viável),
  separada da pesquisa. Observabilidade: métricas de latência fim-a-fim,
  reconciliação de fills, kill-switch de risco.
