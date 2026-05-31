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
- `sources/mt5_source.py`: conector MetaTrader 5 (produção/forward test).
- `sources/csv_source.py`: loader de histórico (Dukascopy/CSV) para pesquisa.
- `sources/synthetic.py`: gerador GBM com microestrutura, para testes/CI sem rede.
- `resample.py`: agregação de timeframe 100% Polars (right-closed, label correto).
- `storage.py`: `ParquetStore` particionado, idempotente, com upsert por range.
- `pipeline.py`: orquestra fetch → validate → store, com checkpoint/retomada.

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

### L5 — `research`
- `overfitting.py`: Probabilistic & Deflated Sharpe Ratio (Bailey & López de
  Prado) — desconta o número de tentativas para separar skill de sorte.
- *(próximas fases)* mineração de padrões e modelos (XGBoost → atenção).

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
- **Fase 1**: Conector MT5 real + ingestão de histórico Dukascopy + storage.
- **Fase 2**: Biblioteca de features e detecção de padrões de movimento.
- **Fase 4**: Camada de IA (modelos) sobre o mesmo backtest.
- **Fase 5**: Forward test em demo → infra cloud (AWS/GCP) → execução live.

> A Fase 3 foi priorizada antes da 2: gestão de risco precisa estar perfeita
> antes de minerar padrões, para que toda métrica de estratégia já nasça sob as
> mesmas restrições de margem que a conta real enfrentará.

## 8. Infra cloud (alvo)

- Ingestão e backtests pesados em instâncias compute-optimized (spot) com
  storage em S3/GCS (Parquet). Orquestração por jobs idempotentes.
- Execução live em VM low-latency próxima ao broker (colocation quando viável),
  separada da pesquisa. Observabilidade: métricas de latência fim-a-fim,
  reconciliação de fills, kill-switch de risco.
