# INNOVA EA

Plataforma modular de **pesquisa e backtest** para trading algorítmico no
mercado Forex, projetada com rigor de mesa institucional: custos reais,
ausência de look-ahead por construção e performance (Polars + NumPy + Numba).

> ⚠️ **Aviso**: trading de Forex/CFDs envolve alto risco. Esta plataforma é
> ferramenta de *pesquisa*. Nenhum resultado de backtest garante lucro futuro.
> Faça forward test em conta demo antes de qualquer capital real.

## Por que ela é diferente

A maioria dos "robôs" falha por overfitting e por backtests irrealistas. Aqui o
realismo é o default:

- **Custos sempre aplicados**: meio-spread + slippage embutidos no preço de
  execução, comissão round-turn por lote e swap por noite carregada.
- **Risco implacável (estilo MT5)**: margem exigida por alavancagem, **bloqueio
  de ordens** quando a margem livre é insuficiente e **stop-out** (liquidação
  forçada) no nível da corretora.
- **Sem look-ahead**: o sinal decidido no fechamento da barra `i` só executa na
  abertura de `i+1`. A API torna impossível "ver o futuro".
- **Pior preço**: toda execução piora a favor da corretora.
- **Walk-forward + anti-overfitting**: avaliação out-of-sample por janelas e
  Deflated Sharpe Ratio para descontar o número de tentativas.
- **Métricas institucionais**: Sharpe, Sortino, Calmar, max drawdown e duração,
  VaR/CVaR, Profit Factor, expectancy — sempre líquidas de custos.

## Arquitetura

Camadas independentes e testáveis (detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)):

```
core      → tipos fundamentais (Instrument, Bars, Timeframe)
data      → pipeline (DataSource → resample → ParquetStore)
backtest  → engine realista (custos · margem/stop-out · walk-forward · métricas)
features  → engenharia de features (volatilidade OHLC · direção · sessões)
strategy  → geração de sinais sem look-ahead
research  → rotulagem (triple-barrier) · mineração de padrões · PSR/DSR
```

## Instalação

```bash
cd innova_ea
pip install -r requirements.txt          # núcleo (roda em Linux/CI/cloud)
# No Windows, para dados/execução ao vivo:
pip install MetaTrader5
```

## Uso rápido

```python
from datetime import datetime, timezone
from innova_ea.core import Timeframe, get_instrument
from innova_ea.data import DataPipeline, ParquetStore, SyntheticSource, get_mt5_source
from innova_ea.backtest import BacktestEngine, CostModel
from innova_ea.strategy import MovingAverageCrossover

# 1) Fonte de dados: sintética (pesquisa/CI) ou MT5 (produção).
source = SyntheticSource(seed=7)               # ou: get_mt5_source(login=..., server=...)
store = ParquetStore("./data")

# 2) Ingestão: baixa M15 e deriva H1/H4/D1 em Parquet particionado.
pipe = DataPipeline(source, store)
pipe.ingest("EURUSD", Timeframe.M15,
            datetime(2015, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            derive=[Timeframe.H1, Timeframe.H4, Timeframe.D1])

# 3) Backtest com custos + risco realista (margem/stop-out estilo MT5).
from innova_ea.backtest import AccountConfig
bars = store.read("EURUSD", Timeframe.H1)
engine = BacktestEngine(get_instrument("EURUSD"), CostModel(slippage_points=2.0),
                        AccountConfig(leverage=100, stop_out_level=0.5),
                        initial_capital=10_000, max_lots=0.1)
result = engine.run(bars, MovingAverageCrossover(20, 50), Timeframe.H1)
print(result.report.summary())
print(result.risk.summary())   # nível de margem mín., stop-outs, ordens rejeitadas
```

### Walk-forward + Deflated Sharpe

```python
from innova_ea.backtest import WalkForward
from innova_ea.research import deflated_sharpe_ratio, deannualize_sharpe
from innova_ea.backtest.metrics import periods_per_year

wf = WalkForward(train_size=8000, test_size=4000)
res = wf.run(bars,
             strategy_factory=lambda train: MovingAverageCrossover(20, 50),
             engine_factory=lambda: engine, timeframe=Timeframe.H1)
print(res.summary())   # estabilidade do Sharpe entre janelas OOS

ppy = periods_per_year(Timeframe.H1)
trials = [deannualize_sharpe(s, ppy) for s in res.oos_sharpes]
print(deflated_sharpe_ratio(trials, n_obs=bars.height))  # prob. de edge real
```

### Engenharia de features + mineração de padrões

```python
from innova_ea.features import FeatureSet, YangZhang, EfficiencyRatio, SessionOpenFeature
from innova_ea.features.sessions import LONDON
from innova_ea.research import forward_return, scan_conditions
import polars as pl

feats = FeatureSet([
    YangZhang(20), EfficiencyRatio(20),
    SessionOpenFeature(LONDON, opening_range_bars=4),
]).transform(bars)
feats = feats.with_columns(forward_return(bars, horizon=4).alias("fwd_ret"))

# Minera padrões nos horários de liquidez (média, hit rate, t-stat vs baseline).
patterns = scan_conditions(feats, "fwd_ret", {
    "londres_thrust_alta": (pl.col("london_active") == 1) & (pl.col("london_thrust_norm") > 0),
    "londres_breakout":     pl.col("london_or_breakout") == 1,
}, min_samples=50)
for p in patterns:
    print(p.summary())
```

Demos completas offline:

```bash
PYTHONPATH=src python examples/run_backtest_demo.py        # backtest + risco + walk-forward
PYTHONPATH=src python examples/run_pattern_mining_demo.py  # features + mineração de sessões
```

## Testes

```bash
PYTHONPATH=src python -m pytest
```

A suíte valida P&L exato, ausência de look-ahead (lag de 1 barra), impacto dos
custos, contabilidade de posições short, a fórmula de margem MT5, stop-out e
bloqueio de ordens, o walk-forward e o Deflated Sharpe Ratio.

## Conectando ao MetaTrader 5

`MT5Source` pagina o histórico e normaliza tudo para o schema canônico (UTC).
Funciona apenas no Windows com um terminal MT5 instalado e logado:

```python
src = get_mt5_source(login=12345678, password="...", server="MeuBroker-Demo")
```

Para pesquisa de longo prazo, ingira uma vez para Parquet e leia de lá — é
ordens de magnitude mais rápido que reconsultar o terminal.

## Roadmap

- [x] **Fase 0** — Core + Data Pipeline + Backtest Engine + Métricas
- [x] **Fase 3** — Risco/margem/stop-out (MT5) + walk-forward + anti-overfitting
- [x] **Fase 2** — Features avançadas (volatilidade/direção/sessões) + rotulagem
  (triple-barrier) + mineração de padrões por estatística condicional
- [ ] **Fase 1** — Conector MT5 real + ingestão Dukascopy 2015→hoje
- [ ] **Fase 4** — Camada de IA (modelos) sobre o mesmo backtest
- [ ] **Fase 5** — Forward test em demo → infra cloud (AWS/GCP) → execução live

## Licença

MIT (ver `LICENSE` na raiz do repositório).
