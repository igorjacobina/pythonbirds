# Forex Quant

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
- **Sem look-ahead**: o sinal decidido no fechamento da barra `i` só executa na
  abertura de `i+1`. A API torna impossível "ver o futuro".
- **Pior preço**: toda execução piora a favor da corretora.
- **Métricas institucionais**: Sharpe, Sortino, Calmar, max drawdown e duração,
  VaR/CVaR, Profit Factor, expectancy — sempre líquidas de custos.

## Arquitetura

Camadas independentes e testáveis (detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)):

```
core      → tipos fundamentais (Instrument, Bars, Timeframe)
data      → pipeline (DataSource → resample → ParquetStore)
backtest  → engine realista (CostModel · execução · métricas)
strategy  → geração de sinais sem look-ahead
```

## Instalação

```bash
cd forex_quant
pip install -r requirements.txt          # núcleo (roda em Linux/CI/cloud)
# No Windows, para dados/execução ao vivo:
pip install MetaTrader5
```

## Uso rápido

```python
from datetime import datetime, timezone
from forex_quant.core import Timeframe, get_instrument
from forex_quant.data import DataPipeline, ParquetStore, SyntheticSource, get_mt5_source
from forex_quant.backtest import BacktestEngine, CostModel
from forex_quant.strategy import MovingAverageCrossover

# 1) Fonte de dados: sintética (pesquisa/CI) ou MT5 (produção).
source = SyntheticSource(seed=7)               # ou: get_mt5_source(login=..., server=...)
store = ParquetStore("./data")

# 2) Ingestão: baixa M15 e deriva H1/H4/D1 em Parquet particionado.
pipe = DataPipeline(source, store)
pipe.ingest("EURUSD", Timeframe.M15,
            datetime(2015, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            derive=[Timeframe.H1, Timeframe.H4, Timeframe.D1])

# 3) Backtest com custos realistas.
bars = store.read("EURUSD", Timeframe.H1)
engine = BacktestEngine(get_instrument("EURUSD"), CostModel(slippage_points=2.0),
                        initial_capital=10_000, max_lots=0.1)
result = engine.run(bars, MovingAverageCrossover(20, 50), Timeframe.H1)
print(result.report.summary())
```

Demo completa offline:

```bash
PYTHONPATH=src python examples/run_backtest_demo.py
```

## Testes

```bash
PYTHONPATH=src python -m pytest
```

A suíte valida P&L exato, ausência de look-ahead (lag de 1 barra), impacto dos
custos, contabilidade de posições short e a integridade do pipeline de dados.

## Conectando ao MetaTrader 5

`MT5Source` pagina o histórico e normaliza tudo para o schema canônico (UTC).
Funciona apenas no Windows com um terminal MT5 instalado e logado:

```python
src = get_mt5_source(login=12345678, password="...", server="MeuBroker-Demo")
```

Para pesquisa de longo prazo, ingira uma vez para Parquet e leia de lá — é
ordens de magnitude mais rápido que reconsultar o terminal.

## Roadmap

- [x] **Fase 0** — Core + Data Pipeline + Backtest Engine + Métricas (esta entrega)
- [ ] **Fase 1** — Conector MT5 real + ingestão Dukascopy 2015→hoje
- [ ] **Fase 2** — Biblioteca de features e detecção de padrões de movimento
- [ ] **Fase 3** — Walk-forward + risco/margem/stop-out + controle de overfitting
- [ ] **Fase 4** — Camada de IA (modelos) sobre o mesmo backtest
- [ ] **Fase 5** — Forward test em demo → infra cloud (AWS/GCP) → execução live

## Licença

MIT (ver `LICENSE` na raiz do repositório).
