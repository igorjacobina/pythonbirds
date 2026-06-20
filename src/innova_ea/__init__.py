"""INNOVA EA — plataforma de pesquisa e backtest para trading algorítmico.

Camadas:
    core      — tipos e contratos fundamentais (instrumentos, barras).
    data      — pipeline de dados (fontes, resample, storage Parquet).
    backtest  — engine realista (custos, margem/risco, execução, métricas).
    strategy  — geração de sinais sem look-ahead.
    research  — controle de overfitting (PSR/DSR) e mineração de padrões.
"""
from innova_ea.backtest import (
    AccountConfig,
    BacktestEngine,
    CostModel,
    PerformanceReport,
    RiskReport,
    WalkForward,
)
from innova_ea.core import Instrument, Timeframe, get_instrument
from innova_ea.data import DataPipeline, ParquetStore, SyntheticSource
from innova_ea.research import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from innova_ea.strategy import MovingAverageCrossover, Strategy

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AccountConfig",
    "BacktestEngine",
    "CostModel",
    "PerformanceReport",
    "RiskReport",
    "WalkForward",
    "Instrument",
    "Timeframe",
    "get_instrument",
    "DataPipeline",
    "ParquetStore",
    "SyntheticSource",
    "Strategy",
    "MovingAverageCrossover",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]
