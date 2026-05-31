"""Forex Quant — plataforma de pesquisa e backtest para trading algorítmico.

Camadas:
    core      — tipos e contratos fundamentais (instrumentos, barras).
    data      — pipeline de dados (fontes, resample, storage Parquet).
    backtest  — engine realista (custos, execução, métricas).
    strategy  — geração de sinais sem look-ahead.
"""
from forex_quant.backtest import BacktestEngine, CostModel, PerformanceReport
from forex_quant.core import Instrument, Timeframe, get_instrument
from forex_quant.data import DataPipeline, ParquetStore, SyntheticSource
from forex_quant.strategy import MovingAverageCrossover, Strategy

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BacktestEngine",
    "CostModel",
    "PerformanceReport",
    "Instrument",
    "Timeframe",
    "get_instrument",
    "DataPipeline",
    "ParquetStore",
    "SyntheticSource",
    "Strategy",
    "MovingAverageCrossover",
]
