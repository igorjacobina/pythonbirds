"""Camada L3 — backtest realista (custos, execução, métricas)."""
from forex_quant.backtest.costs import CostModel
from forex_quant.backtest.engine import BacktestEngine, BacktestResult
from forex_quant.backtest.metrics import PerformanceReport, compute_metrics, periods_per_year

__all__ = [
    "CostModel",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceReport",
    "compute_metrics",
    "periods_per_year",
]
