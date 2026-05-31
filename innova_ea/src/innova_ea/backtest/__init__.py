"""Camada L3 — backtest realista (custos, margem/risco, execução, métricas)."""
from innova_ea.backtest.account import AccountConfig
from innova_ea.backtest.costs import CostModel
from innova_ea.backtest.engine import BacktestEngine, BacktestResult
from innova_ea.backtest.metrics import (
    PerformanceReport,
    RiskReport,
    compute_metrics,
    compute_risk,
    periods_per_year,
)
from innova_ea.backtest.walkforward import (
    WalkForward,
    WalkForwardResult,
    WalkForwardWindow,
)

__all__ = [
    "AccountConfig",
    "CostModel",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceReport",
    "RiskReport",
    "compute_metrics",
    "compute_risk",
    "periods_per_year",
    "WalkForward",
    "WalkForwardResult",
    "WalkForwardWindow",
]
