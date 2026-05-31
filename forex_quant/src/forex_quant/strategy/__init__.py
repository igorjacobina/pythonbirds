"""Camada L4 — estratégias (geração de sinais sem look-ahead)."""
from forex_quant.strategy.base import (
    MovingAverageCrossover,
    SignalFromColumn,
    Strategy,
)

__all__ = ["Strategy", "SignalFromColumn", "MovingAverageCrossover"]
