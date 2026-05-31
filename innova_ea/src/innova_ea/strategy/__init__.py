"""Camada L4 — estratégias (geração de sinais sem look-ahead)."""
from innova_ea.strategy.base import (
    MovingAverageCrossover,
    SignalFromColumn,
    Strategy,
)

__all__ = ["Strategy", "SignalFromColumn", "MovingAverageCrossover"]
