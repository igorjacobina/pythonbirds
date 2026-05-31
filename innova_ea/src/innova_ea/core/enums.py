"""Tipos enumerados fundamentais do domínio de trading."""
from __future__ import annotations

from enum import Enum


class Timeframe(str, Enum):
    """Timeframes suportados. O valor textual é o usado no storage/paths."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    @property
    def minutes(self) -> int:
        """Duração do timeframe em minutos (W1/MN1 são aproximações de calendário)."""
        return {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN1": 43200,
        }[self.value]

    @property
    def polars_every(self) -> str:
        """String de período aceita por ``polars.DataFrame.group_by_dynamic``."""
        return {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "M30": "30m",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
            "W1": "1w",
            "MN1": "1mo",
        }[self.value]


class Side(int, Enum):
    """Direção de uma posição. O valor inteiro é o sinal aplicado ao P&L."""

    LONG = 1
    SHORT = -1
    FLAT = 0


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
