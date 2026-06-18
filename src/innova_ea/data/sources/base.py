"""Contrato abstrato de fonte de dados.

Qualquer origem (MT5, Dukascopy, CSV, sintético) implementa a mesma interface,
de modo que o pipeline e o backtest são agnósticos à procedência dos dados.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import polars as pl

from innova_ea.core.enums import Timeframe


class DataSource(ABC):
    """Fonte de barras OHLCV.

    Implementações devem retornar dados já no schema canônico
    (ver ``innova_ea.core.bars``), em UTC e ordenados.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Busca barras de ``symbol`` em ``[start, end)`` no ``timeframe`` dado."""

    def close(self) -> None:  # opcional: liberar conexões/recursos
        return None

    def __enter__(self) -> "DataSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
