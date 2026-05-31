"""Orquestração da ingestão: fetch → validate → (resample) → store.

Idempotente e retomável: como o ``ParquetStore`` faz upsert por ano, reexecutar
a ingestão de um range já baixado não duplica dados. Permite ingerir o timeframe
base uma vez e derivar os superiores localmente, sem rebaixar tudo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from forex_quant.core.enums import Timeframe
from forex_quant.data.resample import resample
from forex_quant.data.sources.base import DataSource
from forex_quant.data.storage import ParquetStore


@dataclass
class IngestionReport:
    symbol: str
    base_timeframe: Timeframe
    bars_fetched: int = 0
    bars_written: dict[str, int] = field(default_factory=dict)


class DataPipeline:
    """Liga uma ``DataSource`` a um ``ParquetStore``."""

    def __init__(self, source: DataSource, store: ParquetStore) -> None:
        self.source = source
        self.store = store

    def ingest(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        start: datetime,
        end: datetime,
        *,
        derive: list[Timeframe] | None = None,
    ) -> IngestionReport:
        """Ingere o timeframe base e, opcionalmente, deriva timeframes superiores.

        Args:
            derive: timeframes a gerar por agregação a partir do base
                (devem ser >= base_timeframe).
        """
        report = IngestionReport(symbol=symbol, base_timeframe=base_timeframe)

        base = self.source.fetch(symbol, base_timeframe, start, end)
        report.bars_fetched = base.height
        if base.height == 0:
            return report

        report.bars_written[base_timeframe.value] = self.store.write(
            symbol, base_timeframe, base
        )

        for tf in derive or []:
            if tf.minutes < base_timeframe.minutes:
                raise ValueError(
                    f"não é possível derivar {tf.value} (menor) "
                    f"de {base_timeframe.value}"
                )
            if tf == base_timeframe:
                continue
            agg = resample(base, tf)
            report.bars_written[tf.value] = self.store.write(symbol, tf, agg)

        return report

    def ingest_many(
        self,
        symbols: list[str],
        base_timeframe: Timeframe,
        start: datetime,
        end: datetime,
        *,
        derive: list[Timeframe] | None = None,
    ) -> list[IngestionReport]:
        return [
            self.ingest(s, base_timeframe, start, end, derive=derive)
            for s in symbols
        ]
