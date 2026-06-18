"""Loader de histórico em CSV (Dukascopy export, HistData.com, exports de broker).

Mapeia colunas heterogêneas para o schema canônico. Suporta header opcional,
separador configurável, timestamps ISO/epoch/combinados e conversão de fuso
(zona IANA ou offset fixo, ex. HistData em EST = UTC-5).
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from innova_ea.core.bars import validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.base import DataSource


class CsvSource(DataSource):
    """Fonte de barras a partir de um único arquivo CSV.

    Args:
        path: caminho do CSV.
        column_map: mapeia nomes do CSV -> canônico
            (chaves: time, open, high, low, close, volume).
        time_format: formato strptime; ``None`` tenta epoch e depois ISO.
        time_unit_epoch: 's' ou 'ms' quando o tempo for epoch numérico.
        separator: separador de campos (ex. ';' no HistData).
        has_header: se o CSV tem cabeçalho.
        new_columns: nomes a atribuir às colunas (útil quando ``has_header=False``).
        source_timezone: fuso IANA de origem (naive → este fuso → UTC).
        source_utc_offset: offset fixo em horas da origem (ex. -5 para EST).
    """

    def __init__(
        self,
        path: str,
        *,
        column_map: dict[str, str] | None = None,
        time_format: str | None = None,
        time_unit_epoch: str = "s",
        separator: str = ",",
        has_header: bool = True,
        new_columns: list[str] | None = None,
        source_timezone: str | None = None,
        source_utc_offset: float | None = None,
    ) -> None:
        self.path = path
        self.column_map = column_map or {
            "time": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        }
        self.time_format = time_format
        self.time_unit_epoch = time_unit_epoch
        self.separator = separator
        self.has_header = has_header
        self.new_columns = new_columns
        self.source_timezone = source_timezone
        self.source_utc_offset = source_utc_offset

    def _parse_time(self, col: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        if dtype.is_numeric():
            return pl.from_epoch(col, time_unit=self.time_unit_epoch)
        if self.time_format:
            return col.str.strptime(pl.Datetime, self.time_format, strict=False)
        return col.str.to_datetime(strict=False)

    def _to_utc(self, df: pl.DataFrame) -> pl.DataFrame:
        tz = df.schema["time"].time_zone
        if tz is not None:                                   # já tem fuso → UTC
            return df.with_columns(pl.col("time").dt.convert_time_zone("UTC"))
        if self.source_utc_offset is not None:               # naive em offset fixo
            mins = int(round(self.source_utc_offset * 60))
            return df.with_columns(
                (pl.col("time") - pl.duration(minutes=mins)).dt.replace_time_zone("UTC")
            )
        if self.source_timezone is not None:                 # naive em fuso IANA
            return df.with_columns(
                pl.col("time").dt.replace_time_zone(self.source_timezone, ambiguous="earliest")
                .dt.convert_time_zone("UTC")
            )
        return df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))  # assume UTC

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        raw = pl.read_csv(
            self.path, separator=self.separator, has_header=self.has_header,
            new_columns=self.new_columns,
        )
        inv = {v: k for k, v in self.column_map.items()}  # csv_name -> canônico
        raw = raw.rename({k: v for k, v in inv.items() if k in raw.columns})

        df = raw.with_columns(self._parse_time(pl.col("time"), raw.schema["time"]).alias("time"))
        df = self._to_utc(df)
        df = df.with_columns(
            pl.col("time").dt.cast_time_unit("us"),
            pl.col(["open", "high", "low", "close", "volume"]).cast(pl.Float64),
        )
        df = df.filter((pl.col("time") >= start) & (pl.col("time") < end)).sort("time")
        return validate_bars(df, symbol=symbol)
