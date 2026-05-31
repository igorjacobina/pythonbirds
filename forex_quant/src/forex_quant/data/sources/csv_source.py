"""Loader de histórico em CSV (ex.: Dukascopy, exports de corretora).

Mapeia colunas heterogêneas para o schema canônico. Suporta timestamps em
string ISO ou epoch (s/ms). Para pesquisa de longo prazo, prefira converter o
CSV para Parquet uma vez via ``ParquetStore`` e ler de lá depois.
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from forex_quant.core.bars import validate_bars
from forex_quant.core.enums import Timeframe
from forex_quant.data.sources.base import DataSource


class CsvSource(DataSource):
    """Fonte de barras a partir de um único arquivo CSV.

    Args:
        path: caminho do CSV.
        column_map: mapeia nomes do CSV -> canônico
            (chaves: time, open, high, low, close, volume).
        time_format: formato strptime; ``None`` tenta epoch e depois ISO.
        time_unit_epoch: 's' ou 'ms' quando o tempo for epoch numérico.
    """

    def __init__(
        self,
        path: str,
        *,
        column_map: dict[str, str] | None = None,
        time_format: str | None = None,
        time_unit_epoch: str = "s",
    ) -> None:
        self.path = path
        self.column_map = column_map or {
            "time": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        }
        self.time_format = time_format
        self.time_unit_epoch = time_unit_epoch

    def _parse_time(self, col: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        if dtype.is_numeric():
            return pl.from_epoch(col, time_unit=self.time_unit_epoch)
        if self.time_format:
            return col.str.strptime(pl.Datetime, self.time_format)
        return col.str.to_datetime(strict=False)

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

        raw = pl.read_csv(self.path)
        inv = {v: k for k, v in self.column_map.items()}  # csv_name -> canônico
        raw = raw.rename({k: v for k, v in inv.items() if k in raw.columns})

        time_dtype = raw.schema["time"]
        df = raw.with_columns(self._parse_time(pl.col("time"), time_dtype).alias("time"))

        # Normaliza para UTC us.
        if df.schema["time"].time_zone is None:
            df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        else:
            df = df.with_columns(pl.col("time").dt.convert_time_zone("UTC"))
        df = df.with_columns(
            pl.col("time").dt.cast_time_unit("us"),
            pl.col(["open", "high", "low", "close", "volume"]).cast(pl.Float64),
        )

        df = (
            df.filter((pl.col("time") >= start) & (pl.col("time") < end))
            .sort("time")
        )
        return validate_bars(df, symbol=symbol)
