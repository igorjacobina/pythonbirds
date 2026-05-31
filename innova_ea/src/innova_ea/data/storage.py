"""Storage colunar particionado em Parquet.

Layout em disco (compatível com S3/GCS via pyarrow):

    root/symbol=EURUSD/timeframe=M1/year=2015/data.parquet

Particionar por symbol/timeframe/ano permite *predicate pushdown* (ler só os
anos pedidos) e upserts idempotentes por range. O schema é o canônico de barras.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from innova_ea.core.bars import empty_bars, validate_bars
from innova_ea.core.enums import Timeframe


class ParquetStore:
    """Persistência de barras em árvore de diretórios particionada."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _partition_dir(self, symbol: str, tf: Timeframe, year: int) -> Path:
        return (
            self.root
            / f"symbol={symbol.upper()}"
            / f"timeframe={tf.value}"
            / f"year={year}"
        )

    def write(self, symbol: str, tf: Timeframe, df: pl.DataFrame) -> int:
        """Grava barras, fazendo upsert por ano (idempotente).

        Para cada ano presente em ``df``, mescla com o que já existe no disco,
        deduplica por ``time`` (mantendo o dado novo) e reescreve a partição.

        Returns:
            Número total de barras gravadas (após dedupe).
        """
        df = validate_bars(df, symbol=symbol)
        if df.height == 0:
            return 0

        df = df.with_columns(pl.col("time").dt.year().alias("_year"))
        written = 0
        for (year,), part in df.group_by(["_year"], maintain_order=True):
            part = part.drop("_year")
            pdir = self._partition_dir(symbol, tf, int(year))
            pdir.mkdir(parents=True, exist_ok=True)
            fpath = pdir / "data.parquet"

            if fpath.exists():
                existing = pl.read_parquet(fpath)
                part = (
                    pl.concat([existing, part])
                    .unique(subset=["time"], keep="last")
                    .sort("time")
                )
            part = validate_bars(part, symbol=symbol)
            part.write_parquet(fpath, compression="zstd")
            written += part.height
        return written

    def read(
        self,
        symbol: str,
        tf: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Lê barras de ``[start, end)``. Sem limites, lê todo o histórico."""
        base = self.root / f"symbol={symbol.upper()}" / f"timeframe={tf.value}"
        if not base.exists():
            return empty_bars()

        lf = pl.scan_parquet(base / "year=*" / "data.parquet")
        if start is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            lf = lf.filter(pl.col("time") >= start)
        if end is not None:
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            lf = lf.filter(pl.col("time") < end)

        df = lf.sort("time").collect()
        return validate_bars(df, symbol=symbol)

    def available_symbols(self) -> list[str]:
        return sorted(
            p.name.split("=", 1)[1]
            for p in self.root.glob("symbol=*")
            if p.is_dir()
        )
