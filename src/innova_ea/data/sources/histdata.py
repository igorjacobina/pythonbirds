"""Fonte HistData.com — M1 grátis em CSV, reutilizando o ``CsvSource``.

O HistData publica M1 em arquivos mensais "Generic ASCII"
(``DAT_ASCII_{CODE}_M1_{YYYYMM}.csv``), formato ``YYYYMMDD HHMMSS;O;H;L;C;V``,
separador ``;``, sem cabeçalho, timestamps em **EST (UTC-5, sem horário de verão)**.

Esta fonte localiza os arquivos mensais de um diretório e delega a leitura ao
``CsvSource`` (um por arquivo), concatenando e convertendo para UTC.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import polars as pl

from innova_ea.core.bars import empty_bars, validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.base import DataSource
from innova_ea.data.sources.csv_source import CsvSource

# Símbolo (nosso) -> código HistData. US30 (Dow) NÃO existe no HistData → use Dukascopy.
HISTDATA_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCHF": "USDCHF", "USDCAD": "USDCAD",
    "XAUUSD": "XAUUSD", "XAGUSD": "XAGUSD",
    "US100": "NSXUSD",   # Nasdaq 100
    "US500": "SPXUSD",   # S&P 500
    "DE40":  "GRXEUR",   # DAX
    "JP225": "JPXJPY",   # Nikkei 225
    "UK100": "UKXGBP",   # FTSE 100
    "HK50":  "HKXHKD",   # Hang Seng
}

_HISTDATA_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


class HistDataSource(DataSource):
    """Lê M1 do HistData a partir dos arquivos mensais em ``directory``.

    Args:
        directory: pasta com os CSVs ``DAT_ASCII_{CODE}_M1_*.csv``.
        symbols: override do mapa símbolo → código HistData.
        est_offset: offset fixo da origem (EST = -5; HistData não usa DST).
    """

    def __init__(
        self,
        directory: str,
        *,
        symbols: dict[str, str] | None = None,
        est_offset: float = -5.0,
    ) -> None:
        self.directory = directory
        self.symbols = symbols or HISTDATA_SYMBOLS
        self.est_offset = est_offset

    def _files(self, code: str) -> list[str]:
        pattern = os.path.join(self.directory, f"*{code}_M1_*.csv")
        return sorted(glob.glob(pattern))

    def _csv(self, path: str) -> CsvSource:
        return CsvSource(
            path,
            new_columns=_HISTDATA_COLUMNS,
            column_map={c: c for c in _HISTDATA_COLUMNS},
            time_format="%Y%m%d %H%M%S",
            separator=";",
            has_header=False,
            source_utc_offset=self.est_offset,
        )

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        if symbol.upper() not in self.symbols:
            raise ValueError(f"símbolo '{symbol}' sem mapeamento HistData "
                             "(ex. US30/Dow não existe no HistData — use Dukascopy)")
        files = self._files(self.symbols[symbol.upper()])
        if not files:
            return empty_bars()
        frames = [self._csv(f).fetch(symbol, timeframe, start, end) for f in files]
        frames = [f for f in frames if f.height]
        if not frames:
            return empty_bars()
        combined = pl.concat(frames).unique(subset=["time"], keep="last").sort("time")
        return validate_bars(combined, symbol=symbol)
