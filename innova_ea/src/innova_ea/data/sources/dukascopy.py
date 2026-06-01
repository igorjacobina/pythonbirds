"""Fonte de dados Dukascopy — histórico profundo (tick → M1) em UTC.

A Dukascopy publica dados históricos de tick gratuitamente em URLs determinísticas
(um arquivo ``.bi5`` por hora UTC), cobrindo FX, metais e vários índices CFD desde
~2003. Cada ``.bi5`` é LZMA-comprimido; descomprimido, é uma sequência de registros
de 20 bytes **big-endian**: ``(ms_desde_a_hora:uint32, ask:int32, bid:int32,
ask_vol:float32, bid_vol:float32)``. Os preços inteiros são escalados por um fator
por instrumento (ex. 1e5 para FX de 5 dígitos, 1e3 para JPY/metais).

Diferente do MT5, os timestamps da Dukascopy já são **UTC** — sem conversão de
fuso. A decodificação é pura e testável offline; o download é injetável.
"""
from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import polars as pl

from innova_ea.core.bars import empty_bars, validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.base import DataSource

_BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Registro de tick big-endian: ms(uint32), ask(int32), bid(int32), askv(f32), bidv(f32).
_TICK_DTYPE = np.dtype([
    ("ms", ">u4"), ("ask", ">i4"), ("bid", ">i4"), ("askv", ">f4"), ("bidv", ">f4"),
])
_RECORD_SIZE = 20

# Mapa símbolo (nosso) -> (código no datafeed Dukascopy, fator de escala de preço).
# FX e metais: confiáveis. Índices: melhor esforço — CONFIRME o código no datafeed
# antes de confiar (a simbologia de CFD da Dukascopy é peculiar).
DUKASCOPY_INSTRUMENTS: dict[str, tuple[str, float]] = {
    "EURUSD": ("EURUSD", 1e5),
    "GBPUSD": ("GBPUSD", 1e5),
    "USDJPY": ("USDJPY", 1e3),
    "AUDUSD": ("AUDUSD", 1e5),
    "USDCHF": ("USDCHF", 1e5),
    "USDCAD": ("USDCAD", 1e5),
    "XAUUSD": ("XAUUSD", 1e3),
    "XAGUSD": ("XAGUSD", 1e3),
    # Índices (verificar códigos no datafeed Dukascopy):
    "US30":  ("USA30IDXUSD", 1e3),
    "US100": ("USATECHIDXUSD", 1e3),
    "US500": ("USA500IDXUSD", 1e3),
    "DE40":  ("DEUIDXEUR", 1e3),
    "JP225": ("JPNIDXJPY", 1e3),
    "UK100": ("GBRIDXGBP", 1e3),
    "HK50":  ("HKGIDXHKD", 1e3),
}


def decode_bi5(raw: bytes) -> np.ndarray:
    """Descomprime (LZMA) e decodifica um ``.bi5`` em array estruturado de ticks."""
    if not raw:
        return np.empty(0, dtype=_TICK_DTYPE)
    try:
        data = lzma.decompress(raw, format=lzma.FORMAT_AUTO)
    except lzma.LZMAError:
        data = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    n = len(data) // _RECORD_SIZE
    if n == 0:
        return np.empty(0, dtype=_TICK_DTYPE)
    return np.frombuffer(data[: n * _RECORD_SIZE], dtype=_TICK_DTYPE)


def ticks_to_frame(ticks: np.ndarray, hour_start: datetime, price_scale: float) -> pl.DataFrame:
    """Converte ticks de UMA hora em DataFrame (time UTC, mid, volume)."""
    if ticks.shape[0] == 0:
        return pl.DataFrame(schema={
            "time": pl.Datetime("us", "UTC"), "mid": pl.Float64, "volume": pl.Float64,
        })
    if hour_start.tzinfo is None:
        hour_start = hour_start.replace(tzinfo=timezone.utc)
    base_us = int(hour_start.timestamp() * 1_000_000)
    time_us = base_us + ticks["ms"].astype(np.int64) * 1000
    bid = ticks["bid"].astype(np.float64) / price_scale
    ask = ticks["ask"].astype(np.float64) / price_scale
    mid = (bid + ask) / 2.0
    vol = (ticks["askv"].astype(np.float64) + ticks["bidv"].astype(np.float64))
    return pl.DataFrame({"time": time_us, "mid": mid, "volume": vol}).with_columns(
        pl.from_epoch(pl.col("time"), time_unit="us").dt.replace_time_zone("UTC").alias("time")
    )


def _default_downloader(url: str) -> bytes | None:
    """Baixa um ``.bi5`` (None em 404/erro). Substituível em testes."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - rede
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError:  # pragma: no cover - rede
        return None


class DukascopySource(DataSource):
    """Histórico Dukascopy (tick → barras) para o ``DataPipeline``.

    Args:
        instruments: override do mapa símbolo → (código, escala).
        downloader: função ``url -> bytes|None`` (injetável p/ testes).
        max_workers: downloads horários concorrentes por requisição.
    """

    def __init__(
        self,
        *,
        instruments: dict[str, tuple[str, float]] | None = None,
        downloader: Callable[[str], bytes | None] | None = None,
        max_workers: int = 8,
    ) -> None:
        self.instruments = instruments or DUKASCOPY_INSTRUMENTS
        self.downloader = downloader or _default_downloader
        self.max_workers = max_workers

    def _url(self, code: str, dt: datetime) -> str:
        # Mês é 0-indexado na URL da Dukascopy (janeiro = 00).
        return f"{_BASE_URL}/{code}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"

    def _hours(self, start: datetime, end: datetime):
        cur = start.replace(minute=0, second=0, microsecond=0)
        while cur < end:
            yield cur
            cur += timedelta(hours=1)

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        if symbol.upper() not in self.instruments:
            raise ValueError(f"símbolo '{symbol}' sem mapeamento Dukascopy")
        code, scale = self.instruments[symbol.upper()]
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        hours = list(self._hours(start, end))
        frames: list[pl.DataFrame] = []
        for hour in hours:
            raw = self.downloader(self._url(code, hour))
            if not raw:
                continue                      # fim de semana/feriado/sem dados
            ticks = decode_bi5(raw)
            if ticks.shape[0]:
                frames.append(ticks_to_frame(ticks, hour, scale))

        if not frames:
            return empty_bars()

        ticks_df = pl.concat(frames).sort("time")
        bars = (
            ticks_df.group_by_dynamic("time", every=timeframe.polars_every, closed="left", label="left")
            .agg(
                pl.col("mid").first().alias("open"),
                pl.col("mid").max().alias("high"),
                pl.col("mid").min().alias("low"),
                pl.col("mid").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            )
            .filter((pl.col("time") >= start) & (pl.col("time") < end))
            .sort("time")
        )
        return validate_bars(bars, symbol=symbol)


def pack_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    """Empacota+comprime ticks no formato ``.bi5`` (usado em testes/ferramentas)."""
    raw = b"".join(struct.pack(">Iiiff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)
