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


def _make_downloader(
    timeout: float = 30.0, max_retries: int = 4, retry_wait: float = 1.5
) -> Callable[[str], bytes | None]:
    """Cria um downloader RESILIENTE de ``.bi5``.

    Trata 404 (→ None, sem dados), e re-tenta com backoff em timeouts/erros de
    rede (``TimeoutError``/``URLError``/``OSError``). Após esgotar as tentativas,
    retorna None (pula a hora) — assim uma hora lenta NÃO derruba a ingestão; o
    ``ParquetStore`` é idempotente e completa o que faltar numa nova execução.
    """
    import time
    import urllib.error
    import urllib.request

    headers = {"User-Agent": "Mozilla/5.0 (innova_ea data ingest)"}

    def download(url: str) -> bytes | None:
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None            # fim de semana/feriado/sem dados
            except (urllib.error.URLError, TimeoutError, OSError):
                pass                        # transitório → re-tenta
            if attempt < max_retries - 1:
                time.sleep(retry_wait * (attempt + 1))
        return None                         # esgotou tentativas → pula a hora

    return download


# Downloader padrão (compatibilidade); use _make_downloader p/ parametrizar.
def _default_downloader(url: str) -> bytes | None:  # pragma: no cover - rede
    return _make_downloader()(url)


class DukascopySource(DataSource):
    """Histórico Dukascopy (tick → barras) para o ``DataPipeline``.

    Args:
        instruments: override do mapa símbolo → (código, escala).
        downloader: função ``url -> bytes|None`` (injetável p/ testes); se ``None``,
            usa um downloader resiliente com ``timeout``/``max_retries``.
        max_workers: downloads horários CONCORRENTES por requisição.
        timeout/max_retries/retry_wait: política do downloader padrão.
    """

    def __init__(
        self,
        *,
        instruments: dict[str, tuple[str, float]] | None = None,
        downloader: Callable[[str], bytes | None] | None = None,
        max_workers: int = 8,
        timeout: float = 30.0,
        max_retries: int = 4,
        retry_wait: float = 1.5,
    ) -> None:
        self.instruments = instruments or DUKASCOPY_INSTRUMENTS
        self.downloader = downloader or _make_downloader(timeout, max_retries, retry_wait)
        self.max_workers = max_workers

    def _url(self, code: str, dt: datetime) -> str:
        # Mês é 0-indexado na URL da Dukascopy (janeiro = 00).
        return f"{_BASE_URL}/{code}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"

    def _hours(self, start: datetime, end: datetime):
        cur = start.replace(minute=0, second=0, microsecond=0)
        while cur < end:
            yield cur
            cur += timedelta(hours=1)

    @staticmethod
    def _aggregate(ticks_frame: pl.DataFrame, timeframe: Timeframe) -> pl.DataFrame:
        return ticks_frame.group_by_dynamic(
            "time", every=timeframe.polars_every, closed="left", label="left"
        ).agg(
            pl.col("mid").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        from concurrent.futures import ThreadPoolExecutor

        if symbol.upper() not in self.instruments:
            raise ValueError(f"símbolo '{symbol}' sem mapeamento Dukascopy")
        code, scale = self.instruments[symbol.upper()]
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        hours = list(self._hours(start, end))
        # Downloads CONCORRENTES (IO-bound); uma hora lenta não bloqueia o resto.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            raws = list(pool.map(lambda h: self.downloader(self._url(code, h)), hours))

        # Agrega CADA hora a M1 já (memória-segura: não acumula todos os ticks de
        # anos). Para M1, os buckets de minuto não cruzam a fronteira da hora.
        bars_list: list[pl.DataFrame] = []
        for hour, raw in zip(hours, raws):
            if not raw:
                continue                      # fim de semana/feriado/sem dados
            ticks = decode_bi5(raw)
            if ticks.shape[0] == 0:
                continue
            frame = ticks_to_frame(ticks, hour, scale)
            bars_list.append(self._aggregate(frame, timeframe))

        if not bars_list:
            return empty_bars()

        bars = (
            pl.concat(bars_list)
            .filter((pl.col("time") >= start) & (pl.col("time") < end))
            .unique(subset=["time"], keep="last")
            .sort("time")
        )
        return validate_bars(bars, symbol=symbol)


def pack_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    """Empacota+comprime ticks no formato ``.bi5`` (usado em testes/ferramentas)."""
    raw = b"".join(struct.pack(">Iiiff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)
