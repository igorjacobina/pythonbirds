"""Fonte sintética de barras (GBM + microestrutura intrabar).

Serve para testes, CI e demos sem dependência de rede ou de MT5. O processo é um
movimento browniano geométrico com volatilidade configurável; o high/low de cada
vela é construído de forma coerente (high >= max(open,close), etc.).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from innova_ea.core.bars import validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.base import DataSource


class SyntheticSource(DataSource):
    """Gera barras pseudo-realistas de forma determinística (via seed)."""

    def __init__(
        self,
        *,
        start_price: float = 1.10,
        annual_vol: float = 0.08,
        annual_drift: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.seed = seed

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

        step = timedelta(minutes=timeframe.minutes)
        n = max(0, int((end - start) / step))
        if n == 0:
            return validate_bars(pl.DataFrame(), symbol=symbol)

        # Seed estável por símbolo+timeframe para reprodutibilidade.
        rng = np.random.default_rng(self.seed + hash((symbol, timeframe.value)) % 10_000)

        minutes_per_year = 365.25 * 24 * 60
        dt = timeframe.minutes / minutes_per_year
        mu = (self.annual_drift - 0.5 * self.annual_vol**2) * dt
        sigma = self.annual_vol * np.sqrt(dt)

        log_ret = mu + sigma * rng.standard_normal(n)
        close = self.start_price * np.exp(np.cumsum(log_ret))
        open_ = np.empty(n)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        # Wicks: amplitude intrabar proporcional a sigma.
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        wick = np.abs(rng.standard_normal(n)) * sigma * close
        high = body_hi + wick * rng.uniform(0.0, 1.0, n)
        low = body_lo - wick * rng.uniform(0.0, 1.0, n)
        volume = rng.uniform(100, 1000, n)

        times = [start + i * step for i in range(n)]
        df = pl.DataFrame(
            {
                "time": times,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).with_columns(
            pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
        )
        return validate_bars(df, symbol=symbol)
