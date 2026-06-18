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
        vol_by_hour: dict[int, float] | None = None,
        drift_by_hour: dict[int, float] | None = None,
    ) -> None:
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.seed = seed
        # Sazonalidade intradiária OPCIONAL (ground-truth para validar a Fase 2):
        # multiplicador de volatilidade e deslocamento de drift por hora UTC.
        self.vol_by_hour = vol_by_hour
        self.drift_by_hour = drift_by_hour

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

        times = [start + i * step for i in range(n)]
        hours = np.array([t.hour for t in times], dtype=np.int64)

        # Aplica sazonalidade intradiária por hora UTC (se configurada).
        sigma_arr = np.full(n, sigma, dtype=np.float64)
        mu_arr = np.full(n, mu, dtype=np.float64)
        if self.vol_by_hour:
            mult = np.array([self.vol_by_hour.get(int(h), 1.0) for h in hours])
            sigma_arr = sigma * mult
        if self.drift_by_hour:
            mu_arr = mu + np.array(
                [self.drift_by_hour.get(int(h), 0.0) * dt for h in hours]
            )

        log_ret = mu_arr + sigma_arr * rng.standard_normal(n)
        close = self.start_price * np.exp(np.cumsum(log_ret))
        open_ = np.empty(n)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        # Wicks: amplitude intrabar proporcional à sigma (sazonal) da barra.
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        wick = np.abs(rng.standard_normal(n)) * sigma_arr * close
        high = body_hi + wick * rng.uniform(0.0, 1.0, n)
        low = body_lo - wick * rng.uniform(0.0, 1.0, n)
        volume = rng.uniform(100, 1000, n)

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
