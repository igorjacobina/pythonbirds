"""Estimadores de volatilidade de alta eficiência (OHLC) + regime de volatilidade.

O desvio-padrão de retornos de fechamento descarta a informação contida nas
máximas/mínimas de cada barra. Os estimadores abaixo (Parkinson, Garman-Klass,
Rogers-Satchell, Yang-Zhang) usam todo o OHLC e são *várias vezes mais
eficientes* — convergem com menos dados. Todos são janelas trailing (causais) e
retornam a volatilidade estimada por barra (não anualizada).

Referências: Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991),
Yang & Zhang (2000).
"""
from __future__ import annotations

import math

import polars as pl

from innova_ea.features.base import Feature

_LN2 = math.log(2.0)


def _hl(expr_h: str = "high", expr_l: str = "low") -> pl.Expr:
    return (pl.col(expr_h) / pl.col(expr_l)).log()


class Parkinson(Feature):
    """Volatilidade de Parkinson: usa apenas o range (high/low) de cada barra."""

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        self.window = window
        self._name = name or f"vol_parkinson_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        u2 = _hl() ** 2
        var = u2.rolling_mean(self.window) / (4.0 * _LN2)
        return self._check(bars.select(var.sqrt().alias(self._name)), bars.height)


class GarmanKlass(Feature):
    """Volatilidade de Garman-Klass: combina range e corpo (open/close)."""

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        self.window = window
        self._name = name or f"vol_gk_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        hl2 = _hl() ** 2
        co2 = (pl.col("close") / pl.col("open")).log() ** 2
        per_bar = 0.5 * hl2 - (2.0 * _LN2 - 1.0) * co2
        var = per_bar.rolling_mean(self.window)
        return self._check(bars.select(var.sqrt().alias(self._name)), bars.height)


class RogersSatchell(Feature):
    """Volatilidade de Rogers-Satchell: robusta a drift (tendência)."""

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        self.window = window
        self._name = name or f"vol_rs_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        ho = (pl.col("high") / pl.col("open")).log()
        hc = (pl.col("high") / pl.col("close")).log()
        lo = (pl.col("low") / pl.col("open")).log()
        lc = (pl.col("low") / pl.col("close")).log()
        per_bar = hc * ho + lc * lo
        var = per_bar.rolling_mean(self.window)
        return self._check(bars.select(var.sqrt().alias(self._name)), bars.height)


class YangZhang(Feature):
    """Volatilidade de Yang-Zhang: combina overnight + open-close + Rogers-Satchell.

    É o estimador de menor variância e o único independente de drift e de gaps de
    abertura — referência para volatilidade realizada institucional.
    """

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        if window < 2:
            raise ValueError("window deve ser >= 2")
        self.window = window
        self._name = name or f"vol_yz_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        n = self.window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        overnight = (pl.col("open") / pl.col("close").shift(1)).log()
        open_close = (pl.col("close") / pl.col("open")).log()

        var_on = overnight.rolling_var(n)
        var_oc = open_close.rolling_var(n)

        ho = (pl.col("high") / pl.col("open")).log()
        hc = (pl.col("high") / pl.col("close")).log()
        lo = (pl.col("low") / pl.col("open")).log()
        lc = (pl.col("low") / pl.col("close")).log()
        var_rs = (hc * ho + lc * lo).rolling_mean(n)

        var_yz = var_on + k * var_oc + (1.0 - k) * var_rs
        return self._check(bars.select(var_yz.sqrt().alias(self._name)), bars.height)


class RealizedVol(Feature):
    """Volatilidade realizada clássica: desvio-padrão móvel dos log-retornos."""

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        self.window = window
        self._name = name or f"vol_rv_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        r = pl.col("close").log() - pl.col("close").shift(1).log()
        return self._check(
            bars.select(r.rolling_std(self.window).alias(self._name)), bars.height
        )


class VolatilityRegime(Feature):
    """Razão vol curta / vol longa — detecta expansão (>1) ou contração (<1)."""

    def __init__(self, short: int = 10, long: int = 100, name: str | None = None) -> None:
        if short >= long:
            raise ValueError("short deve ser < long")
        self.short = short
        self.long = long
        self._name = name or f"vol_regime_{short}_{long}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        r = pl.col("close").log() - pl.col("close").shift(1).log()
        ratio = r.rolling_std(self.short) / r.rolling_std(self.long)
        return self._check(bars.select(ratio.alias(self._name)), bars.height)
