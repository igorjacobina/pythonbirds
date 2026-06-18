"""Codificação temporal — sazonalidade intradiária sem fronteiras artificiais.

Hora do dia e dia da semana são cíclicos: 23h está perto de 0h. Encoding linear
cria um "salto" falso na virada. O encoding seno/cosseno preserva a continuidade,
permitindo que o modelo aprenda padrões de horário (ex. expansão na abertura de
Londres) sem descontinuidades.
"""
from __future__ import annotations

import math

import polars as pl

from innova_ea.features.base import Feature

_TWO_PI = 2.0 * math.pi


class TimeOfDayCyclical(Feature):
    """Encoding cíclico (sin/cos) do minuto-do-dia e do dia-da-semana."""

    @property
    def names(self) -> list[str]:
        return ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        minute_of_day = pl.col("time").dt.hour() * 60 + pl.col("time").dt.minute()
        frac_day = minute_of_day / 1440.0
        dow = pl.col("time").dt.weekday()  # 1=segunda ... 7=domingo
        frac_week = (dow - 1) / 7.0
        out = bars.select(
            (frac_day * _TWO_PI).sin().alias("tod_sin"),
            (frac_day * _TWO_PI).cos().alias("tod_cos"),
            (frac_week * _TWO_PI).sin().alias("dow_sin"),
            (frac_week * _TWO_PI).cos().alias("dow_cos"),
        )
        return self._check(out, bars.height)


class VolatilityProfile:
    """Perfil de volatilidade intradiária — sazonalidade média por horário.

    Aprende, em dados de TREINO, a volatilidade realizada média de cada minuto-
    do-dia. Aplicado a novas barras, expõe (a) a vol esperada para aquele horário
    e (b) o desvio da vol corrente frente ao esperado. Como é ajustado apenas com
    dados passados (fit/transform explícitos), não vaza informação do futuro.
    """

    def __init__(self, vol_window: int = 12) -> None:
        self.vol_window = vol_window
        self._profile: pl.DataFrame | None = None

    def _minute_of_day(self) -> pl.Expr:
        return pl.col("time").dt.hour() * 60 + pl.col("time").dt.minute()

    def _realized(self) -> pl.Expr:
        r = pl.col("close").log() - pl.col("close").shift(1).log()
        return r.rolling_std(self.vol_window)

    def fit(self, train_bars: pl.DataFrame) -> "VolatilityProfile":
        self._profile = (
            train_bars.with_columns(
                self._minute_of_day().alias("_mod"),
                self._realized().alias("_rv"),
            )
            .group_by("_mod")
            .agg(pl.col("_rv").mean().alias("expected_vol"))
            .sort("_mod")
        )
        return self

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        if self._profile is None:
            raise RuntimeError("VolatilityProfile deve ser ajustado (fit) antes de transform")
        df = bars.with_columns(
            self._minute_of_day().alias("_mod"),
            self._realized().alias("_rv_now"),
        ).join(self._profile, on="_mod", how="left")
        out = df.select(
            pl.col("expected_vol").alias("tod_expected_vol"),
            (pl.col("_rv_now") / pl.col("expected_vol")).alias("tod_vol_ratio"),
        )
        return out
