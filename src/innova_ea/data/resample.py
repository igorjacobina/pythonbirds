"""Agregação determinística de timeframe (ex.: M1 → H1) em Polars.

Princípio: armazena-se o timeframe mais granular e derivam-se os superiores.
A agregação é *left-labeled* (o timestamp da barra é o início do bucket) e
*left-closed* — consistente com o schema canônico onde `time` = abertura.
"""
from __future__ import annotations

import polars as pl

from innova_ea.core.bars import validate_bars
from innova_ea.core.enums import Timeframe


def resample(df: pl.DataFrame, target: Timeframe) -> pl.DataFrame:
    """Agrega barras para um timeframe superior.

    Args:
        df: barras no schema canônico (validadas), de um timeframe menor.
        target: timeframe de destino (deve ser >= ao de origem).

    Returns:
        Novo DataFrame de barras agregadas, já validado.
    """
    if df.height == 0:
        return df

    out = (
        df.lazy()
        .sort("time")
        .group_by_dynamic(
            "time",
            every=target.polars_every,
            closed="left",
            label="left",
        )
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
        .collect()
    )
    return validate_bars(out)
