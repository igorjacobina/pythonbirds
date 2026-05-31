"""Schema canônico de barras OHLCV e validações de integridade.

Toda a plataforma fala uma única linguagem de "barras": um Polars DataFrame com
colunas e dtypes fixos, timestamps em UTC e ordem cronológica estrita. Validar
isso na fronteira de cada módulo evita bugs silenciosos (gaps, duplicatas,
look-ahead) que corrompem backtests.
"""
from __future__ import annotations

import polars as pl

# Schema canônico. `time` é o instante de ABERTURA da vela, em UTC.
BAR_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

REQUIRED_COLUMNS = tuple(BAR_SCHEMA.keys())


class BarValidationError(ValueError):
    """Levantado quando um DataFrame de barras viola o contrato canônico."""


def validate_bars(df: pl.DataFrame, *, symbol: str | None = None) -> pl.DataFrame:
    """Valida e normaliza um DataFrame de barras.

    Verifica colunas, ordem temporal estritamente crescente, ausência de
    duplicatas, nulos e coerência OHLC (high >= max(open,close,low), etc.).

    Returns:
        O mesmo DataFrame, com colunas reordenadas para o schema canônico.

    Raises:
        BarValidationError: em qualquer violação do contrato.
    """
    ctx = f" [{symbol}]" if symbol else ""

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BarValidationError(f"colunas ausentes{ctx}: {missing}")

    df = df.select(REQUIRED_COLUMNS)

    if df.height == 0:
        return df  # vazio é válido (range sem dados)

    if df["time"].dtype != BAR_SCHEMA["time"]:
        raise BarValidationError(
            f"coluna 'time' deve ser {BAR_SCHEMA['time']} (UTC){ctx}, "
            f"recebido {df['time'].dtype}"
        )

    null_counts = df.null_count().row(0)
    if any(null_counts):
        cols = dict(zip(df.columns, null_counts))
        raise BarValidationError(f"valores nulos encontrados{ctx}: "
                                 f"{ {k: v for k, v in cols.items() if v} }")

    times = df["time"]
    if not times.is_sorted():
        raise BarValidationError(f"timestamps não estão em ordem crescente{ctx}")
    if times.n_unique() != times.len():
        raise BarValidationError(f"timestamps duplicados encontrados{ctx}")

    # Coerência OHLC.
    bad = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    )
    if bad.height > 0:
        raise BarValidationError(
            f"{bad.height} barras com OHLC inconsistente{ctx} "
            f"(ex. high<low). Primeira em {bad['time'][0]}"
        )

    return df


def empty_bars() -> pl.DataFrame:
    """Cria um DataFrame de barras vazio com o schema canônico."""
    return pl.DataFrame(schema=BAR_SCHEMA)
