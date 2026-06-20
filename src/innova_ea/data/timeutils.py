"""Conversão impecável do tempo do servidor da corretora para UTC.

O MetaTrader 5 retorna o campo ``time`` das barras como *epoch* — mas esse epoch
codifica o **relógio de parede do servidor** (tipicamente EET, UTC+2/+3 com DST),
não o instante UTC real. Tratar esse epoch diretamente como UTC é o erro de fuso
mais comum (e silencioso) em backtests de Forex.

Conversão correta:
  1. ``from_epoch`` → datetime ingênuo = relógio de parede do servidor;
  2. atribui o fuso do servidor (``replace_time_zone``);
  3. converte para UTC (``convert_time_zone``), respeitando o DST.

Quando o servidor usa offset FIXO (sem DST), informe ``server_utc_offset`` para
evitar qualquer ambiguidade de horário de verão.
"""
from __future__ import annotations

import polars as pl


def server_epoch_to_utc(
    df: pl.DataFrame,
    *,
    epoch_col: str = "time",
    server_timezone: str | None = "Europe/Athens",
    server_utc_offset: float | None = None,
    ambiguous: str = "earliest",
) -> pl.Series:
    """Converte uma coluna de epoch (tempo do servidor) para ``Datetime[us, UTC]``.

    Args:
        df: DataFrame contendo ``epoch_col`` (segundos epoch, inteiro/float).
        server_timezone: fuso IANA do servidor (ex. ``"Europe/Athens"`` para EET
            com DST). Ignorado se ``server_utc_offset`` for fornecido.
        server_utc_offset: offset FIXO em horas (ex. 2 ou 3). Use quando o broker
            não aplica horário de verão — elimina ambiguidade de DST.
        ambiguous: política para horários ambíguos na virada de DST
            (``"earliest"``/``"latest"``/``"raise"``/``"null"``).

    Returns:
        Série ``time`` em UTC, microssegundos.
    """
    if server_utc_offset is None and server_timezone is None:
        raise ValueError("informe server_timezone ou server_utc_offset")

    # Relógio de parede do servidor como datetime ingênuo (interpretado como UTC).
    naive = pl.from_epoch(pl.col(epoch_col).cast(pl.Int64), time_unit="s")

    if server_utc_offset is not None:
        # Offset fixo: instante UTC = relógio do servidor - offset.
        utc = (naive - pl.duration(minutes=int(round(server_utc_offset * 60)))).dt.replace_time_zone("UTC")
    else:
        utc = (
            naive.dt.replace_time_zone(server_timezone, ambiguous=ambiguous)
            .dt.convert_time_zone("UTC")
        )

    return df.select(utc.dt.cast_time_unit("us").alias("time")).to_series()
