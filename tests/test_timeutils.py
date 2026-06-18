"""Testes da conversão tempo-do-servidor → UTC (o gotcha de fuso do MT5)."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from innova_ea.data.timeutils import server_epoch_to_utc


def _server_epoch(year, month, day, hour):
    """Epoch que o MT5 retornaria para um relógio de parede de servidor dado.

    O MT5 codifica o relógio do servidor como se fosse UTC; reproduzimos isso
    tomando o timestamp do instante interpretado em UTC.
    """
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp())


def test_fixed_offset_conversion():
    # Servidor EET (+2) sem DST: relógio 10:00 → UTC 08:00.
    df = pl.DataFrame({"time": [_server_epoch(2021, 1, 4, 10)]})
    utc = server_epoch_to_utc(df, server_utc_offset=2)
    assert utc.dtype.time_zone == "UTC"
    assert utc[0] == datetime(2021, 1, 4, 8, tzinfo=timezone.utc)


def test_iana_timezone_handles_dst():
    # Europe/Athens: inverno = +2 (EET), verão = +3 (EEST).
    winter = pl.DataFrame({"time": [_server_epoch(2021, 1, 4, 10)]})
    summer = pl.DataFrame({"time": [_server_epoch(2021, 7, 5, 10)]})

    utc_w = server_epoch_to_utc(winter, server_timezone="Europe/Athens")
    utc_s = server_epoch_to_utc(summer, server_timezone="Europe/Athens")

    assert utc_w[0] == datetime(2021, 1, 4, 8, tzinfo=timezone.utc)   # -2h
    assert utc_s[0] == datetime(2021, 7, 5, 7, tzinfo=timezone.utc)   # -3h (DST)


def test_requires_tz_or_offset():
    df = pl.DataFrame({"time": [_server_epoch(2021, 1, 4, 10)]})
    try:
        server_epoch_to_utc(df, server_timezone=None, server_utc_offset=None)
        assert False, "deveria exigir tz ou offset"
    except ValueError:
        pass
