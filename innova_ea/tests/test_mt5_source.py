"""Testes do conector MT5 com um módulo MetaTrader5 FALSO (rodável em Linux).

Valida a conversão de fuso do servidor para UTC, a paginação por chunks e a
deduplicação — sem depender de um terminal MT5 real.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.mt5_source import MT5Source

_RATE_DTYPE = np.dtype([
    ("time", "i8"), ("open", "f8"), ("high", "f8"),
    ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
])


class FakeMT5:
    """Imitação mínima da API MetaTrader5 para testes.

    Gera barras horárias em ``[from, to]`` no TEMPO DO SERVIDOR: epoch = UTC +
    ``offset_hours`` (como o MT5 real codifica o relógio de parede do servidor).
    """

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769
    TIMEFRAME_MN1 = 49153

    def __init__(self, offset_hours: int = 2) -> None:
        self.offset_hours = offset_hours
        self.initialized = False

    def initialize(self, **kwargs) -> bool:
        self.initialized = True
        return True

    def last_error(self):
        return (0, "ok")

    def symbol_select(self, symbol, enable) -> bool:
        return True

    def shutdown(self) -> None:
        self.initialized = False

    def copy_rates_range(self, symbol, timeframe, dfrom, dto):
        rows = []
        t = dfrom
        while t <= dto:
            epoch = int(t.timestamp()) + self.offset_hours * 3600
            rows.append((epoch, 1.10, 1.11, 1.09, 1.105, 100))
            t += timedelta(hours=1)
        return np.array(rows, dtype=_RATE_DTYPE)


def _utc(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_fetch_converts_server_time_to_utc_fixed_offset():
    src = MT5Source(mt5_module=FakeMT5(offset_hours=2), server_utc_offset=2)
    bars = src.fetch("EURUSD", Timeframe.M1, _utc(2021, 1, 4), _utc(2021, 1, 5))

    assert bars.height == 24                       # 00:00..23:00
    assert bars["time"].dtype.time_zone == "UTC"
    assert bars["time"][0] == _utc(2021, 1, 4, 0)  # offset desfeito corretamente
    assert bars["time"][-1] == _utc(2021, 1, 4, 23)


def test_fetch_converts_with_iana_timezone_winter():
    # EET (+2) no inverno via fuso IANA — deve bater com o offset fixo.
    src = MT5Source(mt5_module=FakeMT5(offset_hours=2), server_timezone="Europe/Athens")
    bars = src.fetch("EURUSD", Timeframe.M1, _utc(2021, 1, 4), _utc(2021, 1, 5))
    assert bars["time"][0] == _utc(2021, 1, 4, 0)


def test_fetch_filters_range_and_is_sorted_unique():
    src = MT5Source(mt5_module=FakeMT5(offset_hours=0), server_utc_offset=0)
    bars = src.fetch("EURUSD", Timeframe.M1, _utc(2021, 1, 4), _utc(2021, 1, 5))
    times = bars["time"].to_list()
    assert times == sorted(times)
    assert len(set(times)) == len(times)               # sem duplicatas
    assert all(_utc(2021, 1, 4) <= t < _utc(2021, 1, 5) for t in times)


def test_fetch_chunks_multiple_months_dedup():
    # 3 meses, chunk de 1 mês → múltiplas chamadas com sobreposição (pad) deduplicadas.
    src = MT5Source(mt5_module=FakeMT5(offset_hours=0), server_utc_offset=0, chunk_months=1)
    bars = src.fetch("EURUSD", Timeframe.M1, _utc(2021, 1, 1), _utc(2021, 4, 1))
    times = bars["time"].to_list()
    assert len(set(times)) == len(times)               # dedupe entre chunks
    assert times == sorted(times)
    assert bars["time"][0] == _utc(2021, 1, 1, 0)


def test_connection_retries_then_raises():
    class AlwaysFail(FakeMT5):
        def initialize(self, **kwargs):
            return False

    src = MT5Source(mt5_module=AlwaysFail(), server_utc_offset=0,
                    max_retries=2, retry_wait=0.0)
    try:
        src.fetch("EURUSD", Timeframe.M1, _utc(2021, 1, 4), _utc(2021, 1, 5))
        assert False, "deveria falhar na conexão"
    except ConnectionError as e:
        assert "2 tentativas" in str(e)
