"""Testes do calendário de negociação Forex."""
from __future__ import annotations

from datetime import date, datetime, timezone

from innova_ea.core.enums import Timeframe
from innova_ea.data.calendar import ForexCalendar, common_forex_holidays


def _utc(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_is_trading_scalar_rules():
    cal = ForexCalendar()  # abre dom / fecha sex às 17:00 NY (DST-aware)
    assert cal.is_trading(_utc(2021, 1, 4, 12)) is True    # segunda
    assert cal.is_trading(_utc(2021, 1, 9, 12)) is False   # sábado
    assert cal.is_trading(_utc(2021, 1, 8, 23)) is False   # sexta após fechamento
    assert cal.is_trading(_utc(2021, 1, 8, 12)) is True     # sexta antes do fechamento
    assert cal.is_trading(_utc(2021, 1, 3, 21)) is False    # domingo antes da abertura
    assert cal.is_trading(_utc(2021, 1, 3, 23)) is True     # domingo após a abertura


def test_dst_aware_weekly_boundary():
    cal = ForexCalendar()
    # Verão (EDT, UTC-4): abertura de domingo é às 21:00 UTC (= 17:00 NY).
    assert cal.is_trading(_utc(2021, 7, 4, 20)) is False    # 16:00 NY → fechado
    assert cal.is_trading(_utc(2021, 7, 4, 21)) is True     # 17:00 NY → abre
    # Inverno (EST, UTC-5): abertura de domingo é às 22:00 UTC.
    assert cal.is_trading(_utc(2021, 1, 3, 21)) is False    # 16:00 NY → fechado
    assert cal.is_trading(_utc(2021, 1, 3, 22)) is True     # 17:00 NY → abre


def test_is_trading_excludes_holidays():
    cal = ForexCalendar(holidays=frozenset({date(2021, 12, 25)}))
    assert cal.is_trading(_utc(2021, 12, 24, 12)) is True
    assert cal.is_trading(_utc(2021, 12, 25, 12)) is False  # Natal (sábado, de todo modo)


def test_trading_grid_excludes_weekends():
    cal = ForexCalendar()
    grid = cal.trading_grid(_utc(2021, 1, 4), _utc(2021, 1, 11), Timeframe.H1)
    weekdays = grid["time"].dt.weekday().to_list()
    assert 6 not in weekdays  # nenhum sábado (Polars: 6=sábado)
    assert grid.height > 0


def test_trading_grid_excludes_holiday_day():
    holidays = common_forex_holidays(range(2021, 2022))
    cal = ForexCalendar(holidays=holidays)
    grid = cal.trading_grid(_utc(2021, 1, 1), _utc(2021, 1, 5), Timeframe.H1)
    dates = set(grid["time"].dt.date().to_list())
    assert date(2021, 1, 1) not in dates  # Ano Novo excluído
