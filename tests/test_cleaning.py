"""Testes da limpeza defensiva e da análise de gaps."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data.calendar import ForexCalendar
from innova_ea.data.cleaning import analyze_gaps, clean_bars


def _m1(times, opens=None, highs=None, lows=None, closes=None):
    n = len(times)
    opens = opens or [1.10] * n
    highs = highs or [1.11] * n
    lows = lows or [1.09] * n
    closes = closes or [1.105] * n
    return pl.DataFrame(
        {
            "time": times,
            "open": list(map(float, opens)),
            "high": list(map(float, highs)),
            "low": list(map(float, lows)),
            "close": list(map(float, closes)),
            "volume": [100.0] * n,
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def test_clean_bars_removes_dirt():
    t0 = datetime(2021, 1, 4, 12, tzinfo=timezone.utc)
    times = [t0, t0, t0 + timedelta(minutes=1), t0 + timedelta(minutes=2),
             t0 + timedelta(minutes=3)]
    # idx0/1 duplicados; idx3 com OHLC inconsistente; idx4 com preço <= 0.
    df = _m1(
        times,
        highs=[1.11, 1.11, 1.11, 0.5, 1.11],   # idx3: high < low → inconsistente
        lows=[1.09, 1.09, 1.09, 1.09, 1.09],
        opens=[1.10, 1.10, 1.10, 1.10, -1.0],  # idx4: preço negativo
        closes=[1.105, 1.105, 1.105, 1.105, 1.105],
    )
    clean, rep = clean_bars(df)
    assert rep.raw_rows == 5
    assert rep.duplicates_removed == 1
    assert rep.ohlc_violations_removed == 1
    assert rep.nonpositive_removed == 1
    assert clean.height == 2  # sobram idx2 e idx3-original? não: idx0(dedupe), idx2 válidos


def test_analyze_gaps_detects_intrasession_hole():
    t0 = datetime(2021, 1, 4, 12, tzinfo=timezone.utc)  # segunda
    end = t0 + timedelta(minutes=10)
    all_times = [t0 + timedelta(minutes=i) for i in range(10)]
    # Remove 12:03 e 12:04 (buraco de 2 barras).
    present = [t for i, t in enumerate(all_times) if i not in (3, 4)]
    bars = _m1(present)
    cal = ForexCalendar()
    rep = analyze_gaps(bars, cal, Timeframe.M1, t0, end)

    assert rep.expected_bars == 10
    assert rep.missing_bars == 2
    assert rep.n_gap_runs == 1
    assert rep.bars_outside_calendar == 0
    assert rep.coverage == 0.8
    assert rep.largest_gap.missing == 2


def test_analyze_gaps_flags_bars_outside_calendar():
    # Janela cruzando o fechamento de sexta (22:00 UTC).
    start = datetime(2021, 1, 8, 21, 55, tzinfo=timezone.utc)  # sexta
    end = datetime(2021, 1, 8, 22, 5, tzinfo=timezone.utc)
    trading = [start + timedelta(minutes=i) for i in range(5)]   # 21:55..21:59
    stray = datetime(2021, 1, 8, 22, 2, tzinfo=timezone.utc)     # após fechamento
    bars = _m1(trading + [stray])
    cal = ForexCalendar()
    rep = analyze_gaps(bars, cal, Timeframe.M1, start, end)

    assert rep.expected_bars == 5          # só 21:55..21:59 são negociáveis
    assert rep.missing_bars == 0
    assert rep.bars_outside_calendar == 1  # alerta de fuso/dados
