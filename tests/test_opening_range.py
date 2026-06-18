"""Testes do Opening Range Breakout (método Andrea Unger)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from innova_ea.research.opening_range import orb_backtest, orb_outcomes


def _intraday(rows_per_day, days, *, start=datetime(2021, 1, 5, tzinfo=timezone.utc)):
    """Gera barras intradiárias (M60) para vários dias (terça em diante)."""
    times, o, h, l, c = [], [], [], [], []
    t = start
    for _ in range(days):
        for b in range(rows_per_day):
            times.append(t)
            t += timedelta(hours=1)
        t = (t + timedelta(days=1)).replace(hour=0)
    return times


def _bars(times, opens, highs, lows, closes):
    return pl.DataFrame({
        "time": times, "open": list(map(float, opens)), "high": list(map(float, highs)),
        "low": list(map(float, lows)), "close": list(map(float, closes)),
        "volume": [1.0] * len(times),
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def test_orb_long_breakout_one_day():
    # Um dia, 6 barras. or_bars=2 → range de abertura = barras 0,1 (high 11, low 9).
    # barra 3 rompe p/ cima (high 12 >= 11) → long em 11; sobe até o alvo.
    t0 = datetime(2021, 1, 5, tzinfo=timezone.utc)  # terça
    times = [t0 + timedelta(hours=i) for i in range(6)]
    bars = _bars(
        times,
        opens=[10, 10, 10, 11, 11, 11],
        highs=[11, 10.5, 10.8, 12, 13, 13],
        lows=[9, 9.5, 10, 10.5, 11, 11],
        closes=[10, 10, 10.5, 11.5, 12.5, 12.5],
    )
    out = orb_outcomes(bars, or_bars=2, reward_risk=1.0, cost_frac=0.0, skip_monday=True)
    assert out["triggered"].sum() == 1
    idx = int(out["triggered"].to_numpy().argmax())
    assert out["direction"][idx] == 1
    assert out["net_ret"][idx] > 0       # entrada 11, alvo = 11 + (11-9) = 13 atingido


def test_orb_skips_monday():
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)  # SEGUNDA-feira
    times = [t0 + timedelta(hours=i) for i in range(6)]
    bars = _bars(
        times,
        opens=[10, 10, 10, 11, 11, 11],
        highs=[11, 10.5, 10.8, 12, 13, 13],
        lows=[9, 9.5, 10, 10.5, 11, 11],
        closes=[10, 10, 10.5, 11.5, 12.5, 12.5],
    )
    out = orb_outcomes(bars, or_bars=2, skip_monday=True)
    assert out["triggered"].sum() == 0   # segunda é pulada (filtro de Unger)


def test_orb_no_breakout_no_trade():
    t0 = datetime(2021, 1, 5, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(6)]
    bars = _bars(
        times,
        opens=[10, 10, 10, 10, 10, 10],
        highs=[11, 10.5, 10.6, 10.7, 10.6, 10.5],   # nunca rompe a máxima (11)
        lows=[9, 9.5, 9.6, 9.7, 9.6, 9.5],          # nem a mínima (9)
        closes=[10, 10, 10, 10, 10, 10],
    )
    out = orb_outcomes(bars, or_bars=2, skip_monday=True)
    assert out["triggered"].sum() == 0


def test_orb_backtest_runs():
    rng = np.random.default_rng(0)
    days, rpd = 40, 8
    t0 = datetime(2021, 1, 5, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(days * rpd)]
    close = 100 + np.cumsum(rng.normal(0, 0.5, days * rpd))
    high = close + np.abs(rng.normal(0, 0.3, days * rpd))
    low = close - np.abs(rng.normal(0, 0.3, days * rpd))
    open_ = close + rng.normal(0, 0.2, days * rpd)
    bars = _bars(times, open_, np.maximum.reduce([high, open_, close]),
                 np.minimum.reduce([low, open_, close]), close)
    out = orb_outcomes(bars, or_bars=3, reward_risk=1.0, cost_frac=0.0004, skip_monday=False)
    equity, pos, trades = orb_backtest(out, initial_capital=10_000.0)
    assert equity.shape[0] == bars.height
    assert np.isfinite(equity).all()
