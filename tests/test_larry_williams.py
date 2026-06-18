"""Testes do método Larry Williams (volatility breakout)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from innova_ea.research.larry_williams import larry_backtest, larry_outcomes


def _bars(opens, highs, lows, closes):
    n = len(closes)
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)
    times = [t0 + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({
        "time": times, "open": list(map(float, opens)), "high": list(map(float, highs)),
        "low": list(map(float, lows)), "close": list(map(float, closes)),
        "volume": [1.0] * n,
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def test_long_breakout_profitable():
    # Dia 0 define o range (10). Dia 1 rompe p/ cima e a abertura do dia 2 é lucrativa.
    # k=0.5 → buy = open[1](100) + 0.5*range(10) = 105. high[1]=106 dispara.
    bars = _bars(
        opens=[100, 100, 110],
        highs=[105, 106, 112],
        lows=[95, 99, 108],
        closes=[100, 106, 111],
    )
    out = larry_outcomes(bars, k=0.5, max_hold=3, cost_frac=0.0)
    assert out["triggered"][1] == 1 and out["direction"][1] == 1
    # entrada 105; saída na 1ª abertura lucrativa = open[2]=110 → lucro.
    assert out["net_ret"][1] > 0


def test_no_breakout_no_trade():
    # Dia 1 fica DENTRO do range (não rompe nem p/ cima nem p/ baixo).
    bars = _bars(
        opens=[100, 100, 100],
        highs=[105, 103, 103],
        lows=[95, 97, 97],
        closes=[100, 100, 100],
    )
    out = larry_outcomes(bars, k=0.5, max_hold=3, cost_frac=0.0)
    assert out["triggered"][1] == 0


def test_costs_reduce_returns():
    bars = _bars([100, 100, 110], [105, 106, 112], [95, 99, 108], [100, 106, 111])
    free = larry_outcomes(bars, k=0.5, cost_frac=0.0)["net_ret"][1]
    costed = larry_outcomes(bars, k=0.5, cost_frac=0.02)["net_ret"][1]
    assert costed < free


def test_backtest_runs_and_is_finite():
    rng = np.random.default_rng(0)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    bars = _bars(open_, np.maximum.reduce([high, open_, close]),
                 np.minimum.reduce([low, open_, close]), close)
    out = larry_outcomes(bars, k=0.5, max_hold=5, cost_frac=0.0004)
    equity, pos, trades = larry_backtest(out, initial_capital=10_000.0)
    assert equity.shape[0] == bars.height
    assert np.isfinite(equity).all()
    assert len(trades) > 0
