"""Testes de rotulagem: forward returns e triple-barrier."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from innova_ea.research.labeling import forward_return, triple_barrier_labels


def _bars(opens, highs, lows, closes):
    n = len(closes)
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "time": times,
            "open": list(map(float, opens)),
            "high": list(map(float, highs)),
            "low": list(map(float, lows)),
            "close": list(map(float, closes)),
            "volume": [1.0] * n,
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def test_forward_return_simple():
    bars = _bars([1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8])
    fr = forward_return(bars, horizon=1, log=False)
    vals = fr.to_list()
    assert vals[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert vals[3] is None  # sem futuro na última barra


def test_triple_barrier_upper_lower_time():
    # 5 barras de close=100; barreiras absolutas de ±5, horizonte 3.
    close = [100, 100, 100, 100, 100]
    # bar0: high[1]=106 toca a barreira superior → +1 em 1 barra.
    # bar1: low[2]=94 toca a inferior (high[2]<105) → -1 em 1 barra.
    # bar2: nada dentro do horizonte → 0 (barreira de tempo).
    high = [100, 106, 101, 101, 101]
    low = [100, 99, 94, 99, 99]
    bars = _bars(close, high, low, close)
    out = triple_barrier_labels(bars, width=5.0, max_horizon=3, width_is_pct=False)

    assert out["label"][0] == 1 and out["bars_to_touch"][0] == 1
    assert out["label"][1] == -1 and out["bars_to_touch"][1] == 1
    assert out["label"][2] == 0 and out["bars_to_touch"][2] == -1


def test_triple_barrier_pct_and_array_width():
    bars = _bars([100] * 4, [100, 100, 100, 100], [100, 90, 100, 100], [100] * 4)
    # width como array (por barra) em pct.
    out = triple_barrier_labels(
        bars, width=[0.05, 0.05, 0.05, 0.05], max_horizon=2, width_is_pct=True
    )
    # bar0: low[1]=90 <= 100*0.95=95 → -1.
    assert out["label"][0] == -1


def test_triple_barrier_validates_horizon():
    bars = _bars([100], [100], [100], [100])
    with pytest.raises(ValueError):
        triple_barrier_labels(bars, width=0.01, max_horizon=0)
