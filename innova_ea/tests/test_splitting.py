"""Testes do walk-forward purgado com embargo."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from innova_ea.research.splitting import PurgedWalkForward


def _frame(n_times: int, assets: int = 2):
    t0 = datetime(2021, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(minutes=15 * i) for i in range(n_times)]
    rows = []
    for a in range(assets):
        for t in times:
            rows.append({"time": t, "asset_id": a, "label": 0})
    return pl.DataFrame(rows).with_columns(
        pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )


def test_n_splits_count():
    frame = _frame(300)
    wf = PurgedWalkForward(train_size=100, test_size=50, horizon=5, embargo=2)
    assert wf.n_splits(frame) == 4  # test_lo em 100,150,200,250


def test_purge_and_embargo_create_gap():
    frame = _frame(300)
    horizon, embargo = 5, 3
    wf = PurgedWalkForward(train_size=100, test_size=50, horizon=horizon, embargo=embargo)
    train_df, test_df, _ = next(wf.split(frame))

    train_max = train_df["time"].max()
    test_min = test_df["time"].min()
    # Deve haver uma folga >= (horizon+embargo) barras de 15min entre treino e teste.
    gap = (test_min - train_max).total_seconds() / 60.0
    assert gap >= (horizon + embargo) * 15
    # Treino e teste são temporalmente disjuntos.
    assert train_max < test_min


def test_test_windows_are_non_overlapping():
    frame = _frame(300)
    wf = PurgedWalkForward(train_size=100, test_size=50, horizon=5)
    test_starts = [t["time"].min() for _, t, _ in wf.split(frame)]
    assert test_starts == sorted(test_starts)
    assert len(set(test_starts)) == len(test_starts)


def test_raises_on_insufficient_data():
    frame = _frame(80)
    wf = PurgedWalkForward(train_size=100, test_size=50)
    with pytest.raises(ValueError, match="insuficientes"):
        next(wf.split(frame))
