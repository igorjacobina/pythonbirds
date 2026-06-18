"""Fixtures compartilhadas."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.synthetic import SyntheticSource


@pytest.fixture
def synthetic_m15() -> pl.DataFrame:
    src = SyntheticSource(seed=123)
    return src.fetch(
        "EURUSD",
        Timeframe.M15,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 2, 1, tzinfo=timezone.utc),
    )
