"""Fixtures compartilhadas."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from forex_quant.core.enums import Timeframe
from forex_quant.data.sources.synthetic import SyntheticSource


@pytest.fixture
def synthetic_m15() -> pl.DataFrame:
    src = SyntheticSource(seed=123)
    return src.fetch(
        "EURUSD",
        Timeframe.M15,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 2, 1, tzinfo=timezone.utc),
    )
