"""Testes da camada core: instrumentos e validação de barras."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from forex_quant.core.bars import BarValidationError, validate_bars
from forex_quant.core.enums import Timeframe
from forex_quant.core.instruments import Instrument, get_instrument


def test_pip_and_point():
    eur = get_instrument("EURUSD")
    assert eur.point == pytest.approx(1e-5)
    assert eur.pip == pytest.approx(1e-4)
    assert eur.points_per_pip == pytest.approx(10.0)

    jpy = get_instrument("USDJPY")
    assert jpy.pip == pytest.approx(0.01)


def test_pnl_quote_one_pip_standard_lot():
    eur = get_instrument("EURUSD")
    # 1 pip (0.0001) em 1.0 lote (100k) = 10 da moeda de cotação.
    assert eur.pnl_quote(0.0001, 1.0) == pytest.approx(10.0)


def test_round_lots_respects_step_and_limits():
    inst = Instrument("X", min_lot=0.01, max_lot=5.0, lot_step=0.01)
    assert inst.round_lots(0.123) == pytest.approx(0.12)
    assert inst.round_lots(0.0) == pytest.approx(0.01)
    assert inst.round_lots(99.0) == pytest.approx(5.0)


def test_timeframe_minutes():
    assert Timeframe.H1.minutes == 60
    assert Timeframe.H4.minutes == 240
    assert Timeframe.D1.minutes == 1440


def _good_df():
    return pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "open": [1.0, 1.1],
            "high": [1.2, 1.2],
            "low": [0.9, 1.0],
            "close": [1.1, 1.15],
            "volume": [100.0, 120.0],
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))


def test_validate_bars_ok():
    df = validate_bars(_good_df())
    assert df.columns == ["time", "open", "high", "low", "close", "volume"]


def test_validate_bars_rejects_unsorted():
    df = _good_df().reverse()
    with pytest.raises(BarValidationError, match="ordem"):
        validate_bars(df)


def test_validate_bars_rejects_bad_ohlc():
    df = _good_df().with_columns(pl.lit(0.5).alias("high"))  # high < low
    with pytest.raises(BarValidationError, match="OHLC"):
        validate_bars(df)


def test_validate_bars_rejects_nulls():
    df = _good_df().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("close")).alias("close")
    )
    with pytest.raises(BarValidationError, match="nulos"):
        validate_bars(df)
