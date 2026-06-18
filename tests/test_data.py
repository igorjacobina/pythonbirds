"""Testes do pipeline de dados: resample, storage e ingestão."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data.pipeline import DataPipeline
from innova_ea.data.resample import resample
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.data.storage import ParquetStore


def test_resample_m15_to_h1_aggregates_correctly(synthetic_m15):
    h1 = resample(synthetic_m15, Timeframe.H1)
    # 4 barras M15 por H1.
    assert h1.height == synthetic_m15.height // 4

    # Primeira H1 deve combinar as 4 primeiras M15.
    first4 = synthetic_m15.head(4)
    assert h1["open"][0] == first4["open"][0]
    assert h1["close"][0] == first4["close"][3]
    assert h1["high"][0] == first4["high"].max()
    assert h1["low"][0] == first4["low"].min()
    assert h1["volume"][0] == first4["volume"].sum()


def test_resample_preserves_utc(synthetic_m15):
    h4 = resample(synthetic_m15, Timeframe.H4)
    assert h4["time"].dtype.time_zone == "UTC"


def test_storage_roundtrip(tmp_path, synthetic_m15):
    store = ParquetStore(tmp_path)
    written = store.write("EURUSD", Timeframe.M15, synthetic_m15)
    assert written == synthetic_m15.height

    back = store.read("EURUSD", Timeframe.M15)
    assert back.height == synthetic_m15.height
    assert back["close"].to_list() == synthetic_m15["close"].to_list()
    assert store.available_symbols() == ["EURUSD"]


def test_storage_upsert_is_idempotent(tmp_path, synthetic_m15):
    store = ParquetStore(tmp_path)
    store.write("EURUSD", Timeframe.M15, synthetic_m15)
    store.write("EURUSD", Timeframe.M15, synthetic_m15)  # reescreve mesmo range
    back = store.read("EURUSD", Timeframe.M15)
    assert back.height == synthetic_m15.height  # sem duplicatas


def test_storage_partial_read_range(tmp_path, synthetic_m15):
    store = ParquetStore(tmp_path)
    store.write("EURUSD", Timeframe.M15, synthetic_m15)
    mid = synthetic_m15["time"][synthetic_m15.height // 2]
    back = store.read("EURUSD", Timeframe.M15, start=mid)
    assert back["time"].min() >= mid


def test_pipeline_ingest_and_derive(tmp_path):
    src = SyntheticSource(seed=5)
    store = ParquetStore(tmp_path)
    pipe = DataPipeline(src, store)
    rep = pipe.ingest(
        "GBPUSD",
        Timeframe.M15,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 10, tzinfo=timezone.utc),
        derive=[Timeframe.H1, Timeframe.D1],
    )
    assert rep.bars_fetched > 0
    assert "H1" in rep.bars_written and "D1" in rep.bars_written
    assert store.read("GBPUSD", Timeframe.H1).height > 0
