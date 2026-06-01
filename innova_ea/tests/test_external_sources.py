"""Testes das fontes de histórico externo: Dukascopy (offline) e HistData (CSV)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.csv_source import CsvSource
from innova_ea.data.sources.dukascopy import (
    DukascopySource,
    decode_bi5,
    pack_bi5,
    ticks_to_frame,
)
from innova_ea.data.sources.histdata import HistDataSource


# --------------------------------------------------------------- Dukascopy
def test_decode_bi5_roundtrip():
    recs = [(0, 110010, 109990, 1.0, 1.0), (30000, 110050, 110030, 2.0, 1.5)]
    arr = decode_bi5(pack_bi5(recs))
    assert arr.shape[0] == 2
    assert int(arr["ask"][0]) == 110010 and int(arr["bid"][0]) == 109990
    assert int(arr["ms"][1]) == 30000


def test_decode_bi5_empty():
    assert decode_bi5(b"").shape[0] == 0


def test_ticks_to_frame_scales_and_times():
    recs = [(0, 110010, 109990, 1.0, 1.0), (60000, 110030, 110010, 1.0, 1.0)]
    arr = decode_bi5(pack_bi5(recs))
    hour = datetime(2020, 1, 6, 10, tzinfo=timezone.utc)
    fr = ticks_to_frame(arr, hour, price_scale=1e5)
    assert fr["time"][0] == hour
    assert fr["time"][1] == datetime(2020, 1, 6, 10, 1, tzinfo=timezone.utc)
    assert fr["mid"][0] == pytest.approx(1.1)        # (1.1001+1.0999)/2


def test_dukascopy_fetch_builds_m1_bars():
    raw = pack_bi5([(0, 110010, 109990, 1.0, 1.0),       # 10:00 → mid 1.1
                    (30000, 110050, 110030, 2.0, 1.0),    # 10:00 → mid 1.1004
                    (3_540_000, 109900, 109880, 1.0, 1.0)])  # 10:59 → mid 1.0989

    def fake_dl(url):
        return raw if url.endswith("2020/00/06/10h_ticks.bi5") else None

    src = DukascopySource(downloader=fake_dl)
    bars = src.fetch("EURUSD", Timeframe.M1,
                     datetime(2020, 1, 6, 10, tzinfo=timezone.utc),
                     datetime(2020, 1, 6, 11, tzinfo=timezone.utc))
    assert bars.columns == ["time", "open", "high", "low", "close", "volume"]
    first = bars.row(0, named=True)
    assert first["open"] == pytest.approx(1.1)
    assert first["high"] == pytest.approx(1.1004)
    assert bars["time"].dtype.time_zone == "UTC"


def test_dukascopy_unknown_symbol_raises():
    src = DukascopySource(downloader=lambda url: None)
    with pytest.raises(ValueError, match="mapeamento Dukascopy"):
        src.fetch("FOOBAR", Timeframe.M1,
                  datetime(2020, 1, 1, tzinfo=timezone.utc),
                  datetime(2020, 1, 2, tzinfo=timezone.utc))


# --------------------------------------------------------------- HistData / CSV
_HISTDATA_ROWS = (
    "20200106 100000;1.1000;1.1010;1.0990;1.1005;0\n"
    "20200106 100100;1.1005;1.1015;1.1000;1.1010;0\n"
)


def test_csv_source_fixed_offset_est_to_utc(tmp_path):
    path = tmp_path / "DAT_ASCII_EURUSD_M1_202001.csv"
    path.write_text(_HISTDATA_ROWS)
    src = CsvSource(
        str(path), new_columns=["time", "open", "high", "low", "close", "volume"],
        column_map={c: c for c in ["time", "open", "high", "low", "close", "volume"]},
        time_format="%Y%m%d %H%M%S", separator=";", has_header=False, source_utc_offset=-5,
    )
    bars = src.fetch("EURUSD", Timeframe.M1,
                     datetime(2020, 1, 6, tzinfo=timezone.utc),
                     datetime(2020, 1, 7, tzinfo=timezone.utc))
    # 10:00 EST → 15:00 UTC.
    assert bars["time"][0] == datetime(2020, 1, 6, 15, 0, tzinfo=timezone.utc)
    assert bars["close"][0] == pytest.approx(1.1005)


def test_histdata_source_reads_monthly_files(tmp_path):
    (tmp_path / "DAT_ASCII_EURUSD_M1_202001.csv").write_text(_HISTDATA_ROWS)
    src = HistDataSource(str(tmp_path))
    bars = src.fetch("EURUSD", Timeframe.M1,
                     datetime(2020, 1, 6, tzinfo=timezone.utc),
                     datetime(2020, 1, 7, tzinfo=timezone.utc))
    assert bars.height == 2
    assert bars["time"][0] == datetime(2020, 1, 6, 15, 0, tzinfo=timezone.utc)


def test_histdata_unknown_symbol_raises(tmp_path):
    src = HistDataSource(str(tmp_path))
    with pytest.raises(ValueError, match="HistData"):
        src.fetch("US30", Timeframe.M1,
                  datetime(2020, 1, 1, tzinfo=timezone.utc),
                  datetime(2020, 1, 2, tzinfo=timezone.utc))
