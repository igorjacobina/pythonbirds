"""Testes do motor de straddle de volatilidade (direção-agnóstico)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.features import CloseLocationValue, EfficiencyRatio, FeatureSet, YangZhang
from innova_ea.research.straddle import (
    _straddle_kernel,
    build_straddle_panel,
    straddle_backtest,
    straddle_outcomes,
)


def _arr(x):
    return np.array(x, dtype=np.float64)


def test_kernel_long_win():
    high = _arr([10, 10.5, 12.0, 12, 12, 12])
    low = _arr([9, 9.5, 11.0, 11, 11, 11])
    close = high.copy()
    band_hi = _arr([10, 0, 0, 0, 0, 0])
    band_lo = _arr([9, 0, 0, 0, 0, 0])
    tgt = _arr([2, 0, 0, 0, 0, 0])
    stp = _arr([1, 0, 0, 0, 0, 0])
    label, net, resolve, trig = _straddle_kernel(high, low, close, band_hi, band_lo, tgt, stp, 0.0, 5)
    assert trig[0] == 1 and label[0] == 1
    assert net[0] == 2.0           # tgt - 2*cost(0)
    assert resolve[0] == 2


def test_kernel_long_loss():
    high = _arr([10, 10.5, 10.6, 10.6, 10.6, 10.6])
    low = _arr([9, 9.5, 8.5, 8.5, 8.5, 8.5])
    close = high.copy()
    band_hi = _arr([10, 0, 0, 0, 0, 0])
    band_lo = _arr([9, 0, 0, 0, 0, 0])
    tgt = _arr([2, 0, 0, 0, 0, 0])
    stp = _arr([1, 0, 0, 0, 0, 0])
    label, net, resolve, trig = _straddle_kernel(high, low, close, band_hi, band_lo, tgt, stp, 0.0, 5)
    assert trig[0] == 1 and label[0] == 0
    assert net[0] == -1.0          # -stp - 2*cost(0)


def test_kernel_no_breakout():
    high = _arr([10, 10.2, 10.3, 10.1, 10.2, 10.0])
    low = _arr([9, 9.2, 9.3, 9.1, 9.2, 9.0])
    close = high.copy()
    band_hi = _arr([11, 0, 0, 0, 0, 0])   # nunca rompe
    band_lo = _arr([8, 0, 0, 0, 0, 0])
    tgt = _arr([2, 0, 0, 0, 0, 0])
    stp = _arr([1, 0, 0, 0, 0, 0])
    label, net, resolve, trig = _straddle_kernel(high, low, close, band_hi, band_lo, tgt, stp, 0.0, 5)
    assert trig[0] == 0 and resolve[0] == 0 and label[0] == 0


def test_kernel_cost_reduces_net():
    high = _arr([10, 10.5, 12.0, 12, 12, 12])
    low = _arr([9, 9.5, 11.0, 11, 11, 11])
    close = high.copy()
    band_hi = _arr([10, 0, 0, 0, 0, 0]); band_lo = _arr([9, 0, 0, 0, 0, 0])
    tgt = _arr([2, 0, 0, 0, 0, 0]); stp = _arr([1, 0, 0, 0, 0, 0])
    _, net, _, _ = _straddle_kernel(high, low, close, band_hi, band_lo, tgt, stp, 0.1, 5)
    assert net[0] == 2.0 - 0.2     # tgt - 2*cost


def _synth(symbol, vol, seed):
    return SyntheticSource(seed=seed, annual_vol=vol).fetch(
        symbol, Timeframe.H1,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 6, 1, tzinfo=timezone.utc))


def test_straddle_outcomes_columns():
    bars = _synth("EURUSD", 0.1, 1)
    out = straddle_outcomes(bars, horizon=12, band_lookback=8)
    assert out.columns == ["label", "net_price", "resolve", "triggered"]
    assert out.height == bars.height
    assert set(out["label"].unique().to_list()).issubset({0, 1})


def test_build_straddle_panel():
    bars_by = {"EURUSD": _synth("EURUSD", 0.1, 1), "XAUUSD": _synth("XAUUSD", 0.15, 2)}
    cls = {"EURUSD": "forex", "XAUUSD": "metal"}
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    panel = build_straddle_panel(bars_by, fs, asset_class_of=lambda s: cls[s],
                                 horizon=12, warmup=120)
    assert "label" in panel.frame.columns
    assert panel.frame.select(panel.feature_names).null_count().to_numpy().sum() == 0
    assert set(panel.frame["label"].unique().to_list()).issubset({0, 1})


def test_straddle_backtest_no_overlap():
    bars = _synth("EURUSD", 0.1, 3)
    out = straddle_outcomes(bars, horizon=12, band_lookback=8)
    arm = np.ones(bars.height, dtype=bool)     # tenta armar sempre
    equity, in_trade, pnls = straddle_backtest(bars, arm, out, 100_000.0,
                                               lots=0.1, initial_capital=10_000.0)
    assert equity.shape[0] == bars.height
    assert np.isfinite(equity).all()
    # Sem sobreposição: nº de trades <= barras / 1 (cada trade ocupa >=1 barra).
    assert len(pnls) <= bars.height
