"""Testes do engine de backtest: P&L correto, sem look-ahead, custos aplicados."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from forex_quant.backtest.costs import CostModel
from forex_quant.backtest.engine import BacktestEngine
from forex_quant.core.enums import Timeframe
from forex_quant.core.instruments import Instrument
from forex_quant.strategy.base import MovingAverageCrossover, SignalFromColumn


def _bars(opens, closes, highs=None, lows=None, target=None):
    n = len(opens)
    highs = highs or [max(o, c) for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) for o, c in zip(opens, closes)]
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)  # segunda-feira
    times = [t0 + timedelta(hours=i) for i in range(n)]
    data = {
        "time": times,
        "open": list(map(float, opens)),
        "high": list(map(float, highs)),
        "low": list(map(float, lows)),
        "close": list(map(float, closes)),
        "volume": [100.0] * n,
    }
    if target is not None:
        data["target"] = list(map(float, target))
    return pl.DataFrame(data).with_columns(
        pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )


def _frictionless_instrument():
    return Instrument(
        "TEST", digits=5, contract_size=100_000.0,
        typical_spread_points=0.0, swap_long_points=0.0, swap_short_points=0.0,
    )


def _no_costs():
    return CostModel(spread_points=0.0, slippage_points=0.0,
                     commission_per_lot=0.0, apply_swap=False)


def test_pnl_is_exact_without_costs():
    # Sempre long; preço sobe de 1.0 para 2.0 na abertura da barra 2.
    bars = _bars(
        opens=[1.0, 1.0, 2.0, 2.0],
        closes=[1.0, 1.0, 2.0, 2.0],
        target=[1.0, 1.0, 1.0, 1.0],
    )
    engine = BacktestEngine(_frictionless_instrument(), _no_costs(),
                            initial_capital=10_000.0, max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)

    # Entrada na abertura da barra 1 (=1.0); ganho de 1.0 * 100k = 100_000.
    assert res.equity[-1] == pytest.approx(110_000.0)


def test_no_look_ahead_lag_of_one_bar():
    # Sinal aparece na barra 0; o salto de preço ocorre na abertura da barra 1.
    # Com lag de 1 barra, a entrada é em open[1]=2.0 → NÃO captura o salto.
    bars = _bars(
        opens=[1.0, 2.0, 2.0],
        closes=[1.0, 2.0, 2.0],
        target=[1.0, 1.0, 1.0],
    )
    engine = BacktestEngine(_frictionless_instrument(), _no_costs(),
                            initial_capital=10_000.0, max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)
    # Entrou a 2.0 e preço ficou a 2.0 → zero lucro (sem look-ahead).
    assert res.equity[-1] == pytest.approx(10_000.0)


def test_flat_strategy_keeps_capital_and_no_trades():
    bars = _bars(opens=[1.0, 1.1, 1.2], closes=[1.05, 1.15, 1.25],
                 target=[0.0, 0.0, 0.0])
    engine = BacktestEngine(_frictionless_instrument(), _no_costs(),
                            initial_capital=10_000.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)
    assert np.allclose(res.equity, 10_000.0)
    assert res.report.n_trades == 0


def test_costs_reduce_final_equity():
    bars = _bars(opens=[1.0, 1.0, 2.0, 2.0, 1.0],
                 closes=[1.0, 1.0, 2.0, 2.0, 1.0],
                 target=[1.0, 1.0, -1.0, -1.0, 0.0])
    inst = _frictionless_instrument()
    free = BacktestEngine(inst, _no_costs(), max_lots=1.0).run(
        bars, SignalFromColumn("target"), Timeframe.H1)
    costed = BacktestEngine(
        inst, CostModel(spread_points=10.0, slippage_points=2.0, commission_per_lot=7.0),
        max_lots=1.0,
    ).run(bars, SignalFromColumn("target"), Timeframe.H1)
    assert costed.equity[-1] < free.equity[-1]


def test_short_position_profits_when_price_falls():
    bars = _bars(opens=[2.0, 2.0, 1.0, 1.0],
                 closes=[2.0, 2.0, 1.0, 1.0],
                 target=[-1.0, -1.0, -1.0, -1.0])
    engine = BacktestEngine(_frictionless_instrument(), _no_costs(), max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)
    # Vendido a 2.0, preço cai p/ 1.0 → ganho 1.0 * 100k.
    assert res.equity[-1] == pytest.approx(110_000.0)


def test_engine_runs_on_synthetic(synthetic_m15):
    inst = _frictionless_instrument()
    engine = BacktestEngine(inst, CostModel(), initial_capital=10_000.0)
    res = engine.run(synthetic_m15, MovingAverageCrossover(10, 30), Timeframe.M15)
    assert res.equity.shape[0] == synthetic_m15.height
    assert np.isfinite(res.equity).all()
    assert res.report.n_trades >= 0
