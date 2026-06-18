"""Testes do framework de walk-forward."""
from __future__ import annotations

import pytest

from innova_ea.backtest.account import AccountConfig
from innova_ea.backtest.costs import CostModel
from innova_ea.backtest.engine import BacktestEngine
from innova_ea.backtest.walkforward import WalkForward
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument
from innova_ea.strategy.base import MovingAverageCrossover


def test_split_rolling_counts_and_indices():
    wf = WalkForward(train_size=100, test_size=50, step=50)
    windows = wf.split(300)
    # test_start em 100,150,200,250 → 4 janelas.
    assert len(windows) == 4
    assert windows[0].train_start == 0 and windows[0].test_start == 100
    assert windows[1].train_start == 50  # rolling: anda com o passo
    for w in windows:
        assert w.test_end - w.test_start == 50
        assert w.train_end == w.test_start  # treino termina onde o teste começa


def test_split_anchored_keeps_train_start_zero():
    wf = WalkForward(train_size=100, test_size=50, anchored=True)
    windows = wf.split(300)
    assert all(w.train_start == 0 for w in windows)


def test_split_raises_when_insufficient():
    wf = WalkForward(train_size=200, test_size=100)
    with pytest.raises(ValueError, match="insuficientes"):
        wf.split(150)


def test_walkforward_run_produces_oos_report(synthetic_m15):
    inst = Instrument("TEST", typical_spread_points=0.0)

    def strategy_factory(train):
        # Estratégia fixa (não calibrada) — o framework suporta calibração real.
        return MovingAverageCrossover(10, 30)

    def engine_factory():
        return BacktestEngine(inst, CostModel(), AccountConfig(leverage=100),
                              initial_capital=10_000.0, max_lots=0.1)

    wf = WalkForward(train_size=500, test_size=250)
    result = wf.run(synthetic_m15, strategy_factory, engine_factory, Timeframe.M15)

    assert result.n_windows >= 1
    assert len(result.window_reports) == result.n_windows
    assert len(result.oos_sharpes) == result.n_windows
    assert result.oos_report.initial_capital == 10_000.0
    assert "Janelas OOS" in result.summary()
