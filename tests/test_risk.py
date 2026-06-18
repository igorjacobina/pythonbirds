"""Testes de gestão de risco: margem MT5, stop-out e bloqueio de ordens."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from innova_ea.backtest.account import AccountConfig
from innova_ea.backtest.costs import CostModel
from innova_ea.backtest.engine import BacktestEngine
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument
from innova_ea.strategy.base import SignalFromColumn


def _bars(opens, closes, target):
    n = len(opens)
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "time": times,
            "open": list(map(float, opens)),
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": list(map(float, closes)),
            "volume": [100.0] * n,
            "target": list(map(float, target)),
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def _usd_base_instrument():
    # Base = conta = USD → margem independe do preço (constante por lote).
    return Instrument(
        "USDTST", digits=5, contract_size=100_000.0, base_currency="USD",
        quote_currency="USD", typical_spread_points=0.0,
        swap_long_points=0.0, swap_short_points=0.0,
    )


def _no_costs():
    return CostModel(spread_points=0.0, slippage_points=0.0,
                     commission_per_lot=0.0, apply_swap=False)


# ---------- Cálculo de margem no Instrument ----------

def test_margin_requires_price_depends_on_base_currency():
    eur = Instrument("EURUSD")          # base EUR, conta USD → usa preço
    jpy = Instrument("USDJPY", base_currency="USD")  # base USD → não usa preço
    assert eur.margin_requires_price("USD") is True
    assert jpy.margin_requires_price("USD") is False


def test_margin_required_formula():
    inst = _usd_base_instrument()
    # 1 lote, leverage 100: 100_000 / 100 = 1_000 (sem conversão de preço).
    assert inst.margin_required(1.0, price=1.23, leverage=100) == pytest.approx(1000.0)
    eur = Instrument("EURUSD")
    # base EUR: 100_000/100 * preço.
    assert eur.margin_required(1.0, price=1.10, leverage=100) == pytest.approx(1100.0)


# ---------- Stop-out ----------

def test_stop_out_liquidates_position():
    # Long 1 lote; preço cai 1% → perda de 1000 numa conta de 1100 (margem 1000).
    bars = _bars(opens=[1.0, 1.0, 0.990], closes=[1.0, 1.0, 0.990], target=[1, 1, 1])
    acct = AccountConfig(leverage=100, stop_out_level=0.5, margin_call_level=1.0)
    engine = BacktestEngine(_usd_base_instrument(), _no_costs(), acct,
                            initial_capital=1100.0, max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)

    assert res.risk.n_stopouts == 1
    assert res.position_lots[-1] == pytest.approx(0.0)  # liquidado à força


# ---------- Bloqueio de ordens por margem livre ----------

def test_order_rejected_when_insufficient_free_margin():
    # Abrir 1 lote exige margem 1000; conta de 500 < 1000 → ordem bloqueada.
    bars = _bars(opens=[1.0, 1.0, 1.0], closes=[1.0, 1.0, 1.0], target=[1, 1, 1])
    acct = AccountConfig(leverage=100, margin_call_level=1.0)
    engine = BacktestEngine(_usd_base_instrument(), _no_costs(), acct,
                            initial_capital=500.0, max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)

    assert res.risk.n_rejected_orders >= 1
    assert res.report.n_trades == 0
    assert all(p == 0.0 for p in res.position_lots)


def test_order_allowed_with_sufficient_margin():
    bars = _bars(opens=[1.0, 1.0, 1.0], closes=[1.0, 1.0, 1.0], target=[1, 1, 1])
    acct = AccountConfig(leverage=100, margin_call_level=1.0)
    engine = BacktestEngine(_usd_base_instrument(), _no_costs(), acct,
                            initial_capital=5000.0, max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)
    assert res.risk.n_rejected_orders == 0
    assert res.position_lots[-1] == pytest.approx(1.0)


# ---------- Alavancagem afeta margem ----------

def test_higher_leverage_uses_less_margin():
    bars = _bars(opens=[1.0, 1.0, 1.0], closes=[1.0, 1.0, 1.0], target=[1, 1, 1])
    inst = _usd_base_instrument()
    low = BacktestEngine(inst, _no_costs(), AccountConfig(leverage=100),
                         initial_capital=10_000.0, max_lots=1.0).run(
        bars, SignalFromColumn("target"), Timeframe.H1)
    high = BacktestEngine(inst, _no_costs(), AccountConfig(leverage=500),
                          initial_capital=10_000.0, max_lots=1.0).run(
        bars, SignalFromColumn("target"), Timeframe.H1)
    assert high.risk.max_used_margin == pytest.approx(low.risk.max_used_margin / 5.0)


# ---------- Detecção de conta zerada ----------

def test_blew_account_flag_without_stopout_guard():
    # Desliga o gating (stop_out=0, margin_call=0); perda gigante zera a conta.
    bars = _bars(opens=[1.0, 1.0, 0.5], closes=[1.0, 1.0, 0.5], target=[1, 1, 1])
    acct = AccountConfig(leverage=100, stop_out_level=0.0, margin_call_level=0.0)
    engine = BacktestEngine(_usd_base_instrument(), _no_costs(), acct,
                            initial_capital=1000.0, max_lots=1.0)
    res = engine.run(bars, SignalFromColumn("target"), Timeframe.H1)
    assert res.risk.blew_account is True


def test_account_config_validates_levels():
    with pytest.raises(ValueError):
        AccountConfig(stop_out_level=1.5, margin_call_level=1.0)  # stop > call
    with pytest.raises(ValueError):
        AccountConfig(leverage=0)
