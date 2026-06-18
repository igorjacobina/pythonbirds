"""Testes do Serviço de Execução: conciliação, event-driven, kill-switch, margem.

Usa um Broker FALSO (sem MT5) para validar toda a lógica crítica no Linux/CI.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from innova_ea.backtest.account import AccountConfig
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument, get_instrument
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.execution.broker import (
    AccountState,
    Broker,
    BrokerError,
    OrderResult,
    Position,
)
from innova_ea.execution.reconciliation import reconcile
from innova_ea.execution.risk import RiskLimits, RiskManager
from innova_ea.execution.service import ExecutionConfig, ExecutionService
from innova_ea.strategy.base import Strategy

FIXED_NOW = datetime(2021, 6, 1, 12, 0, tzinfo=timezone.utc)


class FixedTarget(Strategy):
    """Estratégia de teste: alvo constante em [-1, 1]."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.name = "fixed"

    def generate_targets(self, bars, inst):
        return np.full(bars.height, self.value)


class FakeBroker(Broker):
    def __init__(self, bars_by, instruments, account):
        self._bars = bars_by
        self._instr = instruments
        self._account = account
        self._positions: dict[str, float] = {}
        self.orders: list[tuple[str, float]] = []
        self._cursor = {s: 100 for s in bars_by}
        self.connected = True
        self.fail_orders = False
        self.connect_fails = False

    # estado de teste
    def set_account(self, account):
        self._account = account

    def set_position(self, symbol, lots):
        self._positions[symbol] = lots

    def advance(self, symbol, n=1):
        self._cursor[symbol] += n

    # interface Broker
    def is_connected(self):
        return self.connected

    def connect(self):
        if self.connect_fails:
            raise BrokerError("reconexão falhou")
        self.connected = True

    def disconnect(self):
        self.connected = False

    def instrument(self, symbol):
        return self._instr[symbol]

    def account_state(self):
        return self._account

    def position(self, symbol):
        lots = self._positions.get(symbol, 0.0)
        return Position(symbol, lots) if lots != 0.0 else None

    def open_positions(self):
        return {s: Position(s, l) for s, l in self._positions.items() if l != 0.0}

    def recent_closed_bars(self, symbol, timeframe, count):
        return self._bars[symbol].head(self._cursor[symbol]).tail(count)

    def send_market_order(self, symbol, delta_lots):
        if self.fail_orders:
            return OrderResult(False, symbol, delta_lots, message="rejeitado")
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta_lots
        self.orders.append((symbol, delta_lots))
        return OrderResult(True, symbol, delta_lots, price=1.1)


def _bars(symbol="EURUSD"):
    return SyntheticSource(seed=1, annual_vol=0.1).fetch(
        symbol, Timeframe.M15,
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 8, tzinfo=timezone.utc),
    )


def _account(equity=10_000.0, margin_free=10_000.0, margin_used=0.0):
    lvl = (equity / margin_used) if margin_used > 0 else float("inf")
    return AccountState(balance=equity, equity=equity, margin_used=margin_used,
                        margin_free=margin_free, margin_level=lvl)


def _service(target, *, dry_run=False, max_lots=0.1, limits=None, broker=None,
             instrument=None):
    inst = instrument or get_instrument("EURUSD")
    broker = broker or FakeBroker({"EURUSD": _bars()}, {"EURUSD": inst}, _account())
    risk = RiskManager(AccountConfig(leverage=100), limits or RiskLimits())
    cfg = ExecutionConfig(Timeframe.M15, lookback_bars=300, max_lots=max_lots, dry_run=dry_run)
    svc = ExecutionService(broker, {"EURUSD": FixedTarget(target)}, risk, cfg,
                           now_fn=lambda: FIXED_NOW)
    return svc, broker, risk


# --------------------------------------------------------------- conciliação
def test_reconcile_kinds():
    inst = get_instrument("EURUSD")
    assert reconcile("X", 0.1, 0.0, inst).kind == "open"
    assert reconcile("X", 0.0, 0.1, inst).kind == "close"
    assert reconcile("X", 0.2, 0.1, inst).kind == "increase"
    assert reconcile("X", 0.1, 0.2, inst).kind == "reduce"
    assert reconcile("X", -0.1, 0.1, inst).kind == "reverse"
    assert reconcile("X", 0.1, 0.1, inst).is_noop


# --------------------------------------------------------------- event-driven
def test_acts_only_on_new_bar():
    svc, broker, _ = _service(1.0, dry_run=True)
    first = svc.on_bar("EURUSD")
    assert first.new_bar is True
    again = svc.on_bar("EURUSD")
    assert again.new_bar is False          # mesma barra → ocioso
    broker.advance("EURUSD")
    third = svc.on_bar("EURUSD")
    assert third.new_bar is True           # nova barra fechada → processa


def test_no_duplicate_order_when_in_sync():
    svc, broker, _ = _service(1.0, dry_run=False, max_lots=0.1)
    broker.set_position("EURUSD", 0.1)     # já está no alvo (0.1 lote)
    out = svc.on_bar("EURUSD")
    assert out.action == "none"
    assert broker.orders == []             # nenhuma ordem duplicada


def test_opens_position_to_reach_target():
    svc, broker, _ = _service(1.0, dry_run=False, max_lots=0.1)
    out = svc.on_bar("EURUSD")
    assert out.sent and out.action == "open"
    assert broker._positions["EURUSD"] == pytest.approx(0.1)
    assert broker.orders == [("EURUSD", pytest.approx(0.1))]


def test_reconciles_orphan_position_to_flat():
    # Algo quer ficar FLAT (target 0) mas há posição órfã na corretora.
    svc, broker, _ = _service(0.0, dry_run=False)
    broker.set_position("EURUSD", 0.3)
    out = svc.on_bar("EURUSD")
    assert out.action == "close"
    assert broker._positions["EURUSD"] == pytest.approx(0.0)


def test_dry_run_sends_no_orders():
    svc, broker, _ = _service(1.0, dry_run=True)
    out = svc.on_bar("EURUSD")
    assert out.action == "open" and out.sent is False
    assert broker.orders == []


# --------------------------------------------------------------- kill-switch
def test_kill_switch_on_daily_loss_flattens_and_blocks():
    svc, broker, risk = _service(1.0, dry_run=False,
                                 limits=RiskLimits(daily_max_loss_pct=0.05))
    # Barra 1: estabelece o equity de início do dia (10k) e abre posição.
    svc.on_bar("EURUSD")
    assert broker._positions["EURUSD"] != 0.0
    # Barra 2: equity despenca 10% (> limite de 5%) → kill-switch.
    broker.advance("EURUSD")
    broker.set_account(_account(equity=9_000.0))
    out = svc.on_bar("EURUSD")
    assert risk.tripped and "perda diária" in risk.state.trip_reason
    assert "kill-switch" in out.blocked
    assert broker._positions["EURUSD"] == pytest.approx(0.0)   # carteira zerada
    # Barra 3: continua bloqueado (não reabre).
    broker.advance("EURUSD")
    svc.on_bar("EURUSD")
    assert broker._positions["EURUSD"] == pytest.approx(0.0)


def test_kill_switch_on_connection_instability():
    svc, broker, risk = _service(1.0, dry_run=False,
                                 limits=RiskLimits(max_consecutive_errors=3))
    broker.connected = False
    broker.connect_fails = True             # reconexão persistentemente falha
    for _ in range(3):
        svc.step()                          # cada step sem conexão = 1 erro
    assert risk.tripped and "conexão" in risk.state.trip_reason


def test_reset_rearms_after_trip():
    _, _, risk = _service(1.0)
    risk.trip("teste")
    assert risk.tripped
    risk.reset()
    assert not risk.tripped


# --------------------------------------------------------------- margem
def test_margin_block_prevents_opening():
    inst = Instrument("EURUSD")  # base EUR → margem depende do preço
    broker = FakeBroker({"EURUSD": _bars()}, {"EURUSD": inst},
                        _account(equity=10_000, margin_free=0.0))  # sem margem livre
    svc, _, risk = _service(1.0, dry_run=False, broker=broker, instrument=inst)
    out = svc.on_bar("EURUSD")
    assert out.blocked == "margem"
    assert broker.orders == []                # ordem de abertura bloqueada


def test_margin_allows_reduction_even_without_free_margin():
    inst = Instrument("EURUSD")
    broker = FakeBroker({"EURUSD": _bars()}, {"EURUSD": inst},
                        _account(equity=10_000, margin_free=0.0))
    broker.set_position("EURUSD", 0.5)        # posição grande aberta
    # Alvo menor (reduzir) deve passar mesmo sem margem livre.
    svc, _, _ = _service(0.1, dry_run=False, broker=broker, instrument=inst, max_lots=0.1)
    out = svc.on_bar("EURUSD")
    assert out.action == "reduce" and out.sent
