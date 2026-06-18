"""Testes do serviço de execução de straddle (broker falso, sem MT5)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from innova_ea.backtest.account import AccountConfig
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import get_instrument
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.execution.broker import AccountState, PendingOrder, Position
from innova_ea.execution.risk import RiskLimits, RiskManager
from innova_ea.execution.straddle_service import StraddleConfig, StraddleExecutionService
from innova_ea.features import CloseLocationValue, EfficiencyRatio, FeatureSet

FIXED_NOW = datetime(2021, 6, 1, 12, tzinfo=timezone.utc)


class StubMeta:
    """Meta-modelo falso: P(lucro) constante."""

    def __init__(self, p, feature_names):
        self.p = p
        self.feature_names = feature_names
        self.asset_vocab = ["EURUSD"]
        self.class_vocab = ["forex"]

    def predict_meta(self, df):
        return np.full(df.height, self.p)


class FakeStraddleBroker:
    def __init__(self, bars):
        self._bars = bars
        self._positions: dict[str, float] = {}
        self._pending: dict[str, list[PendingOrder]] = {}
        self.placed: list[PendingOrder] = []
        self.cancelled: list[int] = []
        self.market: list[tuple[str, float]] = []
        self._cursor = 100
        self._ticket = 0
        self.connected = True

    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def account_state(self):
        return AccountState(10_000, 10_000, 0.0, 10_000, float("inf"))

    def position(self, symbol):
        lots = self._positions.get(symbol, 0.0)
        return Position(symbol, lots) if lots != 0.0 else None

    def open_positions(self):
        return {s: Position(s, l) for s, l in self._positions.items() if l != 0.0}

    def pending_orders(self, symbol):
        return list(self._pending.get(symbol, []))

    def recent_closed_bars(self, symbol, tf, count):
        return self._bars.head(self._cursor).tail(count)

    def place_stop(self, symbol, side, price, lots, sl, tp):
        self._ticket += 1
        po = PendingOrder(self._ticket, symbol, side, price, lots, sl, tp)
        self._pending.setdefault(symbol, []).append(po)
        self.placed.append(po)
        return self._ticket

    def cancel_order(self, ticket):
        self.cancelled.append(ticket)
        for s in self._pending:
            self._pending[s] = [p for p in self._pending[s] if p.ticket != ticket]

    def send_market_order(self, symbol, delta):
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta
        self.market.append((symbol, delta))

    def advance(self, n=1):
        self._cursor += n


def _bars():
    return SyntheticSource(seed=1, annual_vol=0.1).fetch(
        "EURUSD", Timeframe.H1,
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 20, tzinfo=timezone.utc))


def _service(p, *, dry_run=False, threshold=0.5, horizon=12, limits=None):
    fs = FeatureSet([EfficiencyRatio(20), CloseLocationValue()])
    model = StubMeta(p, fs.names)
    broker = FakeStraddleBroker(_bars())
    risk = RiskManager(AccountConfig(leverage=100), limits or RiskLimits())
    cfg = StraddleConfig(Timeframe.H1, lookback_bars=200, horizon=horizon,
                         threshold=threshold, dry_run=dry_run)
    svc = StraddleExecutionService(broker, model, fs, ["EURUSD"],
                                   lambda s: "forex", risk, cfg, now_fn=lambda: FIXED_NOW)
    return svc, broker, risk


def test_arms_straddle_on_signal():
    svc, broker, _ = _service(0.8)
    out = svc.on_bar("EURUSD")
    assert out.state == "armed"
    assert len(broker.placed) == 2                       # buy-stop + sell-stop
    sides = sorted(p.side for p in broker.placed)
    assert sides == [-1, 1]
    # buy-stop acima, sell-stop abaixo.
    buy = next(p for p in broker.placed if p.side == 1)
    sell = next(p for p in broker.placed if p.side == -1)
    assert buy.price > sell.price
    assert buy.tp > buy.price > buy.sl                   # TP acima, SL abaixo
    assert sell.tp < sell.price < sell.sl


def test_no_signal_no_orders():
    svc, broker, _ = _service(0.3, threshold=0.5)
    out = svc.on_bar("EURUSD")
    assert out.state == "no_signal"
    assert broker.placed == []


def test_dry_run_places_no_orders():
    svc, broker, _ = _service(0.9, dry_run=True)
    out = svc.on_bar("EURUSD")
    assert out.state == "armed" and broker.placed == []


def test_oco_cancels_remaining_leg_when_filled():
    svc, broker, _ = _service(0.9)
    svc.on_bar("EURUSD")                                  # arma 2 pendentes
    assert len(broker._pending["EURUSD"]) == 2
    # Simula a perna de COMPRA disparando → vira posição; resta a venda pendente.
    broker._positions["EURUSD"] = 0.1
    broker._pending["EURUSD"] = [p for p in broker._pending["EURUSD"] if p.side == -1]
    broker.advance()
    out = svc.on_bar("EURUSD")
    assert out.state == "in_position"
    assert broker._pending["EURUSD"] == []               # OCO cancelou a outra perna


def test_expiry_cancels_both_when_no_breakout():
    svc, broker, _ = _service(0.9, horizon=12)
    svc.on_bar("EURUSD")                                  # arma; expira em 12h
    assert len(broker._pending["EURUSD"]) == 2
    broker.advance(13)                                    # passou o horizonte, sem fill
    out = svc.on_bar("EURUSD")
    assert out.state == "expired_cancelled"
    assert broker._pending["EURUSD"] == []
    assert len(broker.cancelled) == 2


def test_killswitch_cancels_and_flattens():
    svc, broker, risk = _service(0.9)
    svc.on_bar("EURUSD")                                  # arma
    broker._positions["EURUSD"] = 0.1                     # finge posição aberta
    risk.trip("teste")
    broker.advance()
    out = svc.on_bar("EURUSD")
    assert out.state == "killswitch"
    assert broker._pending["EURUSD"] == []               # pendentes canceladas
    assert ("EURUSD", -0.1) in broker.market             # carteira zerada
