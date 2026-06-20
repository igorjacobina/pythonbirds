"""Serviço de execução do STRADDLE ao vivo (Fase 6) — direção-agnóstico.

A cada barra fechada, se o meta-modelo prevê P(rompimento lucrativo) ≥ limiar e o
símbolo está livre, arma um straddle: **buy-stop** acima do range + **sell-stop**
abaixo, cada um com SL/TP embutidos. Gerencia OCO (a perna não-acionada é
cancelada quando a outra dispara) e a expiração (cancela ambas se ninguém romper
dentro do horizonte). Sob as mesmas travas de margem + kill-switch da Fase 3/5.

Event-driven e *paper mode* (``dry_run=True``) por padrão.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.execution.risk import RiskManager

logger = logging.getLogger("innova_ea.execution.straddle")


@dataclass(frozen=True, slots=True)
class StraddleConfig:
    timeframe: Timeframe
    lookback_bars: int = 300
    band_lookback: int = 12
    horizon: int = 12            # nº de barras até expirar o straddle não-acionado
    target_mult: float = 2.5
    stop_mult: float = 2.0
    atr_window: int = 24
    threshold: float = 0.50
    lots: float = 0.1
    poll_seconds: float = 30.0
    dry_run: bool = True


@dataclass(frozen=True, slots=True)
class StraddleOutcome:
    symbol: str
    state: str            # idle | no_signal | armed | armed_waiting | in_position |
                          # expired_cancelled | killswitch
    pwin: float = float("nan")
    band_hi: float = 0.0
    band_lo: float = 0.0


class StraddleExecutionService:
    """Executa o straddle de volatilidade ao vivo (modelo meta-rotulado)."""

    def __init__(
        self,
        broker,
        model,                       # MetaLabelModel treinado p/ rótulo de straddle
        feature_set,
        symbols: list[str],
        asset_class_of: Callable[[str], str],
        risk: RiskManager,
        config: StraddleConfig,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.broker = broker
        self.model = model
        self.feature_set = feature_set
        self.symbols = symbols
        self.asset_class_of = asset_class_of
        self.risk = risk
        self.cfg = config
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._last_bar: dict[str, datetime] = {}
        self._armed_until: dict[str, datetime] = {}
        self._stop = False

    # ----------------------------------------------------------- sinais/níveis
    def _predict(self, symbol: str, bars: pl.DataFrame) -> float:
        feats = self.feature_set.transform(bars).with_columns(
            [pl.col(c).fill_null(0) for c in self.model.feature_names]
        ).with_columns(
            pl.lit(self.model.asset_vocab.index(symbol), dtype=pl.Int32).alias("asset_id"),
            pl.lit(self.model.class_vocab.index(self.asset_class_of(symbol)),
                   dtype=pl.Int32).alias("asset_class_id"),
        )
        return float(self.model.predict_meta(feats)[-1])

    def _levels(self, bars: pl.DataFrame) -> tuple[float, float, float]:
        bl, aw = self.cfg.band_lookback, self.cfg.atr_window
        band_hi = float(bars["high"].tail(bl).max())
        band_lo = float(bars["low"].tail(bl).min())
        atr = float((bars["high"] - bars["low"]).tail(aw).mean())
        return band_hi, band_lo, atr

    def _cancel_all(self, symbol: str) -> None:
        for p in self.broker.pending_orders(symbol):
            self.broker.cancel_order(p.ticket)

    # --------------------------------------------------------------- por barra
    def on_bar(self, symbol: str) -> StraddleOutcome:
        bars = self.broker.recent_closed_bars(symbol, self.cfg.timeframe, self.cfg.lookback_bars)
        if bars.height == 0:
            return StraddleOutcome(symbol, "idle")
        last_t = bars["time"][-1]
        new_bar = self._last_bar.get(symbol) != last_t
        self._last_bar[symbol] = last_t

        self.risk.on_account(self.broker.account_state(), self._now())
        pos = self.broker.position(symbol)
        pendings = self.broker.pending_orders(symbol)

        # Kill-switch: cancela pendentes, zera posição, bloqueia.
        if self.risk.tripped:
            self._cancel_all(symbol)
            if pos and pos.lots != 0.0 and not self.cfg.dry_run:
                self.broker.send_market_order(symbol, -pos.lots)
            self._armed_until.pop(symbol, None)
            return StraddleOutcome(symbol, "killswitch")

        # Em posição: uma perna disparou → cancela a outra (OCO). SL/TP cuidam da saída.
        if pos and pos.lots != 0.0:
            if pendings:
                self._cancel_all(symbol)
            self._armed_until.pop(symbol, None)
            return StraddleOutcome(symbol, "in_position")

        # Armado e aguardando rompimento: expira se passou o horizonte.
        if pendings:
            expiry = self._armed_until.get(symbol)
            if expiry is not None and last_t >= expiry:
                self._cancel_all(symbol)
                self._armed_until.pop(symbol, None)
                return StraddleOutcome(symbol, "expired_cancelled")
            return StraddleOutcome(symbol, "armed_waiting")

        # Livre: arma só em barra nova e com sinal.
        if not new_bar:
            return StraddleOutcome(symbol, "idle")
        pwin = self._predict(symbol, bars)
        if pwin < self.cfg.threshold:
            return StraddleOutcome(symbol, "no_signal", pwin=pwin)

        band_hi, band_lo, atr = self._levels(bars)
        if not (band_hi > 0 and band_lo > 0 and atr > 0):
            return StraddleOutcome(symbol, "idle", pwin=pwin)
        tgt = self.cfg.target_mult * atr
        stp = self.cfg.stop_mult * atr
        lots = self.cfg.lots
        self._armed_until[symbol] = last_t + timedelta(minutes=self.cfg.horizon * self.cfg.timeframe.minutes)

        if self.cfg.dry_run:
            logger.info("[DRY] %s arma straddle | buy-stop %.5f (sl %.5f tp %.5f) | "
                        "sell-stop %.5f (sl %.5f tp %.5f) | P=%.2f",
                        symbol, band_hi, band_hi - stp, band_hi + tgt,
                        band_lo, band_lo + stp, band_lo - tgt, pwin)
            return StraddleOutcome(symbol, "armed", pwin, band_hi, band_lo)

        self.broker.place_stop(symbol, +1, band_hi, lots, band_hi - stp, band_hi + tgt)
        self.broker.place_stop(symbol, -1, band_lo, lots, band_lo + stp, band_lo - tgt)
        return StraddleOutcome(symbol, "armed", pwin, band_hi, band_lo)

    # --------------------------------------------------------------- ciclo
    def step(self) -> list[StraddleOutcome]:
        from innova_ea.execution.broker import BrokerError

        if not self.broker.is_connected():
            self.risk.on_error()
            try:
                self.broker.connect()
            except BrokerError:
                logger.error("falha ao reconectar")
            return []
        out = []
        for s in self.symbols:
            try:
                out.append(self.on_bar(s))
                self.risk.on_ok()
            except BrokerError as exc:
                self.risk.on_error()
                logger.error("erro em %s: %s", s, exc)
        return out

    def run(self) -> None:  # pragma: no cover - loop de produção
        import time
        self._stop = False
        if not self.broker.is_connected():
            self.broker.connect()
        logger.info("execução de straddle iniciada (dry_run=%s) — %d símbolos",
                    self.cfg.dry_run, len(self.symbols))
        while not self._stop:
            self.step()
            time.sleep(self.cfg.poll_seconds)

    def stop(self) -> None:
        self._stop = True
