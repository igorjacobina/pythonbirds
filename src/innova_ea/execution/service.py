"""Serviço de Execução Live — loop event-driven a cada barra fechada.

Orquestra o ciclo institucional em produção:

    nova barra fechada → features + ModelStrategy → alvo de posição
        → risco/kill-switch → CONCILIAÇÃO contra a corretora → ordem (delta)
        → auditoria do fill

Princípios:
  * **Event-driven**: só age quando uma NOVA barra fecha (compara o timestamp da
    última barra fechada); entre barras, fica ocioso (latência mínima por ciclo).
  * **Conciliação implacável**: a posição real na corretora é a verdade; a ordem
    é sempre o delta até o alvo → sem duplicidade, sem órfãs.
  * **Segurança por padrão**: ``dry_run=True`` (paper) — não envia ordens até ser
    explicitamente desligado. Kill-switch zera a carteira e bloqueia entradas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument
from innova_ea.execution.broker import Broker, BrokerError, OrderResult
from innova_ea.execution.reconciliation import ReconcileAction, reconcile
from innova_ea.execution.risk import RiskManager
from innova_ea.strategy.base import Strategy

logger = logging.getLogger("innova_ea.execution")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    timeframe: Timeframe
    lookback_bars: int = 300       # janela passada p/ features (>= warmup + janela)
    max_lots: float = 0.1          # lotes correspondentes a target = ±1.0
    poll_seconds: float = 5.0
    dry_run: bool = True           # paper mode (não envia ordens) — padrão seguro


@dataclass(frozen=True, slots=True)
class BarOutcome:
    symbol: str
    new_bar: bool
    desired_lots: float = 0.0
    actual_lots: float = 0.0
    action: str = "none"
    sent: bool = False
    blocked: str = ""              # motivo do bloqueio (margem/kill-switch), se houver


class ExecutionService:
    """Serviço de execução plugável (um modelo treinado + travas de risco)."""

    def __init__(
        self,
        broker: Broker,
        strategies: dict[str, Strategy],
        risk: RiskManager,
        config: ExecutionConfig,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.broker = broker
        self.strategies = strategies
        self.risk = risk
        self.cfg = config
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._last_bar: dict[str, datetime] = {}
        self._stop = False

    # ------------------------------------------------------------- utilitários
    def _round_signed(self, inst: Instrument, lots: float) -> float:
        if abs(lots) < inst.min_lot / 2.0:
            return 0.0
        return (1.0 if lots > 0 else -1.0) * inst.round_lots(abs(lots))

    def _flatten_all(self) -> None:
        """Kill-switch: zera todas as posições abertas (fecha o delta até 0)."""
        for symbol, pos in self.broker.open_positions().items():
            if pos.lots == 0.0:
                continue
            if self.cfg.dry_run:
                logger.warning("[DRY] flatten %s lots=%.2f", symbol, pos.lots)
            else:
                self.broker.send_market_order(symbol, -pos.lots)

    def _audit_fill(self, symbol: str, desired: float, inst: Instrument) -> None:
        """Auditoria pós-ordem: relê a posição e alerta se divergir do alvo."""
        pos = self.broker.position(symbol)
        actual = pos.lots if pos else 0.0
        if abs(actual - desired) >= inst.lot_step:
            logger.error("DESSINCRONIA %s: alvo=%.2f, real=%.2f após ordem",
                         symbol, desired, actual)

    # --------------------------------------------------------------- por barra
    def on_bar(self, symbol: str) -> BarOutcome:
        """Processa um símbolo; age apenas se há uma NOVA barra fechada."""
        inst = self.broker.instrument(symbol)
        bars = self.broker.recent_closed_bars(symbol, self.cfg.timeframe, self.cfg.lookback_bars)
        if bars.height == 0:
            return BarOutcome(symbol, new_bar=False)

        last_t = bars["time"][-1]
        if self._last_bar.get(symbol) == last_t:
            return BarOutcome(symbol, new_bar=False)   # nada novo → ocioso
        self._last_bar[symbol] = last_t

        account = self.broker.account_state()
        self.risk.on_account(account, self._now())
        if self.risk.tripped:
            self._flatten_all()
            return BarOutcome(symbol, new_bar=True, blocked=f"kill-switch: {self.risk.state.trip_reason}")

        target = float(self.strategies[symbol].generate_targets(bars, inst)[-1])
        desired = self._round_signed(inst, target * self.cfg.max_lots)
        pos = self.broker.position(symbol)
        actual = pos.lots if pos else 0.0

        action: ReconcileAction = reconcile(symbol, desired, actual, inst)
        if action.is_noop:
            return BarOutcome(symbol, True, desired, actual, "none")  # já sincronizado

        price = float(bars["close"][-1])
        if action.increases_exposure and not self.risk.can_open(inst, desired, actual, price, account):
            logger.warning("ordem bloqueada por margem: %s alvo=%.2f", symbol, desired)
            return BarOutcome(symbol, True, desired, actual, action.kind, blocked="margem")

        if self.cfg.dry_run:
            logger.info("[DRY] %s %s delta=%.2f (alvo=%.2f, real=%.2f)",
                        symbol, action.kind, action.delta_lots, desired, actual)
            return BarOutcome(symbol, True, desired, actual, action.kind, sent=False)

        result: OrderResult = self.broker.send_market_order(symbol, action.delta_lots)
        if not result.ok:
            raise BrokerError(f"ordem rejeitada {symbol}: {result.message}")
        self._audit_fill(symbol, desired, inst)
        return BarOutcome(symbol, True, desired, actual, action.kind, sent=True)

    # --------------------------------------------------------------- ciclo
    def step(self) -> list[BarOutcome]:
        """Uma varredura de todos os símbolos (chamada pelo ``run``)."""
        if not self.broker.is_connected():
            self.risk.on_error()
            try:
                self.broker.connect()
            except BrokerError:
                logger.error("falha ao reconectar à corretora")
            if self.risk.tripped:
                self._flatten_all()
            return []

        outcomes: list[BarOutcome] = []
        for symbol in self.strategies:
            try:
                outcomes.append(self.on_bar(symbol))
                self.risk.on_ok()
            except BrokerError as exc:
                self.risk.on_error()
                logger.error("erro em %s: %s", symbol, exc)
        if self.risk.tripped:
            self._flatten_all()
        return outcomes

    def run(self) -> None:  # pragma: no cover - loop de produção
        import time

        self._stop = False
        if not self.broker.is_connected():
            self.broker.connect()
        logger.info("execução iniciada (dry_run=%s) — %d símbolos",
                    self.cfg.dry_run, len(self.strategies))
        while not self._stop:
            self.step()
            time.sleep(self.cfg.poll_seconds)

    def stop(self) -> None:
        self._stop = True


def build_model_strategies(
    model,
    feature_set,
    symbols: list[str],
    asset_class_of: Callable[[str], str],
    *,
    threshold: float = 0.15,
    warmup: int = 150,
) -> dict[str, Strategy]:
    """Cria uma ``ModelStrategy`` por símbolo a partir de UM modelo universal."""
    from innova_ea.strategy.model_strategy import ModelStrategy

    return {
        s: ModelStrategy(model, feature_set, s, asset_class_of(s),
                         threshold=threshold, warmup=warmup)
        for s in symbols
    }
