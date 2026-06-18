"""Gestão de risco ao vivo + Kill-Switch global.

Traz as travas da Fase 3 (margem/alavancagem estilo MT5) para a execução real e
adiciona um desarme de emergência:

  * **Kill-Switch**: bloqueia novas posições e ZERA a carteira quando a perda
    diária atinge o limite OU quando há instabilidade crítica de conexão (erros
    consecutivos). Permanece armado até reset manual/rollover de dia.
  * **Margem**: antes de abrir/aumentar, exige que a margem livre cubra a
    margem adicional (respeitando o nível de margin call).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from innova_ea.backtest.account import AccountConfig
from innova_ea.core.instruments import Instrument
from innova_ea.execution.broker import AccountState


@dataclass(frozen=True, slots=True)
class RiskLimits:
    daily_max_loss_pct: float = 0.05        # 5% do equity de início do dia
    max_consecutive_errors: int = 5         # erros seguidos → instabilidade crítica
    margin_buffer: float = 1.0              # multiplicador sobre a margem exigida


@dataclass
class RiskState:
    tripped: bool = False
    trip_reason: str = ""
    day: date | None = None
    day_start_equity: float = 0.0
    consecutive_errors: int = 0
    daily_pnl: float = 0.0


class RiskManager:
    """Avalia permissões de risco e administra o kill-switch."""

    def __init__(
        self,
        account_cfg: AccountConfig,
        limits: RiskLimits | None = None,
    ) -> None:
        self.account_cfg = account_cfg
        self.limits = limits or RiskLimits()
        self.state = RiskState()

    # ---------------------------------------------------------- ciclo diário
    def on_account(self, account: AccountState, now: datetime) -> None:
        """Atualiza P&L diário e dispara kill-switch se a perda exceder o limite."""
        today = now.date()
        if self.state.day != today:
            # Novo dia (UTC): rebaseia e REARMA (libera o robô para o pregão).
            self.state.day = today
            self.state.day_start_equity = account.equity
            self.state.daily_pnl = 0.0
            self.state.consecutive_errors = 0
            self.state.tripped = False
            self.state.trip_reason = ""

        self.state.daily_pnl = account.equity - self.state.day_start_equity
        max_loss = self.limits.daily_max_loss_pct * max(self.state.day_start_equity, 1e-9)
        if self.state.daily_pnl <= -max_loss:
            self.trip(f"perda diária {self.state.daily_pnl:.2f} excedeu limite {-max_loss:.2f}")

    # ---------------------------------------------------------- saúde de conexão
    def on_error(self) -> None:
        self.state.consecutive_errors += 1
        if self.state.consecutive_errors >= self.limits.max_consecutive_errors:
            self.trip(f"{self.state.consecutive_errors} erros consecutivos (conexão instável)")

    def on_ok(self) -> None:
        self.state.consecutive_errors = 0

    # ---------------------------------------------------------- kill-switch
    def trip(self, reason: str) -> None:
        if not self.state.tripped:
            self.state.tripped = True
            self.state.trip_reason = reason

    def reset(self) -> None:
        self.state.tripped = False
        self.state.trip_reason = ""
        self.state.consecutive_errors = 0

    @property
    def tripped(self) -> bool:
        return self.state.tripped

    # ---------------------------------------------------------- margem (Fase 3)
    def can_open(
        self,
        inst: Instrument,
        desired_lots: float,
        actual_lots: float,
        price: float,
        account: AccountState,
    ) -> bool:
        """True se há margem livre para AUMENTAR a exposição até ``desired_lots``.

        Reduções/fechamentos liberam margem e são sempre permitidos (não passam
        por aqui). Usa a fórmula de margem MT5 do ``Instrument``.
        """
        if self.tripped:
            return False
        lev = self.account_cfg.leverage
        ccy = self.account_cfg.account_currency
        req_desired = inst.margin_required(abs(desired_lots), price, lev, ccy)
        req_actual = inst.margin_required(abs(actual_lots), price, lev, ccy)
        additional = max(0.0, req_desired - req_actual)
        if additional <= 0.0:
            return True
        return account.margin_free >= additional * self.limits.margin_buffer
