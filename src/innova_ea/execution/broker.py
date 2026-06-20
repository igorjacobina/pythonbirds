"""Abstração de corretora — contrato que o serviço de execução usa.

O MetaTrader 5 implementa esta interface em produção (Windows); um broker falso a
implementa nos testes (Linux/CI). Assim, toda a lógica crítica de execução
(loop, conciliação, kill-switch) é validada sem um terminal real.

Convenção de posição: modo NETTING (uma posição líquida por símbolo, com sinal
positivo = comprado, negativo = vendido) — coerente com o engine de backtest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument


class BrokerError(Exception):
    """Falha de comunicação/execução com a corretora."""


@dataclass(frozen=True, slots=True)
class AccountState:
    """Fotografia da conta (na moeda do depósito)."""

    balance: float
    equity: float
    margin_used: float
    margin_free: float
    margin_level: float  # equity/margin_used (fração); inf se sem posição

    @property
    def has_positions(self) -> bool:
        return self.margin_used > 0.0


@dataclass(frozen=True, slots=True)
class Position:
    """Posição líquida de um símbolo (lots com sinal)."""

    symbol: str
    lots: float          # + comprado, - vendido, 0 = flat
    avg_price: float = 0.0
    profit: float = 0.0


@dataclass(frozen=True, slots=True)
class OrderResult:
    ok: bool
    symbol: str
    delta_lots: float    # volume negociado com sinal (+ compra, - venda)
    price: float = 0.0
    message: str = ""


@dataclass(frozen=True, slots=True)
class PendingOrder:
    """Ordem pendente (stop) — usada pelo straddle de rompimento."""

    ticket: int
    symbol: str
    side: int            # +1 buy-stop, -1 sell-stop
    price: float         # nível de disparo
    lots: float
    sl: float = 0.0      # stop loss
    tp: float = 0.0      # take profit



class Broker(ABC):
    """Interface mínima exigida pelo ``ExecutionService``."""

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def instrument(self, symbol: str) -> Instrument:
        """Especificação do símbolo (fonte de verdade do broker: point, lote, margem)."""

    @abstractmethod
    def account_state(self) -> AccountState: ...

    @abstractmethod
    def position(self, symbol: str) -> Position | None:
        """Posição líquida atual do símbolo (None se flat)."""

    @abstractmethod
    def open_positions(self) -> dict[str, Position]:
        """Todas as posições abertas (para conciliação/órfãs)."""

    @abstractmethod
    def recent_closed_bars(self, symbol: str, timeframe: Timeframe, count: int) -> pl.DataFrame:
        """Últimas ``count`` barras FECHADAS (schema canônico, UTC) — sem a barra em formação."""

    @abstractmethod
    def send_market_order(self, symbol: str, delta_lots: float) -> OrderResult:
        """Executa a mercado: ``delta_lots`` > 0 compra, < 0 vende. 0 é no-op."""
