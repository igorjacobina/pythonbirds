"""Camada de execução ao vivo — loop event-driven, conciliação e kill-switch."""
from innova_ea.execution.broker import (
    AccountState,
    Broker,
    BrokerError,
    OrderResult,
    Position,
)
from innova_ea.execution.reconciliation import ReconcileAction, reconcile
from innova_ea.execution.risk import RiskLimits, RiskManager, RiskState
from innova_ea.execution.service import (
    BarOutcome,
    ExecutionConfig,
    ExecutionService,
    build_model_strategies,
)

__all__ = [
    "Broker",
    "BrokerError",
    "AccountState",
    "Position",
    "OrderResult",
    "reconcile",
    "ReconcileAction",
    "RiskManager",
    "RiskLimits",
    "RiskState",
    "ExecutionService",
    "ExecutionConfig",
    "BarOutcome",
    "build_model_strategies",
    "get_mt5_broker",
]


def get_mt5_broker(**kwargs):
    """Instancia o ``MT5Broker`` sob demanda (Windows + terminal MT5)."""
    from innova_ea.execution.mt5_broker import MT5Broker

    return MT5Broker(**kwargs)
