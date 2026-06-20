"""Configuração da conta de negociação — parâmetros de margem da corretora.

Reproduz o modelo do MetaTrader 5: alavancagem, nível de margin call e nível de
stop-out. Estes parâmetros governam o gating de ordens (bloqueio por margem
livre insuficiente) e a liquidação forçada (stop-out) durante o backtest.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccountConfig:
    """Parâmetros de conta/margem no estilo MT5.

    Attributes:
        leverage: Alavancagem da conta (ex. 100 para 1:100, 500 para 1:500).
        account_currency: Moeda do depósito (define a conversão de margem/P&L).
        margin_call_level: Nível de margem (fração) abaixo do qual NOVAS ordens
            de aumento de exposição são bloqueadas. 1.0 = 100%.
        stop_out_level: Nível de margem (fração) no qual a corretora liquida
            posições à força. 0.5 = 50% (default típico de varejo).
        hedging: Se False (modo netting do MT5), posições no mesmo símbolo se
            compensam — comportamento já assumido pelo engine.
    """

    leverage: float = 100.0
    account_currency: str = "USD"
    margin_call_level: float = 1.0   # 100%
    stop_out_level: float = 0.5      # 50%
    hedging: bool = False

    def __post_init__(self) -> None:
        if self.leverage <= 0:
            raise ValueError("leverage deve ser > 0")
        if not (0.0 <= self.stop_out_level <= self.margin_call_level):
            raise ValueError(
                "exige-se 0 <= stop_out_level <= margin_call_level"
            )
