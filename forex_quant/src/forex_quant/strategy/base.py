"""Contrato de estratégia — projetado para impedir look-ahead por construção.

Uma estratégia produz, para cada barra, um *target de posição* normalizado em
[-1, 1] (fração do risco máximo): +1 comprado pleno, -1 vendido pleno, 0 fora.
O engine garante que o alvo decidido a partir das barras até ``t`` só é
executado na abertura de ``t+1`` — nunca com informação da própria vela.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from forex_quant.core.instruments import Instrument


class Strategy(ABC):
    """Base para estratégias vetorizadas.

    A implementação recebe o histórico completo de barras e devolve um array de
    targets alinhado às barras. O engine aplica o lag de 1 barra; a estratégia
    NÃO deve tentar antecipar isso.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_targets(self, bars: pl.DataFrame, inst: Instrument) -> np.ndarray:
        """Retorna um array float em [-1, 1] com len == bars.height.

        target[i] é a posição desejada considerando informação disponível
        *até o fechamento da barra i*. O engine a executará em i+1.
        """


class SignalFromColumn(Strategy):
    """Estratégia trivial de teste: usa uma coluna pré-computada como target."""

    def __init__(self, column: str, name: str = "from_column") -> None:
        self.column = column
        self.name = name

    def generate_targets(self, bars: pl.DataFrame, inst: Instrument) -> np.ndarray:
        return bars[self.column].cast(pl.Float64).to_numpy().clip(-1.0, 1.0)


class MovingAverageCrossover(Strategy):
    """Cruzamento de médias móveis — exemplo canônico, totalmente vetorizado.

    Compra quando a MA rápida cruza acima da lenta; vende no contrário. Serve de
    baseline honesto (com custos) para comparar estratégias mais sofisticadas.
    """

    def __init__(self, fast: int = 20, slow: int = 50, name: str | None = None) -> None:
        if fast >= slow:
            raise ValueError("fast deve ser < slow")
        self.fast = fast
        self.slow = slow
        self.name = name or f"ma_cross_{fast}_{slow}"

    def generate_targets(self, bars: pl.DataFrame, inst: Instrument) -> np.ndarray:
        fast = bars["close"].rolling_mean(self.fast)
        slow = bars["close"].rolling_mean(self.slow)
        target = (
            pl.when(fast > slow).then(1.0)
            .when(fast < slow).then(-1.0)
            .otherwise(0.0)
        )
        out = bars.select(target.alias("t"))["t"].to_numpy().copy()
        # Período de aquecimento das médias → fora do mercado.
        out[: self.slow] = 0.0
        return out
