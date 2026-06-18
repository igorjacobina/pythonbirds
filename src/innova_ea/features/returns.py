"""Features baseadas em retornos — impulso normalizado, sem suavização atrasada."""
from __future__ import annotations

import polars as pl

from innova_ea.features.base import ExprFeature, Feature


def log_return_expr(period: int = 1) -> pl.Expr:
    """Log-retorno em ``period`` barras: ln(close_t / close_{t-period})."""
    return (pl.col("close").log() - pl.col("close").shift(period).log())


class LogReturn(ExprFeature):
    """Log-retorno simples de ``period`` barras."""

    def __init__(self, period: int = 1, name: str | None = None) -> None:
        super().__init__(name or f"logret_{period}", log_return_expr(period))


class ReturnZScore(Feature):
    """Z-score do retorno: r_t / desvio-padrão móvel de r — impulso normalizado.

    Mede quão "anormal" é o movimento atual frente à sua volatilidade recente,
    sem o atraso de uma média móvel. Útil para detectar distorções/expansões.
    """

    def __init__(self, window: int = 50, period: int = 1, name: str | None = None) -> None:
        self.window = window
        self.period = period
        self._name = name or f"ret_z_{period}_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        r = log_return_expr(self.period)
        z = r / r.rolling_std(self.window)
        out = bars.select(z.alias(self._name))
        return self._check(out, bars.height)


class CumulativeReturn(ExprFeature):
    """Retorno acumulado nas últimas ``window`` barras (deslocamento líquido)."""

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        expr = pl.col("close").log() - pl.col("close").shift(window).log()
        super().__init__(name or f"cumret_{window}", expr)
