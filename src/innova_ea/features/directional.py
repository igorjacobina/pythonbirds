"""Features de direção e microestrutura — tendência vs. ruído, pressão, exaustão.

Capturam a *qualidade* do movimento (não apenas a magnitude): se o deslocamento
é eficiente (tendência) ou errático (ruído), onde o preço fecha dentro da barra
(pressão compradora/vendedora) e a anatomia de corpo/pavios (exaustão).
"""
from __future__ import annotations

import polars as pl

from innova_ea.features.base import ExprFeature, Feature

# Range protegido contra divisão por zero (barras de range nulo).
_RANGE = pl.when((pl.col("high") - pl.col("low")) > 0).then(
    pl.col("high") - pl.col("low")
).otherwise(None)


class EfficiencyRatio(Feature):
    """Efficiency Ratio (Kaufman): deslocamento líquido / caminho percorrido.

    ER = |Δclose em N barras| / Σ|Δclose por barra|  ∈ [0, 1].
    Próximo de 1 → movimento direcional eficiente (tendência); próximo de 0 →
    vai-e-volta (ruído/lateralização). Não tem atraso de média móvel.
    """

    def __init__(self, window: int = 20, name: str | None = None) -> None:
        self.window = window
        self._name = name or f"eff_ratio_{window}"

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        net = (pl.col("close") - pl.col("close").shift(self.window)).abs()
        path = pl.col("close").diff().abs().rolling_sum(self.window)
        er = pl.when(path > 0).then(net / path).otherwise(0.0)
        return self._check(bars.select(er.alias(self._name)), bars.height)


class CloseLocationValue(ExprFeature):
    """Close Location Value: onde o close fecha dentro do range da barra.

    CLV = ((close-low) - (high-close)) / (high-low) ∈ [-1, 1].
    +1 = fechou na máxima (pressão compradora); -1 = fechou na mínima.
    """

    def __init__(self, name: str = "clv") -> None:
        expr = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / _RANGE
        super().__init__(name, expr)


class BodyRatio(ExprFeature):
    """Proporção do corpo no range: |close-open| / (high-low) ∈ [0, 1].

    Corpo grande = convicção direcional; corpo pequeno = indecisão/exaustão.
    """

    def __init__(self, name: str = "body_ratio") -> None:
        expr = (pl.col("close") - pl.col("open")).abs() / _RANGE
        super().__init__(name, expr)


class UpperWick(ExprFeature):
    """Pavio superior normalizado: rejeição de preços altos (exaustão de alta)."""

    def __init__(self, name: str = "upper_wick") -> None:
        body_top = pl.max_horizontal("open", "close")
        expr = (pl.col("high") - body_top) / _RANGE
        super().__init__(name, expr)


class LowerWick(ExprFeature):
    """Pavio inferior normalizado: rejeição de preços baixos (exaustão de baixa)."""

    def __init__(self, name: str = "lower_wick") -> None:
        body_bot = pl.min_horizontal("open", "close")
        expr = (body_bot - pl.col("low")) / _RANGE
        super().__init__(name, expr)


class GapOpen(ExprFeature):
    """Gap de abertura: log(open_t / close_{t-1}) — descontinuidade entre barras."""

    def __init__(self, name: str = "gap_open") -> None:
        expr = (pl.col("open") / pl.col("close").shift(1)).log()
        super().__init__(name, expr)
