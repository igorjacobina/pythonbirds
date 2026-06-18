"""Framework de features — contrato causal e composição em matriz.

Toda feature transforma barras OHLCV (schema canônico) em uma ou mais colunas
numéricas *alinhadas* às barras, usando exclusivamente informação disponível até
o fechamento de cada barra (janelas trailing, shifts com lag positivo). Isso
garante ausência de look-ahead por construção — pré-requisito para que qualquer
padrão minerado seja operável na vida real.

``FeatureSet`` compõe várias features numa única matriz (Polars DataFrame),
preservando a coluna ``time`` para junção com rótulos e sessões.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class Feature(ABC):
    """Transformação causal de barras em colunas de feature.

    Subclasses implementam ``transform`` retornando um DataFrame com exatamente
    as colunas declaradas em ``names`` e a mesma altura das barras de entrada.
    """

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """Nomes das colunas produzidas por esta feature."""

    @abstractmethod
    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Calcula as colunas de feature (mesma altura de ``bars``)."""

    def _check(self, out: pl.DataFrame, n: int) -> pl.DataFrame:
        if out.height != n:
            raise ValueError(
                f"{type(self).__name__} produziu {out.height} linhas, esperado {n}"
            )
        if list(out.columns) != self.names:
            raise ValueError(
                f"{type(self).__name__} produziu colunas {out.columns}, "
                f"esperado {self.names}"
            )
        return out


class ExprFeature(Feature):
    """Feature simples definida por uma expressão Polars sobre as colunas de barra.

    A expressão deve ser causal (rolling/shift trailing). Atalho para a maioria
    das features que são uma única coluna derivada vetorialmente.
    """

    def __init__(self, name: str, expr: pl.Expr) -> None:
        self._name = name
        self._expr = expr

    @property
    def names(self) -> list[str]:
        return [self._name]

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        out = bars.select(self._expr.alias(self._name))
        return self._check(out, bars.height)


class FeatureSet:
    """Compõe múltiplas features numa matriz alinhada às barras."""

    def __init__(self, features: list[Feature] | None = None) -> None:
        self.features: list[Feature] = list(features or [])

    def add(self, feature: Feature) -> "FeatureSet":
        self.features.append(feature)
        return self

    @property
    def names(self) -> list[str]:
        out: list[str] = []
        for f in self.features:
            out.extend(f.names)
        if len(set(out)) != len(out):
            raise ValueError(f"nomes de feature duplicados: {out}")
        return out

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Retorna ``time`` + todas as colunas de feature."""
        matrix = bars.select("time")
        for f in self.features:
            matrix = matrix.hstack(f.transform(bars))
        return matrix
