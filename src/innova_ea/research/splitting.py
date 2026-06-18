"""Walk-forward PURGADO com embargo — validação sem vazamento para rótulos.

Rótulos triple-barrier olham até ``max_horizon`` barras à frente, então uma
amostra de treino próxima do início do teste "enxerga" o futuro do teste. Para
não vazar:

  * **Purga**: remove do treino as amostras cujo horizonte de rótulo invade o
    período de teste.
  * **Embargo**: deixa uma folga após o treino, antes do teste, para cortar
    autocorrelação residual.

As janelas são definidas sobre os TIMESTAMPS ÚNICOS do painel (todos os ativos
compartilham o eixo temporal), e cada split devolve os DataFrames de treino/teste
já filtrados por pertencimento de tempo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class PurgedWindow:
    index: int
    train_times: np.ndarray
    test_times: np.ndarray


class PurgedWalkForward:
    """Gera splits walk-forward purgados sobre o eixo de timestamps do painel.

    Args:
        train_size: nº de timestamps únicos de treino por janela.
        test_size: nº de timestamps únicos de teste por janela.
        horizon: horizonte do rótulo (barras) a purgar antes do teste.
        embargo: folga adicional (barras) entre treino e teste.
        step: avanço entre janelas (default = test_size, sem sobreposição OOS).
        anchored: se True, o treino sempre começa no início (expanding).
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        *,
        horizon: int = 0,
        embargo: int = 0,
        step: int | None = None,
        anchored: bool = False,
    ) -> None:
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size e test_size devem ser > 0")
        self.train_size = train_size
        self.test_size = test_size
        self.horizon = max(0, horizon)
        self.embargo = max(0, embargo)
        self.step = step or test_size
        self.anchored = anchored

    def _windows(self, n_times: int) -> list[tuple[int, int, int, int]]:
        """Retorna (train_lo, train_hi, test_lo, test_hi) em índices de timestamp."""
        out = []
        test_lo = self.train_size
        while test_lo + self.test_size <= n_times:
            test_hi = test_lo + self.test_size
            train_lo = 0 if self.anchored else test_lo - self.train_size
            # Purga + embargo: encerra o treino antes do teste com folga.
            train_hi = test_lo - self.horizon - self.embargo
            if train_hi - train_lo >= 1:
                out.append((train_lo, train_hi, test_lo, test_hi))
            test_lo += self.step
        return out

    def split(
        self, frame: pl.DataFrame, *, time_col: str = "time"
    ) -> Iterator[tuple[pl.DataFrame, pl.DataFrame, PurgedWindow]]:
        times = frame[time_col].unique().sort()
        n = times.len()
        wins = self._windows(n)
        if not wins:
            raise ValueError(
                f"timestamps insuficientes ({n}) para train_size={self.train_size}"
                f" + test_size={self.test_size} (+ purga/embargo)"
            )
        col = pl.col(time_col)
        for i, (tlo, thi, slo, shi) in enumerate(wins):
            # Cada bloco é uma fatia CONTÍGUA de timestamps únicos → filtro por
            # intervalo seleciona exatamente as linhas do bloco (mais rápido que is_in).
            train_lo, train_hi = times[tlo], times[thi - 1]
            test_lo, test_hi = times[slo], times[shi - 1]
            train_df = frame.filter((col >= train_lo) & (col <= train_hi))
            test_df = frame.filter((col >= test_lo) & (col <= test_hi))
            window = PurgedWindow(
                i,
                times.slice(tlo, thi - tlo).to_numpy(),
                times.slice(slo, shi - slo).to_numpy(),
            )
            yield train_df, test_df, window

    def n_splits(self, frame: pl.DataFrame, *, time_col: str = "time") -> int:
        n = frame[time_col].n_unique()
        return len(self._windows(n))
