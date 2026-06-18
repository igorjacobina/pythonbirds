"""Walk-forward analysis — o padrão-ouro de avaliação out-of-sample.

Em vez de um único backtest (propenso a overfitting), divide-se o histórico em
janelas sucessivas de *treino* (onde a estratégia é calibrada) e *teste*
(out-of-sample, OOS, nunca visto na calibração). Os resultados reportados são
sempre a concatenação dos trechos OOS — a única medida honesta de generalização.

Suporta janela deslizante (rolling) e ancorada (expanding/anchored).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from innova_ea.backtest.engine import BacktestEngine
from innova_ea.backtest.metrics import PerformanceReport, compute_metrics
from innova_ea.core.enums import Timeframe
from innova_ea.strategy.base import Strategy

# Recebe as barras de treino e devolve uma estratégia calibrada.
StrategyFactory = Callable[[pl.DataFrame], Strategy]
# Devolve um engine novo (capital reinicia a cada janela OOS).
EngineFactory = Callable[[], BacktestEngine]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """Índices de linha (fim exclusivo) de uma janela treino/teste."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    window_reports: list[PerformanceReport]
    oos_report: PerformanceReport
    oos_sharpes: list[float]
    initial_capital: float

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def summary(self) -> str:
        sr = np.array(self.oos_sharpes, dtype=np.float64)
        consistency = float(np.mean(sr > 0)) if sr.size else 0.0
        return (
            f"Janelas OOS        : {self.n_windows}\n"
            f"Sharpe OOS (médio) : {sr.mean():.2f}\n"
            f"Sharpe OOS (dp)    : {sr.std(ddof=1) if sr.size > 1 else 0.0:.2f}\n"
            f"Janelas positivas  : {consistency:.0%}\n"
            f"--- Agregado OOS (compostas) ---\n"
            f"{self.oos_report.summary()}"
        )


class WalkForward:
    """Gera e executa janelas walk-forward.

    Args:
        train_size: nº de barras de treino por janela.
        test_size: nº de barras de teste (OOS) por janela.
        step: avanço entre janelas (default = ``test_size`` → OOS sem sobreposição).
        anchored: se True, o treino sempre começa em 0 (expanding); senão rolling.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        *,
        step: int | None = None,
        anchored: bool = False,
    ) -> None:
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size e test_size devem ser > 0")
        self.train_size = train_size
        self.test_size = test_size
        self.step = step or test_size
        self.anchored = anchored

    def split(self, n_rows: int) -> list[WalkForwardWindow]:
        windows: list[WalkForwardWindow] = []
        test_start = self.train_size
        idx = 0
        while test_start + self.test_size <= n_rows:
            train_start = 0 if self.anchored else test_start - self.train_size
            windows.append(
                WalkForwardWindow(
                    index=idx,
                    train_start=train_start,
                    train_end=test_start,
                    test_start=test_start,
                    test_end=test_start + self.test_size,
                )
            )
            test_start += self.step
            idx += 1
        if not windows:
            raise ValueError(
                f"barras insuficientes ({n_rows}) para train_size="
                f"{self.train_size} + test_size={self.test_size}"
            )
        return windows

    def run(
        self,
        bars: pl.DataFrame,
        strategy_factory: StrategyFactory,
        engine_factory: EngineFactory,
        timeframe: Timeframe,
    ) -> WalkForwardResult:
        windows = self.split(bars.height)
        window_reports: list[PerformanceReport] = []
        oos_sharpes: list[float] = []
        oos_returns: list[np.ndarray] = []
        oos_positions: list[np.ndarray] = []
        oos_trades: list[np.ndarray] = []

        for w in windows:
            train = bars[w.train_start : w.train_end]
            test = bars[w.test_start : w.test_end]

            strategy = strategy_factory(train)
            engine = engine_factory()
            res = engine.run(test, strategy, timeframe)

            window_reports.append(res.report)
            oos_sharpes.append(res.report.sharpe)

            eq = res.equity
            if eq.size > 1:
                oos_returns.append(np.diff(eq) / eq[:-1])
            oos_positions.append(res.position_lots)
            oos_trades.append(res.trade_pnls)

        # Curva OOS agregada: composição encadeada dos retornos de cada janela.
        initial_capital = engine_factory().initial_capital
        if oos_returns:
            all_rets = np.concatenate(oos_returns)
            all_rets = np.nan_to_num(all_rets, nan=0.0, posinf=0.0, neginf=0.0)
            oos_equity = initial_capital * np.cumprod(1.0 + all_rets)
            oos_equity = np.insert(oos_equity, 0, initial_capital)
        else:
            oos_equity = np.array([initial_capital], dtype=np.float64)

        oos_positions_arr = (
            np.concatenate(oos_positions) if oos_positions else np.zeros(0)
        )
        oos_trades_arr = np.concatenate(oos_trades) if oos_trades else np.zeros(0)

        oos_report = compute_metrics(
            oos_equity, oos_positions_arr, oos_trades_arr, timeframe, initial_capital
        )
        return WalkForwardResult(
            windows=windows,
            window_reports=window_reports,
            oos_report=oos_report,
            oos_sharpes=oos_sharpes,
            initial_capital=initial_capital,
        )
