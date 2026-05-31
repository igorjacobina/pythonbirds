"""Engine de backtest event-driven (vetorizado por barra) com hot loop em Numba.

Garantias de realismo:
  * **Sem look-ahead**: o target decidido no fechamento da barra ``i`` só é
    executado na ABERTURA da barra ``i+1``.
  * **Custos sempre aplicados**: meio-spread + slippage embutidos no preço de
    execução; comissão round-turn por lote; swap por noite carregada.
  * **Pior preço**: toda execução piora o preço a favor da corretora.
  * **Mark-to-market**: a equity é remarcada ao fechamento (mid) de cada barra.

O kernel opera sobre arrays NumPy puros — nenhuma alocação no loop quente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from forex_quant.backtest.costs import CostModel
from forex_quant.backtest.metrics import PerformanceReport, compute_metrics
from forex_quant.core.enums import Timeframe
from forex_quant.core.instruments import Instrument
from forex_quant.strategy.base import Strategy
from forex_quant.utils.jit import njit


@njit
def _run_kernel(
    open_: np.ndarray,
    close: np.ndarray,
    day_idx: np.ndarray,
    target: np.ndarray,
    contract: float,
    half_cost_price: float,
    commission_per_lot: float,
    swap_long_price: float,
    swap_short_price: float,
    max_lots: float,
    initial_capital: float,
):
    n = open_.shape[0]
    equity = np.empty(n, dtype=np.float64)
    pos_arr = np.empty(n, dtype=np.float64)
    trade_pnl = np.empty(n, dtype=np.float64)
    n_trades = 0

    cash = initial_capital
    pos = 0.0          # lotes com sinal (+ comprado, - vendido)
    avg_fill = 0.0     # preço médio de execução da posição corrente
    prev_day = day_idx[0]

    for i in range(n):
        # Alvo decidido no fechamento de i-1, executado na abertura de i.
        desired = 0.0 if i == 0 else target[i - 1] * max_lots
        price_open = open_[i]

        # --- Swap das noites carregadas desde a barra anterior ---
        if pos != 0.0 and day_idx[i] > prev_day:
            nights = day_idx[i] - prev_day
            if pos > 0.0:
                cash += swap_long_price * contract * pos * nights
            else:
                cash += swap_short_price * contract * (-pos) * nights

        # --- Decompõe a mudança de posição em "fechar" e "abrir" ---
        if desired != pos:
            close_lots = 0.0
            open_lots = 0.0
            if pos == 0.0:
                open_lots = desired
            elif (pos > 0.0 and desired >= 0.0) or (pos < 0.0 and desired <= 0.0):
                if abs(desired) >= abs(pos):
                    open_lots = desired - pos        # aumenta no mesmo sentido
                else:
                    close_lots = desired - pos       # reduz parcialmente
            else:
                close_lots = -pos                    # inverte: fecha tudo...
                open_lots = desired                  # ...e abre o novo sentido

            # Fase 1: fechar/reduzir (realiza P&L)
            if close_lots != 0.0:
                s = 1.0 if close_lots > 0.0 else -1.0
                fill = price_open + s * half_cost_price
                realized = (fill - avg_fill) * contract * (-close_lots)
                comm = commission_per_lot * abs(close_lots)
                cash += realized - comm
                pos += close_lots
                trade_pnl[n_trades] = realized - comm
                n_trades += 1
                if pos == 0.0:
                    avg_fill = 0.0

            # Fase 2: abrir/aumentar (atualiza preço médio)
            if open_lots != 0.0:
                s = 1.0 if open_lots > 0.0 else -1.0
                fill = price_open + s * half_cost_price
                new_mag = abs(pos) + abs(open_lots)
                avg_fill = (avg_fill * abs(pos) + fill * abs(open_lots)) / new_mag
                pos += open_lots
                cash -= commission_per_lot * abs(open_lots)

        # --- Mark-to-market ao fechamento ---
        unrealized = (close[i] - avg_fill) * contract * pos
        equity[i] = cash + unrealized
        pos_arr[i] = pos
        prev_day = day_idx[i]

    return equity, pos_arr, trade_pnl[:n_trades]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    report: PerformanceReport
    equity: np.ndarray
    position_lots: np.ndarray
    trade_pnls: np.ndarray
    times: pl.Series

    def equity_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"time": self.times, "equity": self.equity, "position_lots": self.position_lots}
        )


class BacktestEngine:
    """Executa uma ``Strategy`` sobre barras com custos realistas.

    Args:
        instrument: especificação do símbolo (P&L, custos).
        cost_model: parâmetros de spread/slippage/comissão/swap.
        initial_capital: capital inicial da conta.
        max_lots: tamanho de posição correspondente a target = ±1.0.
    """

    def __init__(
        self,
        instrument: Instrument,
        cost_model: CostModel | None = None,
        *,
        initial_capital: float = 10_000.0,
        max_lots: float = 1.0,
    ) -> None:
        self.instrument = instrument
        self.cost_model = cost_model or CostModel()
        self.initial_capital = initial_capital
        self.max_lots = max_lots

    def run(self, bars: pl.DataFrame, strategy: Strategy, timeframe: Timeframe) -> BacktestResult:
        if bars.height == 0:
            raise ValueError("não há barras para backtest")

        inst = self.instrument
        targets = np.asarray(strategy.generate_targets(bars, inst), dtype=np.float64)
        if targets.shape[0] != bars.height:
            raise ValueError(
                f"estratégia retornou {targets.shape[0]} targets "
                f"para {bars.height} barras"
            )
        targets = np.clip(np.nan_to_num(targets), -1.0, 1.0)

        open_ = bars["open"].to_numpy().astype(np.float64)
        close = bars["close"].to_numpy().astype(np.float64)
        # Índice de dia (dias desde epoch) para contabilizar swap por noite.
        day_idx = (
            bars["time"].dt.epoch(time_unit="d").to_numpy().astype(np.int64)
        )

        half_cost_points = self.cost_model.effective_spread(inst) / 2.0 + self.cost_model.slippage_points
        half_cost_price = inst.points_to_price(half_cost_points)
        commission_half = self.cost_model.effective_commission(inst) / 2.0  # meio em cada perna
        swap_long_price = inst.points_to_price(inst.swap_long_points) if self.cost_model.apply_swap else 0.0
        swap_short_price = inst.points_to_price(inst.swap_short_points) if self.cost_model.apply_swap else 0.0

        equity, pos_arr, trade_pnls = _run_kernel(
            open_,
            close,
            day_idx,
            targets,
            inst.contract_size,
            half_cost_price,
            commission_half,
            swap_long_price,
            swap_short_price,
            self.max_lots,
            self.initial_capital,
        )

        report = compute_metrics(
            equity, pos_arr, trade_pnls, timeframe, self.initial_capital
        )
        return BacktestResult(
            report=report,
            equity=equity,
            position_lots=pos_arr,
            trade_pnls=trade_pnls,
            times=bars["time"],
        )
