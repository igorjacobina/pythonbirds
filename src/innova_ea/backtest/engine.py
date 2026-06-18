"""Engine de backtest event-driven (vetorizado por barra) com hot loop em Numba.

Garantias de realismo:
  * **Sem look-ahead**: o target decidido no fechamento da barra ``i`` só é
    executado na ABERTURA da barra ``i+1``.
  * **Custos sempre aplicados**: meio-spread + slippage embutidos no preço de
    execução; comissão round-turn por lote; swap por noite carregada.
  * **Pior preço**: toda execução piora o preço a favor da corretora.
  * **Mark-to-market**: a equity é remarcada ao fechamento (mid) de cada barra.
  * **Margem & stop-out (estilo MT5)**: ordens que aumentam exposição são
    BLOQUEADAS se a margem livre for insuficiente; posições são LIQUIDADAS à
    força quando o nível de margem cai até o stop-out da corretora.

O kernel opera sobre arrays NumPy puros — nenhuma alocação no loop quente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from innova_ea.backtest.account import AccountConfig
from innova_ea.backtest.costs import CostModel
from innova_ea.backtest.metrics import (
    PerformanceReport,
    RiskReport,
    compute_metrics,
    compute_risk,
)
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument
from innova_ea.strategy.base import Strategy
from innova_ea.utils.jit import njit

# Sentinela para "sem posição" no array de nível de margem (margem infinita).
_NO_MARGIN_LEVEL = 1.0e12


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
    margin_factor: float,       # contract_size / leverage
    margin_uses_price: float,   # 1.0 se a margem depende do preço, senão 0.0
    margin_call_level: float,
    stop_out_level: float,
):
    n = open_.shape[0]
    equity = np.empty(n, dtype=np.float64)
    pos_arr = np.empty(n, dtype=np.float64)
    margin_level_arr = np.empty(n, dtype=np.float64)
    used_margin_arr = np.empty(n, dtype=np.float64)
    trade_pnl = np.empty(n, dtype=np.float64)
    n_trades = 0
    n_rejected = 0   # ordens bloqueadas por margem livre insuficiente
    n_stopouts = 0   # liquidações forçadas pela corretora

    cash = initial_capital
    pos = 0.0          # lotes com sinal (+ comprado, - vendido)
    avg_fill = 0.0     # preço médio de execução da posição corrente
    prev_day = day_idx[0]

    for i in range(n):
        # Alvo decidido no fechamento de i-1, executado na abertura de i.
        desired = 0.0 if i == 0 else target[i - 1] * max_lots
        price_open = open_[i]
        pm_open = price_open if margin_uses_price > 0.5 else 1.0

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

            # Fase 1: fechar/reduzir (sempre permitido — libera margem)
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

            # Fase 2: abrir/aumentar — sujeito a checagem de MARGEM LIVRE
            if open_lots != 0.0:
                new_pos = pos + open_lots
                new_margin = margin_factor * abs(new_pos) * pm_open
                # Equity no instante da ordem (mark-to-market na abertura).
                eq_now = cash + (price_open - avg_fill) * contract * pos
                # MT5: bloqueia se o nível de margem resultante < margin_call.
                if new_margin <= 0.0 or eq_now >= margin_call_level * new_margin:
                    s = 1.0 if open_lots > 0.0 else -1.0
                    fill = price_open + s * half_cost_price
                    new_mag = abs(pos) + abs(open_lots)
                    avg_fill = (avg_fill * abs(pos) + fill * abs(open_lots)) / new_mag
                    pos += open_lots
                    cash -= commission_per_lot * abs(open_lots)
                else:
                    n_rejected += 1  # ordem rejeitada; posição inalterada

        # --- Mark-to-market ao fechamento ---
        unrealized = (close[i] - avg_fill) * contract * pos
        eq = cash + unrealized
        pm_close = close[i] if margin_uses_price > 0.5 else 1.0
        used_margin = margin_factor * abs(pos) * pm_close

        # --- Stop-out: liquidação forçada da corretora ---
        if used_margin > 0.0 and eq <= stop_out_level * used_margin:
            s = -1.0 if pos > 0.0 else 1.0          # fecha no sentido oposto
            fill = close[i] + s * half_cost_price
            realized = (fill - avg_fill) * contract * pos
            comm = commission_per_lot * abs(pos)
            cash += realized - comm
            trade_pnl[n_trades] = realized - comm
            n_trades += 1
            n_stopouts += 1
            pos = 0.0
            avg_fill = 0.0
            unrealized = 0.0
            eq = cash
            used_margin = 0.0

        equity[i] = eq
        pos_arr[i] = pos
        used_margin_arr[i] = used_margin
        margin_level_arr[i] = (eq / used_margin) if used_margin > 0.0 else _NO_MARGIN_LEVEL
        prev_day = day_idx[i]

    return (
        equity,
        pos_arr,
        trade_pnl[:n_trades],
        margin_level_arr,
        used_margin_arr,
        n_rejected,
        n_stopouts,
    )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    report: PerformanceReport
    risk: RiskReport
    equity: np.ndarray
    position_lots: np.ndarray
    trade_pnls: np.ndarray
    margin_level: np.ndarray
    used_margin: np.ndarray
    times: pl.Series

    def equity_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": self.times,
                "equity": self.equity,
                "position_lots": self.position_lots,
                "margin_level": self.margin_level,
                "used_margin": self.used_margin,
            }
        )


class BacktestEngine:
    """Executa uma ``Strategy`` sobre barras com custos e risco realistas.

    Args:
        instrument: especificação do símbolo (P&L, custos, moeda base).
        cost_model: parâmetros de spread/slippage/comissão/swap.
        account: configuração de margem/alavancagem/stop-out (estilo MT5).
        initial_capital: capital inicial da conta.
        max_lots: tamanho de posição correspondente a target = ±1.0.
    """

    def __init__(
        self,
        instrument: Instrument,
        cost_model: CostModel | None = None,
        account: AccountConfig | None = None,
        *,
        initial_capital: float = 10_000.0,
        max_lots: float = 1.0,
    ) -> None:
        self.instrument = instrument
        self.cost_model = cost_model or CostModel()
        self.account = account or AccountConfig()
        self.initial_capital = initial_capital
        self.max_lots = max_lots

    def run(self, bars: pl.DataFrame, strategy: Strategy, timeframe: Timeframe) -> BacktestResult:
        if bars.height == 0:
            raise ValueError("não há barras para backtest")

        inst = self.instrument
        acct = self.account
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
        day_idx = bars["time"].dt.epoch(time_unit="d").to_numpy().astype(np.int64)

        half_cost_points = (
            self.cost_model.effective_spread(inst) / 2.0 + self.cost_model.slippage_points
        )
        half_cost_price = inst.points_to_price(half_cost_points)
        commission_half = self.cost_model.effective_commission(inst) / 2.0  # meio por perna
        swap_long_price = inst.points_to_price(inst.swap_long_points) if self.cost_model.apply_swap else 0.0
        swap_short_price = inst.points_to_price(inst.swap_short_points) if self.cost_model.apply_swap else 0.0

        margin_factor = inst.contract_size / acct.leverage
        margin_uses_price = 1.0 if inst.margin_requires_price(acct.account_currency) else 0.0

        (
            equity,
            pos_arr,
            trade_pnls,
            margin_level,
            used_margin,
            n_rejected,
            n_stopouts,
        ) = _run_kernel(
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
            margin_factor,
            margin_uses_price,
            acct.margin_call_level,
            acct.stop_out_level,
        )

        report = compute_metrics(
            equity, pos_arr, trade_pnls, timeframe, self.initial_capital
        )
        risk = compute_risk(
            equity,
            margin_level,
            used_margin,
            int(n_rejected),
            int(n_stopouts),
            sentinel=_NO_MARGIN_LEVEL,
        )
        return BacktestResult(
            report=report,
            risk=risk,
            equity=equity,
            position_lots=pos_arr,
            trade_pnls=trade_pnls,
            margin_level=margin_level,
            used_margin=used_margin,
            times=bars["time"],
        )
