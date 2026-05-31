"""Métricas de performance e risco de nível institucional.

Calculadas a partir da curva de equity (mark-to-market) e dos retornos por
barra. Tudo é anualizado de forma consistente com o timeframe operado.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from forex_quant.core.enums import Timeframe

_TRADING_MINUTES_PER_YEAR = 252 * 24 * 60  # convenção 24h, 252 dias úteis


def periods_per_year(tf: Timeframe) -> float:
    return _TRADING_MINUTES_PER_YEAR / tf.minutes


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Resumo de performance de um backtest (todos os valores líquidos de custos)."""

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    exposure: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Capital inicial : {self.initial_capital:,.2f}\n"
            f"Equity final    : {self.final_equity:,.2f}\n"
            f"Retorno total   : {self.total_return:+.2%}\n"
            f"CAGR            : {self.cagr:+.2%}\n"
            f"Vol. anual      : {self.annual_volatility:.2%}\n"
            f"Sharpe          : {self.sharpe:.2f}\n"
            f"Sortino         : {self.sortino:.2f}\n"
            f"Calmar          : {self.calmar:.2f}\n"
            f"Max drawdown    : {self.max_drawdown:.2%} "
            f"(dur. {self.max_drawdown_duration} barras)\n"
            f"VaR 95% / CVaR  : {self.var_95:.2%} / {self.cvar_95:.2%}\n"
            f"Exposição       : {self.exposure:.1%}\n"
            f"Trades          : {self.n_trades}\n"
            f"Win rate        : {self.win_rate:.1%}\n"
            f"Profit factor   : {self.profit_factor:.2f}\n"
            f"Expectancy      : {self.expectancy:,.2f} / trade\n"
            f"Avg win / loss  : {self.avg_win:,.2f} / {self.avg_loss:,.2f}"
        )


def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """Retorna (max drawdown como fração negativa, duração em barras)."""
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min()) if drawdown.size else 0.0

    # Duração: maior sequência abaixo de um pico anterior.
    longest = cur = 0
    for dd in drawdown:
        if dd < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return max_dd, longest


def compute_metrics(
    equity: np.ndarray,
    position_lots: np.ndarray,
    trade_pnls: np.ndarray,
    timeframe: Timeframe,
    initial_capital: float,
) -> PerformanceReport:
    """Calcula o relatório completo a partir das saídas do engine."""
    equity = np.asarray(equity, dtype=np.float64)
    n = equity.size
    ppy = periods_per_year(timeframe)

    rets = np.diff(equity) / equity[:-1] if n > 1 else np.zeros(0)
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    final_equity = float(equity[-1]) if n else initial_capital
    total_return = final_equity / initial_capital - 1.0

    years = n / ppy if ppy else 0.0
    # CAGR em log-space evita overflow em janelas muito curtas (ex.: testes).
    ratio = final_equity / initial_capital
    if years > 0 and ratio > 0:
        with np.errstate(over="ignore"):  # janelas curtas → CAGR pode estourar p/ inf
            cagr = float(np.expm1(np.log(ratio) / years))
    else:
        cagr = 0.0

    ann_vol = float(rets.std(ddof=1) * np.sqrt(ppy)) if rets.size > 1 else 0.0
    mean_ret = float(rets.mean()) if rets.size else 0.0
    sharpe = (mean_ret / rets.std(ddof=1) * np.sqrt(ppy)) if rets.size > 1 and rets.std(ddof=1) > 0 else 0.0

    downside = rets[rets < 0]
    dd_std = downside.std(ddof=1) if downside.size > 1 else 0.0
    sortino = (mean_ret / dd_std * np.sqrt(ppy)) if dd_std > 0 else 0.0

    max_dd, dd_dur = _max_drawdown(equity)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    var_95 = float(np.percentile(rets, 5)) if rets.size else 0.0
    tail = rets[rets <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else 0.0

    position_lots = np.asarray(position_lots, dtype=np.float64)
    exposure = float(np.mean(position_lots != 0.0)) if position_lots.size else 0.0

    trade_pnls = np.asarray(trade_pnls, dtype=np.float64)
    n_trades = int(trade_pnls.size)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    win_rate = wins.size / n_trades if n_trades else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    expectancy = float(trade_pnls.mean()) if n_trades else 0.0
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    return PerformanceReport(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        cagr=cagr,
        annual_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_dur,
        var_95=var_95,
        cvar_95=cvar_95,
        exposure=exposure,
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
