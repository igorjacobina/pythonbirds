"""Andrea Unger — Opening Range Breakout (tetracampeão mundial de trading).

Andrea Unger é o ÚNICO tetracampeão do World Cup Trading Championship (2008,
2009, 2010, 2012), 100% sistemático. Seu operacional clássico é o **rompimento
do range da primeira hora**: mede a máxima/mínima das primeiras barras do dia e
opera o rompimento desse range no restante do pregão, com filtros lógicos
(ex.: "deixar a segunda-feira de fora").

Filosofia (idêntica à do INNOVA EA): começar com ideias simples e robustas
(breakout), validar o edge estatístico básico, e só então adicionar filtros
LÓGICOS de contexto — nunca parâmetros otimizados no escuro.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.utils.jit import njit


@njit
def _orb_kernel(open_, high, low, close, day_id, bod, or_high, or_low, weekday,
                or_bars, reward_risk, cost_frac, skip_monday):
    n = open_.shape[0]
    net = np.zeros(n, dtype=np.float64)
    trig = np.zeros(n, dtype=np.int64)
    direction = np.zeros(n, dtype=np.int64)

    i = 0
    while i < n:
        d = day_id[i]
        j = i
        while j < n and day_id[j] == d:
            j += 1
        day_end = j                                   # exclusivo

        if skip_monday and weekday[i] == 1:           # Polars: 1 = segunda
            i = day_end
            continue
        oh = or_high[i]
        ol = or_low[i]
        if not (oh > ol):                             # range de abertura inválido
            i = day_end
            continue

        entered = 0
        dd = 0
        entry = 0.0
        stop = 0.0
        target = 0.0
        trade_bar = -1
        for t in range(i, day_end):
            if bod[t] < or_bars:                      # ainda dentro do range de abertura
                continue
            if entered == 0:
                up = high[t] >= oh
                dn = low[t] <= ol
                if up and dn:
                    continue                          # rompe os dois → ambíguo, espera
                if up:
                    entered = 1; dd = 1; entry = oh; stop = ol
                    target = entry + reward_risk * (entry - stop); trade_bar = t
                elif dn:
                    entered = 1; dd = -1; entry = ol; stop = oh
                    target = entry - reward_risk * (stop - entry); trade_bar = t
            else:
                if dd == 1:
                    if low[t] <= stop:
                        net[trade_bar] = (stop - entry) / entry - cost_frac
                        entered = 2; break
                    if high[t] >= target:
                        net[trade_bar] = (target - entry) / entry - cost_frac
                        entered = 2; break
                else:
                    if high[t] >= stop:
                        net[trade_bar] = (entry - stop) / entry - cost_frac
                        entered = 2; break
                    if low[t] <= target:
                        net[trade_bar] = (entry - target) / entry - cost_frac
                        entered = 2; break

        if entered == 1:                              # ainda na posição → fecha no fim do dia
            exit_close = close[day_end - 1]
            net[trade_bar] = dd * (exit_close - entry) / entry - cost_frac
        if entered >= 1:
            trig[trade_bar] = 1
            direction[trade_bar] = dd
        i = day_end

    return net, trig, direction


def orb_outcomes(
    bars: pl.DataFrame, *, or_bars: int = 4, reward_risk: float = 1.0,
    cost_frac: float = 0.0004, skip_monday: bool = True,
) -> pl.DataFrame:
    """Resultado do Opening Range Breakout por barra (1 trade/dia no máximo).

    ``or_bars``: nº de barras que formam o range de abertura (ex. 4×M15 = 1ª hora).
    ``reward_risk``: alvo em unidades de risco (risco = largura do range de abertura).
    ``cost_frac``: custo de rodada (2 pernas) como fração. ``skip_monday``: filtro de Unger.
    """
    df = bars.with_columns(
        pl.col("time").dt.epoch(time_unit="d").alias("_day"),
        pl.col("time").dt.weekday().alias("_wd"),
    )
    df = df.with_columns(pl.int_range(pl.len()).over("_day").alias("_bod"))
    df = df.with_columns(
        pl.when(pl.col("_bod") < or_bars).then(pl.col("high")).otherwise(None).max().over("_day").alias("_orh"),
        pl.when(pl.col("_bod") < or_bars).then(pl.col("low")).otherwise(None).min().over("_day").alias("_orl"),
    )
    net, trig, direction = _orb_kernel(
        df["open"].to_numpy().astype(np.float64),
        df["high"].to_numpy().astype(np.float64),
        df["low"].to_numpy().astype(np.float64),
        df["close"].to_numpy().astype(np.float64),
        df["_day"].to_numpy().astype(np.int64),
        df["_bod"].to_numpy().astype(np.int64),
        np.nan_to_num(df["_orh"].to_numpy().astype(np.float64)),
        np.nan_to_num(df["_orl"].to_numpy().astype(np.float64)),
        df["_wd"].to_numpy().astype(np.int64),
        int(or_bars), float(reward_risk), float(cost_frac), bool(skip_monday),
    )
    return pl.DataFrame({"net_ret": net, "triggered": trig, "direction": direction})


def orb_backtest(outcomes: pl.DataFrame, *, initial_capital: float = 10_000.0):
    """Equity do ORB (1 trade/dia, sem sobreposição). Retorna (equity, pos, trade_rets)."""
    net = outcomes["net_ret"].to_numpy()
    trig = outcomes["triggered"].to_numpy()
    equity = initial_capital * np.cumprod(1.0 + net)
    pos = trig.astype(np.float64)
    trade_rets = net[net != 0.0]
    return equity, pos, trade_rets
