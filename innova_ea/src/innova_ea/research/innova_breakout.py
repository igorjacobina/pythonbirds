"""INNOVA Breakout — operacional próprio fundindo Larry Williams + Andrea Unger.

Combina, num só sistema configurável, o que há de validado nos dois campeões
mundiais (ambos da família de ROMPIMENTO DE VOLATILIDADE — a única que o INNOVA
EA aprovou de forma independente):

  * **Entrada (Andrea Unger)**: rompimento do range da primeira hora do dia
    (opening range breakout).
  * **Gatilho de volatilidade (Larry Williams)**: só opera se o range de abertura
    for ao menos ``min_or_frac`` do range do dia ANTERIOR — exige expansão real
    de volatilidade, a essência do método de Williams.
  * **Filtros de contexto (Unger)**: "deixar a segunda-feira de fora" e pular dias
    após um range extremo no dia anterior.
  * **Saída honesta**: alvo/stop em unidades de risco + fechamento no fim do dia
    (sem o viés do "primeira abertura lucrativa").

Direção-agnóstico (opera o lado que romper), e validado com o mesmo rigor:
custos reais, walk-forward, Deflated Sharpe.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.research.opening_range import orb_backtest  # noqa: F401 (reexport p/ conveniência)
from innova_ea.utils.jit import njit


@njit
def _innova_kernel(open_, high, low, close, day_id, bod, or_high, or_low, weekday,
                   prev_range, avg_range, or_bars, reward_risk, cost_frac,
                   skip_monday, min_or_frac, max_prior_mult):
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
        day_end = j

        skip = False
        if skip_monday and weekday[i] == 1:                       # Unger: sem segunda
            skip = True
        pr = prev_range[i]
        ar = avg_range[i]
        # Unger: pula dia após range extremo no dia anterior.
        if max_prior_mult > 0.0 and ar > 0.0 and pr > max_prior_mult * ar:
            skip = True
        oh = or_high[i]
        ol = or_low[i]
        if not (oh > ol):
            skip = True
        # Williams: exige expansão — opening range >= fração do range anterior.
        if (not skip) and min_or_frac > 0.0 and pr > 0.0 and (oh - ol) < min_or_frac * pr:
            skip = True
        if skip:
            i = day_end
            continue

        entered = 0
        dd = 0
        entry = 0.0
        stop = 0.0
        target = 0.0
        trade_bar = -1
        for t in range(i, day_end):
            if bod[t] < or_bars:
                continue
            if entered == 0:
                up = high[t] >= oh
                dn = low[t] <= ol
                if up and dn:
                    continue
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

        if entered == 1:
            exit_close = close[day_end - 1]
            net[trade_bar] = dd * (exit_close - entry) / entry - cost_frac
        if entered >= 1:
            trig[trade_bar] = 1
            direction[trade_bar] = dd
        i = day_end

    return net, trig, direction


def innova_breakout_outcomes(
    bars: pl.DataFrame, *, or_bars: int = 4, reward_risk: float = 1.0,
    cost_frac: float = 0.0004, skip_monday: bool = True,
    min_or_frac: float = 0.5, max_prior_mult: float = 3.0, avg_window: int = 20,
) -> pl.DataFrame:
    """Resultado do INNOVA Breakout por barra (1 trade/dia no máximo).

    Args:
        or_bars: barras do range de abertura (ex. 4×M15 = 1ª hora).
        reward_risk: alvo em unidades de risco (risco = largura do range de abertura).
        cost_frac: custo de rodada (2 pernas) como fração.
        skip_monday: filtro de Unger.
        min_or_frac: gatilho de Williams — opening range deve ser ≥ esta fração do
            range do dia anterior (exige expansão de volatilidade). 0 desliga.
        max_prior_mult: filtro de Unger — pula dia se o range anterior > este
            múltiplo do range médio. 0 desliga.
        avg_window: janela do range médio (dias) para o filtro acima.
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
    # Range diário (high-low do dia), range do dia anterior e range médio (causais).
    daily = (
        df.group_by("_day", maintain_order=True)
        .agg((pl.col("high").max() - pl.col("low").min()).alias("_drange"))
        .with_columns(
            pl.col("_drange").shift(1).alias("_prev_range"),
            pl.col("_drange").rolling_mean(avg_window).shift(1).alias("_avg_range"),
        )
    )
    df = df.join(daily.select("_day", "_prev_range", "_avg_range"), on="_day", how="left")

    net, trig, direction = _innova_kernel(
        df["open"].to_numpy().astype(np.float64),
        df["high"].to_numpy().astype(np.float64),
        df["low"].to_numpy().astype(np.float64),
        df["close"].to_numpy().astype(np.float64),
        df["_day"].to_numpy().astype(np.int64),
        df["_bod"].to_numpy().astype(np.int64),
        np.nan_to_num(df["_orh"].to_numpy().astype(np.float64)),
        np.nan_to_num(df["_orl"].to_numpy().astype(np.float64)),
        df["_wd"].to_numpy().astype(np.int64),
        np.nan_to_num(df["_prev_range"].to_numpy().astype(np.float64)),
        np.nan_to_num(df["_avg_range"].to_numpy().astype(np.float64)),
        int(or_bars), float(reward_risk), float(cost_frac), bool(skip_monday),
        float(min_or_frac), float(max_prior_mult),
    )
    return pl.DataFrame({"net_ret": net, "triggered": trig, "direction": direction})
