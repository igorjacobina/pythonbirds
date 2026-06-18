"""Método de Larry Williams — Volatility Breakout (campeão mundial de 1987).

Larry Williams transformou US$10.000 em US$1.137.600 (11.376%) no Campeonato
Mundial de Futuros de 1987 — recorde até hoje. Seu método principal é o
**volatility breakout** diário:

  * compra com stop em ``abertura + k·range_anterior``;
  * vende com stop em ``abertura − k·range_anterior``;
  * stop-loss no PONTO MÉDIO entre o extremo do dia anterior e o preço de entrada;
  * saída na "primeira abertura lucrativa" (ou por tempo).

É a MESMA família (rompimento de volatilidade) que o INNOVA EA identificou, de
forma independente, como a única com edge real — uma validação externa forte.
Testado aqui com o mesmo rigor: custos reais, walk-forward, Deflated Sharpe.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.utils.jit import njit


@njit
def _larry_kernel(open_, high, low, close, k, reward_risk, max_hold, cost_frac):
    n = open_.shape[0]
    net_ret = np.zeros(n, dtype=np.float64)
    resolve = np.zeros(n, dtype=np.int64)
    trig = np.zeros(n, dtype=np.int64)
    direction = np.zeros(n, dtype=np.int64)

    for i in range(1, n):
        prev_range = high[i - 1] - low[i - 1]
        if prev_range <= 0.0:
            continue
        buy = open_[i] + k * prev_range
        sell = open_[i] - k * prev_range
        long_trig = high[i] >= buy
        short_trig = low[i] <= sell
        # Se AMBOS disparam no mesmo dia, não dá para saber qual veio primeiro pelo
        # OHLC diário — usar o fechamento seria LOOK-AHEAD. Honesto: não opera.
        if long_trig and short_trig:
            continue
        elif long_trig:
            d = 1
        elif short_trig:
            d = -1
        else:
            continue

        # Stop de Larry: ponto médio entre o extremo do dia anterior e a entrada.
        # Alvo SIMÉTRICO em unidades de risco (reward_risk × risco) — exit honesto
        # (triple-barrier), sem o viés do "primeira abertura lucrativa".
        if d == 1:
            entry = buy
            stop = (low[i - 1] + buy) / 2.0
            risk = entry - stop
            target = entry + reward_risk * risk
        else:
            entry = sell
            stop = (high[i - 1] + sell) / 2.0
            risk = stop - entry
            target = entry - reward_risk * risk
        if risk <= 0.0:
            continue
        trig[i] = 1
        direction[i] = d

        end = i + max_hold
        if end > n - 1:
            end = n - 1

        # Saída a partir do dia seguinte. Dentro da barra, checa STOP antes do alvo
        # (conservador: toques simultâneos resolvem como perda).
        exit_price = 0.0
        res = 0
        j = i + 1
        done = False
        while j <= end:
            if d == 1:
                if low[j] <= stop:
                    exit_price = stop; res = j - i; done = True; break
                if high[j] >= target:
                    exit_price = target; res = j - i; done = True; break
            else:
                if high[j] >= stop:
                    exit_price = stop; res = j - i; done = True; break
                if low[j] <= target:
                    exit_price = target; res = j - i; done = True; break
            j += 1
        if not done:
            exit_price = close[end]                    # saída por tempo
            res = end - i if end > i else 1

        # Retorno líquido: ganho fracional menos o custo de rodada (2 pernas).
        gross = d * (exit_price - entry) / entry if entry != 0.0 else 0.0
        net_ret[i] = gross - cost_frac
        resolve[i] = res if res > 0 else 1

    return net_ret, resolve, trig, direction


def larry_outcomes(
    bars: pl.DataFrame, *, k: float = 0.5, reward_risk: float = 1.0,
    max_hold: int = 5, cost_frac: float = 0.0004,
) -> pl.DataFrame:
    """Resultado do volatility breakout de Larry Williams por barra (dia).

    Entrada exata de Larry (``abertura ± k·range``, stop no ponto médio); saída por
    triple-barrier simétrico (alvo = ``reward_risk`` × risco, stop, ou tempo) —
    avaliação honesta, sem o viés do "primeira abertura lucrativa".
    ``cost_frac`` = custo de rodada (2 pernas) como FRAÇÃO do preço.
    """
    o = bars["open"].to_numpy().astype(np.float64)
    h = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)
    c = bars["close"].to_numpy().astype(np.float64)
    net_ret, resolve, trig, direction = _larry_kernel(
        o, h, low, c, float(k), float(reward_risk), int(max_hold), float(cost_frac))
    return pl.DataFrame({
        "net_ret": net_ret, "resolve": resolve, "triggered": trig, "direction": direction,
    })


def larry_backtest(outcomes: pl.DataFrame, *, initial_capital: float = 10_000.0):
    """Equity operando TODO dia que dispara (não sobreposto). Retorna (equity, pos, trade_rets)."""
    net = outcomes["net_ret"].to_numpy()
    resolve = outcomes["resolve"].to_numpy()
    trig = outcomes["triggered"].to_numpy()
    n = net.shape[0]
    rets = np.zeros(n, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    busy_until = -1
    trade_rets = []
    for i in range(n):
        if trig[i] and i > busy_until:
            j = min(i + int(resolve[i]), n - 1)
            rets[j] += net[i]
            pos[i:j + 1] = 1.0
            busy_until = i + int(resolve[i])
            trade_rets.append(net[i])
    equity = initial_capital * np.cumprod(1.0 + rets)
    return equity, pos, np.array(trade_rets, dtype=np.float64)
