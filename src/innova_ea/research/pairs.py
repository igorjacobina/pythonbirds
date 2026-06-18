"""Relative-value / pares (statistical arbitrage) — lucra com a RELAÇÃO entre ativos.

Dois ativos cointegrados (ex. ouro/prata) andam juntos. O *spread* entre eles
(``log A − β·log B``) oscila em torno de uma média. Quando o spread estica além de
``entry`` desvios-padrão, apostamos na reversão (vende o caro, compra o barato);
saímos quando volta para perto da média (``exit``), com um stop de divergência.

Tudo causal: β e as estatísticas do z-score vêm de janelas TRAILING (sem
look-ahead). O backtest cobra custo nas DUAS pernas a cada virada de posição.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from innova_ea.utils.jit import njit


def align_log_prices(bars_a: pl.DataFrame, bars_b: pl.DataFrame):
    """Junta dois ativos pelos timestamps comuns → (times, logA, logB)."""
    j = (
        bars_a.select(["time", "close"])
        .join(bars_b.select(["time", "close"]), on="time", how="inner", suffix="_b")
        .sort("time")
    )
    la = np.log(j["close"].to_numpy())
    lb = np.log(j["close_b"].to_numpy())
    return j["time"], la, lb


@dataclass(frozen=True, slots=True)
class PairConfig:
    beta_window: int = 250      # janela p/ estimar o hedge ratio β
    z_window: int = 100         # janela p/ média/desvio do spread (z-score)
    entry: float = 2.0          # |z| de entrada
    exit: float = 0.5           # |z| de saída (reversão)
    stop: float = 4.0           # |z| de stop (divergência — corta a perda)
    max_hold: int = 100         # nº máx. de barras no trade


def estimate_beta(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """Hedge ratio β via OLS (slope de logA sobre logB) — relação ESTRUTURAL.

    Estimado UMA vez (não rolante): regressão rolante entre passeios aleatórios é
    espúria/instável e injeta tendência no spread. Em produção, estime no período
    de TREINO e fixe no teste (o ``study_pairs`` faz isso).
    """
    a = np.asarray(log_a, dtype=np.float64)
    b = np.asarray(log_b, dtype=np.float64)
    bm = b.mean()
    denom = ((b - bm) ** 2).sum()
    if denom <= 0:
        return 1.0
    return float(((a - a.mean()) * (b - bm)).sum() / denom)


def spread_zscore(log_a: np.ndarray, log_b: np.ndarray, cfg: PairConfig,
                  beta: float | None = None):
    """Spread ``logA − β·logB`` (β fixo) e seu z-score ROLANTE (causal).

    Se ``beta`` não for dado, estima por OLS sobre o array. O sinal operável é o
    z-score (rolante/causal); β é estrutural.
    """
    if beta is None:
        beta = estimate_beta(log_a, log_b)
    spread = np.asarray(log_a, dtype=np.float64) - beta * np.asarray(log_b, dtype=np.float64)
    s = pl.Series(spread)
    zmean = s.rolling_mean(cfg.z_window)
    zstd = s.rolling_std(cfg.z_window)
    z = ((s - zmean) / zstd).shift(1).to_numpy()   # decisão em t usa stats até t-1
    return spread, z, float(beta)


@njit
def _signal_kernel(z, entry, exit_, stop, max_hold):
    n = z.shape[0]
    sig = np.zeros(n, dtype=np.int64)
    pos = 0
    held = 0
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            pos = 0
            held = 0
            sig[t] = 0
            continue
        if pos == 0:
            if zt > entry:
                pos = -1          # spread alto → vende o spread
                held = 0
            elif zt < -entry:
                pos = 1            # spread baixo → compra o spread
                held = 0
        else:
            held += 1
            if abs(zt) < exit_ or abs(zt) > stop or held >= max_hold:
                pos = 0
                held = 0
        sig[t] = pos
    return sig


def pairs_signal(z: np.ndarray, cfg: PairConfig) -> np.ndarray:
    """Posição no spread por barra: +1 (comprado), -1 (vendido), 0 (fora)."""
    return _signal_kernel(np.ascontiguousarray(z, dtype=np.float64),
                          cfg.entry, cfg.exit, cfg.stop, cfg.max_hold)


def pairs_backtest(
    spread: np.ndarray,
    signal: np.ndarray,
    *,
    cost_frac_per_turn: float = 0.0004,
    initial_capital: float = 10_000.0,
):
    """Equity do spread (log-retornos), com custo nas 2 pernas a cada virada.

    ``cost_frac_per_turn`` = custo de UMA virada (cruzar o spread nas duas pernas),
    como fração. Retorna (equity, position, trade_pnls).
    """
    n = spread.shape[0]
    dspread = np.zeros(n)
    dspread[1:] = spread[1:] - spread[:-1]
    dspread = np.nan_to_num(dspread)

    ret = np.zeros(n)
    ret[1:] = signal[:-1] * dspread[1:]            # posição de t-1 aplicada ao movimento de t

    turns = np.zeros(n)
    turns[1:] = np.abs(signal[1:] - signal[:-1])   # mudança de posição → custo
    net = ret - turns * cost_frac_per_turn
    equity = initial_capital * np.exp(np.cumsum(net))

    # P&L por trade (em dinheiro), somando net dentro de cada período em posição.
    trade_pnls = []
    acc = 0.0
    prev_eq = initial_capital
    in_pos = False
    for t in range(n):
        if signal[t] != 0:
            acc += net[t] * prev_eq
            in_pos = True
        elif in_pos:
            trade_pnls.append(acc)
            acc = 0.0
            in_pos = False
        prev_eq = equity[t]
    if in_pos:
        trade_pnls.append(acc)

    position = signal.astype(np.float64)
    return equity, position, np.array(trade_pnls, dtype=np.float64)
