"""Estratégia de VOLATILIDADE — straddle de rompimento (direção-agnóstico).

Direção em majors é imprevisível (3 paredes confirmadas). Volatilidade NÃO é:
o *clustering* de volatilidade é um dos fenômenos mais robustos do mercado. Esta
camada aposta no TAMANHO do movimento, não no sinal:

  * define um range de referência (máx/mín das últimas N barras);
  * arma um straddle: rompeu para cima → compra; para baixo → vende;
  * o trade vence se o movimento ANDAR o suficiente (alvo) antes de reverter
    (stop), líquido do custo de rodada (≈ spread + 2× slippage).

O ML prevê P(rompimento lucrativo). Tudo computado de forma causal; o resultado
do straddle (rótulo) é simulado para frente sobre o OHLC (look-ahead só no alvo).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from innova_ea.core.bars import validate_bars  # noqa: F401  (uso futuro/validação)
from innova_ea.features.base import FeatureSet
from innova_ea.research.dataset import UniversalPanel
from innova_ea.utils.jit import njit


@njit
def _straddle_kernel(high, low, close, band_hi, band_lo, tgt, stp, cost, horizon):
    """Para cada barra, simula o straddle e devolve (label, net_price, resolve, trig).

    Convenções de custo (preço): entrada e saída pioram em ``cost`` cada (slippage/
    meio-spread), totalizando ~2× ``cost`` de ida e volta. Alvo/stop medidos a
    partir da BANDA de rompimento.
    """
    n = high.shape[0]
    label = np.zeros(n, dtype=np.int64)
    net = np.zeros(n, dtype=np.float64)
    resolve = np.zeros(n, dtype=np.int64)   # 0 = não disparou (sem trade)
    trig = np.zeros(n, dtype=np.int64)

    for i in range(n):
        hi = band_hi[i]
        lo = band_lo[i]
        t = tgt[i]
        s = stp[i]
        c = cost
        if not (hi > 0.0 and lo > 0.0 and t > 0.0):
            continue
        end = i + horizon
        if end > n - 1:
            end = n - 1

        entered = 0
        direction = 0
        j = i + 1
        while j <= end:
            if entered == 0:
                if high[j] >= hi:                 # rompimento de alta → long
                    entered = 1
                    direction = 1
                    tgt_lvl = hi + t
                    stp_lvl = hi - s
                elif low[j] <= lo:                # rompimento de baixa → short
                    entered = 1
                    direction = -1
                    tgt_lvl = lo - t
                    stp_lvl = lo + s
                if entered == 1:
                    trig[i] = 1
            else:
                if direction == 1:
                    if low[j] <= stp_lvl:          # stop primeiro (conservador)
                        net[i] = -s - 2.0 * c
                        resolve[i] = j - i
                        break
                    if high[j] >= tgt_lvl:
                        net[i] = t - 2.0 * c
                        label[i] = 1
                        resolve[i] = j - i
                        break
                else:
                    if high[j] >= stp_lvl:
                        net[i] = -s - 2.0 * c
                        resolve[i] = j - i
                        break
                    if low[j] <= tgt_lvl:
                        net[i] = t - 2.0 * c
                        label[i] = 1
                        resolve[i] = j - i
                        break
            j += 1

        if entered == 1 and resolve[i] == 0:       # barreira de tempo: sai no close
            exit_close = close[end]
            if direction == 1:
                net[i] = (exit_close - hi) - 2.0 * c
            else:
                net[i] = (lo - exit_close) - 2.0 * c
            label[i] = 1 if net[i] > 0.0 else 0
            resolve[i] = end - i

    return label, net, resolve, trig


def straddle_outcomes(
    bars: pl.DataFrame,
    *,
    band_lookback: int = 12,
    horizon: int = 24,
    target_mult: float = 1.5,
    stop_mult: float = 1.5,
    atr_window: int = 24,
    cost_points: float = 12.0,
    point: float = 1e-5,
) -> pl.DataFrame:
    """Simula o straddle por barra → colunas label, net_price, resolve, triggered."""
    band_hi = bars.select(pl.col("high").rolling_max(band_lookback).shift(1))["high"].to_numpy()
    band_lo = bars.select(pl.col("low").rolling_min(band_lookback).shift(1))["low"].to_numpy()
    atr = bars.select(
        (pl.col("high") - pl.col("low")).rolling_mean(atr_window).shift(1).alias("a")
    )["a"].to_numpy()

    band_hi = np.nan_to_num(band_hi)
    band_lo = np.nan_to_num(band_lo)
    atr = np.nan_to_num(atr)
    tgt = atr * target_mult
    stp = atr * stop_mult
    cost = cost_points * point

    high = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)
    close = bars["close"].to_numpy().astype(np.float64)

    label, net, resolve, trig = _straddle_kernel(
        high, low, close, band_hi.astype(np.float64), band_lo.astype(np.float64),
        tgt.astype(np.float64), stp.astype(np.float64), float(cost), int(horizon),
    )
    return pl.DataFrame({
        "label": label, "net_price": net, "resolve": resolve, "triggered": trig,
    })


def build_straddle_panel(
    bars_by_symbol: dict[str, pl.DataFrame],
    feature_set: FeatureSet,
    *,
    asset_class_of: Callable[[str], str],
    horizon: int = 24,
    band_lookback: int = 12,
    target_mult: float = 1.5,
    stop_mult: float = 1.5,
    atr_window: int = 24,
    cost_points: float = 12.0,
    warmup: int = 150,
) -> UniversalPanel:
    """Painel multi-ativo cujo rótulo é P(straddle lucrativo) — direção-agnóstico."""
    symbols = sorted(bars_by_symbol)
    asset_index = {s: i for i, s in enumerate(symbols)}
    class_vocab = sorted({asset_class_of(s) for s in symbols})
    class_index = {c: i for i, c in enumerate(class_vocab)}
    feat_names = feature_set.names
    frames: list[pl.DataFrame] = []

    for symbol in symbols:
        bars = bars_by_symbol[symbol]
        if bars.height <= horizon + warmup:
            continue
        feats = feature_set.transform(bars)
        inst_point = 0.01 if symbol.upper().endswith("JPY") else 1e-5
        out = straddle_outcomes(
            bars, band_lookback=band_lookback, horizon=horizon,
            target_mult=target_mult, stop_mult=stop_mult, atr_window=atr_window,
            cost_points=cost_points, point=inst_point,
        )
        cls = asset_class_of(symbol)
        df = feats.with_columns(
            out["label"].alias("label"),
            pl.lit(symbol).alias("asset"),
            pl.lit(asset_index[symbol], dtype=pl.Int32).alias("asset_id"),
            pl.lit(class_index[cls], dtype=pl.Int32).alias("asset_class_id"),
        )
        df = df.head(df.height - horizon).slice(warmup)
        df = df.with_columns([pl.col(c).fill_null(0) for c in feat_names])
        frames.append(df)

    if not frames:
        raise ValueError("dados insuficientes para o painel de straddle")
    panel = pl.concat(frames, how="vertical").sort(["time", "asset_id"])
    return UniversalPanel(panel, feat_names, symbols, class_vocab)


def straddle_backtest(
    bars: pl.DataFrame,
    arm: np.ndarray,
    outcomes: pl.DataFrame,
    contract_size: float,
    *,
    lots: float = 0.1,
    initial_capital: float = 10_000.0,
    commission_per_lot: float = 7.0,
):
    """Equity de straddles NÃO sobrepostos: arma quando ``arm`` e está livre.

    Realiza o P&L (líquido de spread/slippage no ``net_price`` + comissão) na barra
    de resolução do trade. Retorna (equity, in_trade, trade_pnls).
    """
    net = outcomes["net_price"].to_numpy()
    resolve = outcomes["resolve"].to_numpy()
    n = bars.height
    realized = np.zeros(n, dtype=np.float64)
    in_trade = np.zeros(n, dtype=np.float64)
    busy_until = -1
    for i in range(n):
        if arm[i] and i > busy_until and resolve[i] > 0:
            j = min(i + int(resolve[i]), n - 1)
            pnl = net[i] * contract_size * lots - commission_per_lot * lots
            realized[j] += pnl
            in_trade[i:j + 1] = 1.0
            busy_until = i + int(resolve[i])
    equity = initial_capital + np.cumsum(realized)
    trade_pnls = realized[realized != 0.0]
    return equity, in_trade, trade_pnls
