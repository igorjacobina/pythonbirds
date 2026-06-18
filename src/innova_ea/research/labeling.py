"""Rotulagem de eventos — alvos honestos para descoberta de padrões.

ATENÇÃO: rótulos OLHAM O FUTURO por definição (o que aconteceu *depois* da
barra). São o alvo do aprendizado e NUNCA devem ser usados como feature de
entrada. Mantê-los neste módulo, separados de ``features``, evita o erro.

Inclui o método **triple-barrier** (López de Prado): para cada barra, observa-se
qual barreira é tocada primeiro dentro de um horizonte — alvo de lucro (cima),
stop (baixo) ou tempo (nenhuma). É muito mais fiel à operação real do que um
retorno fixo de N barras.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.utils.jit import njit


def forward_return(bars: pl.DataFrame, horizon: int, *, log: bool = True) -> pl.Series:
    """Retorno futuro em ``horizon`` barras (close→close). NaN no fim da série."""
    c = pl.col("close")
    expr = (c.shift(-horizon) / c).log() if log else (c.shift(-horizon) / c - 1.0)
    return bars.select(expr.alias(f"fwd_ret_{horizon}")).to_series()


@njit
def _triple_barrier(high, low, upper, lower, max_h):
    n = high.shape[0]
    label = np.zeros(n, dtype=np.int8)
    touch = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        ub = upper[i]
        lb = lower[i]
        end = i + max_h
        if end > n - 1:
            end = n - 1
        lab = 0
        t = -1
        for j in range(i + 1, end + 1):
            # Convenção conservadora: checa o alvo de lucro e o stop na ordem
            # cima→baixo; toques simultâneos na mesma barra resolvem como +1.
            if high[j] >= ub:
                lab = 1
                t = j - i
                break
            if low[j] <= lb:
                lab = -1
                t = j - i
                break
        label[i] = lab
        touch[i] = t
    return label, touch


def triple_barrier_labels(
    bars: pl.DataFrame,
    *,
    width: float | np.ndarray,
    max_horizon: int,
    width_is_pct: bool = True,
) -> pl.DataFrame:
    """Rótulos triple-barrier por barra.

    Args:
        width: meia-largura das barreiras. Escalar (mesmo para todas as barras)
            ou array por barra (ex. múltiplo de volatilidade → barreiras
            adaptativas, recomendado).
        max_horizon: nº máximo de barras à frente (barreira de tempo).
        width_is_pct: se True, ``width`` é fração do preço (ex. 0.001 = 0,1%);
            se False, é distância absoluta de preço.

    Returns:
        DataFrame com colunas ``label`` (-1/0/1) e ``bars_to_touch`` (-1 se tempo).
    """
    if max_horizon < 1:
        raise ValueError("max_horizon deve ser >= 1")
    close = bars["close"].to_numpy().astype(np.float64)
    high = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)

    w = np.asarray(width, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(close.shape[0], float(width))
    if w.shape[0] != close.shape[0]:
        raise ValueError("array 'width' deve ter o mesmo tamanho das barras")

    if width_is_pct:
        upper = close * (1.0 + w)
        lower = close * (1.0 - w)
    else:
        upper = close + w
        lower = close - w

    label, touch = _triple_barrier(high, low, upper, lower, int(max_horizon))
    return pl.DataFrame({"label": label, "bars_to_touch": touch})
