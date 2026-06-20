"""Avaliação de MESA PROPRIETÁRIA (prop firm challenge).

Simula, sobre o histórico real de um operacional, a regra de uma mesa
(ex.: FTMO): bater a META de lucro antes de violar a perda DIÁRIA ou a perda
TOTAL. Varre níveis de alavancagem e mede, de forma honesta:

  * probabilidade de PASSAR (começando o desafio em muitos pontos do histórico)
  * dias medianos até passar
  * retorno anual e pior drawdown da estratégia naquele nível de risco

Granularidade DIÁRIA: o limite de perda diária é checado no fim do dia. Mesas
reais checam intradiário/tick, então o resultado aqui é um limite SUPERIOR
otimista para o limite diário — trate como referência, não garantia.
"""
from __future__ import annotations

import numpy as np

# Preset clássico estilo FTMO (fase 1): 10% de meta, 5% perda diária, 10% total.
FTMO_CHALLENGE = dict(profit_target=0.10, daily_limit=0.05, total_limit=0.10)


def simulate_challenge(daily: np.ndarray, start: int, lev: float, *,
                       profit_target: float, daily_limit: float,
                       total_limit: float, max_days: int = 0):
    """Simula UM desafio a partir de ``start``.

    Retorna ``(status, days, equity)`` com status: 1=passou, -1=quebrou,
    0=acabou o dado/tempo sem passar. ``max_days<=0`` = sem limite de tempo.
    """
    equity = 1.0
    floor_total = 1.0 - total_limit
    target = 1.0 + profit_target
    n = daily.shape[0]
    days = 0
    i = start
    while i < n:
        if max_days > 0 and days >= max_days:
            return 0, days, equity
        day_start = equity
        equity *= (1.0 + daily[i] * lev)
        days += 1
        i += 1
        if equity <= floor_total or equity <= day_start * (1.0 - daily_limit):
            return -1, days, equity
        if equity >= target:
            return 1, days, equity
    return 0, days, equity


def equity_metrics(daily: np.ndarray, lev: float, *, days_per_year: int = 252):
    """CAGR e pior drawdown da estratégia (curva cheia) naquele ``lev``."""
    r = daily * lev
    eq = np.cumprod(1.0 + r)
    if eq.size == 0:
        return float("nan"), 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1.0).min())
    years = eq.size / days_per_year
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else float("nan")
    return cagr, max_dd


def challenge_stats(daily: np.ndarray, lev: float, *, profit_target: float = 0.10,
                    daily_limit: float = 0.05, total_limit: float = 0.10,
                    max_days: int = 0, stride: int = 1):
    """Estatística do desafio começando em vários pontos do histórico (rolling)."""
    n = daily.shape[0]
    passes = fails = timeouts = 0
    pass_days: list[int] = []
    for s in range(0, n, max(1, stride)):
        status, days, _ = simulate_challenge(
            daily, s, lev, profit_target=profit_target, daily_limit=daily_limit,
            total_limit=total_limit, max_days=max_days)
        if status == 1:
            passes += 1
            pass_days.append(days)
        elif status == -1:
            fails += 1
        else:
            timeouts += 1
    total = passes + fails + timeouts
    cagr, max_dd = equity_metrics(daily, lev)
    return {
        "lev": lev,
        "pass_rate": passes / total if total else 0.0,
        "fail_rate": fails / total if total else 0.0,
        "timeout_rate": timeouts / total if total else 0.0,
        "median_pass_days": float(np.median(pass_days)) if pass_days else float("nan"),
        "cagr": cagr,
        "max_drawdown": max_dd,
        "n_starts": total,
    }


def sweep_leverage(daily: np.ndarray, levs, **kwargs):
    """Roda ``challenge_stats`` para cada alavancagem em ``levs``."""
    return [challenge_stats(daily, float(lv), **kwargs) for lv in levs]
