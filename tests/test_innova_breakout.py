"""Testes do INNOVA Breakout (fusão Larry Williams + Andrea Unger)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from innova_ea.research import orb_backtest
from innova_ea.research.innova_breakout import innova_breakout_outcomes


def _bars(times, opens, highs, lows, closes):
    return pl.DataFrame({
        "time": times, "open": list(map(float, opens)), "high": list(map(float, highs)),
        "low": list(map(float, lows)), "close": list(map(float, closes)),
        "volume": [1.0] * len(times),
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def test_innova_long_breakout_with_expansion():
    # Terça, 6 barras. or_bars=2 → range de abertura = barras 0,1 (high 11, low 9, largura 2).
    # Sem dia anterior (prev_range=0) → gatilho de Williams desligado nesse dia.
    # barra 3 rompe p/ cima → long em 11; alvo 13 atingido.
    t0 = datetime(2021, 1, 5, tzinfo=timezone.utc)  # terça
    times = [t0 + timedelta(hours=i) for i in range(6)]
    bars = _bars(
        times,
        opens=[10, 10, 10, 11, 11, 11],
        highs=[11, 10.5, 10.8, 12, 13, 13],
        lows=[9, 9.5, 10, 10.5, 11, 11],
        closes=[10, 10, 10.5, 11.5, 12.5, 12.5],
    )
    out = innova_breakout_outcomes(bars, or_bars=2, reward_risk=1.0, cost_frac=0.0,
                                   skip_monday=True, min_or_frac=0.0, max_prior_mult=0.0)
    assert out["triggered"].sum() == 1
    idx = int(out["triggered"].to_numpy().argmax())
    assert out["direction"][idx] == 1
    assert out["net_ret"][idx] > 0


def test_innova_skips_monday():
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)  # SEGUNDA-feira
    times = [t0 + timedelta(hours=i) for i in range(6)]
    bars = _bars(
        times,
        opens=[10, 10, 10, 11, 11, 11],
        highs=[11, 10.5, 10.8, 12, 13, 13],
        lows=[9, 9.5, 10, 10.5, 11, 11],
        closes=[10, 10, 10.5, 11.5, 12.5, 12.5],
    )
    out = innova_breakout_outcomes(bars, or_bars=2, skip_monday=True,
                                   min_or_frac=0.0, max_prior_mult=0.0)
    assert out["triggered"].sum() == 0


def test_innova_williams_volatility_gate_blocks_narrow_open():
    # Dois dias. min_or_frac alto exige que o range de abertura seja grande
    # frente ao range do dia anterior. No 2º dia o range de abertura é estreito
    # → gatilho de Williams bloqueia a operação.
    t0 = datetime(2021, 1, 5, tzinfo=timezone.utc)  # terça
    day1_t = [t0 + timedelta(hours=i) for i in range(6)]
    day2_t = [t0 + timedelta(days=1, hours=i) for i in range(6)]
    # Dia 1: range diário grande (high 20, low 0 → range 20).
    d1 = ([10, 10, 10, 11, 11, 11], [11, 10.5, 10.8, 12, 20, 13],
          [9, 9.5, 10, 10.5, 0, 11], [10, 10, 10.5, 11.5, 12.5, 12.5])
    # Dia 2: range de abertura estreito (high 10.1, low 9.9 → largura 0.2),
    # muito menor que min_or_frac * prev_range (0.5 * 20 = 10).
    d2 = ([10, 10, 10, 11, 11, 11], [10.1, 10.05, 10.08, 12, 13, 13],
          [9.9, 9.95, 9.98, 10.5, 11, 11], [10, 10, 10.5, 11.5, 12.5, 12.5])
    bars = _bars(
        day1_t + day2_t,
        d1[0] + d2[0], d1[1] + d2[1], d1[2] + d2[2], d1[3] + d2[3],
    )
    out = innova_breakout_outcomes(bars, or_bars=2, reward_risk=1.0, cost_frac=0.0,
                                   skip_monday=False, min_or_frac=0.5, max_prior_mult=0.0)
    # Só o dia 1 pode operar (no dia 2 o gatilho de volatilidade bloqueia).
    assert out["triggered"].sum() <= 1


def test_innova_backtest_runs():
    rng = np.random.default_rng(0)
    days, rpd = 40, 8
    t0 = datetime(2021, 1, 5, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(days * rpd)]
    close = 100 + np.cumsum(rng.normal(0, 0.5, days * rpd))
    high = close + np.abs(rng.normal(0, 0.3, days * rpd))
    low = close - np.abs(rng.normal(0, 0.3, days * rpd))
    open_ = close + rng.normal(0, 0.2, days * rpd)
    bars = _bars(times, open_, np.maximum.reduce([high, open_, close]),
                 np.minimum.reduce([low, open_, close]), close)
    out = innova_breakout_outcomes(bars, or_bars=3, reward_risk=1.0, cost_frac=0.0004,
                                   skip_monday=False, min_or_frac=0.0, max_prior_mult=0.0)
    equity, pos, trades = orb_backtest(out, initial_capital=10_000.0)
    assert equity.shape[0] == bars.height
    assert np.isfinite(equity).all()
