"""Testes do motor de relative-value / pares (statistical arbitrage)."""
from __future__ import annotations

import numpy as np

from innova_ea.research.pairs import (
    PairConfig,
    pairs_backtest,
    pairs_signal,
    spread_zscore,
)


def test_signal_entry_exit_logic():
    z = np.array([0, 0, 2.5, 2.5, 1.0, 0.3, 0.0, -2.5, -0.2], dtype=np.float64)
    cfg = PairConfig(entry=2.0, exit=0.5, stop=4.0, max_hold=100)
    sig = pairs_signal(z, cfg)
    # entra vendido em z>2; segura; sai quando |z|<0.5; entra comprado em z<-2; sai.
    assert sig.tolist() == [0, 0, -1, -1, -1, 0, 0, 1, 0]


def test_signal_divergence_stop():
    z = np.array([0, 2.5, 5.0, 5.0], dtype=np.float64)   # estoura o stop (|z|>4)
    cfg = PairConfig(entry=2.0, exit=0.5, stop=4.0, max_hold=100)
    sig = pairs_signal(z, cfg)
    assert sig[1] == -1 and sig[2] == 0          # cortou na divergência


def _cointegrated_pair(n=3000, seed=0, phi=0.95):
    """B = random walk; A = B + ruído MEAN-REVERTING (β=1) → par cointegrado."""
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = phi * ou[t - 1] + rng.normal(0, 0.01)   # Ornstein-Uhlenbeck
    log_a = log_b + ou
    return log_a, log_b


def _run(log_a, log_b, cfg, cost):
    spread, z, _ = spread_zscore(log_a, log_b, cfg)
    sig = pairs_signal(z, cfg)
    return pairs_backtest(spread, sig, cost_frac_per_turn=cost)[0]


def test_pairs_profits_on_cointegrated_pair():
    log_a, log_b = _cointegrated_pair()
    cfg = PairConfig(z_window=50, entry=1.5, exit=0.3, stop=4.0)
    spread, z, beta = spread_zscore(log_a, log_b, cfg)
    assert 0.7 < beta < 1.3                      # β estrutural ~ 1 (par real)
    equity, _, trades = pairs_backtest(spread, pairs_signal(z, cfg),
                                       cost_frac_per_turn=0.0, initial_capital=10_000.0)
    assert equity[-1] > 10_000.0                 # arbitrar a reversão LUCRA (sem custo)
    assert len(trades) > 0


def test_costs_reduce_pairs_equity():
    log_a, log_b = _cointegrated_pair(seed=1)
    cfg = PairConfig(z_window=50, entry=1.5, exit=0.3)
    assert _run(log_a, log_b, cfg, 0.001)[-1] < _run(log_a, log_b, cfg, 0.0)[-1]


def test_cointegrated_beats_independent_walks():
    cfg = PairConfig(z_window=50, entry=1.5, exit=0.3)
    coint = _run(*_cointegrated_pair(seed=7), cfg, 0.001)
    rng = np.random.default_rng(7)
    la = np.cumsum(rng.normal(0, 0.01, 3000))
    lb = np.cumsum(rng.normal(0, 0.01, 3000))
    indep = _run(la, lb, cfg, 0.001)
    # O par REALMENTE cointegrado supera passeios independentes (sem reversão real).
    assert coint[-1] > indep[-1]
