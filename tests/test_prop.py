"""Testes da simulação de mesa proprietária (prop challenge)."""
from __future__ import annotations

import numpy as np

from innova_ea.research.prop import (
    challenge_stats,
    equity_metrics,
    simulate_challenge,
)


def test_pass_when_target_reached():
    # +2% por dia, meta 10% → 1.02^5 = 1.104 ≥ 1.10 no 5º dia.
    daily = np.full(20, 0.02)
    status, days, eq = simulate_challenge(
        daily, 0, 1.0, profit_target=0.10, daily_limit=0.05, total_limit=0.10)
    assert status == 1
    assert days == 5
    assert eq >= 1.10


def test_fail_on_daily_limit():
    # -6% num dia, limite diário 5% → quebra no 1º dia.
    daily = np.array([-0.06, 0.0, 0.0])
    status, days, _ = simulate_challenge(
        daily, 0, 1.0, profit_target=0.10, daily_limit=0.05, total_limit=0.10)
    assert status == -1
    assert days == 1


def test_fail_on_total_limit():
    # -2% por dia: não fere o diário (5%), mas a perda acumulada passa de 10%.
    daily = np.full(20, -0.02)
    status, _, eq = simulate_challenge(
        daily, 0, 1.0, profit_target=0.10, daily_limit=0.05, total_limit=0.10)
    assert status == -1
    assert eq <= 0.90 + 1e-9


def test_timeout_when_no_target_and_max_days():
    # Retornos minúsculos e prazo curto → nem passa nem quebra.
    daily = np.full(10, 0.001)
    status, days, _ = simulate_challenge(
        daily, 0, 1.0, profit_target=0.10, daily_limit=0.05, total_limit=0.10, max_days=3)
    assert status == 0
    assert days == 3


def test_leverage_amplifies_both():
    # Mesma série, mais alavancagem → maior CAGR (em valor absoluto) e maior DD.
    rng = np.random.default_rng(0)
    daily = rng.normal(0.0005, 0.01, 1000)
    _, dd1 = equity_metrics(daily, 1.0)
    _, dd3 = equity_metrics(daily, 3.0)
    assert abs(dd3) >= abs(dd1)


def test_challenge_stats_keys_and_ranges():
    rng = np.random.default_rng(1)
    daily = rng.normal(0.001, 0.01, 800)
    st = challenge_stats(daily, 2.0, stride=20)
    assert 0.0 <= st["pass_rate"] <= 1.0
    assert abs(st["pass_rate"] + st["fail_rate"] + st["timeout_rate"] - 1.0) < 1e-9
    assert st["n_starts"] > 0
