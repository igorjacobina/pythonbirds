"""Testes do controle de overfitting: PSR e Deflated Sharpe Ratio."""
from __future__ import annotations

import pytest

from innova_ea.research.overfitting import (
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_psr_half_at_zero_sharpe():
    # Sharpe observado = benchmark = 0 → probabilidade 50%.
    assert probabilistic_sharpe_ratio(0.0, n_obs=100) == pytest.approx(0.5, abs=1e-9)


def test_psr_increases_with_sharpe_and_sample():
    low = probabilistic_sharpe_ratio(0.1, n_obs=100)
    high = probabilistic_sharpe_ratio(0.3, n_obs=100)
    more_data = probabilistic_sharpe_ratio(0.1, n_obs=1000)
    assert high > low > 0.5
    assert more_data > low  # mais amostra → mais confiança


def test_expected_max_sharpe_grows_with_trials():
    e10 = expected_max_sharpe(sharpe_std=0.1, n_trials=10)
    e1000 = expected_max_sharpe(sharpe_std=0.1, n_trials=1000)
    assert e1000 > e10 > 0.0
    assert expected_max_sharpe(0.1, n_trials=1) == 0.0


def test_deflation_reduces_confidence_vs_naive_psr():
    # Muitas tentativas com Sharpes ruidosos: o melhor é provavelmente sorte.
    trials = [0.02 * i for i in range(-25, 25)]  # média ~0, algum disperso
    selected = max(trials)
    naive = probabilistic_sharpe_ratio(selected, n_obs=500)
    dsr = deflated_sharpe_ratio(trials, n_obs=500, selected_sharpe=selected)
    assert 0.0 <= dsr <= 1.0
    assert dsr < naive  # deflação penaliza o número de tentativas


def test_deannualize_sharpe():
    # Sharpe anual 2.0 com 252 períodos/ano → por período ~0.126.
    assert deannualize_sharpe(2.0, 252) == pytest.approx(2.0 / (252 ** 0.5))


def test_deflated_sharpe_raises_on_empty():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([], n_obs=100)
