"""Camada L5 — pesquisa: controle de overfitting e (futuro) mineração de padrões."""
from innova_ea.research.overfitting import (
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]
