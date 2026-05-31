"""Camada L5 — pesquisa: rotulagem, mineração de padrões e controle de overfitting."""
from innova_ea.research.labeling import forward_return, triple_barrier_labels
from innova_ea.research.overfitting import (
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from innova_ea.research.patterns import (
    PatternStats,
    conditional_stats,
    scan_conditions,
)

__all__ = [
    "forward_return",
    "triple_barrier_labels",
    "PatternStats",
    "conditional_stats",
    "scan_conditions",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]
