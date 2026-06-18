"""Testes do motor de mineração de padrões (estatística condicional)."""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.research.patterns import conditional_stats, scan_conditions


def _data_with_edge(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    feat = rng.integers(0, 2, n)  # 0/1
    # Onde feat==1, retorno futuro positivo (+0.01); senão centrado em 0.
    fwd = np.where(feat == 1, 0.01, 0.0) + 0.002 * rng.standard_normal(n)
    return pl.DataFrame({"feat": feat, "fwd_ret": fwd})


def test_conditional_stats_detects_positive_edge():
    data = _data_with_edge()
    stats = conditional_stats(data, "fwd_ret", pl.col("feat") == 1, name="feat_on")
    assert stats.n > 800
    assert stats.mean > 0.008          # ~0.01 esperado
    assert stats.hit_rate > 0.9
    assert stats.t_stat > 5.0          # altamente significativo
    assert stats.p_value < 1e-6
    assert stats.edge_vs_baseline > 0  # melhor que o baseline geral


def test_conditional_stats_no_edge_is_insignificant():
    rng = np.random.default_rng(1)
    n = 2000
    data = pl.DataFrame(
        {"feat": rng.integers(0, 2, n), "fwd_ret": 0.002 * rng.standard_normal(n)}
    )
    stats = conditional_stats(data, "fwd_ret", pl.col("feat") == 1)
    assert abs(stats.t_stat) < 3.0     # sem edge real


def test_conditional_stats_ignores_null_targets():
    data = pl.DataFrame(
        {"feat": [1, 1, 1, 1], "fwd_ret": [0.01, None, 0.02, None]}
    )
    stats = conditional_stats(data, "fwd_ret", pl.col("feat") == 1)
    assert stats.n == 2  # apenas alvos não-nulos contam


def test_scan_conditions_filters_and_sorts():
    data = _data_with_edge()
    conditions = {
        "feat_on": pl.col("feat") == 1,
        "feat_off": pl.col("feat") == 0,
        "tiny": pl.col("feat") > 99,  # amostra zero → descartado por min_samples
    }
    results = scan_conditions(data, "fwd_ret", conditions, min_samples=30)
    names = [r.name for r in results]
    assert "tiny" not in names
    assert "feat_on" in names and "feat_off" in names
    # Ordenado por |t_stat| desc.: feat_on (forte) antes de feat_off.
    assert abs(results[0].t_stat) >= abs(results[-1].t_stat)
