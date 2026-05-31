"""Mineração de padrões — estatística condicional de retorno futuro.

Dado um conjunto de features causais e um alvo (retorno futuro/rótulo), avalia
*condições* (ex. "killzone de Londres E expansão de vol > 1,5 E thrust > 0") e
mede a distribuição do alvo no subconjunto: amostra, média, hit rate, expectancy
e significância (t-stat, p-valor) frente ao baseline. É a ponte entre as
features e uma hipótese de estratégia operável.

O t-stat aqui é exploratório; a confirmação anti-overfitting (Deflated Sharpe,
walk-forward) acontece na camada de validação — não confie em um único p-valor.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import polars as pl

_N = NormalDist()


@dataclass(frozen=True, slots=True)
class PatternStats:
    """Estatísticas de um padrão (subconjunto condicionado) sobre o alvo."""

    name: str
    n: int
    coverage: float          # fração das barras válidas que satisfazem a condição
    mean: float
    std: float
    hit_rate: float          # fração de alvos positivos
    expectancy: float        # média do alvo (= mean)
    t_stat: float            # vs. zero
    p_value: float           # bilateral (aprox. normal)
    edge_vs_baseline: float  # mean(condição) - mean(baseline)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"[{self.name}] n={self.n} ({self.coverage:.1%} cobertura) | "
            f"média={self.mean:+.5f} | hit={self.hit_rate:.1%} | "
            f"t={self.t_stat:+.2f} (p={self.p_value:.3f}) | "
            f"edge={self.edge_vs_baseline:+.5f}"
        )


def _stats_from_array(name: str, target: np.ndarray, n_valid_total: int,
                      baseline_mean: float) -> PatternStats:
    target = target[np.isfinite(target)]
    n = int(target.size)
    if n == 0:
        return PatternStats(name, 0, 0.0, math.nan, math.nan, math.nan,
                            math.nan, math.nan, math.nan, math.nan)
    mean = float(target.mean())
    std = float(target.std(ddof=1)) if n > 1 else 0.0
    hit = float(np.mean(target > 0))
    t_stat = (mean / (std / math.sqrt(n))) if std > 0 else 0.0
    p_value = 2.0 * (1.0 - _N.cdf(abs(t_stat)))
    coverage = n / n_valid_total if n_valid_total else 0.0
    return PatternStats(
        name=name, n=n, coverage=coverage, mean=mean, std=std, hit_rate=hit,
        expectancy=mean, t_stat=t_stat, p_value=p_value,
        edge_vs_baseline=mean - baseline_mean,
    )


def conditional_stats(
    data: pl.DataFrame,
    target_col: str,
    condition: pl.Expr,
    *,
    name: str = "pattern",
) -> PatternStats:
    """Estatísticas do alvo no subconjunto onde ``condition`` é verdadeira.

    Args:
        data: DataFrame com as features e a coluna-alvo (ex. retorno futuro).
        target_col: nome da coluna-alvo.
        condition: expressão booleana Polars sobre as features.
        name: rótulo do padrão para o relatório.
    """
    valid = data.filter(pl.col(target_col).is_not_null())
    n_valid = valid.height
    baseline_mean = float(valid[target_col].mean()) if n_valid else 0.0

    subset = valid.filter(condition.fill_null(False))
    target = subset[target_col].to_numpy().astype(np.float64)
    return _stats_from_array(name, target, n_valid, baseline_mean)


def scan_conditions(
    data: pl.DataFrame,
    target_col: str,
    conditions: dict[str, pl.Expr],
    *,
    min_samples: int = 30,
    sort_by: str = "t_stat",
) -> list[PatternStats]:
    """Avalia várias condições e devolve as significativas, ordenadas.

    Args:
        conditions: mapeia nome→expressão booleana.
        min_samples: descarta padrões com amostra menor que isto.
        sort_by: campo de ``PatternStats`` para ordenar (desc. por |valor|).
    """
    results = [
        conditional_stats(data, target_col, expr, name=name)
        for name, expr in conditions.items()
    ]
    results = [r for r in results if r.n >= min_samples]
    results.sort(key=lambda r: abs(getattr(r, sort_by)), reverse=True)
    return results
