"""Controle de overfitting — quanto de um bom Sharpe é skill vs. sorte de busca.

Implementa o Probabilistic Sharpe Ratio (PSR) e o Deflated Sharpe Ratio (DSR)
de Bailey & López de Prado. A intuição:

  * Quanto MAIS estratégias você testa, MAIOR o melhor Sharpe esperado só por
    acaso. O DSR desconta exatamente esse efeito (número de tentativas).
  * Um Sharpe alto obtido após mil tentativas pode ter DSR ~0.5 (indistinguível
    de ruído). Só DSR alto (ex. > 0.95) sugere edge real.

Todas as funções operam sobre Sharpe *por período* (não anualizado). Use
``deannualize_sharpe`` para converter um Sharpe anual.

Referência: Bailey, D. & López de Prado, M. (2014), "The Deflated Sharpe Ratio".
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def deannualize_sharpe(annual_sharpe: float, periods_per_year: float) -> float:
    """Converte Sharpe anualizado em Sharpe por período."""
    return annual_sharpe / math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR: probabilidade de o Sharpe verdadeiro exceder ``benchmark``.

    Ajusta para tamanho de amostra e para os momentos superiores (assimetria e
    curtose) dos retornos — retornos com cauda gorda exigem mais evidência.

    Args:
        sharpe: Sharpe observado por período.
        n_obs: número de retornos na amostra.
        benchmark: Sharpe de referência a superar (default 0).
        skew: assimetria dos retornos.
        kurtosis: curtose dos retornos (3 = normal).

    Returns:
        Probabilidade em [0, 1].
    """
    if n_obs < 2:
        return float("nan")
    denom = math.sqrt(
        max(1e-12, 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2)
    )
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / denom
    return _N.cdf(z)


def expected_max_sharpe(sharpe_std: float, n_trials: int) -> float:
    """Sharpe máximo ESPERADO sob a hipótese nula, dado ``n_trials`` tentativas.

    Aproximação de estatística de extremos (Bailey & López de Prado):
        E[max SR] ≈ σ_SR · [(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))]
    onde γ é a constante de Euler-Mascheroni.
    """
    if n_trials < 1:
        raise ValueError("n_trials deve ser >= 1")
    if n_trials == 1:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    trial_sharpes: list[float] | np.ndarray,
    n_obs: int,
    *,
    selected_sharpe: float | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR: PSR usando como benchmark o máximo Sharpe esperado por puro acaso.

    Args:
        trial_sharpes: Sharpes (por período) de TODAS as configurações testadas.
        n_obs: número de retornos da estratégia selecionada.
        selected_sharpe: Sharpe da estratégia escolhida (default = max dos trials).
        skew, kurtosis: momentos dos retornos da estratégia selecionada.

    Returns:
        Probabilidade em [0, 1]. DSR alto (> 0.95) = edge provavelmente real.
    """
    sr = np.asarray(trial_sharpes, dtype=np.float64)
    n_trials = sr.size
    if n_trials == 0:
        raise ValueError("trial_sharpes não pode ser vazio")
    sr_std = float(sr.std(ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(sr_std, n_trials)
    observed = float(sr.max()) if selected_sharpe is None else selected_sharpe
    return probabilistic_sharpe_ratio(
        observed, n_obs, benchmark=sr0, skew=skew, kurtosis=kurtosis
    )
