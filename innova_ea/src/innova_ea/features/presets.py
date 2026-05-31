"""Conjuntos de features de produção — compartilhados entre treino e execução.

CRÍTICO: o modelo deve ver, na execução ao vivo, EXATAMENTE as mesmas features
(mesma ordem e parâmetros) usadas no treino. Centralizar a definição aqui evita
divergência treino/produção (uma fonte clássica de erro silencioso).
"""
from __future__ import annotations

from innova_ea.features.base import FeatureSet
from innova_ea.features.directional import (
    BodyRatio,
    CloseLocationValue,
    EfficiencyRatio,
    GapOpen,
    LowerWick,
    UpperWick,
)
from innova_ea.features.returns import CumulativeReturn, LogReturn, ReturnZScore
from innova_ea.features.sessions import (
    LONDON,
    LONDON_KILLZONE,
    NEW_YORK,
    NEWYORK_KILLZONE,
    TOKYO,
    SessionOpenFeature,
)
from innova_ea.features.time_features import TimeOfDayCyclical
from innova_ea.features.volatility import (
    GarmanKlass,
    Parkinson,
    RealizedVol,
    VolatilityRegime,
    YangZhang,
)


def default_feature_set() -> FeatureSet:
    """Conjunto de features de produção do INNOVA EA (causal, sem look-ahead).

    Cobre as quatro famílias da Fase 2: volatilidade (estimadores OHLC),
    direção/microestrutura, retorno normalizado e sessões/horário de liquidez.
    """
    return FeatureSet([
        # --- Volatilidade (estimadores OHLC + regime) ---
        YangZhang(20),
        Parkinson(20),
        GarmanKlass(20),
        RealizedVol(20),
        VolatilityRegime(10, 100),
        # --- Direção e microestrutura ---
        EfficiencyRatio(20),
        EfficiencyRatio(60, name="eff_ratio_60"),
        CloseLocationValue(),
        BodyRatio(),
        UpperWick(),
        LowerWick(),
        GapOpen(),
        # --- Retorno normalizado ---
        LogReturn(1),
        ReturnZScore(50),
        CumulativeReturn(20),
        # --- Tempo e sessões de liquidez ---
        TimeOfDayCyclical(),
        SessionOpenFeature(TOKYO, opening_range_bars=4),
        SessionOpenFeature(LONDON, opening_range_bars=4),
        SessionOpenFeature(NEW_YORK, opening_range_bars=4),
        SessionOpenFeature(LONDON_KILLZONE, opening_range_bars=4),
        SessionOpenFeature(NEWYORK_KILLZONE, opening_range_bars=4),
    ])
