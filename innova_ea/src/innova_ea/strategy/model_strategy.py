"""Ponte modelo → estratégia: previsões viram alvo de posição no backtest.

Fecha o ciclo institucional: o mesmo modelo universal validado na pesquisa é
avaliado pelo MESMO engine de risco da Fase 3 (spread, swap, margem, stop-out).
Não há caminho privilegiado para a IA — o sinal só vale se sobreviver a custos e
risco e, depois, ao Deflated Sharpe.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.core.instruments import Instrument
from innova_ea.features.base import FeatureSet
from innova_ea.research.models.base import UniversalModel
from innova_ea.strategy.base import Strategy


class ModelStrategy(Strategy):
    """Transforma previsões de um ``UniversalModel`` em alvos de posição [-1, 1].

    Args:
        model: modelo universal treinado.
        feature_set: o MESMO ``FeatureSet`` usado para montar o painel.
        asset: símbolo deste backtest (deve estar no vocabulário do modelo).
        asset_class: classe do ativo (deve estar no vocabulário de classes).
        threshold: zona morta — sinais com |valor| < threshold viram 0 (não opera).
        scale: amplificação do sinal antes do clip em [-1, 1].
        warmup: nº de barras iniciais sem operar (aquecimento das features), igual
            ao usado no painel de treino para consistência treino/execução.
    """

    def __init__(
        self,
        model: UniversalModel,
        feature_set: FeatureSet,
        asset: str,
        asset_class: str,
        *,
        threshold: float = 0.0,
        scale: float = 1.0,
        warmup: int = 150,
        name: str | None = None,
    ) -> None:
        if asset not in model.asset_vocab:
            raise ValueError(f"ativo '{asset}' fora do vocabulário do modelo")
        if asset_class not in model.class_vocab:
            raise ValueError(f"classe '{asset_class}' fora do vocabulário do modelo")
        self.model = model
        self.feature_set = feature_set
        self.asset = asset
        self.asset_id = model.asset_vocab.index(asset)
        self.asset_class_id = model.class_vocab.index(asset_class)
        self.threshold = threshold
        self.scale = scale
        self.warmup = warmup
        self.name = name or f"model_{asset}"

    def generate_targets(self, bars: pl.DataFrame, inst: Instrument) -> np.ndarray:
        feats = self.feature_set.transform(bars)  # time + features
        # Nulos estruturais (sessão inativa) → 0, exatamente como no treino.
        df = feats.with_columns(
            [pl.col(c).fill_null(0) for c in self.model.feature_names]
        ).with_columns(
            pl.lit(self.asset_id, dtype=pl.Int32).alias("asset_id"),
            pl.lit(self.asset_class_id, dtype=pl.Int32).alias("asset_class_id"),
            pl.lit(self.asset).alias("asset"),
        )

        signal = self.model.predict_signal(df).astype(np.float64)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Aquecimento inicial (janelas rolantes indefinidas) → fora do mercado.
        signal[: self.warmup] = 0.0

        # Zona morta + escala + clip.
        signal = np.where(np.abs(signal) < self.threshold, 0.0, signal * self.scale)
        return np.clip(signal, -1.0, 1.0)
