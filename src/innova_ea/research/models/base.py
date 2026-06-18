"""Contrato do modelo universal — uma interface, vários backbones.

LightGBM (baseline) e o Super Cérebro (Transformer) implementam a MESMA
interface, de modo que a estratégia de backtest e a validação são agnósticas ao
backbone. Toda previsão é uma distribuição de 3 classes: down / flat / up.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from innova_ea.research.dataset import UniversalPanel


class UniversalModel(ABC):
    """Modelo único treinado sobre o painel multi-ativo."""

    def __init__(self, panel: UniversalPanel) -> None:
        self.feature_names = panel.feature_names
        self.asset_vocab = panel.asset_vocab
        self.class_vocab = panel.class_vocab
        self.n_assets = panel.n_assets
        self.n_asset_classes = panel.n_asset_classes

    @abstractmethod
    def fit(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame | None = None,
    ) -> "UniversalModel":
        """Treina no recorte de treino (e usa validação para early stopping)."""

    @abstractmethod
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Probabilidades por classe, shape (n, 3) = [down, flat, up]."""

    def predict_signal(self, df: pl.DataFrame) -> np.ndarray:
        """Sinal direcional em [-1, 1]: P(up) - P(down)."""
        proba = self.predict_proba(df)
        return proba[:, 2] - proba[:, 0]
