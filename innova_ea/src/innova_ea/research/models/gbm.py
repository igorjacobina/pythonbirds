"""Baseline universal — LightGBM com identidade de ativo categórica.

O modelo tabular campeão para começar: rápido, imune a outliers e com
*feature importance* imediato — ideal para confirmar se a engenharia de features
da Fase 2 carrega alfa real antes de investir no Super Cérebro (Transformer).

O ``asset_id`` (e a classe do ativo) entram como variáveis CATEGÓRICAS nativas:
as árvores aprendem ajustes por ativo enquanto compartilham splits universais —
um cross-asset learning implícito, sem embeddings explícitos.

Usa a API nativa do LightGBM (``lgb.train``) — sem dependência de scikit-learn
nem pandas.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from innova_ea.research.dataset import UniversalPanel, label_to_class
from innova_ea.research.models.base import UniversalModel


class LightGBMUniversal(UniversalModel):
    """Classificador multiclasse (down/flat/up) sobre o painel unificado."""

    def __init__(
        self,
        panel: UniversalPanel,
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
    ) -> None:
        super().__init__(panel)
        self._cols = list(self.feature_names) + ["asset_id", "asset_class_id"]
        self._cat_idx = [len(self.feature_names), len(self.feature_names) + 1]
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self._params = dict(
            objective="multiclass",
            num_class=3,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_data_in_leaf=min_child_samples,
            bagging_fraction=subsample,
            bagging_freq=1,
            feature_fraction=colsample_bytree,
            lambda_l2=reg_lambda,
            seed=random_state,
            num_threads=0,
            verbosity=-1,
        )
        self._booster = None

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self._cols).to_numpy()

    def fit(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame | None = None,
    ) -> "LightGBMUniversal":
        import lightgbm as lgb

        X = self._matrix(train_df)
        y = label_to_class(train_df["label"].to_numpy())

        # Pesos por classe para lidar com o desbalanceamento (flat domina).
        counts = np.bincount(y, minlength=3).astype(np.float64)
        cw = counts.sum() / (3.0 * np.maximum(counts, 1.0))
        weight = cw[y]

        dtrain = lgb.Dataset(
            X, label=y, weight=weight,
            categorical_feature=self._cat_idx, free_raw_data=False,
        )
        valid_sets = [dtrain]
        callbacks = [lgb.log_evaluation(0)]
        if valid_df is not None and valid_df.height:
            dvalid = lgb.Dataset(
                self._matrix(valid_df),
                label=label_to_class(valid_df["label"].to_numpy()),
                reference=dtrain, free_raw_data=False,
            )
            valid_sets.append(dvalid)
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self._booster = lgb.train(
            self._params, dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("modelo não treinado")
        proba = self._booster.predict(self._matrix(df))
        return np.asarray(proba, dtype=np.float64)

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        if self._booster is None:
            raise RuntimeError("modelo não treinado")
        imp = self._booster.feature_importance(importance_type=importance_type).astype(float)
        return dict(sorted(zip(self._cols, imp), key=lambda kv: kv[1], reverse=True))
