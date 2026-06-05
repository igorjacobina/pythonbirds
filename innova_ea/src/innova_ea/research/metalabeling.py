"""Meta-labeling (López de Prado) — o ML filtra uma regra primária, não adivinha direção.

Vimos que prever DIREÇÃO de majors a partir de OHLC é ~impossível (mercado
eficiente). O meta-labeling muda o problema:

  1. uma **regra primária** transparente decide o LADO da aposta (ex. rompimento
     de canal de Donchian ou do opening range de sessão);
  2. o **meta-modelo** (ML) decide se VALE A PENA tomar aquela aposta — prevê
     P(o trade primário vai ganhar) e dimensiona a posição por essa confiança.

O ML é treinado num problema mais tratável (filtrar falsos positivos de uma regra
conhecida), o que costuma elevar a precisão e transformar uma regra medíocre em
algo operável — sempre sujeito à validação OOS + Deflated Sharpe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from innova_ea.features.base import FeatureSet
from innova_ea.research.dataset import UniversalPanel, _adaptive_width
from innova_ea.research.labeling import forward_return, triple_barrier_labels
from innova_ea.research.models.base import UniversalModel
from innova_ea.strategy.base import Strategy

# Uma regra primária: (bars, features) -> array de lado em {-1, 0, +1} por barra.
Primary = Callable[[pl.DataFrame, pl.DataFrame], np.ndarray]


# ----------------------------------------------------------- regras primárias
def donchian_breakout_primary(lookback: int = 20) -> Primary:
    """Rompimento de canal de Donchian (causal): rompe a máxima de N barras → +1;
    rompe a mínima → -1; senão 0. Regra clássica e transparente."""

    def primary(bars: pl.DataFrame, features: pl.DataFrame) -> np.ndarray:
        hi = pl.col("high").shift(1).rolling_max(lookback)   # só barras passadas
        lo = pl.col("low").shift(1).rolling_min(lookback)
        side = (
            pl.when(pl.col("close") > hi).then(1)
            .when(pl.col("close") < lo).then(-1)
            .otherwise(0)
        )
        return bars.select(side.alias("s"))["s"].fill_null(0).to_numpy().astype(np.int64)

    return primary


def session_breakout_primary(prefix: str = "london") -> Primary:
    """Usa o rompimento do opening range de sessão (feature da Fase 2) como lado."""

    col = f"{prefix}_or_breakout"

    def primary(bars: pl.DataFrame, features: pl.DataFrame) -> np.ndarray:
        if col not in features.columns:
            raise ValueError(f"feature '{col}' ausente — inclua SessionOpenFeature({prefix})")
        return features[col].fill_null(0).to_numpy().astype(np.int64)

    return primary


# ----------------------------------------------------------- meta-rótulos
def meta_labels(
    bars: pl.DataFrame,
    side: np.ndarray,
    *,
    width,
    max_horizon: int,
    width_is_pct: bool = True,
) -> np.ndarray:
    """Meta-rótulo binário: 1 se a aposta primária no ``side`` GANHARIA, senão 0.

    Usa triple-barrier: vitória se a barreira tocada bate com o lado; na barreira
    de tempo, decide pelo sinal do retorno futuro. Onde ``side==0``, meta = 0.
    """
    tb = triple_barrier_labels(bars, width=width, max_horizon=max_horizon,
                               width_is_pct=width_is_pct)["label"].to_numpy()
    fwd = np.nan_to_num(forward_return(bars, max_horizon, log=False).to_numpy())
    side = np.asarray(side, dtype=np.int64)

    meta = np.where(
        tb == side, 1,
        np.where(tb == -side, 0,
                 np.where(side * fwd > 0, 1, 0)),  # barreira de tempo → sinal do retorno
    ).astype(np.int64)
    meta[side == 0] = 0
    return meta


# ----------------------------------------------------------- painel de eventos
def build_metalabel_panel(
    bars_by_symbol: dict[str, pl.DataFrame],
    feature_set: FeatureSet,
    primary: Primary,
    *,
    asset_class_of: Callable[[str], str],
    max_horizon: int = 24,
    atr_window: int = 20,
    vol_mult: float = 1.5,
    warmup: int = 150,
) -> UniversalPanel:
    """Painel multi-ativo só com EVENTOS (barras onde a primária dispara).

    Colunas: features + ``asset_id``/``asset_class_id`` + ``side`` + ``label`` (meta).
    """
    symbols = sorted(bars_by_symbol)
    asset_index = {s: i for i, s in enumerate(symbols)}
    class_vocab = sorted({asset_class_of(s) for s in symbols})
    class_index = {c: i for i, c in enumerate(class_vocab)}
    feat_names = feature_set.names
    frames: list[pl.DataFrame] = []

    for symbol in symbols:
        bars = bars_by_symbol[symbol]
        if bars.height <= max_horizon + warmup:
            continue
        feats = feature_set.transform(bars)
        side = primary(bars, feats)
        width = _adaptive_width(bars, atr_window, vol_mult)
        meta = meta_labels(bars, side, width=width, max_horizon=max_horizon)
        cls = asset_class_of(symbol)
        df = feats.with_columns(
            pl.Series("side", side),
            pl.Series("label", meta),
            pl.lit(symbol).alias("asset"),
            pl.lit(asset_index[symbol], dtype=pl.Int32).alias("asset_id"),
            pl.lit(class_index[cls], dtype=pl.Int32).alias("asset_class_id"),
        )
        df = df.head(df.height - max_horizon).slice(warmup)
        df = df.with_columns([pl.col(c).fill_null(0) for c in feat_names])
        df = df.filter(pl.col("side") != 0)   # só os eventos da regra primária
        frames.append(df)

    if not frames:
        raise ValueError("nenhum evento primário — verifique a regra/dados")
    panel = pl.concat(frames, how="vertical").sort(["time", "asset_id"])
    return UniversalPanel(panel, feat_names, symbols, class_vocab)


# ----------------------------------------------------------- meta-modelo (binário)
class MetaLabelModel(UniversalModel):
    """Classificador binário LightGBM: P(o trade primário vence)."""

    def __init__(self, panel: UniversalPanel, *, n_estimators: int = 400,
                 learning_rate: float = 0.05, num_leaves: int = 63,
                 min_child_samples: int = 100, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 random_state: int = 42, early_stopping_rounds: int = 50) -> None:
        super().__init__(panel)
        self._cols = list(self.feature_names) + ["asset_id", "asset_class_id"]
        self._cat_idx = [len(self.feature_names), len(self.feature_names) + 1]
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self._params = dict(
            objective="binary", learning_rate=learning_rate, num_leaves=num_leaves,
            min_data_in_leaf=min_child_samples, bagging_fraction=subsample,
            bagging_freq=1, feature_fraction=colsample_bytree, lambda_l2=reg_lambda,
            seed=random_state, num_threads=0, verbosity=-1,
        )
        self._booster = None

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self._cols).to_numpy()

    def fit(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None) -> "MetaLabelModel":
        import lightgbm as lgb

        X = self._matrix(train_df)
        y = train_df["label"].to_numpy().astype(np.int64)
        # is_unbalance lida com o desbalanceamento win/loss.
        params = dict(self._params, is_unbalance=True)
        dtrain = lgb.Dataset(X, label=y, categorical_feature=self._cat_idx, free_raw_data=False)
        valid_sets, callbacks = [dtrain], [lgb.log_evaluation(0)]
        if valid_df is not None and valid_df.height:
            dvalid = lgb.Dataset(self._matrix(valid_df), label=valid_df["label"].to_numpy(),
                                 reference=dtrain, free_raw_data=False)
            valid_sets.append(dvalid)
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self._booster = lgb.train(params, dtrain, num_boost_round=self.n_estimators,
                                  valid_sets=valid_sets, callbacks=callbacks)
        return self

    def predict_meta(self, df: pl.DataFrame) -> np.ndarray:
        """P(vitória) ∈ [0, 1] por linha."""
        if self._booster is None:
            raise RuntimeError("modelo não treinado")
        return np.asarray(self._booster.predict(self._matrix(df)), dtype=np.float64)

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        p = self.predict_meta(df)
        return np.column_stack([1.0 - p, p])

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        imp = self._booster.feature_importance(importance_type=importance_type).astype(float)
        return dict(sorted(zip(self._cols, imp), key=lambda kv: kv[1], reverse=True))

    def save(self, path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(p / "model.txt"))
        (p / "meta.json").write_text(json.dumps({
            "kind": "metalabel", "feature_names": self.feature_names,
            "asset_vocab": self.asset_vocab, "class_vocab": self.class_vocab,
        }, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path) -> "MetaLabelModel":
        import lightgbm as lgb
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        panel = UniversalPanel(pl.DataFrame(), meta["feature_names"],
                               meta["asset_vocab"], meta["class_vocab"])
        obj = cls(panel)
        obj._booster = lgb.Booster(model_file=str(p / "model.txt"))
        return obj


# ----------------------------------------------------------- estratégia
class MetaModelStrategy(Strategy):
    """Regra primária × meta-modelo: opera o lado da regra só com P(vitória) alta.

    target = side × P(vitória), aplicado apenas quando P(vitória) ≥ ``threshold``;
    dimensionado pela confiança (posição maior em sinais mais prováveis).
    """

    def __init__(self, model: MetaLabelModel, feature_set: FeatureSet, primary: Primary,
                 asset: str, asset_class: str, *, threshold: float = 0.5,
                 warmup: int = 150, name: str | None = None) -> None:
        if asset not in model.asset_vocab:
            raise ValueError(f"ativo '{asset}' fora do vocabulário do modelo")
        if asset_class not in model.class_vocab:
            raise ValueError(f"classe '{asset_class}' fora do vocabulário do modelo")
        self.model = model
        self.feature_set = feature_set
        self.primary = primary
        self.asset = asset
        self.asset_id = model.asset_vocab.index(asset)
        self.asset_class_id = model.class_vocab.index(asset_class)
        self.threshold = threshold
        self.warmup = warmup
        self.name = name or f"meta_{asset}"

    def generate_targets(self, bars: pl.DataFrame, inst) -> np.ndarray:
        feats = self.feature_set.transform(bars)
        side = self.primary(bars, feats).astype(np.float64)
        df = feats.with_columns(
            [pl.col(c).fill_null(0) for c in self.model.feature_names]
        ).with_columns(
            pl.lit(self.asset_id, dtype=pl.Int32).alias("asset_id"),
            pl.lit(self.asset_class_id, dtype=pl.Int32).alias("asset_class_id"),
        )
        pwin = np.nan_to_num(self.model.predict_meta(df))
        target = np.where((side != 0) & (pwin >= self.threshold), side * pwin, 0.0)
        target[: self.warmup] = 0.0
        return np.clip(target, -1.0, 1.0)
