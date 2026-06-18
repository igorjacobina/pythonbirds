"""Painel unificado multi-ativo — o dataset que alimenta o Super Cérebro.

Monta uma única tabela com TODOS os ativos empilhados: cada linha é um par
``(ativo, tempo)`` com as features causais (Fase 2), a identidade do ativo
(``asset_id``/``asset_class_id`` para embeddings/categóricos) e o rótulo
triple-barrier (Fase 2). É sobre este painel que o modelo único aprende a física
universal do preço, transferindo conhecimento entre ativos.

Rótulos triple-barrier são adaptativos por ativo (barreiras proporcionais à
volatilidade local), pois ouro e EUR/USD têm amplitudes muito diferentes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from innova_ea.features.base import FeatureSet
from innova_ea.research.labeling import triple_barrier_labels

# Mapeamento rótulo triple-barrier (-1/0/+1) -> classe do classificador (0/1/2).
_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = ["down", "flat", "up"]


def label_to_class(labels: np.ndarray) -> np.ndarray:
    """Converte rótulos {-1,0,1} em classes {0,1,2} (down/flat/up)."""
    out = np.ones_like(labels, dtype=np.int64)  # default flat=1
    out[labels == -1] = 0
    out[labels == 1] = 2
    return out


@dataclass(frozen=True, slots=True)
class UniversalPanel:
    """Painel multi-ativo pronto para treino."""

    frame: pl.DataFrame
    feature_names: list[str]
    asset_vocab: list[str]      # índice = asset_id
    class_vocab: list[str]      # índice = asset_class_id

    @property
    def n_assets(self) -> int:
        return len(self.asset_vocab)

    @property
    def n_asset_classes(self) -> int:
        return len(self.class_vocab)

    def features_matrix(self, df: pl.DataFrame | None = None) -> np.ndarray:
        src = self.frame if df is None else df
        return src.select(self.feature_names).to_numpy()

    def labels(self, df: pl.DataFrame | None = None) -> np.ndarray:
        src = self.frame if df is None else df
        return src["label"].to_numpy()


def _adaptive_width(bars: pl.DataFrame, atr_window: int, vol_mult: float) -> np.ndarray:
    """Largura de barreira por barra = vol_mult × ATR fracional (range/preço)."""
    frac_range = ((pl.col("high") - pl.col("low")) / pl.col("close"))
    width = bars.select(
        (frac_range.rolling_mean(atr_window) * vol_mult).alias("w")
    )["w"].to_numpy()
    # Antes do aquecimento do ATR, usa um piso pequeno para não rotular ruído.
    fallback = np.nanmedian(width[np.isfinite(width)]) if np.isfinite(width).any() else 1e-3
    width = np.where(np.isfinite(width) & (width > 0), width, fallback)
    return width


def build_universal_panel(
    bars_by_symbol: dict[str, pl.DataFrame],
    feature_set: FeatureSet,
    *,
    asset_class_of: Callable[[str], str],
    max_horizon: int = 16,
    atr_window: int = 20,
    vol_mult: float = 1.5,
    warmup: int = 150,
) -> UniversalPanel:
    """Constrói o painel unificado a partir de barras por ativo.

    Tratamento de nulos:
      * **aquecimento** (janelas rolantes indefinidas no início) → descarta as
        primeiras ``warmup`` barras de cada ativo, por POSIÇÃO;
      * **nulos estruturais** (features de sessão quando a sessão está inativa)
        → preenche com 0 (o flag ``*_active`` já indica a inatividade). NUNCA
        descarta a linha por isso, senão perderíamos quase todo o painel.

    Args:
        bars_by_symbol: mapeia símbolo → barras (schema canônico, validadas).
        feature_set: a matriz de features causais (Fase 2).
        asset_class_of: função símbolo → classe de ativo (forex/metal/index).
        max_horizon: horizonte (barras) da barreira de tempo do triple-barrier.
        atr_window/vol_mult: parametrizam a largura adaptativa das barreiras.
        warmup: nº de barras iniciais a descartar por ativo (aquecimento).
    """
    symbols = sorted(bars_by_symbol)
    asset_vocab = symbols
    asset_index = {s: i for i, s in enumerate(asset_vocab)}
    class_vocab = sorted({asset_class_of(s) for s in symbols})
    class_index = {c: i for i, c in enumerate(class_vocab)}

    feat_names = feature_set.names
    frames: list[pl.DataFrame] = []

    for symbol in symbols:
        bars = bars_by_symbol[symbol]
        if bars.height <= max_horizon + warmup:
            continue
        feats = feature_set.transform(bars)  # time + features
        width = _adaptive_width(bars, atr_window, vol_mult)
        labels = triple_barrier_labels(
            bars, width=width, max_horizon=max_horizon, width_is_pct=True
        )
        cls = asset_class_of(symbol)
        df = feats.with_columns(
            labels["label"].alias("label"),
            pl.lit(symbol).alias("asset"),
            pl.lit(asset_index[symbol], dtype=pl.Int32).alias("asset_id"),
            pl.lit(cls).alias("asset_class"),
            pl.lit(class_index[cls], dtype=pl.Int32).alias("asset_class_id"),
        )
        # Cauda sem futuro completo (rótulo) e aquecimento inicial → descartados.
        df = df.head(df.height - max_horizon).slice(warmup)
        # Nulos estruturais restantes (sessão inativa) → 0.
        df = df.with_columns([pl.col(c).fill_null(0) for c in feat_names])
        frames.append(df)

    if not frames:
        raise ValueError("nenhum ativo com barras suficientes para montar o painel")

    panel = pl.concat(frames, how="vertical").sort(["time", "asset_id"])
    return UniversalPanel(
        frame=panel,
        feature_names=feat_names,
        asset_vocab=asset_vocab,
        class_vocab=class_vocab,
    )


class PerAssetScaler:
    """Padroniza features por ativo (z-score), ajustado SOMENTE no treino.

    Ouro e EUR/USD têm escalas distintas; padronizar por ativo coloca tudo em
    base comparável para a rede, sem vazar estatísticas do futuro/teste.
    """

    def __init__(self) -> None:
        self._mean: dict[int, np.ndarray] = {}
        self._std: dict[int, np.ndarray] = {}
        self._global_mean: np.ndarray | None = None
        self._global_std: np.ndarray | None = None
        self._features: list[str] = []

    def fit(self, panel: UniversalPanel, train_df: pl.DataFrame) -> "PerAssetScaler":
        self._features = panel.feature_names
        X = train_df.select(self._features).to_numpy()
        self._global_mean = np.nanmean(X, axis=0)
        self._global_std = np.nanstd(X, axis=0) + 1e-8
        aid = train_df["asset_id"].to_numpy()
        for a in np.unique(aid):
            mask = aid == a
            self._mean[int(a)] = np.nanmean(X[mask], axis=0)
            self._std[int(a)] = np.nanstd(X[mask], axis=0) + 1e-8
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self._features).to_numpy().astype(np.float64)
        aid = df["asset_id"].to_numpy()
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            a = int(aid[i])
            mean = self._mean.get(a, self._global_mean)
            std = self._std.get(a, self._global_std)
            out[i] = (X[i] - mean) / std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
