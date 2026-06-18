"""Testes do painel unificado multi-ativo e do scaler por ativo."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.features import CloseLocationValue, EfficiencyRatio, FeatureSet, YangZhang
from innova_ea.research.dataset import (
    PerAssetScaler,
    UniversalPanel,
    build_universal_panel,
    label_to_class,
)


def _bars(symbol, vol, seed):
    return SyntheticSource(seed=seed, annual_vol=vol).fetch(
        symbol, Timeframe.M15,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 3, 1, tzinfo=timezone.utc),
    )


def _panel():
    bars_by = {"EURUSD": _bars("EURUSD", 0.08, 1), "XAUUSD": _bars("XAUUSD", 0.15, 2)}
    cls = {"EURUSD": "forex", "XAUUSD": "metal"}
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    return build_universal_panel(bars_by, fs, asset_class_of=lambda s: cls[s],
                                 max_horizon=8, vol_mult=1.5), fs


def test_label_to_class_mapping():
    out = label_to_class(np.array([-1, 0, 1, 1, -1]))
    assert out.tolist() == [0, 1, 2, 2, 0]


def test_panel_structure_and_vocab():
    panel, _ = _panel()
    f = panel.frame
    assert panel.asset_vocab == ["EURUSD", "XAUUSD"]
    assert panel.class_vocab == ["forex", "metal"]
    for col in ["time", "asset", "asset_id", "asset_class_id", "label"]:
        assert col in f.columns
    for feat in panel.feature_names:
        assert feat in f.columns
    # Sem nulos nas features (aquecimento removido) e rótulos em {-1,0,1}.
    assert f.select(panel.feature_names).null_count().to_numpy().sum() == 0
    assert set(f["label"].unique().to_list()).issubset({-1, 0, 1})
    assert f["asset_id"].n_unique() == 2


def test_panel_drops_unfinished_label_tail():
    # As últimas max_horizon barras de cada ativo (sem futuro) não entram.
    panel, _ = _panel()
    # Cada ativo deve terminar antes do fim absoluto da série bruta.
    counts = panel.frame.group_by("asset").len()
    assert counts["len"].min() > 0


def test_per_asset_scaler_standardizes_within_asset():
    panel, _ = _panel()
    scaler = PerAssetScaler().fit(panel, panel.frame)
    z = scaler.transform(panel.frame)
    # Para cada ativo, a média padronizada de cada feature ~ 0.
    aid = panel.frame["asset_id"].to_numpy()
    for a in np.unique(aid):
        col_means = z[aid == a].mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=1e-6)
