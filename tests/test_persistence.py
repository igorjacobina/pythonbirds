"""Testes de persistência dos modelos (save/load para deploy)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from innova_ea.research.dataset import UniversalPanel
from innova_ea.research.models.gbm import LightGBMUniversal


def _planted_panel(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.standard_normal(n)
    label = np.where(f0 > 0.5, 1, np.where(f0 < -0.5, -1, 0)).astype(np.int64)
    aid = rng.integers(0, 2, n).astype(np.int32)
    t0 = datetime(2021, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame({
        "time": [t0 + timedelta(minutes=15 * i) for i in range(n)],
        "asset_id": aid,
        "asset_class_id": np.zeros(n, dtype=np.int32),
        "f0": f0,
        "f1": rng.standard_normal(n) * 0.1,
        "label": label,
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))
    return UniversalPanel(frame, ["f0", "f1"], ["A", "B"], ["x"])


def test_lightgbm_save_load_roundtrip(tmp_path):
    panel = _planted_panel()
    model = LightGBMUniversal(panel, n_estimators=60, min_child_samples=20).fit(panel.frame)
    before = model.predict_proba(panel.frame)

    model.save(tmp_path / "lgbm")
    assert (tmp_path / "lgbm" / "model.txt").exists()
    assert (tmp_path / "lgbm" / "meta.json").exists()

    reloaded = LightGBMUniversal.load(tmp_path / "lgbm")
    after = reloaded.predict_proba(panel.frame)
    assert np.allclose(before, after, atol=1e-9)
    assert reloaded.feature_names == panel.feature_names
    assert reloaded.asset_vocab == panel.asset_vocab


def test_super_brain_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from datetime import datetime, timezone

    from innova_ea.core.enums import Timeframe
    from innova_ea.data.sources.synthetic import SyntheticSource
    from innova_ea.features import CloseLocationValue, EfficiencyRatio, FeatureSet, YangZhang
    from innova_ea.research import build_universal_panel, get_super_brain

    cls = {"EURUSD": "forex", "XAUUSD": "metal"}
    bars_by = {
        s: SyntheticSource(seed=i, annual_vol=0.1).fetch(
            s, Timeframe.M15,
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 2, 1, tzinfo=timezone.utc))
        for i, s in enumerate(cls)
    }
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    panel = build_universal_panel(bars_by, fs, asset_class_of=lambda s: cls[s],
                                  max_horizon=6, vol_mult=1.5)
    model = get_super_brain(panel, window=12, d_model=24, n_heads=2, n_layers=1,
                            d_ff=48, epochs=2, batch_size=128, seed=0).fit(panel.frame)
    before = model.predict_proba(panel.frame)

    from innova_ea.research.models.super_brain import SuperBrain
    model.save(tmp_path / "sb")
    assert (tmp_path / "sb" / "super_brain.pt").exists()
    reloaded = SuperBrain.load(tmp_path / "sb")
    after = reloaded.predict_proba(panel.frame)
    assert np.allclose(before, after, atol=1e-5)
