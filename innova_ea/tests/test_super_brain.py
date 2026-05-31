"""Testes do Super Cérebro (Transformer causal). Pulados se PyTorch ausente."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("torch")  # smoke test só roda onde houver PyTorch

from innova_ea.core.enums import Timeframe  # noqa: E402
from innova_ea.data.sources.synthetic import SyntheticSource  # noqa: E402
from innova_ea.features import (  # noqa: E402
    CloseLocationValue,
    EfficiencyRatio,
    FeatureSet,
    YangZhang,
)
from innova_ea.research import build_universal_panel, get_super_brain  # noqa: E402


def _panel():
    cls = {"EURUSD": "forex", "XAUUSD": "metal"}
    bars_by = {
        s: SyntheticSource(seed=i, annual_vol=0.1).fetch(
            s, Timeframe.M15,
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 2, 15, tzinfo=timezone.utc),
        )
        for i, s in enumerate(cls)
    }
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    panel = build_universal_panel(bars_by, fs, asset_class_of=lambda s: cls[s],
                                  max_horizon=6, vol_mult=1.5)
    return panel


def _small_model(panel):
    return get_super_brain(panel, window=12, d_model=24, n_heads=2, n_layers=2,
                           d_ff=48, epochs=2, batch_size=128, seed=0)


def test_super_brain_predicts_valid_probabilities():
    panel = _panel()
    frame = panel.frame
    n = frame.height
    train, test = frame[: int(n * 0.7)], frame[int(n * 0.7):]
    model = _small_model(panel).fit(train, test)

    proba = model.predict_proba(test)
    assert proba.shape == (test.height, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_super_brain_signal_in_range():
    panel = _panel()
    model = _small_model(panel).fit(panel.frame)
    sig = model.predict_signal(panel.frame)
    assert np.all(sig >= -1.0) and np.all(sig <= 1.0)


def test_super_brain_is_causal():
    # Previsão de uma barra não muda quando barras FUTURAS são removidas:
    # cada janela usa só o passado → predict(full)[:k] == predict(full[:k]).
    panel = _panel()
    model = _small_model(panel).fit(panel.frame)
    full = panel.frame.sort(["asset_id", "time"])
    k = 300
    p_full = model.predict_proba(full)[:k]
    p_trunc = model.predict_proba(full[:k])
    assert np.allclose(p_full, p_trunc, atol=1e-5)
