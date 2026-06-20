"""Smoke test do pipeline de treino (scripts/train_models.py) sem MT5.

Ingere dados sintéticos num ParquetStore temporário e roda o treino completo
(painel → walk-forward purgado → modelo final salvo), validando o artefato.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np

from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.research.models.gbm import LightGBMUniversal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_models import train  # noqa: E402


def _seed_store(store_path):
    store = ParquetStore(store_path)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 7, 1, tzinfo=timezone.utc)
    for i, sym in enumerate(["EURUSD", "XAUUSD"]):
        bars = SyntheticSource(seed=i, annual_vol=0.1 + 0.05 * i).fetch(
            sym, Timeframe.M15, start, end)
        store.write(sym, Timeframe.M15, bars)
    return start, end


def test_train_lightgbm_end_to_end(tmp_path):
    store_path = tmp_path / "data"
    out_path = tmp_path / "artifacts"
    start, end = _seed_store(store_path)

    summary = train(
        str(store_path),
        symbols=["EURUSD", "XAUUSD"],
        timeframe=Timeframe.M15,
        start=start,
        end=end,
        model_kind="lightgbm",
        out_dir=str(out_path),
        max_horizon=8,
        train_size=4000,
        test_size=4000,
        threshold=0.15,
        gbm_kwargs={"n_estimators": 60, "min_child_samples": 20},
        run_eval=True,
    )

    # Artefato salvo e manifesto presente.
    assert (out_path / "lightgbm" / "model.txt").exists()
    assert (out_path / "lightgbm" / "manifest.json").exists()
    ev = summary["lightgbm"]["evaluation"]
    assert "deflated_sharpe" in ev and ev["n_trials"] > 0

    # O artefato recarrega e prevê com a mesma assinatura de features.
    import polars as pl

    model = LightGBMUniversal.load(out_path / "lightgbm")
    cols = {f: [0.0, 0.1] for f in model.feature_names}
    cols["asset_id"] = [0, 1]
    cols["asset_class_id"] = [0, 0]
    proba = model.predict_proba(pl.DataFrame(cols))
    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
