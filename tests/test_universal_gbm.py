"""Testes do baseline LightGBM universal e da ponte ModelStrategy."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from innova_ea.backtest import AccountConfig, BacktestEngine, CostModel
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import get_instrument
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.features import CloseLocationValue, EfficiencyRatio, FeatureSet, YangZhang
from innova_ea.research.dataset import UniversalPanel, label_to_class
from innova_ea.research.models.base import UniversalModel
from innova_ea.research.models.gbm import LightGBMUniversal
from innova_ea.strategy.model_strategy import ModelStrategy


def _times(n):
    t0 = datetime(2021, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=15 * i) for i in range(n)]


def _planted_panel(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.standard_normal(n)
    label = np.where(f0 > 0.5, 1, np.where(f0 < -0.5, -1, 0)).astype(np.int64)
    aid = rng.integers(0, 2, n).astype(np.int32)
    frame = pl.DataFrame({
        "time": _times(n),
        "asset": ["A" if a == 0 else "B" for a in aid],
        "asset_id": aid,
        "asset_class": ["x"] * n,
        "asset_class_id": np.zeros(n, dtype=np.int32),
        "f0": f0,
        "f1": rng.standard_normal(n) * 0.1,  # ruído
        "label": label,
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))
    panel = UniversalPanel(frame, ["f0", "f1"], ["A", "B"], ["x"])
    return panel


def test_gbm_learns_planted_signal():
    panel = _planted_panel()
    frame = panel.frame
    train, test = frame[:3000], frame[3000:]
    model = LightGBMUniversal(panel, n_estimators=120, min_child_samples=20).fit(train, test)

    proba = model.predict_proba(test)
    assert proba.shape == (test.height, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    pred = proba.argmax(axis=1)
    true = label_to_class(test["label"].to_numpy())
    acc = float((pred == true).mean())
    assert acc > 0.85  # sinal plantado é determinístico → modelo deve acertar


def test_gbm_feature_importance_ranks_signal_over_noise():
    panel = _planted_panel()
    model = LightGBMUniversal(panel, n_estimators=120, min_child_samples=20).fit(panel.frame)
    imp = model.feature_importance()
    assert imp["f0"] > imp["f1"]  # f0 carrega o sinal; f1 é ruído


# ---------------- ModelStrategy ----------------

class _ConstModel(UniversalModel):
    """Modelo de teste que devolve sempre a mesma distribuição."""

    def __init__(self, panel, proba):
        super().__init__(panel)
        self._p = np.asarray(proba, dtype=np.float64)

    def fit(self, train_df, valid_df=None):
        return self

    def predict_proba(self, df):
        return np.tile(self._p, (df.height, 1))


def _dummy_panel(features):
    frame = pl.DataFrame(schema={"time": pl.Datetime("us", "UTC")})
    return UniversalPanel(frame, features, ["EURUSD"], ["forex"])


def _bars():
    return SyntheticSource(seed=3).fetch(
        "EURUSD", Timeframe.M15,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 15, tzinfo=timezone.utc),
    )


def test_model_strategy_targets_and_warmup():
    # YangZhang produz NULL no aquecimento → ModelStrategy zera essas barras.
    fs = FeatureSet([YangZhang(20), CloseLocationValue()])
    panel = _dummy_panel(fs.names)
    model = _ConstModel(panel, [0.2, 0.3, 0.5])  # signal = P(up)-P(down) = 0.3
    strat = ModelStrategy(model, fs, "EURUSD", "forex", threshold=0.0)
    bars = _bars()
    tgt = strat.generate_targets(bars, get_instrument("EURUSD"))

    assert tgt.shape[0] == bars.height
    assert np.all(tgt >= -1.0) and np.all(tgt <= 1.0)
    assert np.allclose(tgt[:15], 0.0)            # aquecimento (features nulas) → fora
    assert tgt[-1] == pytest.approx(0.3, abs=1e-9)


def test_model_strategy_deadzone_threshold():
    fs = FeatureSet([EfficiencyRatio(20), CloseLocationValue()])
    panel = _dummy_panel(fs.names)
    model = _ConstModel(panel, [0.2, 0.3, 0.5])  # |signal|=0.3 < threshold 0.4 → 0
    strat = ModelStrategy(model, fs, "EURUSD", "forex", threshold=0.4)
    tgt = strat.generate_targets(_bars(), get_instrument("EURUSD"))
    assert np.allclose(tgt, 0.0)


def test_model_strategy_rejects_unknown_asset():
    fs = FeatureSet([EfficiencyRatio(20)])
    panel = _dummy_panel(fs.names)
    model = _ConstModel(panel, [0.3, 0.3, 0.4])
    with pytest.raises(ValueError, match="vocabul"):
        ModelStrategy(model, fs, "GBPUSD", "forex")


def test_model_strategy_runs_in_backtest():
    fs = FeatureSet([EfficiencyRatio(20), CloseLocationValue()])
    panel = _dummy_panel(fs.names)
    model = _ConstModel(panel, [0.1, 0.2, 0.7])  # viés comprado
    strat = ModelStrategy(model, fs, "EURUSD", "forex", threshold=0.1)
    engine = BacktestEngine(get_instrument("EURUSD"), CostModel(),
                            AccountConfig(leverage=100), initial_capital=10_000, max_lots=0.1)
    res = engine.run(_bars(), strat, Timeframe.M15)
    assert np.isfinite(res.equity).all()
    # Sinal comprado constante → abre e segura posição (exposição > 0).
    assert np.any(res.position_lots != 0.0)
    assert res.risk.blew_account is False
