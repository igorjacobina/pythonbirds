"""Testes da biblioteca de features: corretude, limites e ausência de look-ahead."""
from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from innova_ea.features.base import FeatureSet
from innova_ea.features.directional import (
    BodyRatio,
    CloseLocationValue,
    EfficiencyRatio,
)
from innova_ea.features.returns import ReturnZScore
from innova_ea.features.sessions import LONDON, SessionOpenFeature
from innova_ea.features.time_features import TimeOfDayCyclical, VolatilityProfile
from innova_ea.features.volatility import (
    GarmanKlass,
    Parkinson,
    RogersSatchell,
    VolatilityRegime,
    YangZhang,
)


def test_feature_set_matrix_shape_and_names(synthetic_m15):
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    m = fs.transform(synthetic_m15)
    assert m.height == synthetic_m15.height
    assert m.columns == ["time", "vol_yz_20", "eff_ratio_20", "clv"]


def test_volatility_estimators_nonnegative(synthetic_m15):
    for feat in (Parkinson(20), GarmanKlass(20), RogersSatchell(20), YangZhang(20)):
        col = feat.transform(synthetic_m15).to_series().drop_nulls()
        assert (col >= 0).all(), f"{feat.names[0]} produziu valor negativo"
        assert np.isfinite(col.to_numpy()).all()


def test_clv_and_body_ratio_bounds(synthetic_m15):
    clv = CloseLocationValue().transform(synthetic_m15).to_series().drop_nulls()
    body = BodyRatio().transform(synthetic_m15).to_series().drop_nulls()
    assert clv.min() >= -1.0 - 1e-9 and clv.max() <= 1.0 + 1e-9
    assert body.min() >= 0.0 and body.max() <= 1.0 + 1e-9


def test_efficiency_ratio_bounds(synthetic_m15):
    er = EfficiencyRatio(20).transform(synthetic_m15).to_series().drop_nulls()
    assert er.min() >= 0.0 and er.max() <= 1.0 + 1e-9


def _causality_check(feature, bars, k):
    """Feature causal: valor em i só depende de barras <= i."""
    full = feature.transform(bars)
    truncated = feature.transform(bars[:k])
    assert_frame_equal(truncated, full[:k])


def test_no_lookahead_rolling_features(synthetic_m15):
    k = 400
    for feat in (YangZhang(20), VolatilityRegime(10, 100),
                 EfficiencyRatio(20), ReturnZScore(50)):
        _causality_check(feat, synthetic_m15, k)


def test_no_lookahead_session_feature(synthetic_m15):
    feat = SessionOpenFeature(LONDON, opening_range_bars=4, pre_window=12)
    _causality_check(feat, synthetic_m15, 400)


def test_session_active_flag_matches_hours(synthetic_m15):
    feat = SessionOpenFeature(LONDON, opening_range_bars=4)
    out = feat.transform(synthetic_m15).hstack(synthetic_m15.select("time"))
    hours = out["time"].dt.hour()
    active = out["london_active"] == 1
    # Londres = 7..16 UTC.
    assert (hours.filter(active) >= 7).all()
    assert (hours.filter(active) < 16).all()


def test_session_opening_range_is_causal_null_then_value(synthetic_m15):
    feat = SessionOpenFeature(LONDON, opening_range_bars=4)
    out = feat.transform(synthetic_m15)
    # Durante a formação do OR (bars_since_open < 4) o breakout é null.
    forming = out.filter(
        (pl.col("london_active") == 1) & (pl.col("london_bars_since_open") < 4)
    )
    assert forming["london_or_breakout"].null_count() == forming.height


def test_time_cyclical_encoding_unit_circle(synthetic_m15):
    out = TimeOfDayCyclical().transform(synthetic_m15)
    r = (out["tod_sin"] ** 2 + out["tod_cos"] ** 2).to_numpy()
    assert np.allclose(r, 1.0)


def test_volatility_profile_fit_transform_no_leakage(synthetic_m15):
    n = synthetic_m15.height
    train = synthetic_m15[: n // 2]
    test = synthetic_m15[n // 2 :]
    prof = VolatilityProfile(vol_window=12).fit(train)
    out = prof.transform(test)
    assert out.height == test.height
    assert "tod_expected_vol" in out.columns and "tod_vol_ratio" in out.columns
