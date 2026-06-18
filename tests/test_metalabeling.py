"""Testes do framework de meta-labeling (regra primária + meta-modelo filtro)."""
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
from innova_ea.research.dataset import UniversalPanel
from innova_ea.research.metalabeling import (
    MetaLabelModel,
    MetaModelStrategy,
    build_metalabel_panel,
    donchian_breakout_primary,
    meta_labels,
    reversal_at_extreme_primary,
)


def _bars(opens, highs, lows, closes):
    n = len(closes)
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame({
        "time": times, "open": list(map(float, opens)), "high": list(map(float, highs)),
        "low": list(map(float, lows)), "close": list(map(float, closes)),
        "volume": [1.0] * n,
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def test_meta_label_win_and_loss():
    # Alta: preço sobe e toca a barreira superior → aposta long VENCE (meta=1).
    up = _bars([100, 101, 103, 104], [100, 101, 103, 104], [100, 101, 103, 104],
               [100, 101, 103, 104])
    side_long = np.array([1, 1, 1, 1])
    m = meta_labels(up, side_long, width=2.0, max_horizon=3, width_is_pct=False)
    assert m[0] == 1

    # Baixa: preço cai e toca a inferior → aposta long PERDE (meta=0).
    down = _bars([100, 99, 97, 96], [100, 99, 97, 96], [100, 99, 97, 96], [100, 99, 97, 96])
    m2 = meta_labels(down, side_long, width=2.0, max_horizon=3, width_is_pct=False)
    assert m2[0] == 0


def test_reversal_primary_bullish_at_bottom():
    # Queda + engolfo de ALTA que faz nova mínima (fundo) → lado +1 (compra).
    bars = _bars(
        opens=[12.0, 11.9, 11.5, 11.1, 10.7, 9.9],
        highs=[12.2, 12.0, 11.6, 11.2, 10.8, 11.0],
        lows=[11.8, 11.4, 11.0, 10.6, 10.0, 9.8],
        closes=[11.9, 11.5, 11.1, 10.7, 10.1, 10.9],   # última: verde, engole a vermelha
    )
    side = reversal_at_extreme_primary(lookback=5)(bars, bars)
    assert side[5] == 1


def test_reversal_primary_bearish_at_top():
    # Alta + engolfo de BAIXA que faz nova máxima (topo) → lado -1 (venda).
    bars = _bars(
        opens=[8.0, 8.1, 8.5, 8.9, 9.3, 10.1],
        highs=[8.2, 8.6, 9.0, 9.4, 9.6, 10.2],
        lows=[7.8, 8.0, 8.4, 8.8, 9.2, 9.0],
        closes=[8.1, 8.5, 8.9, 9.3, 9.5, 9.1],          # última: vermelha, engole a verde
    )
    side = reversal_at_extreme_primary(lookback=5)(bars, bars)
    assert side[5] == -1


def test_donchian_primary_breakouts():
    # lookback=2; máxima das 2 barras anteriores e rompimento.
    bars = _bars(
        opens=[10, 10, 10, 10, 10],
        highs=[10, 11, 10, 10, 10],
        lows=[9, 9, 9, 8, 9],
        closes=[10, 10, 12, 7, 10],   # idx2 rompe p/ cima (>11); idx3 rompe p/ baixo (<9)
    )
    side = donchian_breakout_primary(lookback=2)(bars, bars)
    assert side[2] == 1     # rompeu a máxima recente
    assert side[3] == -1    # rompeu a mínima recente


def _synth(symbol, vol, seed):
    return SyntheticSource(seed=seed, annual_vol=vol).fetch(
        symbol, Timeframe.H1,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 6, 1, tzinfo=timezone.utc))


def test_build_metalabel_panel_events_only():
    bars_by = {"EURUSD": _synth("EURUSD", 0.08, 1), "XAUUSD": _synth("XAUUSD", 0.15, 2)}
    cls = {"EURUSD": "forex", "XAUUSD": "metal"}
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    panel = build_metalabel_panel(bars_by, fs, donchian_breakout_primary(20),
                                  asset_class_of=lambda s: cls[s], max_horizon=12, warmup=120)
    f = panel.frame
    assert "side" in f.columns and "label" in f.columns
    assert (f["side"] != 0).all()                         # só eventos
    assert set(f["label"].unique().to_list()).issubset({0, 1})
    assert f.select(panel.feature_names).null_count().to_numpy().sum() == 0


def _planted_meta_panel(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.standard_normal(n)
    label = (f0 + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)  # vitória ~ f0>0
    aid = rng.integers(0, 2, n).astype(np.int32)
    frame = pl.DataFrame({
        "time": [datetime(2021, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)],
        "asset_id": aid, "asset_class_id": np.zeros(n, dtype=np.int32),
        "f0": f0, "f1": rng.standard_normal(n) * 0.1, "side": np.ones(n, dtype=np.int64),
        "label": label,
    }).with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))
    return UniversalPanel(frame, ["f0", "f1"], ["A", "B"], ["x"])


def test_meta_model_learns_filter():
    panel = _planted_meta_panel()
    model = MetaLabelModel(panel, n_estimators=120, min_child_samples=20).fit(panel.frame)
    pwin = model.predict_meta(panel.frame)
    y = panel.frame["label"].to_numpy()
    # P(vitória) média deve ser maior nos vencedores reais que nos perdedores.
    assert pwin[y == 1].mean() > pwin[y == 0].mean() + 0.1


def test_meta_model_save_load(tmp_path):
    panel = _planted_meta_panel()
    model = MetaLabelModel(panel, n_estimators=60, min_child_samples=20).fit(panel.frame)
    before = model.predict_meta(panel.frame)
    model.save(tmp_path / "meta")
    reloaded = MetaLabelModel.load(tmp_path / "meta")
    assert np.allclose(before, reloaded.predict_meta(panel.frame), atol=1e-9)


def test_meta_strategy_runs_in_backtest():
    bars_by = {"EURUSD": _synth("EURUSD", 0.1, 3)}
    fs = FeatureSet([YangZhang(20), EfficiencyRatio(20), CloseLocationValue()])
    primary = donchian_breakout_primary(20)
    panel = build_metalabel_panel(bars_by, fs, primary,
                                  asset_class_of=lambda s: "forex", max_horizon=12, warmup=120)
    model = MetaLabelModel(panel, n_estimators=80, min_child_samples=20).fit(panel.frame)
    strat = MetaModelStrategy(model, fs, primary, "EURUSD", "forex", threshold=0.55, warmup=120)
    tgt = strat.generate_targets(bars_by["EURUSD"], get_instrument("EURUSD"))

    assert tgt.shape[0] == bars_by["EURUSD"].height
    assert np.all(tgt >= -1.0) and np.all(tgt <= 1.0)
    assert np.allclose(tgt[:120], 0.0)                    # aquecimento

    engine = BacktestEngine(get_instrument("EURUSD"), CostModel(), AccountConfig(leverage=100),
                            initial_capital=10_000, max_lots=0.1)
    res = engine.run(bars_by["EURUSD"], strat, Timeframe.H1)
    assert np.isfinite(res.equity).all()
