"""Demo da Fase 4: modelo universal (Super Cérebro) end-to-end e honesto.

Pipeline: dados multi-ativo → painel unificado (features + triple-barrier) →
walk-forward PURGADO → treina UM modelo sobre TODOS os ativos → previsões viram
``ModelStrategy`` → backtest com risco (Fase 3) → Deflated Sharpe (Fase 3).

Por padrão usa o baseline LightGBM (rápido, CPU). Com ``--super-brain`` usa o
Transformer causal (requer PyTorch). Em dados sintéticos sem edge real, o sistema
deve corretamente NÃO alegar alfa — é a prova do rigor, não da lucratividade.

    python examples/run_super_brain_demo.py
    python examples/run_super_brain_demo.py --super-brain
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import polars as pl

from innova_ea.backtest import AccountConfig, BacktestEngine, CostModel
from innova_ea.backtest.metrics import periods_per_year
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import get_instrument
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.features import (
    CloseLocationValue,
    EfficiencyRatio,
    FeatureSet,
    SessionOpenFeature,
    VolatilityRegime,
    YangZhang,
)
from innova_ea.features.sessions import LONDON, NEW_YORK
from innova_ea.research import (
    LightGBMUniversal,
    PurgedWalkForward,
    build_universal_panel,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    get_super_brain,
    label_to_class,
)
from innova_ea.strategy import ModelStrategy

# Universo reduzido de classes distintas (forex / metal / índice).
UNIVERSE = {
    "EURUSD": ("forex", 0.08, {}),
    "GBPUSD": ("forex", 0.10, {}),
    "XAUUSD": ("metal", 0.15, {7: 1.8, 8: 1.8}),
    "US500": ("index", 0.18, {13: 2.2, 14: 1.8}),
}
TF = Timeframe.M15
MAX_HORIZON = 16


def build_data():
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2021, 1, 1, tzinfo=timezone.utc)
    bars_by = {}
    for sym, (_cls, vol, prof) in UNIVERSE.items():
        src = SyntheticSource(
            seed=hash(sym) % 100, annual_vol=vol,
            vol_by_hour=prof, drift_by_hour={h: 1.2 for h in prof},
        )
        bars_by[sym] = src.fetch(sym, TF, start, end)
    return bars_by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--super-brain", action="store_true",
                    help="usa o Transformer causal (requer PyTorch).")
    args = ap.parse_args()

    bars_by = build_data()
    asset_class = {s: c for s, (c, _, _) in UNIVERSE.items()}
    print("Barras por ativo:", {s: b.height for s, b in bars_by.items()})

    feature_set = FeatureSet([
        YangZhang(20), VolatilityRegime(10, 100), EfficiencyRatio(20),
        CloseLocationValue(),
        SessionOpenFeature(LONDON, opening_range_bars=4),
        SessionOpenFeature(NEW_YORK, opening_range_bars=4),
    ])
    panel = build_universal_panel(
        bars_by, feature_set, asset_class_of=lambda s: asset_class[s],
        max_horizon=MAX_HORIZON, vol_mult=1.5,
    )
    dist = np.bincount(label_to_class(panel.frame["label"].to_numpy()), minlength=3)
    print(f"Painel: {panel.frame.height:,} linhas | {len(panel.feature_names)} features "
          f"| {panel.n_assets} ativos | rótulos down/flat/up = {dist.tolist()}")

    wf = PurgedWalkForward(train_size=6000, test_size=2000,
                           horizon=MAX_HORIZON, embargo=MAX_HORIZON)
    ppy = periods_per_year(TF)
    oos_sharpes: list[float] = []
    last_model = None

    print(f"\nValidação walk-forward purgada — backbone: "
          f"{'Super Cérebro (Transformer)' if args.super_brain else 'LightGBM'}\n")

    for train_df, test_df, win in wf.split(panel.frame):
        if args.super_brain:
            model = get_super_brain(panel, window=32, epochs=8, batch_size=512)
        else:
            model = LightGBMUniversal(panel, n_estimators=300)
        model.fit(train_df, test_df)
        last_model = model

        # Acurácia OOS vs. baseline de classe majoritária.
        proba = model.predict_proba(test_df)
        true = label_to_class(test_df["label"].to_numpy())
        acc = float((proba.argmax(1) == true).mean())
        majority = float(np.bincount(true, minlength=3).max() / true.size)

        # Cada ativo vira um backtest OOS com risco (Fase 3).
        t0 = test_df["time"].min()
        t1 = test_df["time"].max()
        fold_sr = []
        for sym in UNIVERSE:
            sub = bars_by[sym].filter((pl.col("time") >= t0) & (pl.col("time") <= t1))
            if sub.height < 200:
                continue
            strat = ModelStrategy(model, feature_set, sym, asset_class[sym], threshold=0.15)
            engine = BacktestEngine(
                get_instrument(sym), CostModel(), AccountConfig(leverage=100),
                initial_capital=10_000, max_lots=0.1,
            )
            res = engine.run(sub, strat, TF)
            fold_sr.append(res.report.sharpe)
            oos_sharpes.append(res.report.sharpe)

        print(f"  fold {win.index}: acc={acc:.3f} (baseline {majority:.3f}) | "
              f"Sharpe OOS por ativo médio={np.mean(fold_sr):+.2f}")

    # Importância de features do último modelo (valida o alfa da Fase 2).
    if hasattr(last_model, "feature_importance"):
        top = list(last_model.feature_importance().items())[:6]
        print("\nTop features (ganho):")
        for name, val in top:
            print(f"  {name:<26} {val:,.0f}")

    # Deflated Sharpe sobre todos os ensaios (fold × ativo).
    sr = np.array(oos_sharpes)
    trial_sr = [deannualize_sharpe(s, ppy) for s in oos_sharpes]
    dsr = deflated_sharpe_ratio(trial_sr, n_obs=panel.frame.height)
    print(f"\n== Agregado OOS ({len(oos_sharpes)} ensaios) ==")
    print(f"  Sharpe OOS médio : {sr.mean():+.2f}  (dp {sr.std():.2f})")
    print(f"  Ensaios positivos: {np.mean(sr > 0):.0%}")
    print(f"  Deflated Sharpe  : {dsr:.1%}  (prob. de edge real após o nº de ensaios)")
    print(
        "\nDados SINTÉTICOS sem edge embutido sustentável → esperado DSR baixo. "
        "O valor está em provar o pipeline: um modelo, todos os ativos, embeddings/"
        "categóricos para cross-asset, validado com custo, risco e anti-overfitting."
    )


if __name__ == "__main__":
    main()
