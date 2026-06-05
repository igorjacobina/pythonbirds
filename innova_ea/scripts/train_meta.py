#!/usr/bin/env python
"""Treino com META-LABELING — regra primária + meta-modelo (filtro) do INNOVA EA.

Em vez de prever direção (mercado eficiente → ~impossível), uma regra primária
transparente define o LADO e o meta-modelo decide se vale tomar a aposta. Valida
no mesmo walk-forward purgado + backtest com risco + Deflated Sharpe.

Exemplos:
    python scripts/train_meta.py --primary donchian --timeframe H1 --store .\\data --out .\\artifacts_meta
    python scripts/train_meta.py --primary session --session-prefix london --timeframe M15 --store .\\data
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from innova_ea.backtest import AccountConfig, BacktestEngine, CostModel
from innova_ea.backtest.metrics import periods_per_year
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import get_instrument
from innova_ea.data import ParquetStore
from innova_ea.features.presets import default_feature_set
from innova_ea.research import (
    MetaLabelModel,
    MetaModelStrategy,
    PurgedWalkForward,
    build_metalabel_panel,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    donchian_breakout_primary,
    session_breakout_primary,
)
from innova_ea.universe import DEFAULT_UNIVERSE, asset_class_of


def build_primary(args):
    if args.primary == "donchian":
        return donchian_breakout_primary(args.donchian_lookback)
    if args.primary == "session":
        return session_breakout_primary(args.session_prefix)
    raise SystemExit(f"primária desconhecida: {args.primary}")


def load_bars(store, symbols, timeframe, start, end, *, min_bars=5000):
    out = {}
    for s in symbols:
        bars = store.read(s, timeframe, start, end)
        if bars.height >= min_bars:
            out[s] = bars
    if not out:
        raise SystemExit("nenhum ativo com dados suficientes no store")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino meta-labeling (INNOVA EA)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--out", default="./artifacts_meta")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--primary", choices=["donchian", "session"], default="donchian")
    ap.add_argument("--donchian-lookback", type=int, default=20)
    ap.add_argument("--session-prefix", default="london")
    ap.add_argument("--max-horizon", type=int, default=24)
    ap.add_argument("--train-size", type=int, default=8000)
    ap.add_argument("--test-size", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="P(vitória) mínima para operar o sinal primário.")
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    bars_by = load_bars(store, args.symbols, tf, args.start, end)
    print(f"Ativos: {list(bars_by)}")

    feature_set = default_feature_set()
    primary = build_primary(args)
    panel = build_metalabel_panel(bars_by, feature_set, primary,
                                  asset_class_of=asset_class_of, max_horizon=args.max_horizon)
    win_rate = float(panel.frame["label"].mean())
    print(f"Painel de EVENTOS: {panel.frame.height:,} sinais primários | "
          f"{len(panel.feature_names)} features | base win-rate {win_rate:.1%}")

    wf = PurgedWalkForward(args.train_size, args.test_size,
                           horizon=args.max_horizon, embargo=args.max_horizon)
    ppy = periods_per_year(tf)
    oos_sharpes: list[float] = []

    print(f"\nWalk-forward purgado | primária={args.primary} | threshold P(win)={args.threshold}\n")
    for train_df, test_df, win in wf.split(panel.frame):
        model = MetaLabelModel(panel).fit(train_df, test_df)
        # Precisão do filtro: dos sinais que o modelo aprovaria (P>=thr), quantos venceriam?
        pw = model.predict_meta(test_df)
        approved = pw >= args.threshold
        prec = float(test_df["label"].to_numpy()[approved].mean()) if approved.any() else float("nan")

        t0, t1 = test_df["time"].min(), test_df["time"].max()
        fold_sr = []
        for sym in bars_by:
            sub = bars_by[sym].filter((pl.col("time") >= t0) & (pl.col("time") <= t1))
            if sub.height < 100:
                continue
            strat = MetaModelStrategy(model, feature_set, primary, sym, asset_class_of(sym),
                                      threshold=args.threshold)
            engine = BacktestEngine(get_instrument(sym), CostModel(), AccountConfig(leverage=100),
                                    initial_capital=10_000, max_lots=0.1)
            sr = engine.run(sub, strat, tf).report.sharpe
            fold_sr.append(sr)
            oos_sharpes.append(sr)
        print(f"  fold {win.index}: precisão_aprovados={prec:.3f} ({approved.mean():.1%} aprovados) "
              f"| Sharpe OOS médio={np.mean(fold_sr):+.2f}")

    sr = np.array(oos_sharpes)
    trial = [deannualize_sharpe(s, ppy) for s in oos_sharpes]
    dsr = deflated_sharpe_ratio(trial, n_obs=panel.frame.height) if trial else float("nan")
    print(f"\n== Agregado OOS ({sr.size} ensaios) ==")
    print(f"  Sharpe OOS médio : {sr.mean():+.2f}  (positivos {np.mean(sr > 0):.0%})")
    print(f"  Deflated Sharpe  : {dsr:.1%}")

    final = MetaLabelModel(panel).fit(panel.frame)
    dest = Path(args.out) / "metalabel"
    final.save(dest)
    (dest / "manifest.json").write_text(json.dumps({
        "kind": "metalabel", "primary": args.primary, "timeframe": tf.value,
        "threshold": args.threshold, "symbols": list(bars_by),
        "events": panel.frame.height, "base_win_rate": win_rate,
        "oos_mean_sharpe": float(sr.mean()) if sr.size else None, "deflated_sharpe": float(dsr),
    }, ensure_ascii=False, indent=2))
    print(f"\n✓ artefato salvo em {dest} | DSR={dsr:.1%}")
    print("\nTop features (ganho):")
    for name, val in list(final.feature_importance().items())[:6]:
        print(f"  {name:<26} {val:,.0f}")


if __name__ == "__main__":
    main()
