#!/usr/bin/env python
"""Treino da estratégia de VOLATILIDADE (straddle de rompimento) — INNOVA EA.

Direção-agnóstico: o ML prevê P(rompimento lucrativo) e armamos um straddle só
quando a confiança ≥ limiar. Validado em walk-forward purgado + backtest de
straddle (custos honestos) + Deflated Sharpe sobre o AGREGADO.

Exemplo:
    python scripts/train_straddle.py --store .\\data --timeframe H1 --threshold 0.55
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from innova_ea.backtest.metrics import compute_metrics, periods_per_year
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import get_instrument
from innova_ea.data import ParquetStore
from innova_ea.features.presets import default_feature_set
from innova_ea.research import (
    MetaLabelModel,
    PurgedWalkForward,
    build_straddle_panel,
    deflated_sharpe_ratio,
    straddle_backtest,
    straddle_outcomes,
)
from innova_ea.research.overfitting import deannualize_sharpe as _deann
from innova_ea.universe import DEFAULT_UNIVERSE, asset_class_of


def load_bars(store, symbols, tf, start, end, *, min_bars=5000):
    out = {}
    for s in symbols:
        b = store.read(s, tf, start, end)
        if b.height >= min_bars:
            out[s] = b
    if not out:
        raise SystemExit("nenhum ativo com dados suficientes")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino straddle de volatilidade (INNOVA EA)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--out", default="./artifacts_straddle")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--band-lookback", type=int, default=12)
    ap.add_argument("--target-mult", type=float, default=1.5)
    ap.add_argument("--stop-mult", type=float, default=1.5)
    ap.add_argument("--cost-points", type=float, default=12.0)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--test-size", type=int, default=8000)
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    bars_by = load_bars(store, args.symbols, tf, args.start, end)
    print(f"Ativos: {list(bars_by)}")

    fs = default_feature_set()
    straddle_kw = dict(horizon=args.horizon, band_lookback=args.band_lookback,
                       target_mult=args.target_mult, stop_mult=args.stop_mult,
                       cost_points=args.cost_points)
    panel = build_straddle_panel(bars_by, fs, asset_class_of=asset_class_of, **straddle_kw)
    base = float(panel.frame["label"].mean())
    print(f"Painel: {panel.frame.height:,} barras | base P(straddle lucrativo) {base:.1%}")

    # Pré-computa os resultados do straddle por ativo (p/ o backtest OOS).
    outcomes = {}
    for s, b in bars_by.items():
        pt = 0.01 if s.upper().endswith("JPY") else 1e-5
        outcomes[s] = straddle_outcomes(b, point=pt, **straddle_kw)

    wf = PurgedWalkForward(args.train_size, args.test_size,
                           horizon=args.horizon, embargo=args.horizon)
    ppy = periods_per_year(tf)
    oos_sharpes: list[float] = []
    armed_total = 0
    from collections import defaultdict
    per_asset_sr: dict[str, list[float]] = defaultdict(list)
    per_asset_trades: dict[str, int] = defaultdict(int)

    print(f"\nWalk-forward purgado | threshold P(lucro)={args.threshold}\n")
    for train_df, test_df, win in wf.split(panel.frame):
        model = MetaLabelModel(panel).fit(train_df, test_df)
        pw = model.predict_meta(test_df)
        approved = pw >= args.threshold
        prec = float(test_df["label"].to_numpy()[approved].mean()) if approved.any() else float("nan")
        armed_total += int(approved.sum())

        t0, t1 = test_df["time"].min(), test_df["time"].max()
        fold_sr = []
        for s, b in bars_by.items():
            sub_mask = (b["time"] >= t0) & (b["time"] <= t1)
            sub = b.filter(sub_mask)
            if sub.height < 100:
                continue
            feats = fs.transform(sub).with_columns(
                pl.lit(model.asset_vocab.index(s), dtype=pl.Int32).alias("asset_id"),
                pl.lit(model.class_vocab.index(asset_class_of(s)), dtype=pl.Int32).alias("asset_class_id"),
            ).with_columns([pl.col(c).fill_null(0) for c in model.feature_names])
            arm = model.predict_meta(feats) >= args.threshold
            sub_out = outcomes[s].filter(sub_mask)
            inst = get_instrument(s)
            equity, in_trade, trade_pnls = straddle_backtest(
                sub, arm, sub_out, inst.contract_size, lots=0.1, initial_capital=10_000.0)
            rep = compute_metrics(equity, in_trade, trade_pnls, tf, 10_000.0)
            fold_sr.append(rep.sharpe)
            oos_sharpes.append(rep.sharpe)
            per_asset_sr[s].append(rep.sharpe)
            per_asset_trades[s] += int(trade_pnls.size)
        print(f"  fold {win.index}: precisão_aprovados={prec:.3f} ({approved.mean():.1%}) "
              f"| Sharpe OOS médio={np.mean(fold_sr):+.2f}")

    sr = np.array(oos_sharpes)
    agg = float(sr.mean()) if sr.size else 0.0
    trial = [_deann(x, ppy) for x in oos_sharpes]
    dsr = (deflated_sharpe_ratio(trial, n_obs=len(trial), selected_sharpe=_deann(agg, ppy))
           if len(trial) >= 2 and armed_total > 0 else float("nan"))

    print(f"\n== Agregado OOS ({sr.size} ensaios) ==")
    print(f"  Straddles armados: {armed_total:,}")
    print(f"  Sharpe OOS médio : {agg:+.2f}  (positivos {np.mean(sr > 0):.0%})")
    print(f"  Deflated Sharpe  : {dsr:.1%}" if armed_total else "  Deflated Sharpe  : N/A (sem trades)")

    # Robustez: o edge é AMPLO (vários ativos) ou concentrado em poucos sortudos?
    print("\n  -- Por ativo (Sharpe OOS médio | nº trades) --")
    n_pos = 0
    for s in bars_by:
        srs = per_asset_sr.get(s, [])
        m = float(np.mean(srs)) if srs else 0.0
        n_pos += 1 if m > 0 else 0
        print(f"    {s:<8} {m:+.2f}  | {per_asset_trades.get(s, 0):>6,} trades")
    print(f"  Ativos com Sharpe>0: {n_pos}/{len(bars_by)} "
          f"(edge {'AMPLO' if n_pos >= 0.7 * len(bars_by) else 'CONCENTRADO'})")

    final = MetaLabelModel(panel).fit(panel.frame)
    dest = Path(args.out) / "straddle"
    final.save(dest)
    (dest / "manifest.json").write_text(json.dumps({
        "kind": "straddle", "timeframe": tf.value, "threshold": args.threshold,
        "straddle": straddle_kw, "symbols": list(bars_by),
        "base_profit_rate": base, "oos_mean_sharpe": agg, "deflated_sharpe": float(dsr),
    }, ensure_ascii=False, indent=2))
    print(f"\n✓ artefato salvo em {dest}")
    print("\nTop features (ganho):")
    for name, val in list(final.feature_importance().items())[:6]:
        print(f"  {name:<26} {val:,.0f}")


if __name__ == "__main__":
    main()
