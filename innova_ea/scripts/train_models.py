#!/usr/bin/env python
"""Treinamento definitivo do INNOVA EA — baseline LightGBM e Super Cérebro.

Lê os dados reais já ingeridos em Parquet (``ParquetStore``), monta o painel
universal multi-ativo, valida com walk-forward PURGADO + backtest de risco +
Deflated Sharpe, e então treina o modelo FINAL em todo o histórico, salvando o
artefato pronto para execução.

Fluxo:
    1. carrega barras de todos os ativos (ParquetStore)
    2. monta o painel unificado (features de produção + triple-barrier)
    3. walk-forward purgado: treina por janela, mede Sharpe OOS por ativo (com
       custos/risco) e o Deflated Sharpe agregado — a prova honesta de edge
    4. treina o modelo final em TODO o histórico e salva em --out

Exemplos:
    # Baseline LightGBM (CPU, rápido):
    python scripts/train_models.py --model lightgbm --store ./data --out ./artifacts

    # Super Cérebro (Transformer; requer PyTorch, GPU recomendado):
    python scripts/train_models.py --model super_brain --store ./data --out ./artifacts

    # Ambos:
    python scripts/train_models.py --model both --store ./data --out ./artifacts
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
    LightGBMUniversal,
    PurgedWalkForward,
    build_universal_panel,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    get_super_brain,
    label_to_class,
)
from innova_ea.strategy import ModelStrategy
from innova_ea.universe import DEFAULT_UNIVERSE, asset_class_of


def load_bars(store, symbols, timeframe, start, end, *, min_bars=5000):
    """Lê as barras de cada símbolo do Parquet, descartando séries curtas."""
    out = {}
    for sym in symbols:
        bars = store.read(sym, timeframe, start, end)
        if bars.height >= min_bars:
            out[sym] = bars
        else:
            print(f"  aviso: {sym} tem só {bars.height} barras — ignorado")
    if not out:
        raise SystemExit("nenhum ativo com dados suficientes no store")
    return out


def build_model(kind, panel, *, gbm_kwargs=None, sb_kwargs=None):
    if kind == "lightgbm":
        return LightGBMUniversal(panel, **(gbm_kwargs or {}))
    if kind == "super_brain":
        return get_super_brain(panel, **(sb_kwargs or {}))
    raise ValueError(f"modelo desconhecido: {kind}")


def evaluate_walk_forward(panel, feature_set, bars_by, kind, *, timeframe,
                          max_horizon, train_size, test_size, threshold,
                          gbm_kwargs=None, sb_kwargs=None):
    """Walk-forward purgado → Sharpe OOS por ativo (com risco) e Deflated Sharpe."""
    wf = PurgedWalkForward(train_size, test_size, horizon=max_horizon, embargo=max_horizon)
    oos_sharpes: list[float] = []
    ppy = periods_per_year(timeframe)

    for train_df, test_df, win in wf.split(panel.frame):
        model = build_model(kind, panel, gbm_kwargs=gbm_kwargs, sb_kwargs=sb_kwargs)
        model.fit(train_df, test_df)
        true = label_to_class(test_df["label"].to_numpy())
        acc = float((model.predict_proba(test_df).argmax(1) == true).mean())

        t0, t1 = test_df["time"].min(), test_df["time"].max()
        fold_sr = []
        for sym in bars_by:
            sub = bars_by[sym].filter((pl.col("time") >= t0) & (pl.col("time") <= t1))
            if sub.height < 200:
                continue
            strat = ModelStrategy(model, feature_set, sym, asset_class_of(sym),
                                  threshold=threshold)
            engine = BacktestEngine(get_instrument(sym), CostModel(),
                                    AccountConfig(leverage=100),
                                    initial_capital=10_000, max_lots=0.1)
            sr = engine.run(sub, strat, timeframe).report.sharpe
            fold_sr.append(sr)
            oos_sharpes.append(sr)
        print(f"  fold {win.index}: acc={acc:.3f} | Sharpe OOS médio={np.mean(fold_sr):+.2f}")

    trial_sr = [deannualize_sharpe(s, ppy) for s in oos_sharpes]
    dsr = deflated_sharpe_ratio(trial_sr, n_obs=panel.frame.height) if trial_sr else float("nan")
    return {
        "n_trials": len(oos_sharpes),
        "mean_oos_sharpe": float(np.mean(oos_sharpes)) if oos_sharpes else float("nan"),
        "pct_positive": float(np.mean(np.array(oos_sharpes) > 0)) if oos_sharpes else 0.0,
        "deflated_sharpe": float(dsr),
    }


def train(store_path, *, symbols, timeframe, start, end, model_kind, out_dir,
          max_horizon=16, train_size=40000, test_size=15000, threshold=0.15,
          gbm_kwargs=None, sb_kwargs=None, run_eval=True):
    """Pipeline completo: carrega → painel → (valida) → treina final → salva."""
    store = ParquetStore(store_path)
    bars_by = load_bars(store, symbols, timeframe, start, end)
    print(f"Ativos carregados: {list(bars_by)}")

    feature_set = default_feature_set()
    panel = build_universal_panel(bars_by, feature_set, asset_class_of=asset_class_of,
                                  max_horizon=max_horizon, vol_mult=1.5)
    print(f"Painel: {panel.frame.height:,} linhas | {len(panel.feature_names)} features "
          f"| {panel.n_assets} ativos")

    kinds = ["lightgbm", "super_brain"] if model_kind == "both" else [model_kind]
    out_root = Path(out_dir)
    summary = {}

    for kind in kinds:
        print(f"\n=== {kind} ===")
        report = {}
        if run_eval:
            try:
                report = evaluate_walk_forward(
                    panel, feature_set, bars_by, kind, timeframe=timeframe,
                    max_horizon=max_horizon, train_size=train_size, test_size=test_size,
                    threshold=threshold, gbm_kwargs=gbm_kwargs, sb_kwargs=sb_kwargs)
                print(f"  -> OOS Sharpe médio {report['mean_oos_sharpe']:+.2f} | "
                      f"Deflated Sharpe {report['deflated_sharpe']:.1%}")
            except ValueError as e:
                print(f"  walk-forward pulado: {e}")

        # Modelo FINAL treinado em TODO o histórico, para deploy.
        final = build_model(kind, panel, gbm_kwargs=gbm_kwargs, sb_kwargs=sb_kwargs)
        final.fit(panel.frame)
        dest = out_root / kind
        final.save(dest)
        manifest = {
            "kind": kind,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "symbols": list(bars_by),
            "timeframe": timeframe.value,
            "period": [start.isoformat(), end.isoformat()],
            "max_horizon": max_horizon,
            "n_features": len(panel.feature_names),
            "panel_rows": panel.frame.height,
            "evaluation": report,
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"  ✓ artefato salvo em {dest}")
        summary[kind] = manifest

    return summary


def _date(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main():
    ap = argparse.ArgumentParser(description="Treino INNOVA EA (LightGBM / Super Cérebro)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--out", default="./artifacts")
    ap.add_argument("--model", choices=["lightgbm", "super_brain", "both"], default="lightgbm")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--start", type=_date, default=_date("2015-01-01"))
    ap.add_argument("--end", type=_date, default=None)
    ap.add_argument("--max-horizon", type=int, default=16)
    ap.add_argument("--train-size", type=int, default=40000)
    ap.add_argument("--test-size", type=int, default=15000)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--no-eval", action="store_true", help="pula a validação walk-forward")
    args = ap.parse_args()

    summary = train(
        args.store,
        symbols=args.symbols,
        timeframe=Timeframe(args.timeframe),
        start=args.start,
        end=args.end or datetime.now(timezone.utc),
        model_kind=args.model,
        out_dir=args.out,
        max_horizon=args.max_horizon,
        train_size=args.train_size,
        test_size=args.test_size,
        threshold=args.threshold,
        run_eval=not args.no_eval,
    )
    print("\n✓ Treino concluído.")
    for kind, m in summary.items():
        ev = m.get("evaluation", {})
        print(f"  {kind}: DSR={ev.get('deflated_sharpe', float('nan')):.1%} "
              f"| artefato em {Path(args.out) / kind}")


if __name__ == "__main__":
    main()
