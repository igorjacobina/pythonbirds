#!/usr/bin/env python
"""Estudo de RELATIVE-VALUE / pares (statistical arbitrage) sobre o histórico.

Para cada par economicamente motivado (ouro/prata, EUR/GBP, índices US...):
estima o hedge ratio β no TREINO (walk-forward), opera a reversão do spread
(z-score) no out-of-sample com custos, e reporta Sharpe OOS por par + agregado +
Deflated Sharpe + consistência entre folds. Mesmo rigor honesto de sempre.

Exemplo:
    python scripts/study_pairs.py --store .\\data --timeframe H4 --cost 0.0004
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from innova_ea.backtest.metrics import compute_metrics, periods_per_year
from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.research import (
    PairConfig,
    align_log_prices,
    deflated_sharpe_ratio,
    estimate_beta,
    pairs_backtest,
    pairs_signal,
    spread_zscore,
)
from innova_ea.research.overfitting import deannualize_sharpe

# Pares com fundamento econômico (não data-mined).
CANDIDATE_PAIRS = [
    ("XAUUSD", "XAGUSD"),   # ouro / prata (clássico de stat-arb)
    ("EURUSD", "GBPUSD"),   # EUR vs GBP (ambos contra USD)
    ("EURUSD", "USDCHF"),   # EUR vs CHF (inverso)
    ("AUDUSD", "USDCAD"),   # moedas de commodity
    ("US30", "US500"),      # índices US
    ("US500", "US100"),
    ("US30", "US100"),
    ("DE40", "UK100"),      # índices Europa
    ("DE40", "US500"),      # Europa vs US
    ("UK100", "US500"),
]


def _windows(n, train_size, test_size):
    test_lo = train_size
    while test_lo + test_size <= n:
        yield 0, test_lo, test_lo, test_lo + test_size      # treino anchored [0,test_lo)
        test_lo += test_size


def study_pair(la, lb, cfg, tf, *, train_size, test_size, cost):
    """Walk-forward: β do treino, opera o spread no teste. Retorna lista de Sharpes."""
    n = len(la)
    sharpes, n_trades = [], 0
    for tlo, thi, slo, shi in _windows(n, train_size, test_size):
        beta = estimate_beta(la[tlo:thi], lb[tlo:thi])
        seg_lo = max(0, slo - cfg.z_window - 5)             # aquecimento do z
        spread, z, _ = spread_zscore(la[seg_lo:shi], lb[seg_lo:shi], cfg, beta=beta)
        sig = pairs_signal(z, cfg)
        off = slo - seg_lo
        eq, pos, trades = pairs_backtest(spread[off:], sig[off:],
                                         cost_frac_per_turn=cost, initial_capital=10_000.0)
        if eq.shape[0] > 2:
            sharpes.append(compute_metrics(eq, pos, trades, tf, 10_000.0).sharpe)
            n_trades += int(trades.size)
    return sharpes, n_trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Estudo de pares (INNOVA EA)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--timeframe", default="H4")
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--z-window", type=int, default=100)
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=4.0)
    ap.add_argument("--max-hold", type=int, default=100)
    ap.add_argument("--cost", type=float, default=0.0004,
                    help="custo por virada (2 pernas), como fração. Padrão ~4 bps.")
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--test-size", type=int, default=2000)
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    ppy = periods_per_year(tf)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    cfg = PairConfig(z_window=args.z_window, entry=args.entry, exit=args.exit,
                     stop=args.stop, max_hold=args.max_hold)

    all_sharpes: list[float] = []
    per_pair: dict[str, float] = {}
    fold_means: list[float] = []

    print(f"Estudo de pares | {tf.value} | entry={args.entry} exit={args.exit} "
          f"cost={args.cost}\n")
    for a, b in CANDIDATE_PAIRS:
        ba = store.read(a, tf, args.start, end)
        bb = store.read(b, tf, args.start, end)
        if ba.height < 5000 or bb.height < 5000:
            print(f"  {a}/{b}: dados insuficientes — pulando")
            continue
        _, la, lb = align_log_prices(ba, bb)
        if la.shape[0] < args.train_size + args.test_size:
            print(f"  {a}/{b}: histórico comum curto — pulando")
            continue
        sharpes, n_trades = study_pair(la, lb, cfg, tf, train_size=args.train_size,
                                       test_size=args.test_size, cost=args.cost)
        if not sharpes:
            continue
        m = float(np.mean(sharpes))
        per_pair[f"{a}/{b}"] = m
        all_sharpes.extend(sharpes)
        fold_means.append(m)
        print(f"  {a}/{b:<8} Sharpe OOS médio={m:+.2f} | {len(sharpes)} folds | {n_trades:,} trades")

    if not all_sharpes:
        print("\nNenhum par avaliado.")
        return

    sr = np.array(all_sharpes)
    agg = float(sr.mean())
    trial = [deannualize_sharpe(x, ppy) for x in all_sharpes]
    dsr = deflated_sharpe_ratio(trial, n_obs=len(trial), selected_sharpe=deannualize_sharpe(agg, ppy))
    n_pos = sum(1 for v in per_pair.values() if v > 0)

    print(f"\n== Agregado OOS ({sr.size} ensaios, {len(per_pair)} pares) ==")
    print(f"  Sharpe OOS médio : {agg:+.2f}  (positivos {np.mean(sr > 0):.0%})")
    print(f"  Pares com Sharpe>0: {n_pos}/{len(per_pair)} "
          f"(edge {'AMPLO' if n_pos >= 0.7 * len(per_pair) else 'CONCENTRADO'})")
    print(f"  Deflated Sharpe  : {dsr:.1%}")


if __name__ == "__main__":
    main()
