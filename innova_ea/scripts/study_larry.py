#!/usr/bin/env python
"""Estudo do método de LARRY WILLIAMS (volatility breakout) sobre o histórico.

Testa, com o mesmo rigor honesto, o operacional que venceu o Campeonato Mundial
de Futuros de 1987 (US$10k → US$1,14M). É da MESMA família (rompimento de
volatilidade) que o INNOVA EA já validou como a única com edge real.

Para cada ativo (D1 por padrão): roda o breakout, divide em folds OOS, e reporta
Sharpe por ativo + agregado + Deflated Sharpe + consistência entre folds.

Exemplo:
    python scripts/study_larry.py --store .\\data --timeframe D1 --k 0.5 --cost 0.0006
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from innova_ea.backtest.metrics import compute_metrics, periods_per_year
from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.research import deflated_sharpe_ratio, larry_backtest, larry_outcomes
from innova_ea.research.overfitting import deannualize_sharpe
from innova_ea.universe import DEFAULT_UNIVERSE


def _folds(n, fold_size):
    lo = 0
    while lo + fold_size <= n:
        yield lo, lo + fold_size
        lo += fold_size


def study_asset(bars, tf, *, k, max_hold, cost, fold_size):
    out = larry_outcomes(bars, k=k, max_hold=max_hold, cost_frac=cost)
    sharpes, n_trades = [], 0
    for lo, hi in _folds(bars.height, fold_size):
        eq, pos, trades = larry_backtest(out[lo:hi], initial_capital=10_000.0)
        if eq.shape[0] > 2 and trades.size > 0:
            sharpes.append(compute_metrics(eq, pos, trades, tf, 10_000.0).sharpe)
            n_trades += int(trades.size)
    return sharpes, n_trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Estudo Larry Williams (INNOVA EA)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--timeframe", default="D1")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--k", type=float, default=0.5, help="multiplicador do range (Larry usava ~0.25–0.6).")
    ap.add_argument("--max-hold", type=int, default=5)
    ap.add_argument("--cost", type=float, default=0.0006, help="custo de rodada (fração).")
    ap.add_argument("--fold-size", type=int, default=400)
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    ppy = periods_per_year(tf)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)

    all_sharpes: list[float] = []
    per_asset: dict[str, float] = {}
    per_trades: dict[str, int] = {}

    print(f"Larry Williams volatility breakout | {tf.value} | k={args.k} cost={args.cost}\n")
    for sym in args.symbols:
        bars = store.read(sym, tf, args.start, end)
        if bars.height < args.fold_size * 2:
            print(f"  {sym}: dados insuficientes — pulando")
            continue
        sharpes, n_trades = study_asset(bars, tf, k=args.k, max_hold=args.max_hold,
                                        cost=args.cost, fold_size=args.fold_size)
        if not sharpes:
            continue
        m = float(np.mean(sharpes))
        per_asset[sym] = m
        per_trades[sym] = n_trades
        all_sharpes.extend(sharpes)
        print(f"  {sym:<8} Sharpe OOS médio={m:+.2f} | {len(sharpes)} folds | {n_trades:,} trades")

    if not all_sharpes:
        print("\nNenhum ativo avaliado.")
        return

    sr = np.array(all_sharpes)
    agg = float(sr.mean())
    trial = [deannualize_sharpe(x, ppy) for x in all_sharpes]
    dsr = deflated_sharpe_ratio(trial, n_obs=len(trial), selected_sharpe=deannualize_sharpe(agg, ppy))
    n_pos = sum(1 for v in per_asset.values() if v > 0)

    print(f"\n== Agregado OOS ({sr.size} ensaios, {len(per_asset)} ativos) ==")
    print(f"  Sharpe OOS médio : {agg:+.2f}  (positivos {np.mean(sr > 0):.0%})")
    print(f"  Ativos com Sharpe>0: {n_pos}/{len(per_asset)} "
          f"(edge {'AMPLO' if n_pos >= 0.7 * len(per_asset) else 'CONCENTRADO'})")
    print(f"  Deflated Sharpe  : {dsr:.1%}")


if __name__ == "__main__":
    main()
