#!/usr/bin/env python
"""Estudo do INNOVA Breakout — nosso operacional próprio (Williams + Unger).

Funde, num só sistema validado, os dois campeões mundiais da família de
rompimento de volatilidade:

  * Entrada: Opening Range Breakout (Andrea Unger, tetracampeão mundial).
  * Gatilho de volatilidade: o range de abertura precisa ser >= ``min_or_frac``
    do range do dia anterior (essência de Larry Williams).
  * Filtros de contexto (Unger): "deixar a segunda-feira de fora" e pular dia
    após range extremo no dia anterior (``max_prior_mult``).
  * Saída honesta: alvo/stop em unidades de risco + fim do dia.

Mesmo rigor: custos reais, walk-forward, Deflated Sharpe.

Exemplo (M15, range de abertura = 1ª hora = 4 barras):
    python scripts/study_innova.py --store .\\data --timeframe M15 --or-bars 4 --cost 0.0004
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from innova_ea.backtest.metrics import compute_metrics, periods_per_year
from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.research import (
    deflated_sharpe_ratio,
    innova_breakout_outcomes,
    orb_backtest,
)
from innova_ea.research.overfitting import deannualize_sharpe
from innova_ea.universe import DEFAULT_UNIVERSE


def _folds(n, fold_size):
    lo = 0
    while lo + fold_size <= n:
        yield lo, lo + fold_size
        lo += fold_size


def study_asset(bars, tf, *, or_bars, reward_risk, cost, skip_monday,
                min_or_frac, max_prior_mult, fold_size):
    out = innova_breakout_outcomes(
        bars, or_bars=or_bars, reward_risk=reward_risk, cost_frac=cost,
        skip_monday=skip_monday, min_or_frac=min_or_frac, max_prior_mult=max_prior_mult,
    )
    sharpes, drawdowns, n_trades = [], [], 0
    for lo, hi in _folds(bars.height, fold_size):
        eq, pos, trades = orb_backtest(out[lo:hi], initial_capital=10_000.0)
        if eq.shape[0] > 2 and trades.size > 0:
            mtr = compute_metrics(eq, pos, trades, tf, 10_000.0)
            sharpes.append(mtr.sharpe)
            drawdowns.append(mtr.max_drawdown)
            n_trades += int(trades.size)
    return sharpes, drawdowns, n_trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Estudo INNOVA Breakout (Williams + Unger)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--or-bars", type=int, default=4, help="barras do range de abertura (M15×4=1h).")
    ap.add_argument("--reward-risk", type=float, default=1.0)
    ap.add_argument("--cost", type=float, default=0.0004)
    ap.add_argument("--min-or-frac", type=float, default=0.5,
                    help="gatilho de Williams: range de abertura >= esta fração do range anterior (0 desliga).")
    ap.add_argument("--max-prior-mult", type=float, default=3.0,
                    help="filtro de Unger: pula dia se range anterior > este múltiplo do range médio (0 desliga).")
    ap.add_argument("--no-skip-monday", action="store_true", help="não pular segunda (filtro de Unger).")
    ap.add_argument("--fold-size", type=int, default=8000)
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    ppy = periods_per_year(tf)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    skip_monday = not args.no_skip_monday

    all_sharpes: list[float] = []
    all_drawdowns: list[float] = []
    per_asset: dict[str, float] = {}

    print(f"INNOVA Breakout (Williams+Unger) | {tf.value} | or_bars={args.or_bars} "
          f"skip_monday={skip_monday} min_or_frac={args.min_or_frac} "
          f"max_prior_mult={args.max_prior_mult} cost={args.cost}\n")
    for sym in args.symbols:
        bars = store.read(sym, tf, args.start, end)
        if bars.height < args.fold_size * 2:
            print(f"  {sym}: dados insuficientes — pulando")
            continue
        sharpes, drawdowns, n_trades = study_asset(
            bars, tf, or_bars=args.or_bars, reward_risk=args.reward_risk, cost=args.cost,
            skip_monday=skip_monday, min_or_frac=args.min_or_frac,
            max_prior_mult=args.max_prior_mult, fold_size=args.fold_size,
        )
        if not sharpes:
            continue
        m = float(np.mean(sharpes))
        worst_dd = min(drawdowns) if drawdowns else 0.0
        per_asset[sym] = m
        all_sharpes.extend(sharpes)
        all_drawdowns.extend(drawdowns)
        print(f"  {sym:<8} Sharpe OOS méd={m:+.2f} | DD pior={worst_dd:+.1%} | "
              f"{len(sharpes)} folds | {n_trades:,} trades")

    if not all_sharpes:
        print("\nNenhum ativo avaliado.")
        return

    sr = np.array(all_sharpes)
    agg = float(sr.mean())
    trial = [deannualize_sharpe(x, ppy) for x in all_sharpes]
    dsr = deflated_sharpe_ratio(trial, n_obs=len(trial), selected_sharpe=deannualize_sharpe(agg, ppy))
    n_pos = sum(1 for v in per_asset.values() if v > 0)

    dd = np.array(all_drawdowns)
    print(f"\n== Agregado OOS ({sr.size} ensaios, {len(per_asset)} ativos) ==")
    print(f"  Sharpe OOS médio : {agg:+.2f}  (positivos {np.mean(sr > 0):.0%})")
    print(f"  Ativos com Sharpe>0: {n_pos}/{len(per_asset)} "
          f"(edge {'AMPLO' if n_pos >= 0.7 * len(per_asset) else 'CONCENTRADO'})")
    print(f"  Drawdown (OOS)   : pior {dd.min():+.1%} | médio {dd.mean():+.1%}")
    print(f"  Deflated Sharpe  : {dsr:.1%}")


if __name__ == "__main__":
    main()
