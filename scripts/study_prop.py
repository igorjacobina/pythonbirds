#!/usr/bin/env python
"""Estudo de MESA PROPRIETÁRIA — o operacional passa no challenge? Com qual risco?

Junta os ativos escolhidos numa CONTA ÚNICA (P&L diário somado, como numa mesa
real), aplica o INNOVA Breakout e simula o desafio da mesa (meta de lucro vs.
perda diária/total) em vários níveis de alavancagem. Reporta, honestamente:

  * % de PASSAR (começando o desafio em centenas de pontos do histórico)
  * dias medianos até passar
  * retorno anual (CAGR) e pior drawdown naquele nível de risco

Exemplo (trio robusto, regra FTMO, sem limite de tempo):
    python scripts/study_prop.py --store .\\data --symbols XAUUSD XAGUSD DE40 --timeframe M15
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.research.innova_breakout import innova_breakout_outcomes
from innova_ea.research.prop import sweep_leverage


def combined_daily_returns(store, symbols, tf, start, end, *, or_bars, reward_risk,
                           cost, skip_monday, min_or_frac, max_prior_mult):
    """P&L diário (fração) da carteira: soma o net por dia entre os ativos (1x cada)."""
    master = None
    used = []
    for sym in symbols:
        bars = store.read(sym, tf, start, end)
        if bars.height < 1000:
            print(f"  {sym}: dados insuficientes — pulando")
            continue
        out = innova_breakout_outcomes(
            bars, or_bars=or_bars, reward_risk=reward_risk, cost_frac=cost,
            skip_monday=skip_monday, min_or_frac=min_or_frac, max_prior_mult=max_prior_mult)
        d = (bars.select(pl.col("time").dt.epoch(time_unit="d").alias("day"))
                 .with_columns(pl.Series("r", out["net_ret"].to_numpy()))
                 .group_by("day").agg(pl.col("r").sum().alias(sym)))
        master = d if master is None else master.join(d, on="day", how="full", coalesce=True)
        used.append(sym)
    if master is None:
        return None, []
    master = master.fill_null(0.0).sort("day")
    sym_cols = [c for c in master.columns if c != "day"]
    combined = master.select(pl.sum_horizontal(sym_cols).alias("r"))["r"].to_numpy()
    return combined, used


def main() -> None:
    ap = argparse.ArgumentParser(description="Estudo de mesa proprietária (INNOVA EA)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD", "XAGUSD", "DE40"])
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--or-bars", type=int, default=4)
    ap.add_argument("--reward-risk", type=float, default=1.0)
    ap.add_argument("--cost", type=float, default=0.0004)
    ap.add_argument("--min-or-frac", type=float, default=0.5)
    ap.add_argument("--max-prior-mult", type=float, default=3.0)
    ap.add_argument("--no-skip-monday", action="store_true")
    # Regras da mesa (default = FTMO fase 1).
    ap.add_argument("--profit-target", type=float, default=0.10)
    ap.add_argument("--daily-limit", type=float, default=0.05)
    ap.add_argument("--total-limit", type=float, default=0.10)
    ap.add_argument("--max-days", type=int, default=0, help="0 = sem limite de tempo.")
    ap.add_argument("--levs", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    skip_monday = not args.no_skip_monday

    print(f"Mesa proprietária | {tf.value} | meta {args.profit_target:.0%} | "
          f"perda diária {args.daily_limit:.0%} | perda total {args.total_limit:.0%} | "
          f"prazo {'sem limite' if args.max_days <= 0 else str(args.max_days)+' dias'}\n")

    daily, used = combined_daily_returns(
        store, args.symbols, tf, args.start, end, or_bars=args.or_bars,
        reward_risk=args.reward_risk, cost=args.cost, skip_monday=skip_monday,
        min_or_frac=args.min_or_frac, max_prior_mult=args.max_prior_mult)
    if daily is None or daily.size == 0:
        print("Nenhum ativo avaliado.")
        return

    n_trade_days = int((daily != 0.0).sum())
    print(f"Carteira: {used} | {daily.size} dias | {n_trade_days} dias com trade\n")

    rows = sweep_leverage(
        daily, args.levs, profit_target=args.profit_target, daily_limit=args.daily_limit,
        total_limit=args.total_limit, max_days=args.max_days)

    print(f"{'Lev':>4} | {'Passa':>7} | {'Quebra':>7} | {'dias p/ passar':>14} | "
          f"{'Retorno/ano':>11} | {'Pior DD':>8}")
    print("-" * 66)
    for r in rows:
        md = "—" if r["median_pass_days"] != r["median_pass_days"] else f"{r['median_pass_days']:.0f}"
        cagr = "—" if r["cagr"] != r["cagr"] else f"{r['cagr']:+.1%}"
        print(f"{r['lev']:>4.0f} | {r['pass_rate']:>7.0%} | {r['fail_rate']:>7.0%} | "
              f"{md:>14} | {cagr:>11} | {r['max_drawdown']:>8.1%}")

    print("\nRegra prática: escolha a alavancagem com ALTA % de passar E retorno/ano "
          "que te sirva, mantendo o pior DD com folga abaixo do limite total.")


if __name__ == "__main__":
    main()
