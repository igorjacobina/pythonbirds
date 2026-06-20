#!/usr/bin/env python
"""BUSCA de operacional para MESA PROPRIETÁRIA — completa e honesta.

Objetivo: achar uma configuração do INNOVA Breakout que, como CARTEIRA
diversificada (equal-weight), passe num challenge de prop firm com drawdown
baixo. Para não se enganar com overfitting, faz:

  1. divide o histórico em TREINO (passado) e TESTE (futuro nunca visto);
  2. varre uma grade de parâmetros (filtros/alvo) no TREINO, monta a carteira
     equal-weight e acha, para cada config, a MELHOR alavancagem (maior % de
     passar mantendo o pior drawdown sob o limite);
  3. ranqueia as configs pelo desempenho no TREINO;
  4. VALIDA a melhor config no TESTE (out-of-sample) — esse é o número honesto.

Se a melhor do treino também passar no teste → operacional validado para mesa.
Se desabar no teste → era overfitting, e o estudo te diz isso na cara.

Exemplo:
    python scripts/study_prop_search.py --store .\\data --timeframe M15 --split 2021-01-01
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.research.innova_breakout import innova_breakout_outcomes
from innova_ea.research.prop import challenge_stats

# Carteira padrão: metais + índices (onde o rompimento tem lógica econômica e
# edge confirmado). Pares de Forex major ficam de fora por falta de edge.
DEFAULT_SYMBOLS = ["XAUUSD", "XAGUSD", "DE40", "US30", "US100", "US500", "JP225", "HK50"]

# Grade de parâmetros do breakout (frequência × qualidade × alvo).
GRID = [
    dict(min_or_frac=m, max_prior_mult=p, reward_risk=rr, skip_monday=sm)
    for m in (0.0, 0.3, 0.5)
    for p in (0.0, 3.0)
    for rr in (1.0, 1.5, 2.0)
    for sm in (True, False)
]


def portfolio_daily(bars_by, *, or_bars, cost, params):
    """Retorno diário (fração) da carteira equal-weight para uma config."""
    master = None
    for sym, bars in bars_by.items():
        out = innova_breakout_outcomes(bars, or_bars=or_bars, cost_frac=cost, **params)
        d = (bars.select(pl.col("time").dt.epoch(time_unit="d").alias("day"))
                 .with_columns(pl.Series("r", out["net_ret"].to_numpy()))
                 .group_by("day").agg(pl.col("r").sum().alias(sym)))
        master = d if master is None else master.join(d, on="day", how="full", coalesce=True)
    if master is None:
        return None, None
    master = master.fill_null(0.0).sort("day")
    cols = [c for c in master.columns if c != "day"]
    days = master["day"].to_numpy()
    rets = master.select((pl.sum_horizontal(cols) / float(len(cols))).alias("r"))["r"].to_numpy()
    return days, rets


def best_leverage(daily, levs, *, dd_cap, **rules):
    """Melhor alavancagem: maior % de passar com pior DD sob ``dd_cap`` e retorno>0."""
    best = None
    for lv in levs:
        st = challenge_stats(daily, lv, **rules)
        ok_dd = st["max_drawdown"] >= -dd_cap
        ok_ret = st["cagr"] == st["cagr"] and st["cagr"] > 0
        if ok_dd and ok_ret:
            if best is None or st["pass_rate"] > best["pass_rate"]:
                best = st
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Busca de operacional para mesa (INNOVA EA)")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--split", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2021, 1, 1, tzinfo=timezone.utc), help="início do TESTE (OOS).")
    ap.add_argument("--or-bars", type=int, default=4)
    ap.add_argument("--cost", type=float, default=0.0004)
    ap.add_argument("--profit-target", type=float, default=0.10)
    ap.add_argument("--daily-limit", type=float, default=0.05)
    ap.add_argument("--total-limit", type=float, default=0.10)
    ap.add_argument("--dd-cap", type=float, default=0.09, help="pior DD tolerado (folga sob o total).")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--levs", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 15])
    ap.add_argument("--stride", type=int, default=2, help="passo dos inícios do desafio (acelera).")
    args = ap.parse_args()

    tf = Timeframe(args.timeframe)
    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    rules = dict(profit_target=args.profit_target, daily_limit=args.daily_limit,
                 total_limit=args.total_limit, max_days=args.max_days, stride=args.stride)

    print(f"BUSCA p/ mesa | {tf.value} | {len(args.symbols)} ativos | split TESTE={args.split.date()} "
          f"| meta {args.profit_target:.0%} / diária {args.daily_limit:.0%} / total {args.total_limit:.0%}")
    print(f"Grade: {len(GRID)} configs\n")

    bars_by = {}
    for sym in args.symbols:
        bars = store.read(sym, tf, args.start, end)
        if bars.height >= 1000:
            bars_by[sym] = bars
        else:
            print(f"  {sym}: dados insuficientes — fora")
    if not bars_by:
        print("Sem dados.")
        return

    split_day = int((args.split - datetime(1970, 1, 1, tzinfo=timezone.utc)).days)

    ranked = []
    for params in GRID:
        days, rets = portfolio_daily(bars_by, or_bars=args.or_bars, cost=args.cost, params=params)
        if days is None:
            continue
        tr = rets[days < split_day]
        if tr.size < 250:
            continue
        bl = best_leverage(tr, args.levs, dd_cap=args.dd_cap, **rules)
        if bl is None:
            continue
        ranked.append((params, bl))

    if not ranked:
        print("Nenhuma config passou no TREINO com DD sob o limite. "
              "O edge não é forte o bastante para mesa nessa configuração — honestamente.")
        return

    ranked.sort(key=lambda x: (x[1]["pass_rate"], x[1]["cagr"]), reverse=True)

    print("== TOP configs no TREINO (in-sample) ==")
    print(f"{'min_or':>6} {'maxP':>5} {'RR':>4} {'segDr':>6} | {'lev':>3} | {'passa':>6} | {'ret/ano':>8} | {'DD':>7}")
    for params, bl in ranked[:8]:
        print(f"{params['min_or_frac']:>6.1f} {params['max_prior_mult']:>5.1f} "
              f"{params['reward_risk']:>4.1f} {str(not params['skip_monday'])[:5]:>6} | "
              f"{bl['lev']:>3.0f} | {bl['pass_rate']:>6.0%} | {bl['cagr']:>+8.1%} | {bl['max_drawdown']:>7.1%}")

    # Validação OOS da melhor config, na MESMA alavancagem escolhida no treino.
    best_params, best_bl = ranked[0]
    lev = best_bl["lev"]
    days, rets = portfolio_daily(bars_by, or_bars=args.or_bars, cost=args.cost, params=best_params)
    te = rets[days >= split_day]
    print(f"\n== VALIDAÇÃO OOS (TESTE, nunca visto) — melhor config @ lev {lev:.0f} ==")
    print(f"  config: {best_params}")
    if te.size < 100:
        print("  TESTE curto demais para validar.")
        return
    oos = challenge_stats(te, lev, **rules)
    veredito = ("VALIDADO ✅" if oos["pass_rate"] >= 0.6 and oos["max_drawdown"] >= -args.total_limit
                else "NÃO segura no OOS ❌ (era overfitting)")
    print(f"  TREINO : passa {best_bl['pass_rate']:>4.0%} | ret/ano {best_bl['cagr']:+.1%} | DD {best_bl['max_drawdown']:.1%}")
    print(f"  TESTE  : passa {oos['pass_rate']:>4.0%} | ret/ano {oos['cagr']:+.1%} | DD {oos['max_drawdown']:.1%}")
    print(f"  --> {veredito}")


if __name__ == "__main__":
    main()
