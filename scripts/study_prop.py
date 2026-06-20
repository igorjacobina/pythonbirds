#!/usr/bin/env python
"""Estudo de MESA PROPRIETÁRIA — o operacional passa no challenge? Com qual risco?

Junta os ativos numa CONTA ÚNICA (P&L diário somado, como numa mesa real),
aplica a estratégia escolhida (``--strategy``) e simula o desafio da mesa
(meta de lucro vs. perda diária/total) em vários níveis de alavancagem.
Reporta, honestamente: % de PASSAR, dias medianos até passar, retorno anual
(CAGR) e pior drawdown por nível de risco.

Estratégias:
  * ``innova``   — INNOVA Breakout (Williams + Unger), raro e seletivo.
  * ``straddle`` — straddle de volatilidade (mais frequente e amplo) → tende a
    ser MELHOR para mesa, pois suaviza a equity com mais trades.

Granularidade DIÁRIA (limite diário checado no fim do dia) — referência, não
garantia (mesas checam intradiário).

Exemplos:
    # straddle, portfólio amplo, regra FTMO, sem limite de tempo:
    python scripts/study_prop.py --strategy straddle --store .\\data --timeframe M15
    # breakout no trio robusto:
    python scripts/study_prop.py --strategy innova --symbols XAUUSD XAGUSD DE40 --store .\\data
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data import ParquetStore
from innova_ea.research.innova_breakout import innova_breakout_outcomes
from innova_ea.research.prop import sweep_leverage
from innova_ea.research.straddle import straddle_outcomes
from innova_ea.universe import DEFAULT_UNIVERSE


def _innova_net(bars, *, or_bars, cost, **_):
    """Net fracionário por barra do INNOVA Breakout (1 trade/dia, já não sobreposto)."""
    out = innova_breakout_outcomes(bars, or_bars=or_bars, reward_risk=1.0, cost_frac=cost,
                                   skip_monday=True, min_or_frac=0.5, max_prior_mult=3.0)
    return out["net_ret"].to_numpy()


def _straddle_net(bars, *, cost, band_lookback=12, horizon=24, target_mult=1.5,
                  stop_mult=1.5, atr_window=24, **_):
    """Net fracionário por barra do straddle, NÃO sobreposto (custo fracionário justo)."""
    out = straddle_outcomes(bars, band_lookback=band_lookback, horizon=horizon,
                            target_mult=target_mult, stop_mult=stop_mult,
                            atr_window=atr_window, cost_points=0.0, point=1e-5)
    net_price = out["net_price"].to_numpy()
    triggered = out["triggered"].to_numpy()
    resolve = out["resolve"].to_numpy()
    close = bars["close"].to_numpy().astype(np.float64)
    n = close.shape[0]
    net = np.zeros(n, dtype=np.float64)
    busy = -1
    for i in range(n):
        if triggered[i] and i > busy and resolve[i] > 0 and close[i] > 0.0:
            net[i] = net_price[i] / close[i] - cost          # fração líquida do custo
            busy = i + int(resolve[i])
    return net


STRATEGIES = {"innova": _innova_net, "straddle": _straddle_net}


def combined_daily_returns(store, symbols, tf, start, end, *, net_fn, or_bars, cost):
    """P&L diário (fração) da carteira: soma o net por dia entre os ativos (1x cada)."""
    master = None
    used = []
    for sym in symbols:
        bars = store.read(sym, tf, start, end)
        if bars.height < 1000:
            print(f"  {sym}: dados insuficientes — pulando")
            continue
        net = net_fn(bars, or_bars=or_bars, cost=cost)
        d = (bars.select(pl.col("time").dt.epoch(time_unit="d").alias("day"))
                 .with_columns(pl.Series("r", net))
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
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="straddle")
    ap.add_argument("--store", default="./data")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=datetime(2015, 1, 1, tzinfo=timezone.utc))
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    default=None)
    ap.add_argument("--or-bars", type=int, default=4)
    ap.add_argument("--cost", type=float, default=0.0004)
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
    net_fn = STRATEGIES[args.strategy]

    print(f"Mesa proprietária | estratégia={args.strategy} | {tf.value} | "
          f"meta {args.profit_target:.0%} | perda diária {args.daily_limit:.0%} | "
          f"perda total {args.total_limit:.0%} | "
          f"prazo {'sem limite' if args.max_days <= 0 else str(args.max_days)+' dias'}\n")

    daily, used = combined_daily_returns(
        store, args.symbols, tf, args.start, end, net_fn=net_fn,
        or_bars=args.or_bars, cost=args.cost)
    if daily is None or daily.size == 0:
        print("Nenhum ativo avaliado.")
        return

    n_trade_days = int((daily != 0.0).sum())
    print(f"Carteira: {len(used)} ativos | {daily.size} dias | {n_trade_days} dias com trade\n")

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
          "que sirva, mantendo o pior DD com folga abaixo do limite total.")


if __name__ == "__main__":
    main()
