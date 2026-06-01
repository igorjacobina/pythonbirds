#!/usr/bin/env python
"""Ingestão de histórico EXTERNO (Dukascopy / HistData) → Parquet.

Plano de DADOS desacoplado da execução: usa um provedor de histórico profundo
(Dukascopy ou HistData) para montar a base densa de treino (M1 desde 2015) de
toda a cesta, gravando no MESMO ``ParquetStore`` que o ``train_models.py`` lê.
A FTMO/MT5 permanece exclusiva do serviço de execução ao vivo.

Exemplos:
    # Dukascopy (download direto; profundidade máxima):
    python scripts/ingest_external.py --source dukascopy --start 2015-01-01 --store ./data

    # HistData (arquivos M1 já baixados num diretório):
    python scripts/ingest_external.py --source histdata --csv-dir ./histdata_raw --store ./data
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from innova_ea.core.enums import Timeframe
from innova_ea.data import (
    DukascopySource,
    ForexCalendar,
    HistDataSource,
    ParquetStore,
    analyze_gaps,
    clean_bars,
    common_forex_holidays,
    resample,
)
from innova_ea.data.sources.dukascopy import DUKASCOPY_INSTRUMENTS
from innova_ea.data.sources.histdata import HISTDATA_SYMBOLS
from innova_ea.universe import DEFAULT_UNIVERSE, asset_class_of, is_exchange_traded

DERIVED_TFS = [Timeframe.M5, Timeframe.M15, Timeframe.M30,
               Timeframe.H1, Timeframe.H4, Timeframe.D1]


def _date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_source(args):
    if args.source == "dukascopy":
        return DukascopySource(max_workers=args.max_workers), set(DUKASCOPY_INSTRUMENTS)
    if args.source == "histdata":
        if not args.csv_dir:
            raise SystemExit("--csv-dir é obrigatório para --source histdata")
        return HistDataSource(args.csv_dir), set(HISTDATA_SYMBOLS)
    raise SystemExit(f"fonte desconhecida: {args.source}")


def ingest_symbol(source, store, fx_calendar, symbol, start, end) -> None:
    cls = asset_class_of(symbol)
    print(f"\n{'='*64}\n  {symbol} [{cls}] — M1 de {start.date()} a {end.date()}\n{'='*64}")
    total = 0
    cursor = start
    while cursor < end:
        year_end = min(datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc), end)
        raw = source.fetch(symbol, Timeframe.M1, cursor, year_end)
        if raw.height:
            clean, _ = clean_bars(raw)
            n = store.write(symbol, Timeframe.M1, clean)
            total += clean.height
            print(f"  {cursor.year}: {clean.height:>8,} barras → gravadas {n:,}")
        else:
            print(f"  {cursor.year}: sem dados")
        cursor = year_end

    full = store.read(symbol, Timeframe.M1, start, end)
    if full.height == 0:
        print("  ⚠ nenhum dado M1 — pulando derivação/gaps.")
        return
    for tf in DERIVED_TFS:
        store.write(symbol, tf, resample(full, tf))
    print(f"  derivados M5..D1 ok | total M1: {total:,}")

    cal = None if is_exchange_traded(symbol) else fx_calendar
    if cal is not None:
        gaps = analyze_gaps(full, cal, Timeframe.M1, start, end)
        print("  -- Gaps (M1) --")
        for line in gaps.summary().splitlines():
            print(f"    {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingestão externa (Dukascopy/HistData) → Parquet")
    ap.add_argument("--source", choices=["dukascopy", "histdata"], required=True)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--start", type=_date, default=_date("2015-01-01"))
    ap.add_argument("--end", type=_date, default=None)
    ap.add_argument("--store", default="./data")
    ap.add_argument("--csv-dir", default=None, help="diretório dos CSVs (HistData).")
    ap.add_argument("--max-workers", type=int, default=8, help="downloads concorrentes (Dukascopy).")
    args = ap.parse_args()

    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    fx_calendar = ForexCalendar(holidays=common_forex_holidays(range(args.start.year, end.year + 1)))
    source, available = build_source(args)

    for symbol in args.symbols:
        if symbol.upper() not in available:
            print(f"\n  pulando {symbol}: sem mapeamento na fonte '{args.source}' "
                  f"(use a outra fonte para este ativo).")
            continue
        ingest_symbol(source, store, fx_calendar, symbol, args.start, end)

    print(f"\n✓ Ingestão externa concluída (fonte={args.source}).")


if __name__ == "__main__":
    main()
