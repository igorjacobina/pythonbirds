#!/usr/bin/env python
"""Pipeline definitivo de ingestão MT5 → Parquet (INNOVA EA, Fase 1).

Extrai o histórico profundo (M1, desde 2015) das paridades majors a partir de um
terminal MetaTrader 5, aplica limpeza defensiva, converte o fuso do servidor para
UTC, grava na estrutura Parquet particionada, deriva os timeframes superiores e
emite relatórios de limpeza e de buracos (gaps).

Pré-requisitos (Windows): terminal MT5 instalado e logado, e ``pip install MetaTrader5``.

Credenciais por variáveis de ambiente (ou argumentos):
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH (opcional), MT5_SERVER_TZ

Exemplos:
    # EUR/USD primeiro (obrigatório), depois as demais majors:
    python scripts/ingest_mt5.py --start 2015-01-01

    # Apenas EUR/USD, fuso de servidor com offset fixo +2 (sem DST):
    python scripts/ingest_mt5.py --symbols EURUSD --server-utc-offset 2
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from innova_ea.core.enums import Timeframe
from innova_ea.data import (
    ForexCalendar,
    ParquetStore,
    analyze_gaps,
    clean_bars,
    common_forex_holidays,
    get_mt5_source,
    resample,
)

# --- Universo de ativos (escopo de fundo macro global) ---
# Nomes de símbolo variam por corretora; ajuste com --symbols se necessário.
# Aliases comuns: US100≈NAS100, DE40≈GER40, UK100≈FTSE100, HK50≈HSI.
FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
METALS = ["XAUUSD", "XAGUSD"]                        # ouro, prata
US_INDICES = ["US30", "US100", "US500"]             # Dow, NASDAQ, S&P 500
GLOBAL_INDICES = ["DE40", "JP225", "UK100", "HK50"]  # DAX, Nikkei, FTSE, Hang Seng

# EUR/USD obrigatoriamente primeiro.
DEFAULT_UNIVERSE = FOREX_MAJORS + METALS + US_INDICES + GLOBAL_INDICES

ASSET_CLASS: dict[str, str] = {
    **{s: "forex" for s in FOREX_MAJORS},
    **{s: "metal" for s in METALS},
    **{s: "index" for s in US_INDICES + GLOBAL_INDICES},
}

DERIVED_TFS = [Timeframe.M5, Timeframe.M15, Timeframe.M30,
               Timeframe.H1, Timeframe.H4, Timeframe.D1]


def asset_class_of(symbol: str) -> str:
    return ASSET_CLASS.get(symbol.upper(), "forex")


def calendar_for(symbol: str, fx_calendar: ForexCalendar) -> ForexCalendar | None:
    """Escolhe o calendário de gaps por classe de ativo.

    Forex e metais negociam ~24/5 → o calendário FX é uma boa referência.
    Índices seguem horários de BOLSA (com pausas, variando por país); aplicar o
    calendário FX geraria milhares de buracos falsos. Até termos calendários de
    bolsa dedicados, a análise por grade é pulada para índices (a limpeza segue).
    """
    return fx_calendar if asset_class_of(symbol) in ("forex", "metal") else None


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def ingest_symbol(source, store, calendar, symbol, start, end) -> None:
    """Ingere M1 ano a ano (memória limitada), deriva TFs e relata limpeza/gaps.

    ``calendar`` pode ser ``None`` (ex. índices) — nesse caso a análise de gaps
    por grade de mercado é pulada, mas a limpeza defensiva é sempre aplicada.
    """
    cls = asset_class_of(symbol)
    print(f"\n{'='*64}\n  {symbol} [{cls}] — M1 de {start.date()} a {end.date()}\n{'='*64}")

    total_raw = total_clean = 0
    cursor = start
    while cursor < end:
        year_end = min(datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc), end)
        raw = source.fetch(symbol, Timeframe.M1, cursor, year_end)
        if raw.height:
            clean, creport = clean_bars(raw)
            written = store.write(symbol, Timeframe.M1, clean)
            total_raw += creport.raw_rows
            total_clean += creport.clean_rows
            print(f"  {cursor.year}: {creport.clean_rows:>8,} barras "
                  f"(remov.: {creport.raw_rows - creport.clean_rows:,}) → gravadas {written:,}")
        else:
            print(f"  {cursor.year}: sem dados")
        cursor = year_end

    # Deriva timeframes superiores a partir do M1 completo já armazenado.
    full = store.read(symbol, Timeframe.M1, start, end)
    if full.height == 0:
        print("  ⚠ nenhum dado M1 — pulando derivação/gaps.")
        return
    for tf in DERIVED_TFS:
        n = store.write(symbol, tf, resample(full, tf))
        print(f"  derivado {tf.value}: {n:,} barras")

    # Relatório de limpeza + (quando aplicável) buracos contra o calendário.
    print(f"\n  -- Limpeza ({total_clean:,}/{total_raw:,} barras mantidas) --")
    if calendar is not None:
        gaps = analyze_gaps(full, calendar, Timeframe.M1, start, end)
        print(f"  -- Gaps (M1) --\n{_indent(gaps.summary())}")
    else:
        print("  -- Gaps (M1): análise por grade pulada (índice usa horário de "
              "bolsa; calendário dedicado é trabalho futuro). --")


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def main() -> None:
    p = argparse.ArgumentParser(description="Ingestão MT5 → Parquet (INNOVA EA)")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE,
                   help="ativos (EUR/USD primeiro). Default: majors + metais + "
                        "índices US e globais. Ajuste os nomes ao seu broker.")
    p.add_argument("--start", type=_parse_date, default=_parse_date("2015-01-01"))
    p.add_argument("--end", type=_parse_date, default=None,
                   help="default: agora (UTC).")
    p.add_argument("--store", default="./data", help="raiz do ParquetStore.")
    p.add_argument("--server-tz", default=os.getenv("MT5_SERVER_TZ", "Europe/Athens"),
                   help="fuso IANA do servidor (EET com DST por padrão).")
    p.add_argument("--server-utc-offset", type=float, default=None,
                   help="offset fixo em horas (use se o broker não tem DST).")
    args = p.parse_args()

    end = args.end or datetime.now(timezone.utc)
    store = ParquetStore(args.store)
    fx_calendar = ForexCalendar(holidays=common_forex_holidays(range(args.start.year, end.year + 1)))

    source = get_mt5_source(
        login=int(os.environ["MT5_LOGIN"]) if os.getenv("MT5_LOGIN") else None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
        server_timezone=None if args.server_utc_offset is not None else args.server_tz,
        server_utc_offset=args.server_utc_offset,
    )

    try:
        for symbol in args.symbols:
            cal = calendar_for(symbol, fx_calendar)
            ingest_symbol(source, store, cal, symbol, args.start, end)
        print(f"\n✓ Ingestão concluída ({len(args.symbols)} ativos). "
              "INNOVA EA está respirando dados reais.")
    finally:
        source.close()


if __name__ == "__main__":
    main()
