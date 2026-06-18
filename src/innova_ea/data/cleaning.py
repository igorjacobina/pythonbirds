"""Limpeza defensiva de barras e análise de buracos (gaps) contra o calendário.

Princípios institucionais:
  * **Nunca fabricar preço.** Forward-fill de candles inexistentes cria dados
    falsos e contamina o backtest. Aqui buracos são DETECTADOS e REPORTADOS,
    não preenchidos com preço inventado.
  * **Distinguir gap legítimo de anômalo.** Fins de semana e feriados não são
    buracos. A grade de negociação (``ForexCalendar``) é a referência.
  * **Detectar fuso errado.** Barras fora do horário de mercado denunciam
    conversão de timezone incorreta — um alerta explícito é emitido.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from innova_ea.core.bars import validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.data.calendar import ForexCalendar


@dataclass(frozen=True, slots=True)
class CleaningReport:
    raw_rows: int
    clean_rows: int
    duplicates_removed: int
    nulls_removed: int
    nonpositive_removed: int
    ohlc_violations_removed: int

    def summary(self) -> str:
        return (
            f"Barras recebidas   : {self.raw_rows:,}\n"
            f"Barras limpas      : {self.clean_rows:,}\n"
            f"  duplicatas       : {self.duplicates_removed:,}\n"
            f"  nulos            : {self.nulls_removed:,}\n"
            f"  preço <= 0       : {self.nonpositive_removed:,}\n"
            f"  OHLC inconsistente: {self.ohlc_violations_removed:,}"
        )


@dataclass(frozen=True, slots=True)
class GapRun:
    start: object   # datetime UTC (primeira barra faltante)
    end: object     # datetime UTC (última barra faltante)
    missing: int    # nº de barras ausentes no intervalo


@dataclass(frozen=True, slots=True)
class GapReport:
    expected_bars: int
    present_bars: int
    missing_bars: int
    coverage: float
    n_gap_runs: int
    bars_outside_calendar: int   # barras em horário não-negociável (alerta de fuso)
    largest_gap: GapRun | None
    top_gaps: list[GapRun] = field(default_factory=list)

    def summary(self) -> str:
        lg = (
            f"{self.largest_gap.missing:,} barras "
            f"({self.largest_gap.start} → {self.largest_gap.end})"
            if self.largest_gap else "—"
        )
        warn = (
            f"\n  ⚠ {self.bars_outside_calendar:,} barras FORA do horário de "
            f"mercado — verifique a conversão de timezone do servidor!"
            if self.bars_outside_calendar else ""
        )
        return (
            f"Cobertura          : {self.coverage:.4%} "
            f"({self.present_bars:,}/{self.expected_bars:,} barras esperadas)\n"
            f"Barras ausentes    : {self.missing_bars:,} em {self.n_gap_runs:,} buracos\n"
            f"Maior buraco       : {lg}{warn}"
        )


def clean_bars(df: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
    """Limpa barras brutas de forma defensiva e devolve (barras_limpas, relatório).

    Remove nulos, preços não-positivos, OHLC inconsistente e timestamps
    duplicados (mantendo a última leitura), e ordena cronologicamente.
    """
    raw = df.height

    nonnull = df.drop_nulls(subset=["time", "open", "high", "low", "close"])
    nulls_removed = raw - nonnull.height

    positive = nonnull.filter(
        (pl.col("open") > 0) & (pl.col("high") > 0)
        & (pl.col("low") > 0) & (pl.col("close") > 0)
    )
    nonpositive_removed = nonnull.height - positive.height

    coherent = positive.filter(
        (pl.col("high") >= pl.col("low"))
        & (pl.col("high") >= pl.col("open")) & (pl.col("high") >= pl.col("close"))
        & (pl.col("low") <= pl.col("open")) & (pl.col("low") <= pl.col("close"))
    )
    ohlc_removed = positive.height - coherent.height

    deduped = coherent.sort("time").unique(subset=["time"], keep="last").sort("time")
    duplicates_removed = coherent.height - deduped.height

    clean = validate_bars(deduped)
    report = CleaningReport(
        raw_rows=raw,
        clean_rows=clean.height,
        duplicates_removed=duplicates_removed,
        nulls_removed=nulls_removed,
        nonpositive_removed=nonpositive_removed,
        ohlc_violations_removed=ohlc_removed,
    )
    return clean, report


def analyze_gaps(
    bars: pl.DataFrame,
    calendar: ForexCalendar,
    timeframe: Timeframe,
    start,
    end,
    *,
    top_n: int = 10,
) -> GapReport:
    """Compara as barras com a grade de mercado e relata os buracos reais.

    Tudo que está na grade de negociação e ausente nos dados é um buraco;
    fins de semana/feriados, por construção, não entram na grade. Barras
    presentes fora da grade indicam provável erro de timezone.
    """
    grid = calendar.trading_grid(start, end, timeframe)
    present = bars.select("time").unique()

    missing = grid.join(present, on="time", how="anti").sort("time")
    outside = present.join(grid, on="time", how="anti")

    expected = grid.height
    present_n = grid.height - missing.height  # presentes que caem na grade
    n_missing = missing.height

    runs: list[GapRun] = []
    if n_missing:
        step_us = timeframe.minutes * 60 * 1_000_000
        # Novo buraco quando o salto entre faltantes consecutivos excede 1 passo.
        marked = missing.with_columns(
            (pl.col("time").diff().dt.total_microseconds() > step_us)
            .fill_null(True)
            .cum_sum()
            .alias("_run")
        )
        agg = marked.group_by("_run").agg(
            pl.col("time").min().alias("start"),
            pl.col("time").max().alias("end"),
            pl.len().alias("missing"),
        ).sort("missing", descending=True)
        for row in agg.head(top_n).iter_rows(named=True):
            runs.append(GapRun(start=row["start"], end=row["end"], missing=row["missing"]))
        n_runs = agg.height
        largest = runs[0] if runs else None
    else:
        n_runs = 0
        largest = None

    coverage = (present_n / expected) if expected else 1.0
    return GapReport(
        expected_bars=expected,
        present_bars=present_n,
        missing_bars=n_missing,
        coverage=coverage,
        n_gap_runs=n_runs,
        bars_outside_calendar=outside.height,
        largest_gap=largest,
        top_gaps=runs,
    )
