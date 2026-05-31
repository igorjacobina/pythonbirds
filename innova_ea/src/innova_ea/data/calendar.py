"""Calendário de negociação Forex — fronteiras de sessão semanal e feriados.

O mercado Forex opera ~24h de domingo à noite (abertura de Sydney) até sexta à
noite (fechamento de Nova York), em UTC. Fora disso não existem barras — e isso
NÃO é um buraco de dados. Este calendário permite distinguir gaps legítimos
(fim de semana/feriado) de buracos anômalos (falha de dados intra-sessão).

As horas são definidas em UTC. Como a maioria das corretoras usa EET (com horário
de verão), as fronteiras reais oscilam ~1h duas vezes por ano; o efeito é restrito
a alguns minutos de borda no domingo/sexta e está documentado. Para rigor máximo,
defina o calendário no fuso do servidor antes de converter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import polars as pl

from innova_ea.core.enums import Timeframe


@dataclass(frozen=True, slots=True)
class ForexCalendar:
    """Janela semanal de negociação (UTC) + feriados de mercado fechado.

    Attributes:
        week_open_hour: hora UTC de abertura no domingo (default 22:00).
        week_close_hour: hora UTC de fechamento na sexta (default 22:00).
        holidays: datas (UTC) de mercado totalmente fechado (ex. 25/12, 01/01).
    """

    week_open_hour: int = 22
    week_close_hour: int = 22
    holidays: frozenset[date] = field(default_factory=frozenset)

    def is_trading_expr(self) -> pl.Expr:
        """Expressão booleana Polars: True se a barra está em horário de mercado.

        Convenção Polars: ``dt.weekday()`` retorna 1=segunda … 7=domingo.
        """
        wd = pl.col("time").dt.weekday()
        hour = pl.col("time").dt.hour()
        trading = (
            (wd <= 4)  # segunda a quinta
            | ((wd == 5) & (hour < self.week_close_hour))   # sexta antes do fechamento
            | ((wd == 7) & (hour >= self.week_open_hour))   # domingo após a abertura
        )
        if self.holidays:
            holiday_list = list(self.holidays)
            trading = trading & ~pl.col("time").dt.date().is_in(holiday_list)
        return trading

    def is_trading(self, dt: datetime) -> bool:
        """Versão escalar (Python: weekday Mon=0 … Sun=6)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.date() in self.holidays:
            return False
        wd = dt.weekday()  # 0=segunda … 6=domingo
        if wd <= 3:
            return True
        if wd == 4:  # sexta
            return dt.hour < self.week_close_hour
        if wd == 6:  # domingo
            return dt.hour >= self.week_open_hour
        return False  # sábado

    def trading_grid(
        self, start: datetime, end: datetime, timeframe: Timeframe
    ) -> pl.DataFrame:
        """Grade COMPLETA de timestamps esperados em ``[start, end)`` (só horário de mercado).

        É a referência contra a qual se detectam buracos: tudo que está na grade e
        não está nos dados é um gap real (fins de semana/feriados já são excluídos).
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        full = pl.datetime_range(
            start,
            end,
            interval=f"{timeframe.minutes}m",
            time_zone="UTC",
            closed="left",
            eager=True,
        ).alias("time")
        return (
            pl.DataFrame(full)
            .filter(self.is_trading_expr())
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
        )


# Feriados globais de Forex mais comuns (mercado essencialmente fechado).
def common_forex_holidays(years: range) -> frozenset[date]:
    """Conjunto básico de feriados (Natal e Ano Novo) para os anos dados.

    Lista mínima e conservadora; amplie com o calendário do seu broker para
    rigor total (ex. feriados bancários de Londres/Nova York).
    """
    days: set[date] = set()
    for y in years:
        days.add(date(y, 1, 1))    # Ano Novo
        days.add(date(y, 12, 25))  # Natal
        days.add(date(y, 12, 26))  # Boxing Day
    return frozenset(days)
