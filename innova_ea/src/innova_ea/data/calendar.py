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
    """Janela semanal de negociação (DST-aware) + feriados de mercado fechado.

    A semana do Forex é definida pelo horário de **Nova York** (abre domingo
    17:00 NY, fecha sexta 17:00 NY). Como NY tem horário de verão, em UTC a
    fronteira oscila entre 21:00 (EDT) e 22:00 (EST) — usar 17:00 no fuso de
    mercado captura isso exatamente, evitando ~1h de borda mal classificada por
    fim de semana (que, com fronteira UTC fixa, gerava milhares de falsos
    "barras fora do horário").

    Attributes:
        market_timezone: fuso de referência do mercado (default America/New_York).
        open_hour: hora (no fuso de mercado) de abertura no domingo (default 17).
        close_hour: hora (no fuso de mercado) de fechamento na sexta (default 17).
        holidays: datas (UTC) de mercado totalmente fechado (ex. 25/12, 01/01).
    """

    market_timezone: str = "America/New_York"
    open_hour: int = 17
    close_hour: int = 17
    holidays: frozenset[date] = field(default_factory=frozenset)

    def is_trading_expr(self) -> pl.Expr:
        """Expressão booleana Polars: True se a barra está em horário de mercado.

        Converte o instante (UTC) para o fuso de mercado e aplica a janela
        semanal. ``dt.weekday()`` (Polars) retorna 1=segunda … 7=domingo.
        """
        local = pl.col("time").dt.convert_time_zone(self.market_timezone)
        wd = local.dt.weekday()
        hour = local.dt.hour()
        trading = (
            (wd <= 4)  # segunda a quinta (no fuso de mercado)
            | ((wd == 5) & (hour < self.close_hour))   # sexta antes do fechamento
            | ((wd == 7) & (hour >= self.open_hour))   # domingo após a abertura
        )
        if self.holidays:
            holiday_list = list(self.holidays)
            trading = trading & ~pl.col("time").dt.date().is_in(holiday_list)
        return trading

    def is_trading(self, dt: datetime) -> bool:
        """Versão escalar. Holiday é avaliado pela data UTC (igual à expressão)."""
        from zoneinfo import ZoneInfo

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.date() in self.holidays:
            return False
        local = dt.astimezone(ZoneInfo(self.market_timezone))
        wd = local.weekday()  # 0=segunda … 6=domingo
        if wd <= 3:
            return True
        if wd == 4:  # sexta
            return local.hour < self.close_hour
        if wd == 6:  # domingo
            return local.hour >= self.open_hour
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
