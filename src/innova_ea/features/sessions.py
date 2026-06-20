"""Features de sessão — a dinâmica nos horários de maior injeção de liquidez.

As maiores distorções e impulsos do Forex acontecem nas aberturas de sessão
(Tóquio, Londres, Nova York) e nas *killzones*. Este módulo mede, de forma
causal e relativa à abertura da sessão de cada barra:

  * **thrust de abertura** — deslocamento desde a abertura (impulso/distorção);
  * **opening range (OR)** e seu **rompimento** (breakout institucional);
  * **expansão de volatilidade** — range acumulado na sessão vs. range
    pré-sessão (mede a injeção de liquidez/exaustão).

Horários são em **UTC** e ignoram horário de verão (DST) — uma aproximação a ser
refinada com o calendário do broker antes de operar. Sessões que cruzam a
meia-noite (ex. Sydney) são suportadas no flag de atividade.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from innova_ea.features.base import Feature


@dataclass(frozen=True, slots=True)
class Session:
    """Janela de sessão em horas UTC (``close_hour`` exclusivo)."""

    name: str
    open_hour: int
    close_hour: int

    def is_active_expr(self) -> pl.Expr:
        h = pl.col("time").dt.hour()
        if self.open_hour < self.close_hour:
            return (h >= self.open_hour) & (h < self.close_hour)
        # Sessão que cruza a meia-noite (ex. 21h–6h).
        return (h >= self.open_hour) | (h < self.close_hour)


# Sessões e killzones (UTC, aproximadas, sem DST).
TOKYO = Session("tokyo", 0, 9)
LONDON = Session("london", 7, 16)
NEW_YORK = Session("newyork", 12, 21)
SYDNEY = Session("sydney", 21, 6)
LONDON_KILLZONE = Session("london_kz", 7, 10)
NEWYORK_KILLZONE = Session("newyork_kz", 12, 15)

PREDEFINED: dict[str, Session] = {
    s.name: s for s in (TOKYO, LONDON, NEW_YORK, SYDNEY, LONDON_KILLZONE, NEWYORK_KILLZONE)
}


class SessionOpenFeature(Feature):
    """Bloco de features causais relativas à abertura de uma ``Session``.

    Produz, com o prefixo do nome da sessão:
        {p}_active         — 1 se a barra está dentro da sessão, senão 0
        {p}_bars_since_open— nº de barras desde a abertura da sessão (null fora)
        {p}_thrust_ret     — log-retorno desde a abertura (impulso direcional)
        {p}_thrust_norm    — deslocamento desde a abertura / range pré-sessão
        {p}_or_breakout    — rompimento do opening range: +1/-1/0 (null até fechar)
        {p}_vol_expansion  — range acumulado na sessão / range pré-sessão

    Args:
        session: a sessão de referência.
        opening_range_bars: nº de barras que formam o opening range.
        pre_window: nº de barras antes da abertura usado como baseline de range.
    """

    def __init__(
        self,
        session: Session,
        *,
        opening_range_bars: int = 4,
        pre_window: int = 12,
    ) -> None:
        if opening_range_bars < 1 or pre_window < 1:
            raise ValueError("opening_range_bars e pre_window devem ser >= 1")
        self.session = session
        self.opening_range_bars = opening_range_bars
        self.pre_window = pre_window
        p = session.name
        self._names = [
            f"{p}_active",
            f"{p}_bars_since_open",
            f"{p}_thrust_ret",
            f"{p}_thrust_norm",
            f"{p}_or_breakout",
            f"{p}_vol_expansion",
        ]

    @property
    def names(self) -> list[str]:
        return self._names

    def transform(self, bars: pl.DataFrame) -> pl.DataFrame:
        active = self.session.is_active_expr()
        df = (
            bars.with_row_index("_idx")
            .with_columns(active.alias("_active"))
            .with_columns(
                (pl.col("_active") & ~pl.col("_active").shift(1, fill_value=False)).alias("_start")
            )
            .with_columns(pl.col("_start").cum_sum().alias("_sid_raw"))
            .with_columns(
                pl.when(pl.col("_active")).then(pl.col("_sid_raw")).otherwise(None).alias("_sid")
            )
            .with_columns(
                (
                    pl.col("high").rolling_max(self.pre_window)
                    - pl.col("low").rolling_min(self.pre_window)
                ).alias("_pre_range")
            )
            .with_columns(
                pl.col("_idx").first().over("_sid").alias("_first_idx"),
                pl.col("open").first().over("_sid").alias("_sopen"),
                pl.col("_pre_range").first().over("_sid").alias("_pre0"),
                pl.col("high").cum_max().over("_sid").alias("_run_high"),
                pl.col("low").cum_min().over("_sid").alias("_run_low"),
            )
            .with_columns((pl.col("_idx") - pl.col("_first_idx")).alias("_bso"))
        )

        within_or = pl.col("_bso") < self.opening_range_bars
        df = df.with_columns(
            pl.when(within_or).then(pl.col("high")).otherwise(None).max().over("_sid").alias("_or_high"),
            pl.when(within_or).then(pl.col("low")).otherwise(None).min().over("_sid").alias("_or_low"),
        )

        act = pl.col("_active")
        or_done = pl.col("_bso") >= self.opening_range_bars
        has_pre = pl.col("_pre0") > 0

        thrust_ret = pl.when(act).then((pl.col("close") / pl.col("_sopen")).log()).otherwise(None)
        thrust_norm = (
            pl.when(act & has_pre)
            .then((pl.col("close") - pl.col("_sopen")) / pl.col("_pre0"))
            .otherwise(None)
        )
        vol_exp = (
            pl.when(act & has_pre)
            .then((pl.col("_run_high") - pl.col("_run_low")) / pl.col("_pre0"))
            .otherwise(None)
        )
        breakout = (
            pl.when(act & or_done & (pl.col("close") > pl.col("_or_high"))).then(1)
            .when(act & or_done & (pl.col("close") < pl.col("_or_low"))).then(-1)
            .when(act & or_done).then(0)
            .otherwise(None)
        )
        bso_out = pl.when(act).then(pl.col("_bso")).otherwise(None)

        out = df.select(
            act.cast(pl.Int8).alias(self._names[0]),
            bso_out.cast(pl.Int32).alias(self._names[1]),
            thrust_ret.alias(self._names[2]),
            thrust_norm.alias(self._names[3]),
            breakout.cast(pl.Int8).alias(self._names[4]),
            vol_exp.alias(self._names[5]),
        )
        return self._check(out, bars.height)
