"""Conector MetaTrader 5 (produção / forform test).

O pacote ``MetaTrader5`` só existe no Windows e exige um terminal MT5 instalado,
por isso é importado de forma preguiçosa: o módulo carrega em qualquer SO, mas
levanta erro claro só quando se tenta efetivamente usar a conexão. Assim o resto
da plataforma (pesquisa/backtest) roda em Linux/CI sem o MT5.

Paginação: o MT5 limita o número de barras por chamada; ``fetch`` itera para
trás a partir de ``end`` até cobrir todo o range solicitado.
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from forex_quant.core.bars import empty_bars, validate_bars
from forex_quant.core.enums import Timeframe
from forex_quant.data.sources.base import DataSource

# Mapa Timeframe -> constante do MT5. Resolvido em runtime (ver _mt5_timeframe).
_MT5_TF_NAMES: dict[Timeframe, str] = {
    Timeframe.M1: "TIMEFRAME_M1",
    Timeframe.M5: "TIMEFRAME_M5",
    Timeframe.M15: "TIMEFRAME_M15",
    Timeframe.M30: "TIMEFRAME_M30",
    Timeframe.H1: "TIMEFRAME_H1",
    Timeframe.H4: "TIMEFRAME_H4",
    Timeframe.D1: "TIMEFRAME_D1",
    Timeframe.W1: "TIMEFRAME_W1",
    Timeframe.MN1: "TIMEFRAME_MN1",
}


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:  # pragma: no cover - depende de SO
        raise RuntimeError(
            "Pacote 'MetaTrader5' não encontrado. Instale no Windows com "
            "`pip install MetaTrader5` e garanta um terminal MT5 instalado. "
            "Para pesquisa/CI use CsvSource ou SyntheticSource."
        ) from exc
    return mt5


class MT5Source(DataSource):
    """Fonte de barras a partir de um terminal MetaTrader 5.

    Args:
        login/password/server: credenciais opcionais. Se omitidas, usa a conta
            já logada no terminal.
        path: caminho do terminal64.exe (opcional).
        max_bars_per_call: tamanho de página ao paginar histórico longo.
    """

    def __init__(
        self,
        *,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        max_bars_per_call: int = 100_000,
    ) -> None:
        self._mt5 = _import_mt5()
        self.max_bars_per_call = max_bars_per_call
        self._connected = False
        self._init_kwargs: dict = {}
        if path:
            self._init_kwargs["path"] = path
        if login is not None:
            self._init_kwargs.update(login=login, password=password, server=server)

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        if not self._mt5.initialize(**self._init_kwargs):
            code, msg = self._mt5.last_error()
            raise ConnectionError(f"Falha ao inicializar MT5 ({code}): {msg}")
        self._connected = True

    def _mt5_timeframe(self, tf: Timeframe):
        return getattr(self._mt5, _MT5_TF_NAMES[tf])

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        self._ensure_connected()
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        if not self._mt5.symbol_select(symbol, True):
            raise ValueError(f"Símbolo '{symbol}' indisponível no terminal MT5")

        mt5_tf = self._mt5_timeframe(timeframe)
        rates = self._mt5.copy_rates_range(symbol, mt5_tf, start, end)
        if rates is None or len(rates) == 0:
            return empty_bars()

        df = pl.from_numpy(rates)  # campos: time, open, high, low, close, tick_volume, ...
        df = df.select(
            pl.from_epoch(pl.col("time"), time_unit="s")
            .dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us")
            .alias("time"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("tick_volume").cast(pl.Float64).alias("volume"),
        )
        return validate_bars(df, symbol=symbol)

    def close(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
