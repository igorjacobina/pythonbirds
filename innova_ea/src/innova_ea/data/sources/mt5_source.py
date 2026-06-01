"""Conector MetaTrader 5 — inicialização, login, fuso do servidor e histórico profundo.

O pacote ``MetaTrader5`` só existe no Windows e exige um terminal MT5 instalado;
é importado de forma preguiçosa para que o resto da plataforma rode em Linux/CI.
Para testes, um módulo MT5 falso pode ser injetado via ``mt5_module``.

Pontos críticos tratados aqui:
  * **Login/inicialização robustos** com tentativas e mensagens claras.
  * **Fuso do servidor → UTC**: o ``time`` do MT5 é o relógio do servidor (EET
    com DST). A conversão correta é delegada a ``timeutils.server_epoch_to_utc``.
  * **Histórico profundo em chunks**: requisições longas (M1 desde 2015 ≈ 3,7M
    barras) são fatiadas por mês para respeitar limites do terminal e dar
    resiliência; o ``ParquetStore`` deduplica na gravação.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from innova_ea.core.bars import empty_bars, validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.base import DataSource
from innova_ea.data.timeutils import server_epoch_to_utc

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


def _add_months(dt: datetime, months: int) -> datetime:
    m = dt.month - 1 + months
    year = dt.year + m // 12
    month = m % 12 + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def _structured_to_polars(rates) -> pl.DataFrame:
    """Converte o array numpy ESTRUTURADO do MT5 em DataFrame Polars.

    ``mt5.copy_rates_*`` devolve um numpy array estruturado (campos nomeados:
    time, open, high, low, close, tick_volume, spread, real_volume). As versões
    recentes do Polars NÃO aceitam isso em ``pl.from_numpy`` (gera ``AsSliceError``
    / PanicException). Convertemos campo a campo para um dict — robusto em
    qualquer versão e sem dependência de pandas.
    """
    names = rates.dtype.names
    if names is None:  # array não-estruturado (fallback improvável)
        return pl.from_numpy(rates)
    # ``rates[name]`` pode ser uma view não contígua; ``np.ascontiguousarray``
    # garante um buffer que o Polars consome com segurança.
    return pl.DataFrame({name: np.ascontiguousarray(rates[name]) for name in names})


class MT5Source(DataSource):
    """Fonte de barras a partir de um terminal MetaTrader 5.

    Args:
        login/password/server: credenciais. Se omitidas, usa a conta já logada.
        path: caminho do ``terminal64.exe`` (opcional).
        server_timezone: fuso IANA do servidor (default ``"Europe/Athens"`` = EET).
        server_utc_offset: offset fixo em horas (use se o broker não tem DST).
        chunk_months: tamanho do chunk de histórico, em meses.
        max_retries / retry_wait: política de reconexão na inicialização.
        mt5_module: injeção do módulo MT5 (para testes); ``None`` importa o real.
    """

    def __init__(
        self,
        *,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        server_timezone: str | None = "Europe/Athens",
        server_utc_offset: float | None = None,
        chunk_months: int = 1,
        max_retries: int = 3,
        retry_wait: float = 2.0,
        mt5_module=None,
    ) -> None:
        self._mt5 = mt5_module if mt5_module is not None else _import_mt5()
        self.server_timezone = server_timezone
        self.server_utc_offset = server_utc_offset
        self.chunk_months = max(1, chunk_months)
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self._connected = False
        self._init_kwargs: dict = {}
        if path:
            self._init_kwargs["path"] = path
        if login is not None:
            self._init_kwargs.update(login=int(login), password=password, server=server)

    # ------------------------------------------------------------------ conexão
    def _ensure_connected(self) -> None:
        if self._connected:
            return
        import time

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            if self._mt5.initialize(**self._init_kwargs):
                self._connected = True
                return
            last_err = self._mt5.last_error()
            if attempt < self.max_retries:
                time.sleep(self.retry_wait * attempt)
        code, msg = last_err if last_err else (-1, "desconhecido")
        raise ConnectionError(
            f"Falha ao inicializar/login no MT5 após {self.max_retries} tentativas "
            f"({code}): {msg}"
        )

    def _mt5_timeframe(self, tf: Timeframe):
        return getattr(self._mt5, _MT5_TF_NAMES[tf])

    # ------------------------------------------------------------------ fetch
    def _rates_to_bars(self, rates, symbol: str) -> pl.DataFrame:
        if rates is None or len(rates) == 0:
            return empty_bars()
        df = _structured_to_polars(rates)  # campos: time, open, high, low, close, tick_volume...
        utc_time = server_epoch_to_utc(
            df,
            epoch_col="time",
            server_timezone=self.server_timezone,
            server_utc_offset=self.server_utc_offset,
        )
        out = pl.DataFrame(
            {
                "time": utc_time,
                "open": df["open"].cast(pl.Float64),
                "high": df["high"].cast(pl.Float64),
                "low": df["low"].cast(pl.Float64),
                "close": df["close"].cast(pl.Float64),
                "volume": df["tick_volume"].cast(pl.Float64),
            }
        ).sort("time")
        return out

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
        # Margem de segurança no pedido (servidor pode estar deslocado do UTC);
        # filtramos com precisão depois de converter para UTC.
        pad = timedelta(days=2)
        frames: list[pl.DataFrame] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(_add_months(chunk_start, self.chunk_months), end)
            rates = self._mt5.copy_rates_range(
                symbol, mt5_tf, chunk_start - pad, chunk_end + pad
            )
            bars = self._rates_to_bars(rates, symbol)
            if bars.height:
                frames.append(bars)
            chunk_start = chunk_end

        if not frames:
            return empty_bars()

        combined = (
            pl.concat(frames)
            .filter((pl.col("time") >= start) & (pl.col("time") < end))
            .unique(subset=["time"], keep="last")
            .sort("time")
        )
        return validate_bars(combined, symbol=symbol)

    def close(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
