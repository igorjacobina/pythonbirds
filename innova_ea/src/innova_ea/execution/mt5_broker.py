"""Adaptador MetaTrader 5 do ``Broker`` (produção, Windows).

Implementa a interface de execução sobre o terminal MT5: estado da conta,
posições (modo netting), barras fechadas (sem a barra em formação) e envio de
ordens a mercado. O pacote ``MetaTrader5`` é Windows-only e importado de forma
preguiçosa; um módulo falso pode ser injetado para testes.

Assume conta em modo NETTING (uma posição líquida por símbolo).
"""
from __future__ import annotations

import polars as pl

from innova_ea.core.bars import empty_bars, validate_bars
from innova_ea.core.enums import Timeframe
from innova_ea.core.instruments import Instrument
from innova_ea.data.sources.mt5_source import (
    _MT5_TF_NAMES,
    _import_mt5,
    _structured_to_polars,
)
from innova_ea.data.timeutils import server_epoch_to_utc
from innova_ea.execution.broker import (
    AccountState,
    Broker,
    BrokerError,
    OrderResult,
    PendingOrder,
    Position,
)


class MT5Broker(Broker):
    """Corretora ao vivo via terminal MetaTrader 5."""

    def __init__(
        self,
        *,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        server_timezone: str | None = "Europe/Athens",
        server_utc_offset: float | None = None,
        deviation_points: int = 20,
        mt5_module=None,
    ) -> None:
        self._mt5 = mt5_module if mt5_module is not None else _import_mt5()
        self.server_timezone = server_timezone
        self.server_utc_offset = server_utc_offset
        self.deviation_points = deviation_points
        self._connected = False
        self._init_kwargs: dict = {}
        if path:
            self._init_kwargs["path"] = path
        if login is not None:
            self._init_kwargs.update(login=int(login), password=password, server=server)

    # ----------------------------------------------------------- conexão
    def is_connected(self) -> bool:
        return self._connected and self._mt5.account_info() is not None

    def connect(self) -> None:
        if not self._mt5.initialize(**self._init_kwargs):
            code, msg = self._mt5.last_error()
            raise BrokerError(f"falha ao conectar MT5 ({code}): {msg}")
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False

    # ----------------------------------------------------------- metadados
    def instrument(self, symbol: str) -> Instrument:
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise BrokerError(f"símbolo '{symbol}' indisponível")
        return Instrument(
            symbol=symbol,
            digits=info.digits,
            contract_size=info.trade_contract_size,
            quote_currency=info.currency_profit,
            base_currency=info.currency_base,
            typical_spread_points=float(getattr(info, "spread", 0.0)),
            swap_long_points=float(getattr(info, "swap_long", 0.0)),
            swap_short_points=float(getattr(info, "swap_short", 0.0)),
            min_lot=info.volume_min,
            max_lot=info.volume_max,
            lot_step=info.volume_step,
        )

    # ----------------------------------------------------------- conta
    def account_state(self) -> AccountState:
        a = self._mt5.account_info()
        if a is None:
            raise BrokerError("account_info indisponível")
        margin_level = (a.margin_level / 100.0) if a.margin > 0 else float("inf")
        return AccountState(
            balance=a.balance, equity=a.equity, margin_used=a.margin,
            margin_free=a.margin_free, margin_level=margin_level,
        )

    # ----------------------------------------------------------- posições
    def _net(self, positions) -> Position | None:
        if not positions:
            return None
        BUY = self._mt5.ORDER_TYPE_BUY
        net = 0.0
        for p in positions:
            net += p.volume if p.type == BUY else -p.volume
        return Position(positions[0].symbol, net, avg_price=positions[0].price_open)

    def position(self, symbol: str) -> Position | None:
        return self._net(self._mt5.positions_get(symbol=symbol) or [])

    def open_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._mt5.positions_get() or []:
            cur = out.get(p.symbol)
            lots = p.volume if p.type == self._mt5.ORDER_TYPE_BUY else -p.volume
            out[p.symbol] = Position(p.symbol, (cur.lots if cur else 0.0) + lots, p.price_open)
        return {s: pos for s, pos in out.items() if pos.lots != 0.0}

    # ----------------------------------------------------------- barras
    def recent_closed_bars(self, symbol: str, timeframe: Timeframe, count: int) -> pl.DataFrame:
        tf = getattr(self._mt5, _MT5_TF_NAMES[timeframe])
        # start_pos=1 pula a barra EM FORMAÇÃO (índice 0); pegamos só as fechadas.
        rates = self._mt5.copy_rates_from_pos(symbol, tf, 1, count)
        if rates is None or len(rates) == 0:
            return empty_bars()
        df = _structured_to_polars(rates)
        utc = server_epoch_to_utc(
            df, epoch_col="time",
            server_timezone=self.server_timezone, server_utc_offset=self.server_utc_offset,
        )
        out = pl.DataFrame({
            "time": utc,
            "open": df["open"].cast(pl.Float64),
            "high": df["high"].cast(pl.Float64),
            "low": df["low"].cast(pl.Float64),
            "close": df["close"].cast(pl.Float64),
            "volume": df["tick_volume"].cast(pl.Float64),
        }).sort("time")
        return validate_bars(out, symbol=symbol)

    # ----------------------------------------------------------- ordens
    def send_market_order(self, symbol: str, delta_lots: float) -> OrderResult:
        if delta_lots == 0.0:
            return OrderResult(True, symbol, 0.0, message="no-op")
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerError(f"sem cotação para {symbol}")
        is_buy = delta_lots > 0
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": abs(delta_lots),
            "type": self._mt5.ORDER_TYPE_BUY if is_buy else self._mt5.ORDER_TYPE_SELL,
            "price": tick.ask if is_buy else tick.bid,
            "deviation": self.deviation_points,
            "type_filling": getattr(self._mt5, "ORDER_FILLING_IOC", 1),
            "comment": "innova_ea",
        }
        result = self._mt5.order_send(request)
        ok = result is not None and result.retcode == self._mt5.TRADE_RETCODE_DONE
        price = getattr(result, "price", 0.0) if result else 0.0
        msg = "" if ok else f"retcode={getattr(result, 'retcode', '?')}"
        return OrderResult(ok, symbol, delta_lots, price, msg)

    # ----------------------------------------------------- ordens pendentes (straddle)
    def place_stop(self, symbol, side, price, lots, sl, tp) -> int:
        """Coloca um stop pendente (side +1 buy-stop, -1 sell-stop) com SL/TP."""
        otype = self._mt5.ORDER_TYPE_BUY_STOP if side > 0 else self._mt5.ORDER_TYPE_SELL_STOP
        request = {
            "action": self._mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": abs(lots),
            "type": otype,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation_points,
            "type_filling": getattr(self._mt5, "ORDER_FILLING_RETURN", 2),
            "comment": "innova_ea_straddle",
        }
        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise BrokerError(f"falha ao colocar stop {symbol}: "
                              f"retcode={getattr(result, 'retcode', '?')}")
        return int(result.order)

    def cancel_order(self, ticket: int) -> None:
        request = {"action": self._mt5.TRADE_ACTION_REMOVE, "order": int(ticket)}
        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise BrokerError(f"falha ao cancelar ordem {ticket}")

    def pending_orders(self, symbol: str) -> list[PendingOrder]:
        orders = self._mt5.orders_get(symbol=symbol) or []
        buy_stop = self._mt5.ORDER_TYPE_BUY_STOP
        out = []
        for o in orders:
            side = 1 if o.type == buy_stop else -1
            out.append(PendingOrder(
                ticket=int(o.ticket), symbol=o.symbol, side=side,
                price=o.price_open, lots=o.volume_current,
                sl=getattr(o, "sl", 0.0), tp=getattr(o, "tp", 0.0),
            ))
        return out

