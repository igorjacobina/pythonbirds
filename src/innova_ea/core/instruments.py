"""Especificação de instrumentos (símbolos) — a fonte de verdade para custos e P&L.

Converter movimento de preço em dinheiro corretamente é a base de qualquer
backtest realista. Cada corretora tem suas especificações; os valores aqui são
defaults razoáveis para majors e devem ser substituídos pelos do broker real
(via ``MT5.symbol_info``) antes de operar capital.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Instrument:
    """Especificação de contrato de um símbolo Forex.

    Attributes:
        symbol: Identificador, ex. ``"EURUSD"``.
        digits: Casas decimais do preço (5 para majors, 3 para pares JPY).
        contract_size: Unidades da moeda base por 1.0 lote (padrão 100_000).
        quote_currency: Moeda em que o P&L é denominado (ex. ``"USD"``).
        commission_per_lot: Comissão round-turn por lote, na ``quote_currency``.
        swap_long_points: Swap diário em points para posições compradas.
        swap_short_points: Swap diário em points para posições vendidas.
        typical_spread_points: Spread típico em points (custo de transação base).
        base_currency: Moeda base do par (ex. ``"EUR"`` em EURUSD). ``None``
            assume os 3 primeiros caracteres do símbolo.
        min_lot / max_lot / lot_step: Limites de volume da corretora.
    """

    symbol: str
    digits: int = 5
    contract_size: float = 100_000.0
    quote_currency: str = "USD"
    commission_per_lot: float = 7.0
    swap_long_points: float = -3.0
    swap_short_points: float = -1.0
    typical_spread_points: float = 8.0
    base_currency: str | None = None
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01

    @property
    def base_ccy(self) -> str:
        """Moeda base resolvida (explícita ou inferida do símbolo)."""
        return (self.base_currency or self.symbol[:3]).upper()

    @property
    def point(self) -> float:
        """Menor incremento de preço. Para 5 dígitos = 0.00001."""
        return 10 ** (-self.digits)

    @property
    def pip(self) -> float:
        """Tamanho de 1 pip. Pares JPY (3 díg.) = 0.01; demais = 0.0001."""
        return 0.01 if self.digits in (2, 3) else 0.0001

    @property
    def points_per_pip(self) -> float:
        return self.pip / self.point

    def price_to_points(self, price_delta: float) -> float:
        """Converte uma diferença de preço em número de points."""
        return price_delta / self.point

    def points_to_price(self, points: float) -> float:
        return points * self.point

    def pnl_quote(self, price_delta: float, lots: float) -> float:
        """P&L (na moeda de cotação) de um movimento ``price_delta`` com ``lots`` lotes.

        Para um par X/USD, contract_size=100_000 e 1.0 lote: cada 0.0001 (pip)
        de movimento vale 10 USD. A fórmula geral é delta_preço * unidades.
        """
        return price_delta * self.contract_size * lots

    def round_lots(self, lots: float) -> float:
        """Arredonda o volume para o ``lot_step`` e aplica limites min/max."""
        stepped = round(lots / self.lot_step) * self.lot_step
        return max(self.min_lot, min(self.max_lot, round(stepped, 8)))

    def margin_requires_price(self, account_currency: str) -> bool:
        """Indica se a margem depende do preço de mercado (conversão base→conta).

        Regra MT5 (Forex): a margem é calculada na *moeda base* e convertida
        para a moeda da conta. Se a moeda base já é a da conta (ex. USDJPY com
        conta USD), não há conversão e a margem independe do preço. Caso
        contrário (ex. EURUSD com conta USD), multiplica-se pelo preço.
        """
        return self.base_ccy != account_currency.upper()

    def margin_required(
        self, lots: float, price: float, leverage: float, account_currency: str = "USD"
    ) -> float:
        """Margem exigida (na moeda da conta) para ``lots`` lotes — fórmula MT5.

        ``Margem = ContractSize * Lots / Leverage`` na moeda base, convertida
        para a moeda da conta. Cobre os casos majors com conta USD; cross-pairs
        cuja cotação não seja a moeda da conta usam ``price`` como aproximação
        de conversão (limitação documentada — requer taxa externa para exatidão).
        """
        base_margin = self.contract_size * abs(lots) / leverage
        if not self.margin_requires_price(account_currency):
            return base_margin
        return base_margin * price


# Catálogo default das principais majors. Substituir pelos valores do broker.
DEFAULT_INSTRUMENTS: dict[str, Instrument] = {
    "EURUSD": Instrument("EURUSD", digits=5, typical_spread_points=6.0),
    "GBPUSD": Instrument("GBPUSD", digits=5, typical_spread_points=9.0),
    "AUDUSD": Instrument("AUDUSD", digits=5, typical_spread_points=8.0),
    "USDJPY": Instrument("USDJPY", digits=3, quote_currency="JPY",
                         typical_spread_points=7.0,
                         swap_long_points=2.0, swap_short_points=-6.0),
    "USDCHF": Instrument("USDCHF", digits=5, quote_currency="CHF",
                         typical_spread_points=10.0),
    "USDCAD": Instrument("USDCAD", digits=5, quote_currency="CAD",
                         typical_spread_points=11.0),
}


def get_instrument(symbol: str) -> Instrument:
    """Retorna o instrumento do catálogo ou um default genérico de 5 dígitos."""
    return DEFAULT_INSTRUMENTS.get(symbol.upper(), Instrument(symbol.upper()))
