"""Modelos de custo de transação — o que separa um backtest sério de fantasia.

Todos os custos são expressos de forma que o engine os aplique de modo
*pessimista* (a favor da corretora). Os defaults são conservadores e devem ser
calibrados com os fills reais do broker.
"""
from __future__ import annotations

from dataclasses import dataclass

from innova_ea.core.enums import Side
from innova_ea.core.instruments import Instrument


@dataclass(frozen=True, slots=True)
class CostModel:
    """Parâmetros de custo aplicados a cada operação.

    Attributes:
        spread_points: spread em points. ``None`` usa o típico do instrumento.
        slippage_points: slippage fixo adicional por execução (entrada e saída).
        commission_per_lot: comissão round-turn por lote (moeda de cotação).
            ``None`` usa a do instrumento.
        apply_swap: se True, cobra swap por noite carregada.
    """

    spread_points: float | None = None
    slippage_points: float = 2.0
    commission_per_lot: float | None = None
    apply_swap: bool = True

    def effective_spread(self, inst: Instrument) -> float:
        return inst.typical_spread_points if self.spread_points is None else self.spread_points

    def effective_commission(self, inst: Instrument) -> float:
        return (
            inst.commission_per_lot
            if self.commission_per_lot is None
            else self.commission_per_lot
        )

    def entry_fill_price(self, inst: Instrument, mid_price: float, side: Side) -> float:
        """Preço de execução de entrada, já penalizado por meio-spread + slippage.

        Compra paga acima do mid; venda recebe abaixo. O slippage sempre piora.
        """
        half_spread = inst.points_to_price(self.effective_spread(inst) / 2.0)
        slip = inst.points_to_price(self.slippage_points)
        return mid_price + side.value * (half_spread + slip)

    def exit_fill_price(self, inst: Instrument, mid_price: float, side: Side) -> float:
        """Preço de execução de saída. Fechar uma posição ``side`` negocia o lado oposto."""
        half_spread = inst.points_to_price(self.effective_spread(inst) / 2.0)
        slip = inst.points_to_price(self.slippage_points)
        return mid_price - side.value * (half_spread + slip)

    def commission_cost(self, inst: Instrument, lots: float) -> float:
        """Comissão round-turn (cobrada uma vez por trade completo)."""
        return self.effective_commission(inst) * lots

    def swap_cost(self, inst: Instrument, side: Side, lots: float, nights: int) -> float:
        """Custo de swap (em moeda de cotação) por ``nights`` noites carregadas.

        Swap negativo = custo. Retorna sempre como despesa positiva a subtrair.
        """
        if not self.apply_swap or nights <= 0 or side is Side.FLAT:
            return 0.0
        points = inst.swap_long_points if side is Side.LONG else inst.swap_short_points
        swap_price = inst.points_to_price(points)
        pnl = inst.pnl_quote(swap_price, lots) * nights
        return -pnl  # despesa positiva (swap costuma ser negativo → custo)
