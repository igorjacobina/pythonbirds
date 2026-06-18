"""Conciliação de posições — a verdade é sempre a posição REAL na corretora.

Antes de enviar qualquer ordem, o serviço lê a posição efetiva no MetaTrader e a
compara com o alvo do algoritmo. A ordem enviada é exatamente o DELTA que leva a
posição real ao alvo. Isso, por construção:

  * **elimina duplicidade**: nunca reenviamos uma ordem já refletida na posição;
  * **corrige órfãs**: se o algo acha que está flat mas há posição na corretora,
    o alvo (flat) gera um delta que a fecha;
  * **resolve dessincronia**: o estado interno nunca é assumido — é sempre
    reconciliado contra a corretora a cada barra.
"""
from __future__ import annotations

from dataclasses import dataclass

from innova_ea.core.instruments import Instrument


@dataclass(frozen=True, slots=True)
class ReconcileAction:
    symbol: str
    desired_lots: float
    actual_lots: float
    delta_lots: float    # volume a negociar (+ compra, - venda)
    kind: str            # none | open | increase | reduce | close | reverse

    @property
    def is_noop(self) -> bool:
        return self.kind == "none"

    @property
    def increases_exposure(self) -> bool:
        """True se a ação ABRE/AUMENTA exposição (sujeita a checagem de margem)."""
        return self.kind in ("open", "increase", "reverse")


def reconcile(
    symbol: str,
    desired_lots: float,
    actual_lots: float,
    inst: Instrument,
) -> ReconcileAction:
    """Calcula o delta para levar a posição real (``actual``) ao alvo (``desired``).

    Diferenças menores que meio passo de lote são tratadas como já sincronizadas
    (no-op) — evita micro-ordens e duplicidade por ruído de arredondamento.
    """
    delta = desired_lots - actual_lots
    tol = inst.lot_step / 2.0
    if abs(delta) < tol:
        return ReconcileAction(symbol, desired_lots, actual_lots, 0.0, "none")

    if actual_lots == 0.0:
        kind = "open"
    elif desired_lots == 0.0:
        kind = "close"
    elif (actual_lots > 0) != (desired_lots > 0):
        kind = "reverse"                       # inverte o sentido
    elif abs(desired_lots) > abs(actual_lots):
        kind = "increase"
    else:
        kind = "reduce"

    return ReconcileAction(symbol, desired_lots, actual_lots, delta, kind)
