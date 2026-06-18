"""Universo de ativos do INNOVA EA — escopo de fundo macro global.

Fonte única de verdade para os símbolos e suas classes, compartilhada pela
ingestão (``scripts/ingest_mt5.py``) e pelo treino (``scripts/train_models.py``).
Nomes de símbolo variam por corretora; aliases comuns indicados nos comentários.
"""
from __future__ import annotations

FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
METALS = ["XAUUSD", "XAGUSD"]                         # ouro, prata
US_INDICES = ["US30", "US100", "US500"]              # Dow, NASDAQ (≈NAS100), S&P 500
GLOBAL_INDICES = ["DE40", "JP225", "UK100", "HK50"]  # DAX(≈GER40), Nikkei, FTSE, Hang Seng

# EUR/USD obrigatoriamente primeiro.
DEFAULT_UNIVERSE = FOREX_MAJORS + METALS + US_INDICES + GLOBAL_INDICES

ASSET_CLASS: dict[str, str] = {
    **{s: "forex" for s in FOREX_MAJORS},
    **{s: "metal" for s in METALS},
    **{s: "index" for s in US_INDICES + GLOBAL_INDICES},
}


def asset_class_of(symbol: str) -> str:
    """Classe do ativo (forex/metal/index). Default 'forex' para desconhecidos."""
    return ASSET_CLASS.get(symbol.upper(), "forex")


def is_exchange_traded(symbol: str) -> bool:
    """True para índices (horário de bolsa) — usado p/ escolher o calendário de gaps."""
    return asset_class_of(symbol) == "index"
