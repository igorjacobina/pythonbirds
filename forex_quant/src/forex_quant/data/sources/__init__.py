"""Fontes de dados concretas."""
from forex_quant.data.sources.base import DataSource
from forex_quant.data.sources.csv_source import CsvSource
from forex_quant.data.sources.synthetic import SyntheticSource

__all__ = ["DataSource", "CsvSource", "SyntheticSource", "get_mt5_source"]


def get_mt5_source(**kwargs):
    """Importa e instancia o ``MT5Source`` sob demanda (Windows + terminal MT5).

    Mantido lazy para o pacote ``MetaTrader5`` (Windows-only) não ser exigido em
    Linux/CI durante import do subpacote.
    """
    from forex_quant.data.sources.mt5_source import MT5Source

    return MT5Source(**kwargs)
