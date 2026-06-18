"""Fontes de dados concretas."""
from innova_ea.data.sources.base import DataSource
from innova_ea.data.sources.csv_source import CsvSource
from innova_ea.data.sources.dukascopy import DukascopySource
from innova_ea.data.sources.histdata import HistDataSource
from innova_ea.data.sources.synthetic import SyntheticSource

__all__ = [
    "DataSource", "CsvSource", "SyntheticSource",
    "DukascopySource", "HistDataSource", "get_mt5_source",
]


def get_mt5_source(**kwargs):
    """Importa e instancia o ``MT5Source`` sob demanda (Windows + terminal MT5).

    Mantido lazy para o pacote ``MetaTrader5`` (Windows-only) não ser exigido em
    Linux/CI durante import do subpacote.
    """
    from innova_ea.data.sources.mt5_source import MT5Source

    return MT5Source(**kwargs)

