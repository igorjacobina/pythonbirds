"""Camada L2 — pipeline de dados (fontes, resample, storage)."""
from innova_ea.data.pipeline import DataPipeline, IngestionReport
from innova_ea.data.resample import resample
from innova_ea.data.sources import CsvSource, DataSource, SyntheticSource, get_mt5_source
from innova_ea.data.storage import ParquetStore

__all__ = [
    "DataPipeline",
    "IngestionReport",
    "resample",
    "ParquetStore",
    "DataSource",
    "CsvSource",
    "SyntheticSource",
    "get_mt5_source",
]
