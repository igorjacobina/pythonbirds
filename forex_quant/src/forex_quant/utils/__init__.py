"""Utilitários transversais."""
from forex_quant.utils.jit import HAS_NUMBA, njit

__all__ = ["njit", "HAS_NUMBA"]
