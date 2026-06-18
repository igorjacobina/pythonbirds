"""Shim de JIT: usa Numba quando disponível, senão degrada para Python puro.

Mantém o engine funcional em ambientes sem Numba (alguns CIs), apenas mais
lento. Em produção/pesquisa o Numba acelera o hot loop em ~1-2 ordens de
grandeza.
"""
from __future__ import annotations

try:
    from numba import njit as _njit  # type: ignore

    HAS_NUMBA = True

    def njit(*args, **kwargs):
        # Defaults sensatos; cache acelera reexecuções.
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", True)
        if args and callable(args[0]) and not kwargs:
            return _njit(cache=True, fastmath=True)(args[0])
        return _njit(*args, **kwargs)

except ImportError:  # pragma: no cover - caminho de fallback
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco
