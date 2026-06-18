"""Modelos universais. LightGBM é eager; o Super Cérebro (torch) é lazy."""
from innova_ea.research.models.base import UniversalModel
from innova_ea.research.models.gbm import LightGBMUniversal

__all__ = ["UniversalModel", "LightGBMUniversal", "get_super_brain"]


def get_super_brain(panel, **kwargs):
    """Importa e instancia o ``SuperBrain`` sob demanda (requer PyTorch).

    Mantido lazy para que ``import innova_ea.research`` funcione sem torch.
    """
    try:
        from innova_ea.research.models.super_brain import SuperBrain
    except ImportError as exc:  # pragma: no cover - depende de torch instalado
        raise RuntimeError(
            "Super Cérebro requer PyTorch. Instale com `pip install torch` "
            "(extra 'dl'). O baseline LightGBM não precisa de torch."
        ) from exc
    return SuperBrain(panel, **kwargs)
