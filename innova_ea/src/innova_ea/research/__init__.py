"""Camada L5 — pesquisa: rotulagem, padrões, dataset universal, modelos e overfitting."""
from innova_ea.research.dataset import (
    CLASS_NAMES,
    PerAssetScaler,
    UniversalPanel,
    build_universal_panel,
    label_to_class,
)
from innova_ea.research.labeling import forward_return, triple_barrier_labels
from innova_ea.research.metalabeling import (
    MetaLabelModel,
    MetaModelStrategy,
    build_metalabel_panel,
    donchian_breakout_primary,
    meta_labels,
    session_breakout_primary,
)
from innova_ea.research.models import LightGBMUniversal, UniversalModel, get_super_brain
from innova_ea.research.overfitting import (
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from innova_ea.research.patterns import (
    PatternStats,
    conditional_stats,
    scan_conditions,
)
from innova_ea.research.splitting import PurgedWalkForward, PurgedWindow

__all__ = [
    "forward_return",
    "triple_barrier_labels",
    "PatternStats",
    "conditional_stats",
    "scan_conditions",
    "UniversalPanel",
    "build_universal_panel",
    "PerAssetScaler",
    "label_to_class",
    "CLASS_NAMES",
    "PurgedWalkForward",
    "PurgedWindow",
    "UniversalModel",
    "LightGBMUniversal",
    "get_super_brain",
    "MetaLabelModel",
    "MetaModelStrategy",
    "build_metalabel_panel",
    "meta_labels",
    "donchian_breakout_primary",
    "session_breakout_primary",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]
