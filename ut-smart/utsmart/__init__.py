"""UT-SMART: Uncertainty-aware Twin-assisted Semantic Multi-Agent edge intelligence for connected AUVs."""
from .config import LABELS, MISSIONS, VARIANTS, SimConfig, Variant
from .env import UnderwaterEnv

__version__ = "1.0.0"
__all__ = ["SimConfig", "Variant", "VARIANTS", "LABELS", "MISSIONS", "UnderwaterEnv", "__version__"]
