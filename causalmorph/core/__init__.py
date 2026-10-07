"""
CausalMorph Core Algorithm Module

Contains the main CausalMorph algorithm implementation including:
- Stage I: MDL-guided linearization
- Stage II: Non-Gaussian synthesis
- Stage III: Orthogonalization

causalmorph_v2: same stages, Stage II gated, sample-wise and deterministic.
"""

from .causalmorph_algorithm import causalMorph
from .causalmorph_v2 import causalmorph_v2

__all__ = ["causalMorph", "causalmorph_v2"]
