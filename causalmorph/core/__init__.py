"""
CausalMorph Core Algorithm Module

Contains the main CausalMorph algorithm implementation including:
- Stage I: MDL-guided linearization
- Stage II: Non-Gaussian synthesis
- Stage III: Orthogonalization
"""

from .causalmorph_algorithm import causalMorph

__all__ = ["causalMorph"]
