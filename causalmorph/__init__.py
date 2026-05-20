"""
CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models

A data preconditioning algorithm that projects observational datasets toward
the Linear Non-Gaussian Acyclic Model (LiNGAM) compatible regime.

Authors: Mario De Los Santos-Hernández, Samuel Montero-Hernández,
         Felipe Orihuela-Espina, L. Enrique Sucar
Paper: CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models
Journal: Knowledge-Based Systems (Under Review)
Manuscript ID: KNOSYS-D-25-17892
"""

__version__ = "0.1.0"
__author__ = "Mario De Los Santos-Hernández"
__email__ = "madlsh3517@gmail.com"

# Import main functions for easy access
from .core.causalmorph_algorithm import causalMorph
from .utils.metrics import normalized_shd, mycomparegraphs
from .utils.non_gaussian import check_non_gaussian
from .data_generation.synthetic_scenarios import causal_graph_synthetic_scenarios

__all__ = [
    "causalMorph",
    "normalized_shd",
    "mycomparegraphs",
    "check_non_gaussian",
    "causal_graph_synthetic_scenarios",
]
