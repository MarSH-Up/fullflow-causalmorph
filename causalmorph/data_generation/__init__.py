"""
CausalMorph Data Generation Module

Contains functions for generating synthetic causal datasets with:
- Linear and nonlinear causal relationships
- Various noise distributions
- Configurable graph structures
"""

from .synthetic_scenarios import causal_graph_synthetic_scenarios

__all__ = ["causal_graph_synthetic_scenarios"]
