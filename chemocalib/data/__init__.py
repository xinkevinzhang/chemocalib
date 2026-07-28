"""
Data loading utilities for real multi-omics datasets.

Supports:
  - E. coli multi-carbon-source experiments
  - S. cerevisiae nitrogen-limitation datasets
  - Generic CSV/HDF5 loaders with validation
"""

from .loader import (
    generate_realistic_e_coli_data,
    generate_realistic_ecoli_ko_data,
    generate_realistic_yeast_stress_data,
    validate_multiblock_coherence,
    load_from_csv,
)

from .fluxome import (
    load_keio_fluxome,
    load_holm_fluxome,
    load_ecoli_combined_fluxome,
    load_yeast_branching_ratios,
    REACTION_NAMES,
    N_REACTIONS,
)

__all__ = [
    "generate_realistic_e_coli_data",
    "generate_realistic_ecoli_ko_data",
    "generate_realistic_yeast_stress_data",
    "validate_multiblock_coherence",
    "load_from_csv",
    "load_keio_fluxome",
    "load_holm_fluxome",
    "load_ecoli_combined_fluxome",
    "load_yeast_branching_ratios",
    "REACTION_NAMES",
    "N_REACTIONS",
]
