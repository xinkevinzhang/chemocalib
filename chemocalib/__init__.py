"""
ChemoCalib v0.2.0
==================
Chemometrics-Calibrated Constraint-Based Metabolic Modeling.

"Quantitatively calibrating metabolic constraints across multi-omics
using chemometric latent variable models."
  — One-sentence research identity

Core pipeline:
  Multi-block PLS (MB-PLS/DIABLO)
    → Multi-constraint GEM (COBRApy FBA)
    → Active learning query selection
    → Virtual-cell "sense-predict-query" loop

Modules:
  models/              - MultiBlockPLS, MultiBlockAligner
  gem/                 - LatentToConstraint, FBASimulator, FVAAnalyzer
  active_learning/     - UncertaintySampler, ExperimentDesigner
  virtual_experiment/  - DoubleKnockoutDesigner, SurrogateModel
  dynamic_layer/       - GlycolysisODE (glycolysis dynamics)
  stats/               - Permutation test, bootstrap CIs
  validation/          - BenchmarkRunner (E-Flux, MADE, GECKO)
  data/                - Realistic multi-omics data loaders
  cross_validation.py  - Grid-search CV, stability selection
"""

__version__ = "0.2.0"
__author__ = "ChemoCalib Authors"

from chemocalib.models.mbpls import MultiBlockPLS
from chemocalib.gem.constraints import LatentToConstraint
from chemocalib.gem.fba import FBASimulator
from chemocalib.gem.fva import FVAAnalyzer
from chemocalib.active_learning.uncertainty import UncertaintySampler
from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner
from chemocalib.virtual_experiment.surrogate import SurrogateModel
from chemocalib.dynamic_layer.ode_solver import GlycolysisODE
from chemocalib.stats import permutation_test, bootstrap_ci
from chemocalib.cross_validation import grid_search_components
from chemocalib.validation.benchmark import BenchmarkRunner

__all__ = [
    "MultiBlockPLS",
    "LatentToConstraint",
    "FBASimulator",
    "FVAAnalyzer",
    "UncertaintySampler",
    "DoubleKnockoutDesigner",
    "SurrogateModel",
    "GlycolysisODE",
    "permutation_test",
    "bootstrap_ci",
    "grid_search_components",
    "BenchmarkRunner",
]
