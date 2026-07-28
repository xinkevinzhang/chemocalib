# ChemoCalib Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
tutorial
api
theory
```

## Project Overview

ChemoCalib bridges **chemometrics** (Multi-Block PLS / DIABLO) and
**constraint-based metabolic modeling** (COBRApy FBA) through a
**latent-variable-to-reaction-bound** mapping, closed by an
**active learning** loop for intelligent virtual experiment design.

```{toctree}
:hidden:
:caption: Links

GitHub Repository <https://github.com/chemocalib/chemocalib>
Issue Tracker <https://github.com/chemocalib/chemocalib/issues>
```

## Citation

```bibtex
@software{chemocalib2026,
  title = {{ChemoCalib}: Chemometrics-Calibrated Constraint-Based Metabolic Modeling},
  author = {Zhang, Xin},
  year = {2026},
  url = {https://github.com/chemocalib/chemocalib},
  organization = {Department of Chemistry, Capital Normal University},
}
```

## Package Layout

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   chemocalib.models
   chemocalib.gem
   chemocalib.active_learning
   chemocalib.virtual_experiment
   chemocalib.dynamic_layer
   chemocalib.stats
   chemocalib.validation
   chemocalib.data
```
