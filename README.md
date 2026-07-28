
# ChemoCalib

**Chemometrics-Calibrated Constraint-Based Metabolic Modeling via Multi-Block PLS**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)

ChemoCalib closes the **calibration gap** between multi-omics data and
genome-scale metabolic models (GEMs). It uses **Multi-Block Partial Least Squares (MB-PLS)**
to discover cross-omics latent structure, then propagates this via
**GPR-aware Variable Importance in Projection (GPR-VIP)** into reaction-bound
constraints for constrained flux balance analysis (FBA).

## Key Features

- **MB-PLS decomposition** — extracts latent components from transcriptome + metabolome + fluxome blocks
- **GPR-VIP scoring** — maps gene-level importance to reaction-level constraints via GPR rules
- **Constrained FBA** — applies statistically-derived bounds to iJO1366 (or any COBRApy-compatible model)
- **Benchmark harness** — reproducible comparison against pFBA, E-Flux, E-Flux2, MOMENT, and SPOT
- **Docker-ready** — single-command reproduction of all results

## Installation

```bash
conda create -n chemocalib python=3.10 -y
conda activate chemocalib
git clone https://github.com/chemocalib/chemocalib.git
cd chemocalib
pip install -e .
```

For detailed instructions, see [INSTALL.md](INSTALL.md).

## Quick Start

```python
from chemocalib.models.mbpls import MultiBlockPLS
from chemocalib.gem.gpr_vip import compute_gpr_vip
from chemocalib.gem.constrained_fba import ConstrainedFBA

# 1. Load multi-omics data
transcriptome = ...  # (n_conditions, n_genes)
metabolome = ...     # (n_conditions, n_metabolites)

# 2. MB-PLS decomposition
mbpls = MultiBlockPLS(n_components=3, scale=True)
mbpls.fit([transcriptome, metabolome], Y)

# 3. GPR-VIP reaction scoring
vip_scores = compute_gpr_vip(mbpls, block_idx=0)
reaction_vip = aggregate_by_gpr(vip_scores, gpr_rules)

# 4. Constrained FBA
cFBA = ConstrainedFBA("iJO1366")
flux_predictions = cFBA.predict(reaction_vip, conditions)
```

Or run the tutorial:
```bash
python examples/example_5min_tutorial.py
```

## Repository Structure

```
chemocalib/
├── chemocalib/              # Core package
│   ├── models/              # MB-PLS and PLS models
│   ├── gem/                 # GEM integration (GPR-VIP, constrained FBA)
│   ├── data/                # Data loading and synthetic data generation
│   ├── stats/               # Statistical utilities (bootstrap, Holm correction)
│   ├── validation/          # Cross-validation and reliability assessment
│   ├── virtual_experiment/  # In-silico experimental design
│   ├── dynamic_layer/       # ODE dynamic modeling layer
│   └── active_learning/     # Active learning for condition selection
├── examples/                # Runnable tutorial scripts
├── scripts/                 # Pipeline and figure-generation scripts
├── data/                    # Example datasets and the Kim 2016 S2 Dataset
├── models/                  # iJO1366 and iMM904 COBRApy model files
├── figures/                 # Generated figures (per-pathway Spearman, etc.)
├── output/                  # Benchmark summary CSV files
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for interactive exploration
├── docs/                    # Extended documentation
├── Dockerfile               # Containerized reproduction
├── pyproject.toml           # Package metadata and dependencies
├── INSTALL.md               # Detailed installation guide
├── CITATION.cff             # Citation metadata
└── README.md                # This file
```

## Benchmark Results

ChemoCalib was evaluated on the iJO1366 genome-scale model against $^{13}$C-MFA
flux measurements (11 *E. coli* conditions: Ishii 2007 + Holm 2010):

| Method     | Spearman $\rho$ (mean) | Pooled $\rho$ | Pearson $r$ | NRMSE | Holm $p$ vs ChemoCalib |
|------------|------------------------|---------------|-------------|-------|--------------------------|
| pFBA       | 0.19 | 0.18 | 0.20 | 0.48 | $2 \times 10^{-4}\;^{**}$ |
| E-Flux     | 0.22 | 0.21 | 0.24 | 0.41 | $3.5 \times 10^{-3}\;^{**}$ |
| E-Flux2    | 0.37 | 0.34 | 0.36 | 0.35 | $1.8 \times 10^{-2}\;^{*}$ |
| MOMENT     | 0.26 | 0.25 | 0.27 | 0.39 | $5.2 \times 10^{-4}\;^{**}$ |
| SPOT       | 0.40 | 0.39 | 0.40 | 0.32 | $8.5 \times 10^{-3}\;^{**}$ |
| **ChemoCalib** | **0.54** | **0.48** | **0.51** | **0.27** | — |

$^{*}p < 0.05$, $^{**}p < 0.01$ Holm-corrected Wilcoxon signed-rank test (paired per-condition $\rho$).

An extended 20-condition benchmark on the Kim et al. (2016) curated set confirms
generalization (Table S3 in the manuscript supplementary).

## Docker Reproduction

```bash
docker build -t chemocalib .
docker run -it chemocalib python examples/example_full_workflow.py
docker run -it chemocalib pytest tests/ -v
```

## Citation

If you use ChemoCalib in your research, please cite:

> Zhang X. (2026) ChemoCalib: Chemometrics-Calibrated Constraint-Based Metabolic
> Modeling via Multi-Block PLS. *Bioinformatics*.

BibTeX:
```bibtex
@article{zhang2026chemocalib,
  title   = {ChemoCalib: Chemometrics-Calibrated Constraint-Based Metabolic
             Modeling via Multi-Block PLS},
  author  = {Zhang, Xin},
  journal = {Bioinformatics},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

## License

MIT License. See `LICENSE` file.

## Contact

Xin Zhang — Department of Chemistry, Capital Normal University
— xinzhang@cnu.edu.cn
