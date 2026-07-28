# Getting Started

## Installation

**Requirements**: Python 3.10+, 8 GB RAM (laptop OK), no GPU needed.

```bash
cd chemocalib
pip install -r requirements.txt
pip install -e ".[notebook]"
```

### GLPK Solver (for COBRApy FBA)

```bash
# Ubuntu/Debian
sudo apt-get install glpk-utils libglpk-dev
# macOS
brew install glpk
# Windows
conda install -c conda-forge glpk
```

### Docker (fully reproducible)

```bash
docker build -t chemocalib .
docker run --rm chemocalib
```

## Verify Installation

```python
import chemocalib
from chemocalib import MultiBlockPLS, FBASimulator
print("ChemoCalib version:", chemocalib.__version__)
```

## Core Concepts

### 1. Multi-Block PLS (MB-PLS)

Multiple omics data blocks (metabolome, transcriptome, proteome) are
decomposed simultaneously into shared latent variables that predict a
response (e.g., growth rate).

### 2. Latent-to-Constraint Mapping

Latent scores are mapped to reaction bound constraints via VIP-weighted
scaling (soft, hard, or adaptive modes).

### 3. Constrained FBA

Chemometrically-constrained flux balance analysis narrows the metabolic
solution space to biologically relevant regions.

### 4. Active Learning Loop

Uncertainty sampling selects the most informative gene pairs for
virtual double-knockout experiments, closing the model-experiment loop.

## Quick Run

```bash
chemocalib pipeline --n-pairs 150 --output-dir ./output
```

Or run the full pipeline from Python:

```python
from chemocalib.scripts.run_pipeline import run_full_pipeline
run_full_pipeline(n_samples=100, n_pairs=80, output_dir="./output")
```

## Environment Dependencies

| Package     | Version    | Purpose                        |
|-------------|-----------|--------------------------------|
| numpy       | >=1.21    | Numerical arrays               |
| scipy       | >=1.7     | Linear algebra, optimization   |
| scikit-learn| >=1.0     | PLS decomposition              |
| cobra       | >=0.26    | Constraint-based modeling (FBA)|
| pandas      | >=1.3     | Data tables, CSV I/O           |
| matplotlib  | >=3.5     | Publication figures            |

## Troubleshooting

**GLPK not found**:
```python
import cobra
print(cobra.util.solver.interface.list_solvers())  # should include "glpk"
```

**COBRApy model download fails**:
```python
# Manually specify solver
from cobra.io import load_model
model = load_model("textbook")
```
