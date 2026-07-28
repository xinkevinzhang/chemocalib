# ChemoCalib Installation Guide

## Quick Install (Recommended)

```bash
# Create conda environment
conda create -n chemocalib python=3.10 -y
conda activate chemocalib

# Install from PyPI (when published)
pip install chemocalib

# Or install from GitHub (for reviewers)
git clone https://github.com/chemocalib/chemocalib.git
cd chemocalib
pip install -e .
```

## Dependencies

Core dependencies are installed automatically:
- `numpy>=1.21`, `scipy>=1.7`, `pandas>=1.3`
- `scikit-learn>=1.0`, `statsmodels>=0.13`
- `cobra>=0.26` (for constraint-based modeling)
- `matplotlib>=3.5`, `seaborn>=0.11`
- `multiblock>=0.2` (for MB-PLS)

Optional:
- `cplex` or `gurobi` (for large-scale FBA, see COBRApy docs)
- GLKP is the default solver (included via cobra)

## Verify Installation

```bash
python -c "import chemocalib; print(chemocalib.__version__)"
```

## Run Example

```bash
python examples/example_5min_tutorial.py
```

## Docker (Alternative)

```bash
docker build -t chemocalib .
docker run -it chemocalib python examples/example_5min_tutorial.py
```

## Troubleshooting

### COBRApy solver issues
COBRApy requires a linear programming solver. The default (GLKP) is
sufficient for the iJO1366 model used in the paper.

If you get solver errors:
```bash
conda install -c conda-forge glpk
```
