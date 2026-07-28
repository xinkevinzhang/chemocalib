# Tutorial: End-to-End Analysis with ChemoCalib

This tutorial walks through the complete ChemoCalib pipeline using
realistic multi-carbon-source E. coli data.

## Step 1: Load Multi-Omics Data

```python
import numpy as np
from chemocalib.data.loader import generate_realistic_e_coli_data

# Generate realistic multi-carbon-source E. coli data
# (mimics Ishii et al. 2007 experimental design)
blocks, growth, meta = generate_realistic_e_coli_data(
    n_conditions=8, seed=42)

print(f"Blocks: {[b.shape for b in blocks]}")
print(f"Carbon sources: {meta['carbon_sources']}")
print(f"Latent factors: {meta['latent_factors']}")
```

## Step 2: Fit Multi-Block PLS Model

```python
from chemocalib.models.mbpls import MultiBlockPLS

mbpls = MultiBlockPLS(
    n_components=3,
    block_names=["Metabolome", "Transcriptome", "Proteome"]
)
mbpls.fit(blocks, growth)

# Inspect block importance
print("Block importance:", mbpls.block_importance)

# Identify driving metabolites
drivers = mbpls.get_driving_metabolites(block_idx=0, top_k=5)
print(f"Top metabolites: {drivers['vip_values'][:5]}")
```

## Step 3: Statistical Validation

```python
from chemocalib.stats import permutation_test, bootstrap_ci

# Permutation test for MB-PLS significance
pt_result = permutation_test(
    model_factory=lambda: MultiBlockPLS(n_components=3),
    blocks=blocks, y=growth, n_permutations=200, seed=42
)
print(f"Permutation p-value: {pt_result['p_value']:.3f}")

# Bootstrap confidence intervals for VIP scores
from chemocalib.stats import bootstrap_vip_ci
vip_ci = bootstrap_vip_ci(
    model_factory=lambda: MultiBlockPLS(n_components=3),
    blocks=blocks, y=growth, block_idx=0, n_bootstrap=100
)
```

## Step 4: Cross-Validation

```python
from chemocalib.cross_validation import grid_search_components, stability_selection

# Grid search for optimal components
gs_result = grid_search_components(
    blocks=blocks, y=growth, model_cls=MultiBlockPLS,
    component_range=[1, 2, 3, 4, 5], n_folds=5, seed=42
)
print(f"Optimal n_components: {gs_result['best_n_components']}")

# Stability selection
ss = stability_selection(
    blocks=blocks, y=growth, model_cls=MultiBlockPLS,
    n_components=3, n_subsamples=50, seed=42
)
```

## Step 5: Map Latent Scores to Metabolic Constraints

```python
from chemocalib.gem.constraints import LatentToConstraint
from chemocalib.gem.fba import FBASimulator

# Initialize simulator and constraint mapper
sim = FBASimulator(model_name="textbook")
sim.load_model()

mapper = LatentToConstraint(scaling_mode="soft")
exchange_ids = [r.replace("EX_", "") for r in sim.get_exchange_reactions()]

# Build feature-to-reaction mapping
names = [f"feature_{i}" for i in range(len(exchange_ids))]
mapper.build_feature_reaction_map(names, exchange_ids)

# Convert latent scores to bounds
latent_mean = mbpls.super_scores.mean(axis=0)
bounds = mapper.latent_to_bounds(latent_mean, n_components=3)

# Run constrained FBA
result = sim.fba_with_chemometric_constraints(bounds)
print(f"FBA status: {result['status']}")
print(f"Constrained growth: {result['objective_value']:.4f}")
```

## Step 6: Flux Variability Analysis

```python
from chemocalib.gem.fva import FVAAnalyzer

fva = FVAAnalyzer(sim)
fva_result = fva.run(bounds)
print(f"Mean flux space contraction: {fva_result['mean_shrinkage_pct']:.1f}%")
```

## Step 7: Virtual Double-Knockout Experiments

```python
from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

genes = sim.get_all_genes()
designer = DoubleKnockoutDesigner(gene_pool=genes)
pairs = designer.generate_pairs(n_pairs=150)

print(f"Generated {len(pairs)} unique gene pairs")
```

## Step 8: Active Learning for Experiment Prioritization

```python
from chemocalib.active_learning.uncertainty import UncertaintySampler

# Compute residual space as uncertainty proxy
residuals = mbpls.residual_space(blocks)

sampler = UncertaintySampler(strategy="hybrid")

# Select top candidates for validation
candidates = sampler.select_double_knockout_candidates(
    all_gene_pairs=pairs,
    pair_features=np.random.randn(len(pairs), 1),  # would use real features
    residuals=residuals,
    n_select=10,
    n_pool=50,
)
print(candidates[["rank", "gene_A", "gene_B"]].head())
```

## Step 9: Benchmark Against Published Methods

```python
from chemocalib.validation.benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    methods=["chemocalib", "eflux", "made", "gecko"],
    n_repeats=5, seed=42
)
summary = runner.run(blocks, growth)
table = runner.comparison_table()
print(table.to_string(index=False))
```

## Step 10: Optional ODE Dynamics Calibration

```python
from chemocalib.dynamic_layer.ode_solver import GlycolysisODE

ode = GlycolysisODE()
ode.calibrate_from_latent(mbpls.super_scores[-1], n_component=0)
result = ode.simulate(t_span=(0, 60), n_points=200)
print(f"Steady-state concentrations: {ode.steady_state()}")
```

---

## Next Steps

- Read the [API Reference](api) for detailed method documentation
- Study the [Theory Background](theory) for mathematical foundations
- Run `scripts/run_pipeline.py` for automated batch execution
