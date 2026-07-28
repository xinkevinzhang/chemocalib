# Changelog

All notable changes to ChemoCalib are documented here.

---

## [0.2.0] - 2026-07-28

### Added
- **Real data validation pipeline**: End-to-end multi-carbon-source E. coli validation script with COBRApy FBA (`scripts/real_data_validation.py`)
- **Upgraded benchmark**: E-Flux/MADE/GECKO comparisons now support actual COBRApy FBA integration and paired t-test statistical comparisons (`BenchmarkRunner.statistical_tests`)
- **Sphinx documentation**: Full 4-page documentation site (Getting Started, Tutorial, API Reference, Theory Background) with `docs/Makefile`
- **Manuscript**: LaTeX methods paper draft with introduction, methods, results, discussion, and bibliography (`manuscript/main.tex`)
- **CITATION.cff**: Standard academic citation metadata for GitHub/Zenodo
- **CODE_OF_CONDUCT.md**: Contributor Covenant CoC
- **Issue/PR templates**: Bug report, feature request, and pull request templates (`/.github/`)
- **.dockerignore**: Optimized Docker build exclusions
- **7 publication-quality figures**: Pipeline schematic, block loadings, latent scores, constraint modes, active learning, benchmark comparison, validation scatter (`scripts/generate_figures.py`)
- **Notebook 02**: Surrogate modeling and benchmarking with uncertainty quantification
- **Modular test suite**: Split from single 783-line file into 7 focused test modules with shared `conftest.py` fixtures
- **CI matrix expansion**: 3 OS × 3 Python versions, lint job, docs build job, PyPI publish workflow, codecov upload

### Changed
- **README**: Added benchmark summary table, updated citation to Zhang Xin / Capital Normal University
- **pyproject.toml**: Updated author info, added project URLs, maintenance status to Beta, added pytest/ruff config
- **LICENSE**: Updated copyright to Zhang Xin (Capital Normal University)
- **docs/conf.py**: Updated author metadata
- **benchmark.py**: Added `use_fba` flag for COBRApy integration, `statistical_tests()` method, `predict_growth_fba()` for E-Flux
- **CI workflow**: Added coverage threshold check, PyPI auto-publish on tag, docs build, lint job

### Fixed
- FBA test `test_fba_with_constraints` private attribute access error
- Boundary validation in hard/adaptive constraint modes ensuring `lb <= ub` always

---

## [0.1.0] - 2026-07-15

### Added
- Initial release
- MultiBlockPLS and DIABLO-like latent variable models
- LatentToConstraint mapper with soft/hard/adaptive modes
- FBASimulator with COBRApy integration
- FVAAnalyzer for flux space contraction analysis
- UncertaintySampler with 4 strategies (residual, entropy, diversity, hybrid)
- DoEDesigner with LHS, CCD, full factorial, Box-Behnken
- DoubleKnockoutDesigner for virtual experiment generation
- SurrogateModel with bootstrap uncertainty
- GlycolysisODE for optional dynamic calibration
- Permutation test and bootstrap CI modules
- Grid search CV and stability selection
- Realistic E. coli data generator
- Docker support
- 73 unit tests
