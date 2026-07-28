"""
Realistic data loaders for multi-omics datasets
=================================================
Generates and loads realistic multi-condition omics data
mimicking published E. coli and S. cerevisiae experiments.

References
----------
- ECOMICS (E. coli multi-omics): Ishii et al. (2007) Science
- Yeast nitrogen limitation: Gutteridge et al. (2010) BMC Syst. Biol.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import os


def generate_realistic_e_coli_data(
    n_conditions: int = 8,
    n_metabolites: int = 60,
    n_transcripts: int = 200,
    n_proteins: int = 80,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """Generate realistic E. coli multi-carbon-source data.

    Simulates growth on different carbon sources (glucose, acetate, glycerol,
    succinate, etc.) with known metabolic shifts.

    Each condition induces a coherent perturbation across metabolome,
    transcriptome, and proteome, driven by underlying pathway activation
    patterns.

    Parameters
    ----------
    n_conditions : int
        Number of conditions (carbon sources), default 8.
    n_metabolites : int
        Number of measured metabolites.
    n_transcripts : int
        Number of measured transcripts.
    n_proteins : int
        Number of measured proteins.
    noise_level : float
        Relative noise level.
    seed : int

    Returns
    -------
    blocks : [X_met, X_txn, X_prot]
        Normalized multi-block data.
    growth_rates : np.ndarray, shape (n_conditions,)
        Measured growth rates (1/h).
    metadata : dict
        Condition names, pathway mask, latent factors.
    """
    rng = np.random.RandomState(seed)

    carbon_sources = [
        "glucose", "acetate", "glycerol", "succinate",
        "fumarate", "pyruvate", "xylose", "lactate",
    ][:n_conditions]

    # True latent factors for each condition
    # Factor 0: glycolytic activity, Factor 1: TCA activity, Factor 2: gluconeogenesis
    pathway_profiles = {
        "glucose":    [2.0, 1.0, 0.2],
        "acetate":    [0.3, 2.0, 1.5],
        "glycerol":   [1.2, 1.5, 0.8],
        "succinate":  [0.2, 1.8, 1.2],
        "fumarate":   [0.3, 1.6, 1.0],
        "pyruvate":   [1.5, 1.2, 0.5],
        "xylose":     [1.0, 0.8, 1.0],
        "lactate":    [0.8, 1.3, 0.7],
    }

    L = np.array([pathway_profiles[cs] for cs in carbon_sources])  # (n_cond, 3)

    # Block-specific loading matrices
    W_met = rng.randn(3, n_metabolites) * 0.3
    W_txn = rng.randn(3, n_transcripts) * 0.25
    W_prot = rng.randn(3, n_proteins) * 0.2

    # Generate blocks
    X_met = L @ W_met + noise_level * rng.randn(n_conditions, n_metabolites)
    X_txn = L @ W_txn + noise_level * rng.randn(n_conditions, n_transcripts)
    X_prot = L @ W_prot + noise_level * rng.randn(n_conditions, n_proteins)

    # Growth rates (primarily driven by glycolytic + TCA activity)
    growth_rates = L @ np.array([0.4, 0.3, 0.15]) + 0.1
    growth_rates += noise_level * 0.1 * rng.randn(n_conditions)
    growth_rates = np.clip(growth_rates, 0.1, 2.5)

    metadata = {
        "carbon_sources": carbon_sources,
        "pathway_profiles": pathway_profiles,
        "n_metabolites": n_metabolites,
        "n_transcripts": n_transcripts,
        "n_proteins": n_proteins,
        "latent_factors": ["glycolysis", "tca", "gluconeogenesis"],
    }

    return [X_met, X_txn, X_prot], growth_rates, metadata


def generate_realistic_ecoli_ko_data(
    n_conditions: int = 8,
    n_metabolites: int = 60,
    n_transcripts: int = 200,
    n_proteins: int = 80,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """Generate realistic E. coli single-gene knockout perturbation data.

    Simulates metabolic states under 8 genetic perturbations (WT + 7 knockouts)
    of key metabolic enzymes, mimicking standard gene deletion phenotyping
    experiments (Baba et al., 2006, Mol. Syst. Biol.).

    Parameters
    ----------
    n_conditions : int
        Number of genetic conditions (WT + knockouts), max 8.
    n_metabolites, n_transcripts, n_proteins : int
        Feature dimensions per block.
    noise_level : float
        Relative noise level.
    seed : int

    Returns
    -------
    blocks, growth_rates, metadata : as in generate_realistic_e_coli_data
    """
    rng = np.random.RandomState(seed)

    strains = [
        "WT", "deltapgi", "deltazwf", "deltapfkA",
        "deltapykA", "delta_ppc", "delta_sdhA", "delta_ackA",
    ][:n_conditions]

    # Latent factors:
    # Factor 0: glycolytic flux (perturbed by pgi, pfkA, pykA KO)
    # Factor 1: TCA cycle activity (perturbed by sdhA KO)
    # Factor 2: overflow metabolism / acetate (perturbed by ackA KO)
    ko_profiles = {
        "WT":          [1.0, 1.0, 0.3],
        "deltapgi":    [0.1, 0.8, 0.5],   # pgi KO: glycolytic block
        "deltazwf":    [0.9, 0.7, 0.4],   # zwf KO: PPP redirect
        "deltapfkA":   [0.2, 0.9, 0.6],   # pfkA KO: glycolytic block
        "deltapykA":   [0.4, 0.8, 0.7],   # pykA KO: pyruvate kinase
        "delta_ppc":   [0.7, 0.3, 1.2],   # ppc KO: anaplerotic loss
        "delta_sdhA":  [0.6, 0.1, 0.8],   # sdhA KO: TCA block
        "delta_ackA":  [0.8, 0.6, 2.0],   # ackA KO: acetate overflow block
    }

    L = np.array([ko_profiles[s] for s in strains])

    W_met = rng.randn(3, n_metabolites) * 0.3
    W_txn = rng.randn(3, n_transcripts) * 0.25
    W_prot = rng.randn(3, n_proteins) * 0.2

    X_met = L @ W_met + noise_level * rng.randn(n_conditions, n_metabolites)
    X_txn = L @ W_txn + noise_level * rng.randn(n_conditions, n_transcripts)
    X_prot = L @ W_prot + noise_level * rng.randn(n_conditions, n_proteins)

    growth_rates = L @ np.array([0.45, 0.25, -0.1]) + 0.6
    growth_rates += noise_level * 0.08 * rng.randn(n_conditions)
    growth_rates = np.clip(growth_rates, 0.05, 2.0)

    metadata = {
        "strains": strains,
        "ko_profiles": ko_profiles,
        "n_metabolites": n_metabolites,
        "n_transcripts": n_transcripts,
        "n_proteins": n_proteins,
        "latent_factors": ["glycolysis", "tca", "overflow"],
        "reference": "Baba et al. (2006) Mol. Syst. Biol. 2:2006.0008",
    }

    return [X_met, X_txn, X_prot], growth_rates, metadata


def generate_realistic_yeast_stress_data(
    n_conditions: int = 8,
    n_metabolites: int = 55,
    n_transcripts: int = 180,
    n_proteins: int = 75,
    noise_level: float = 0.06,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """Generate realistic S. cerevisiae environmental stress multi-omics data.

    Simulates yeast metabolic adaptation under 8 environmental perturbations,
    mimicking chemostat and batch stress experiments (Gasch et al., 2000,
    Mol. Biol. Cell; Daran-Lapujade et al., 2004, J. Biol. Chem.).

    Parameters
    ----------
    n_conditions : int
        Number of stress conditions, max 8.
    n_metabolites, n_transcripts, n_proteins : int
        Feature dimensions per block.
    noise_level : float
        Relative noise level.
    seed : int

    Returns
    -------
    blocks, growth_rates, metadata : as in generate_realistic_e_coli_data
    """
    rng = np.random.RandomState(seed)

    conditions = [
        "glucose_rich", "glucose_limited", "ethanol",
        "heat_shock", "osmotic", "oxidative",
        "nitrogen_lim", "pH_stress",
    ][:n_conditions]

    # Latent factors:
    # Factor 0: fermentative vs respiratory (Crabtree effect)
    # Factor 1: general stress response (ESR)
    # Factor 2: nitrogen/carbon scavenging
    stress_profiles = {
        "glucose_rich":    [2.0, 0.2, 0.1],
        "glucose_limited": [0.3, 1.0, 0.8],
        "ethanol":         [0.1, 1.5, 1.2],
        "heat_shock":      [1.5, 1.8, 0.3],
        "osmotic":         [1.2, 1.6, 0.5],
        "oxidative":       [0.8, 2.0, 0.6],
        "nitrogen_lim":    [0.2, 1.2, 2.0],
        "pH_stress":       [1.0, 1.3, 0.9],
    }

    L = np.array([stress_profiles[c] for c in conditions])

    W_met = rng.randn(3, n_metabolites) * 0.28
    W_txn = rng.randn(3, n_transcripts) * 0.30
    W_prot = rng.randn(3, n_proteins) * 0.22

    X_met = L @ W_met + noise_level * rng.randn(n_conditions, n_metabolites)
    X_txn = L @ W_txn + noise_level * rng.randn(n_conditions, n_transcripts)
    X_prot = L @ W_prot + noise_level * rng.randn(n_conditions, n_proteins)

    growth_rates = L @ np.array([0.35, -0.15, 0.12]) + 0.45
    growth_rates += noise_level * 0.06 * rng.randn(n_conditions)
    growth_rates = np.clip(growth_rates, 0.05, 1.8)

    metadata = {
        "conditions": conditions,
        "stress_profiles": stress_profiles,
        "n_metabolites": n_metabolites,
        "n_transcripts": n_transcripts,
        "n_proteins": n_proteins,
        "latent_factors": ["fermentative_respiratory", "stress_response", "scavenging"],
        "reference": "Gasch et al. (2000) Mol. Biol. Cell 11(12):4241-4257",
    }

    return [X_met, X_txn, X_prot], growth_rates, metadata


def validate_multiblock_coherence(
    blocks: List[np.ndarray],
    y: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Run sanity checks on multi-block data before modeling.

    Parameters
    ----------
    blocks : list of np.ndarray
    y : np.ndarray
    verbose : bool

    Returns
    -------
    report : dict
        Validation metrics.
    """
    from scipy.stats import pearsonr

    report = {
        "n_blocks": len(blocks),
        "n_samples": blocks[0].shape[0],
        "block_shapes": [X.shape for X in blocks],
    }

    # Check sample consistency
    for i, X in enumerate(blocks):
        assert X.shape[0] == report["n_samples"], \
            f"Block {i} has {X.shape[0]} samples, expected {report['n_samples']}"
    assert y.shape[0] == report["n_samples"]

    # Floor: mean pairwise correlation across blocks
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            # Correlation of first PCs
            Ui, _, _ = np.linalg.svd(blocks[i] - blocks[i].mean(0), full_matrices=False)
            Uj, _, _ = np.linalg.svd(blocks[j] - blocks[j].mean(0), full_matrices=False)
            r, _ = pearsonr(Ui[:, 0], Uj[:, 0])
            report[f"pc1_corr_block{i}_vs_block{j}"] = float(r)

    # Y signal check
    report["y_mean"] = float(np.mean(y))
    report["y_std"] = float(np.std(y))
    report["y_cv"] = float(np.std(y) / (abs(np.mean(y)) + 1e-10))

    if verbose:
        print(f"Multiblock data: {report['n_blocks']} blocks, "
              f"{report['n_samples']} samples")
        shapes = " × ".join([str(s[1]) for s in report["block_shapes"]])
        print(f"Features: {shapes}")
        print(f"Growth: {report['y_mean']:.3f} ± {report['y_std']:.3f}")

    return report


def load_from_csv(
    path: str,
    y_column: str = "growth_rate",
    block_patterns: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Load multi-block data from a CSV file.

    The CSV should have columns organized in blocks, with a response column.
    Block columns are identified by prefix patterns.

    Parameters
    ----------
    path : str
        Path to CSV file.
    y_column : str
        Name of response column.
    block_patterns : list of str, optional
        Prefix patterns for each block, e.g. ["met_", "txn_", "prot_"].

    Returns
    -------
    blocks : list of np.ndarray
    y : np.ndarray
    feature_names : list of list of str
    """
    import pandas as pd

    df = pd.read_csv(path)
    y = df[y_column].values

    if block_patterns is None:
        # Auto-detect: all non-y numeric columns as single block
        cols = [c for c in df.columns if c != y_column]
        X = df[cols].values.astype(np.float64)
        return [X], y, [cols]

    blocks = []
    feature_names = []
    for pattern in block_patterns:
        cols = [c for c in df.columns if c.startswith(pattern)]
        if not cols:
            raise ValueError(f"No columns match pattern '{pattern}'")
        X = df[cols].values.astype(np.float64)
        blocks.append(X)
        feature_names.append(cols)

    return blocks, y, feature_names
