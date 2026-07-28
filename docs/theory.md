# Theory Background

## Problem Statement

Integrating multi-omics measurements (metabolomics, transcriptomics,
proteomics) with genome-scale metabolic models (GEMs) is a central
challenge in systems biology. Existing methods (E-Flux, MADE, GECKO) either
rely on single-omics data, impose overly rigid constraints, or require
enzyme kinetic parameters that are rarely available.

ChemoCalib addresses this gap by **chemometrically calibrating**
metabolic constraints from multi-omics latent structure.

## Multi-Block PLS (MB-PLS)

### Model

Given \(K\) omics data blocks \(\{X_1, \dots, X_K\}\) with \(X_k \in
\mathbb{R}^{n \times p_k}\) and a response vector \(y \in \mathbb{R}^n\)
(growth rate), MB-PLS finds:

1. **Block scores**: \(T_k = X_k W_k \in \mathbb{R}^{n \times r}\)
   where \(W_k \in \mathbb{R}^{p_k \times r}\) are block-specific
   weight matrices and \(r\) is the number of latent components.

2. **Super score**: \(T = \sum_{k} T_k \in \mathbb{R}^{n \times r}\),
   the consensus latent representation.

3. **Prediction**: \(\hat{y} = T\beta\) via linear regression of \(y\)
   on \(T\).

### Optimization

The MB-PLS objective maximizes the covariance between block scores
and the response:

\[
\max_{w_k} \sum_k \text{cov}(X_k w_k, y)^2
\quad \text{s.t.} \quad \|w_k\|_2 = 1
\]

### VIP Scores

Variable Importance in Projection for block \(k\), feature \(j\):

\[
\text{VIP}_{kj} = \sqrt{p_k \cdot \sum_{a=1}^{r} w_{k,ja}^2 \cdot
\text{SSY}_a / \sum_{a=1}^{r} \text{SSY}_a}
\]

where \(\text{SSY}_a\) is the sum of squares of \(y\) explained by
component \(a\).

## Latent-to-Constraint Mapping

### Core Idea

Each latent component models a coordinated omics shift. The VIP scores
identify the metabolites driving this shift. By mapping metabolite
drivers to exchange reaction bounds, we translate latent structure into
actionable metabolic constraints.

### Mapping Modes

| Mode | Description | Formula |
|------|-------------|---------|
| **Soft** | Linear modulation with adaptive width | \(b_{r} = b_0 \pm \alpha \cdot l_c \cdot \text{VIP}_{r}\) |
| **Hard** | Strict directionality from latent sign | \(b_{r} \in [\min(0, \beta l_c), \max(0, \beta l_c)]\) |
| **Adaptive** | Data-driven width from latent variance | \(b_{r} = b_0 \cdot (1 + \gamma \cdot l_c \cdot \text{VIP}_{r})\) |

where \(b_r\) is the bound for reaction \(r\), \(l_c\) is the latent
score on component \(c\), and \(\alpha, \beta, \gamma\) are scale
factors.

## Active Learning

### Uncertainty Sampling

The residual space \(R_k = X_k - T \cdot W_k^{T}\) captures
information not explained by the latent model. Uncertainty score:

\[
u_i = \frac{1}{K} \sum_{k=1}^{K} \left\| R_{k,i} \right\|_2
\]

Higher \(u_i\) means sample \(i\) is poorly represented by the current
latent model and should be prioritized for experimental validation.

### Strategy Options

- **Residual**: Pure uncertainty (maximize information gain)
- **Entropy**: Information-theoretic criterion in latent space
- **Diversity**: Maximize coverage in feature space
- **Hybrid**: Combined residual-diversity weighting

## ODE Dynamics Calibration (Optional)

The glycolysis ODE system uses latent component magnitudes to modulate
key enzyme \(V_{\text{max}}\) parameters:

\[
\frac{d[\text{G6P}]}{dt} = v_{\text{HK}} - v_{\text{PGI}}
\]
\[
\frac{d[\text{FBP}]}{dt} = v_{\text{PFK}} - v_{\text{ALD}}
\]
\[
\frac{d[\text{PYR}]}{dt} = v_{\text{PK}} \times 2
\]

with \(v_{\text{enz}} = V_{\text{max,enz}} \cdot
(1 + \theta \cdot l_{\text{enz}})\) where \(l_{\text{enz}}\) is the
latent score mapped to each enzyme.

## Statistical Framework

ChemoCalib provides rigorous statistical inference:

- **Permutation tests**: Shuffle \(y\) to generate null distribution of
  model performance, computing empirical \(p\)-values.
- **Bootstrap CIs**: Resample with replacement to estimate confidence
  intervals for VIP scores and model coefficients.
- **Stability selection**: Subsampling-based feature selection
  probabilities.

## References

1. **MB-PLS**: Wold, S. et al. (1987). Multi-way principal components
   and PLS analysis. *J. Chemometrics*, 1(1), 41-56.
2. **DIABLO**: Singh, A. et al. (2019). DIABLO: an integrative approach
   for identifying key molecular drivers from multi-omics assays.
   *Bioinformatics*, 35(17), 3055-3062.
3. **E-Flux**: Colijn, C. et al. (2009). Interpreting expression data with
   metabolic flux models. *PLoS Comp. Biol.*, 5(8), e1000489.
4. **MADE**: Jensen, P.A. & Papin, J.A. (2011). Functional integration
   of a metabolic network model and expression data. *BMC Syst. Biol.*,
   5, 34.
5. **GECKO**: Sánchez, B.J. et al. (2017). Improving the phenotype
   predictions of a yeast GEM. *Mol. Syst. Biol.*, 13(8), 935.
6. **Active Learning**: Settles, B. (2009). Active learning literature
   survey. *Computer Sciences Technical Report 1648*, UW-Madison.
