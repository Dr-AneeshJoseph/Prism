# PRISM v2.0 Architecture Documentation

## Protocol for Rigorous Investigation of Scientific Mechanisms

**Version:** 2.0 Final  
**Author:** Dr. Aneesh Joseph (Architecture) + Claude (Implementation)  
**Date:** December 2025  
**License:** MIT

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Philosophy](#core-philosophy)
3. [The 5-Layer Architecture](#the-5-layer-architecture)
4. [Statistical Foundations](#statistical-foundations)
5. [Component Reference](#component-reference)
6. [Mathematical Formulations](#mathematical-formulations)
7. [Safety & Governance](#safety--governance)
8. [References](#references)

---

## Executive Summary

PRISM (Protocol for Rigorous Investigation of Scientific Mechanisms) is a hypothesis evaluation engine designed for high-fidelity reasoning. It provides a hardened epistemic structure that forces rigorous weighing of competing hypotheses, systematically eliminating invalid paths through layered statistical analysis.

### Key Innovations in v2.0

| Feature | Principle | Benefit |
|---------|-----------|---------|
| Diagnostic Test Theory | Likelihood ratios from sensitivity/specificity | Properly calibrated evidence weights |
| Correlation-Adjusted Pooling | Variance inflation for dependent evidence | Prevents double-counting |
| Meta-Analytic Synthesis | DerSimonian-Laird random effects | Heterogeneity-aware combination |
| P-Curve Analysis | Distribution shape testing | P-hacking detection |
| Extreme Value Theory | Generalized Pareto Distribution | Tail risk quantification |
| Semantic Independence | NLP-based entity extraction | Catches hidden dependencies |
| Reference Class Forecasting | Empirical base rates | Outside view anchoring |

---

## Core Philosophy

### The Epistemic Problem

Standard analytical tools suffer from **Epistemic Drift**—they make every idea sound plausible. PRISM acts as a **Sincerity Firewall** operating on three principles:

1. **Hypothesis Selection:** Creates a hierarchy of truth, identifying the single best explanation
2. **Fatal Flaw Detection:** Employs a "Kill Switch" for safety/legal violations or feasibility failures  
3. **Layered Filtration:** Analysis occurs in 5 distinct layers, moving from definition to decision

### Design Principles

```
┌─────────────────────────────────────────────────────────────┐
│                    PRISM DESIGN AXIOMS                      │
├─────────────────────────────────────────────────────────────┤
│ 1. Evidence quality > Evidence quantity                     │
│ 2. Independence matters more than sample size               │
│ 3. Base rates anchor all inference                          │
│ 4. Uncertainty must propagate through calculations          │
│ 5. Fatal flaws block—they don't just warn                   │
│ 6. Every number must be explainable                         │
└─────────────────────────────────────────────────────────────┘
```

---

## The 5-Layer Architecture

```
                    ┌──────────────────┐
                    │   USER INPUT     │
                    │  (Hypothesis +   │
                    │   Evidence)      │
                    └────────┬─────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L0: FOUNDATION VALIDATION                                      │
│  ─────────────────────────                                      │
│  • Defines the "Epistemic Target"                               │
│  • Validates What/Why/How structure                             │
│  • Checks feasibility dimensions                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L0.5: PRE-FLIGHT SAFETY                                        │
│  ────────────────────────                                       │
│  • Evidence sufficiency check                                   │
│  • Content scanning (100+ patterns)                             │
│  • Fatal content blocking                                       │
│  • Euphemism detection                                          │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L1: EVIDENCE SYNTHESIS                                         │
│  ──────────────────────                                         │
│  • Bayesian updating with correlation adjustment                │
│  • Meta-analytic pooling (DerSimonian-Laird)                    │
│  • P-curve analysis for p-hacking                               │
│  • Publication bias detection (trim-and-fill)                   │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L2: ADVERSARIAL TESTING                                        │
│  ───────────────────────                                        │
│  • 8 cognitive bias detectors                                   │
│  • Evidence independence verification                           │
│  • Establishment claim verification                             │
│  • Semantic similarity analysis                                 │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L3: SENSITIVITY ANALYSIS                                       │
│  ────────────────────────                                       │
│  • Perturbation testing (±10%, ±20%)                            │
│  • Leave-one-out analysis                                       │
│  • Critical sensitivity flagging                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L4: GATE CHECK                                                 │
│  ───────────────                                                │
│  • Evidence minimum gate                                        │
│  • Fatal content gate                                           │
│  • Severe bias gate                                             │
│  • Stability gate                                               │
│  • BLOCKING if any gate fails                                   │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  L5: DECISION SYNTHESIS                                         │
│  ──────────────────────                                         │
│  • Risk-adjusted utility (CRRA)                                 │
│  • Tail risk adjustment (EVT)                                   │
│  • Decision readiness classification                            │
│  • Final recommendation                                         │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     OUTPUT       │
                    │  (Decision +     │
                    │   Explanation)   │
                    └──────────────────┘
```

---

## Statistical Foundations

### 1. Diagnostic Test Theory

**Source:** Sackett et al. (2000) "Evidence-Based Medicine"

Evidence quality is derived from diagnostic test characteristics:

```
Likelihood Ratio Positive (LR+) = Sensitivity / (1 - Specificity)
Likelihood Ratio Negative (LR-) = (1 - Sensitivity) / Specificity
```

**Study Design Mapping:**

| Study Design | Sensitivity | Specificity | LR+ |
|--------------|-------------|-------------|-----|
| Meta-analysis | 0.92 | 0.90 | 9.2 |
| RCT | 0.85 | 0.85 | 5.7 |
| Cohort | 0.75 | 0.75 | 3.0 |
| Case-control | 0.70 | 0.70 | 2.3 |
| Case study | 0.55 | 0.55 | 1.2 |
| Anecdote | 0.52 | 0.52 | 1.1 |

**Rationale:** This approach grounds evidence quality in empirically-validated diagnostic accuracy rather than arbitrary weights.

---

### 2. Bayesian Updating

**Core Formula:**

```
Posterior Odds = Prior Odds × LR₁ × LR₂ × ... × LRₙ
```

In log-odds space (for numerical stability):

```
log(posterior odds) = log(prior odds) + Σ log(LRᵢ)
```

**Correlation Adjustment:**

For correlated evidence, the effective likelihood ratio is discounted:

```
LR_effective = LR^(1 - ρ)
```

Where ρ is the correlation between the new evidence and prior information.

---

### 3. Meta-Analytic Combination

**Method:** DerSimonian-Laird Random Effects

**Source:** DerSimonian & Laird (1986)

```
Step 1: Fixed-effect weights
        wᵢ = 1/σᵢ²

Step 2: Fixed-effect estimate  
        θ_FE = Σ(wᵢθᵢ) / Σwᵢ

Step 3: Q statistic (heterogeneity)
        Q = Σwᵢ(θᵢ - θ_FE)²

Step 4: Between-study variance (τ²)
        C = Σwᵢ - Σwᵢ²/Σwᵢ
        τ² = max(0, (Q - df) / C)

Step 5: Random-effects weights
        wᵢ* = 1/(σᵢ² + τ²)

Step 6: Random-effects estimate
        θ_RE = Σ(wᵢ*θᵢ) / Σwᵢ*

Step 7: I² (heterogeneity proportion)
        I² = max(0, (Q - df) / Q × 100%)
```

**Interpretation of I²:**

| I² Value | Heterogeneity |
|----------|---------------|
| 0-25% | Low |
| 25-50% | Moderate |
| 50-75% | Substantial |
| >75% | Considerable |

---

### 4. P-Curve Analysis

**Source:** Simonsohn, Nelson & Simmons (2014)

**Principle:** Under true effects, p-values cluster near 0 (right-skewed). Under p-hacking, they cluster near 0.05 (left-skewed).

```
Test: Binomial test on p-values below vs above α/2

If significantly more below α/2 → Evidential value present
If significantly more above α/2 → P-hacking suspected
If neither → Inconclusive
```

**Implementation:**
```python
threshold = α / 2  # typically 0.025
below = count(p < threshold for p in significant_p_values)
above = count(p ≥ threshold for p in significant_p_values)

# Binomial test against 50-50 null
```

---

### 5. Publication Bias Detection

**Method:** Trim and Fill

**Source:** Duval & Tweedie (2000)

Detects asymmetry in the funnel plot by:
1. Calculating the median effect
2. Counting studies above vs below median
3. Estimating missing studies from asymmetry
4. Adjusting pooled estimate

**Warning Threshold:** Asymmetry detected if missing > 20% of observed studies

---

### 6. Correlation-Adjusted Evidence Pooling

**Source:** Borenstein et al. (2009) "Introduction to Meta-Analysis"

**Effective Sample Size Formula:**

```
n_effective = n / (1 + (n-1) × ρ̄)
```

Where ρ̄ is the average pairwise correlation.

**Variance Combination for Correlated Evidence:**

```
Var(combined) = Σᵢσᵢ² + 2Σᵢ<ⱼρᵢⱼσᵢσⱼ
```

---

### 7. Reference Class Forecasting

**Source:** Kahneman & Tversky (1979), Kahneman & Lovallo (1993)

**Principle:** Anchor predictions on empirical base rates from similar cases (the "outside view").

**Empirical Base Rates Used:**

| Category | Base Rate | Source |
|----------|-----------|--------|
| Startup success (5yr) | 10% | Industry data |
| Drug approval (overall) | 10% | FDA statistics |
| Phase 2 trial success | 30% | Clinical trial databases |
| IT project on time | 30% | Standish CHAOS reports |
| M&A value creation | 30% | McKinsey research |
| Replication success | 40% | Reproducibility studies |
| Expert prediction accuracy | 50% | Tetlock's research |

---

### 8. Extreme Value Theory

**Source:** Pickands (1975), Balkema & de Haan (1974)

**Model:** Generalized Pareto Distribution (GPD)

```
F(x) = 1 - (1 + ξx/σ)^(-1/ξ)
```

**Parameters:**
- ξ (shape): Tail behavior
  - ξ > 0: Heavy tail (Pareto-like)
  - ξ = 0: Exponential tail  
  - ξ < 0: Bounded tail
- σ (scale): Spread of exceedances

**Risk Measures:**

```
Value-at-Risk (VaR):
  VaR_p = threshold + (σ/ξ)[(p × n)^(-ξ) - 1]

Conditional VaR (Expected Shortfall):
  CVaR_p = (VaR + σ - ξ×threshold) / (1 - ξ)
```

---

### 9. CRRA Utility

**Source:** Arrow (1965), Pratt (1964)

**Formula:**

```
U(x) = (x^(1-γ) - 1) / (1 - γ)    for γ ≠ 1
U(x) = ln(x)                       for γ = 1
```

Where γ is the coefficient of relative risk aversion.

**Domain Defaults:**

| Domain | γ (Risk Aversion) | Rationale |
|--------|-------------------|-----------|
| Medical | 2.5 | High stakes, conservative |
| Policy | 2.0 | Public impact |
| Business | 1.5 | Balanced |
| Scientific | 1.5 | Balanced |
| Technology | 1.0 | Innovation tolerance |

**Certainty Equivalent:**

```
CE = [(1-γ)E[U] + 1]^(1/(1-γ)) - 1
```

---

### 10. Instrumental Variable Strength Tests

**Source:** Staiger & Stock (1997), Stock & Yogo (2005)

**First-Stage F-Test:**

```
Rule of Thumb: F > 10 indicates strong instrument

F ≥ 10  → Strong (minimal bias)
5 ≤ F < 10 → Moderate (10-20% bias)
F < 5   → Weak (severe bias, possibly worse than OLS)
```

**Stock-Yogo Critical Values (10% maximal bias):**

| # Instruments | Critical Value |
|---------------|----------------|
| 1 | 16.38 |
| 2 | 8.96 |
| 3 | 6.66 |
| 4 | 5.53 |
| 5 | 4.77 |

---

### 11. Wilson Score Interval

**Source:** Wilson (1927) "Probable Inference"

**Formula:**

```
p̂ = successes / trials
z = z-score for confidence level

center = (p̂ + z²/2n) / (1 + z²/n)
margin = z × √[(p̂(1-p̂) + z²/4n) / n] / (1 + z²/n)

CI = [center - margin, center + margin]
```

**Advantage:** More accurate than normal approximation for extreme proportions or small samples.

---

## Component Reference

### Evidence Class

```python
@dataclass
class Evidence:
    # Identifiers
    id: str
    content: str
    source: str
    date: str
    domain: EvidenceDomain
    
    # Study characteristics
    study_design: str  # Maps to diagnostic metrics
    sample_size: int
    causal_level: CausalLevel  # Pearl's hierarchy
    
    # Statistical properties
    p_value: float           # For p-curve
    effect_size: float       # For meta-analysis
    effect_variance: float   # For meta-analysis
    f_statistic: float       # For IV studies
    
    # Computed
    likelihood_ratio: float      # From diagnostic metrics
    information_bits: float      # log₂(LR)
    effective_quality: float     # After penalties
```

### Hypothesis Class

```python
class Hypothesis:
    # Core
    name: str
    domain: EvidenceDomain
    prior: float              # From reference class
    
    # Foundation
    what: (description, confidence)
    why: (description, confidence)
    how: (description, confidence)
    feasibility: {technical, economic, timeline}
    
    # Evidence
    evidence: List[Evidence]
    evidence_correlation_matrix: np.ndarray
    
    # Analysis results
    updater: BayesianUpdater
    mechanism_map: MechanismMap
    p_curve_result: Dict
    meta_analysis_result: Dict
    tail_risk_result: Dict
```

### Bias Detector

Detects 8 cognitive biases:

| Bias | Detection Method |
|------|------------------|
| Confirmation | Evidence direction ratio |
| Anchoring | Prior-posterior movement vs information |
| Availability | Temporal distribution of evidence |
| Overconfidence | Posterior + CI width thresholds |
| Base Rate Neglect | Prior source check |
| Planning Fallacy | Timeline confidence + project keywords |
| Sunk Cost | Language pattern matching |
| Publication Bias | Meta-analysis asymmetry + p-curve |

### Content Scanner

100+ patterns across 6 categories:

| Category | Examples | Severity Range |
|----------|----------|----------------|
| Legal | "illegal", "regulatory gray area" | 0.6-0.95 |
| Safety | "fatal", "adverse event" | 0.7-0.95 |
| Ethical | "fraud", "selective reporting" | 0.6-0.9 |
| Financial | "bankruptcy", "liquidity constraint" | 0.5-0.9 |
| Privacy | "data breach", "GDPR violation" | 0.7-0.85 |
| Environmental | "EPA violation", "contamination" | 0.7-0.85 |

### Mechanism Map

Directed acyclic graph with cycle detection:

```python
class MechanismMap:
    nodes: Dict[str, MechanismNode]  # Causes, mechanisms, outcomes
    edges: List[MechanismEdge]       # Causal relationships
    
    # Tarjan's Algorithm for SCC detection
    def find_cycles_tarjan() -> List[List[str]]  # O(V+E)
    
    # Feedback loop classification
    def identify_feedback_loops() -> List[Dict]
```

---

## Mathematical Formulations

### Information Content

```
Information (bits) = log₂(LR)

Example:
  RCT with LR = 5.7 provides log₂(5.7) ≈ 2.5 bits
  Anecdote with LR = 1.1 provides log₂(1.1) ≈ 0.14 bits
```

### Credible Interval Approximation

Using effective sample size from accumulated bits:

```
n_effective = 2^(total_bits)

Wilson interval with n = n_effective
```

### Sensitivity Analysis

**Perturbation:**
```
For δ ∈ {-0.2, -0.1, +0.1, +0.2}:
    LR'ᵢ = LRᵢ × (1 + δ)
    Recalculate posterior
    
Stable if max|Δposterior| < 0.15
Critical if max|Δposterior| > 0.25
```

**Leave-One-Out:**
```
For each evidence eᵢ:
    posterior_without_i = Bayesian_update(evidence \ {eᵢ})
    impact_i = posterior - posterior_without_i
    
Influential if |impact| > 0.1
```

---

## Safety & Governance

### Hard Limits (Non-Negotiable)

```python
class SafetyLimits:
    MAX_LIKELIHOOD_RATIO = 100.0      # Prevent overconfidence
    MIN_LIKELIHOOD_RATIO = 0.01       # Prevent certainty
    CREDENCE_HARD_CAP = 0.99          # Never 100% certain
    CREDENCE_HARD_FLOOR = 0.01        # Never 0% certain
    MAX_LOG_ODDS = 10.0               # Numerical stability
    MAX_SINGLE_WEIGHT = 3.0           # Prevent dimension gaming
    MAX_WEIGHT_RATIO = 5.0            # Balance requirement
```

### Warning Levels

| Level | Icon | Behavior |
|-------|------|----------|
| FATAL | 💀 | **BLOCKS** analysis |
| CRITICAL | 🚨 | Prominent warning |
| WARNING | ⚠️ | Standard warning |
| INFO | ℹ️ | Informational |

### Blocking Conditions

Analysis is **blocked** (cannot proceed) when:

1. Fatal content detected (severity ≥ 0.9)
2. Weight violations exceed limits
3. Unacknowledged fatal warnings

### Decision Readiness States

```
READY         → High confidence, narrow CI, gates passed
REJECT        → Low probability, narrow CI
NEEDS_MORE_INFO → Wide CI, insufficient evidence
UNCERTAIN     → Moderate confidence
FATAL_FLAW    → Gates failed
BLOCKED       → Fatal issues detected
```

---

## Risk Assessment

### Safe For

- Business decisions up to $5M
- Clinical research (non-treatment)
- Policy analysis
- Strategic planning
- Team decision-making

### Use With Caution

- Medical treatment decisions (require expert review)
- Investments >$5M (require additional validation)
- Regulatory submissions (require human oversight)

### Not Recommended

- Fully autonomous life-critical systems
- Final medical approvals without physician review

---

## References

### Core Statistical Methods

1. **Sackett DL et al.** (2000). *Evidence-Based Medicine: How to Practice and Teach EBM*. Churchill Livingstone.

2. **Borenstein M et al.** (2009). *Introduction to Meta-Analysis*. Wiley.

3. **DerSimonian R, Laird N.** (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3):177-188.

4. **Simonsohn U, Nelson LD, Simmons JP.** (2014). P-curve: A key to the file-drawer. *Journal of Experimental Psychology: General*, 143(2):534-547.

### Decision Theory

5. **Arrow KJ.** (1965). Aspects of the Theory of Risk-Bearing. Yrjö Jahnsson Lectures.

6. **Pratt JW.** (1964). Risk aversion in the small and in the large. *Econometrica*, 32(1-2):122-136.

7. **Kahneman D, Tversky A.** (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2):263-291.

### Extreme Value Theory

8. **Pickands J.** (1975). Statistical inference using extreme order statistics. *Annals of Statistics*, 3(1):119-131.

9. **Balkema AA, de Haan L.** (1974). Residual life time at great age. *Annals of Probability*, 2(5):792-8

10. ### Instrumental Variables

10. **Staiger D, Stock JH.** (1997). Instrumental variables regression with weak instruments. *Econometrica*, 65(3):557-586.

11. **Stock JH, Yogo M.** (2005). Testing for weak instruments in linear IV regression. In *Identification and Inference for Econometric Models*. Cambridge University Press.

### Publication Bias

12. **Duval S, Tweedie R.** (2000). Trim and fill: A simple funnel-plot-based method of testing and adjusting for publication bias in meta-analysis. *Biometrics*, 56(2):455-463.

### Confidence Intervals

13. **Wilson EB.** (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158):209-212.

### Forecasting

14. **Tetlock PE.** (2005). *Expert Political Judgment: How Good Is It? How Can We Know?* Princeton University Press.

15. **Kahneman D, Lovallo D.** (1993). Timid choices and bold forecasts: A cognitive perspective on risk taking. *Management Science*, 39(1):17-31.

---

## Appendix: Pearl's Causal Hierarchy

PRISM uses Pearl's three-level causal hierarchy:

| Level | Name | Query | Evidence Type |
|-------|------|-------|---------------|
| 1 | Association | P(Y\|X) | Observational |
| 2 | Intervention | P(Y\|do(X)) | Experimental (RCT) |
| 3 | Counterfactual | P(Y_x\|X', Y') | Structural models |

**Enforcement:** Study designs are validated against claimed causal levels. A cohort study claiming INTERVENTION-level evidence is automatically downgraded to ASSOCIATION.

---

## Appendix: Tarjan's Algorithm

For cycle detection in mechanism maps:

```
TARJAN-SCC(G):
    index_counter = 0
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    for each node v in G:
        if v not in index:
            STRONGCONNECT(v)
    
    return sccs

STRONGCONNECT(v):
    index[v] = index_counter
    lowlinks[v] = index_counter
    index_counter++
    stack.push(v)
    on_stack[v] = true
    
    for each (v, w) in edges:
        if w not in index:
            STRONGCONNECT(w)
            lowlinks[v] = min(lowlinks[v], lowlinks[w])
        else if on_stack[w]:
            lowlinks[v] = min(lowlinks[v], index[w])
    
    if lowlinks[v] == index[v]:
        scc = []
        repeat:
            w = stack.pop()
            on_stack[w] = false
            scc.append(w)
        until w == v
        
        if |scc| > 1:
            sccs.append(scc)
```

**Complexity:** O(V + E)

---

*Document Version: 1.0*  
*Last Updated: December 2025*  
*PRISM Version: 2.0 Final*
