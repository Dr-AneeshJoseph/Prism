# PRISM v2.2 - Protocol for Rigorous Investigation of Scientific Mechanisms

**Author:** Dr. Aneesh Joseph  
**Implementation:** Claude (Anthropic)  
**Version:** 2.2 | December 2025

---

## When to Use This Skill

Use PRISM when the user asks to:
- Evaluate scientific hypotheses quantitatively
- Compare multiple hypotheses using Bayesian methods
- Perform rigorous evidence synthesis with meta-analysis
- Calculate posterior probabilities from heterogeneous evidence
- Research and prioritize scientific questions
- Assess treatment options, drug efficacy, or technology predictions

**Trigger phrases:**
- "Find the best hypothesis for..."
- "Compare these hypotheses..."
- "What's the probability that..."
- "Analyze the evidence for..."
- "Which treatment is most likely to work..."

---

## Core Philosophy

Traditional hypothesis testing asks: *"Is this result statistically significant?"*

PRISM asks: *"Given all available evidence, what is the probability this hypothesis is true?"*

This shift from binary significance to continuous credence enables:
- Comparison across different evidence types
- Principled evidence accumulation
- Explicit uncertainty quantification
- Optimizer's curse correction for multiple comparisons

---

## Mathematical Framework

### Bayesian Foundation
PRISM uses **Bayes' theorem** with log-odds for numerical stability:

```
log-odds(H|E) = log-odds(H) + log(LR)
```

### Reference Class Priors
Instead of arbitrary priors, PRISM uses **empirical base rates** with Beta distributions:
- `phase2_clinical`: 15.3% [9.0%, 23.0%] (based on FDA 2000-2020)
- `phase3_clinical`: 35.1% [26.2%, 44.7%]
- `replication`: 40.1% [30.8%, 49.8%] (based on OSF 2015)
- `general`: 50.0% (uninformative)

### Hierarchical Correlation Correction
**Critical**: Naive multiplication of likelihood ratios causes exponential overconfidence.

PRISM uses hierarchical correlation:
- Within-cluster ρ = 0.6 (same lab/method)  
- Between-cluster ρ = 0.2 (independent labs)

```
Effective N = Σ(cluster_size / DEFF_within) / √DEFF_between
```

### Key Statistical Methods
1. **REML Meta-Analysis** with Hartung-Knapp adjustment
2. **P-Curve Analysis** for publication bias detection
3. **Kalman Filtering** for temporal evidence integration
4. **Optimizer's Curse Correction** when comparing multiple hypotheses
5. **Sobol Sensitivity Analysis** to identify critical evidence

---

## Implementation Guide

### File Structure
```
/home/claude/
├── prism_v2_2.py          # Core PRISM engine (Dr. Joseph's implementation)
├── prism_session.py       # Session management with checkpointing
└── example_*.py           # Example analyses

/mnt/user-data/outputs/prism_{project}/
├── state.json             # Project state + resume instructions
├── RESUME.md              # Human-readable resume point
├── hypotheses/            # Hypothesis data files
│   ├── h1_*.json
│   └── h2_*.json
└── results/
    ├── comparison.json    # Comparative analysis
    ├── FINAL_REPORT.md    # Complete report
    └── *_results.json     # Individual hypothesis results
```

### Basic Usage Pattern

```python
from prism_session import PRISMSession
from prism_v2_2 import Evidence, Domain

# 1. Create session
session = PRISMSession("project_name")

# 2. Add hypotheses
h1 = session.add_hypothesis(
    hypothesis_id="h1_treatment_a",
    title="Treatment A reduces symptoms by >20%",
    domain=Domain.MEDICAL,
    reference_class="phase2_clinical"
)

# 3. Add evidence (from web_search, papers, etc.)
h1.add_evidence(Evidence(
    id="study1",
    content="RCT shows 25% reduction",
    source="NEJM 2024",
    domain=Domain.MEDICAL,
    study_design="rct",
    sample_size=200,
    supports=True,
    p_value=0.01,
    effect_size=-0.45,
    effect_var=0.0144,  # SE^2
    authors=["Smith"]
))

# Update the saved hypothesis
session._save_hypothesis(h1, 
    session.hypotheses_dir / "h1_treatment_a.json",
    "h1_treatment_a")

# 4. Analyze all hypotheses
session.analyze_all(set_n_compared=True)  # Applies optimizer's curse

# 5. Generate report
session.generate_report()
```

### Workflow for Real Research Questions

When a user asks: **"Find the best hypothesis for treating osteoarthritis"**

1. **Search for hypotheses:**
```python
# Use web_search to find treatment options
web_search("osteoarthritis treatment options 2024 clinical trials")
web_search("osteoarthritis systematic reviews meta-analysis")
```

2. **Create session and define hypotheses:**
```python
session = PRISMSession("osteoarthritis_treatment_2025")

# Add each candidate hypothesis
h1 = session.add_hypothesis(
    "h1_weight_loss",
    "Weight loss reduces knee OA pain",
    Domain.MEDICAL,
    "replication"
)
# ... add more hypotheses
```

3. **Extract evidence from search results:**
```python
# For each relevant study found:
h1.add_evidence(Evidence(
    id="messier_2013",
    content="18-month RCT: 10% weight loss reduced pain 50%",
    source="JAMA 2013;310(12):1263",
    study_design="rct",
    sample_size=454,
    supports=True,
    p_value=0.0001,
    effect_size=-0.48,
    effect_var=0.0144,
    authors=["Messier", "Mihalko"]
))
```

4. **Run complete analysis:**
```python
session.analyze_all(set_n_compared=True)
```

5. **Present results:**
```python
session.generate_report()
present_files([
    session.results_dir / "FINAL_REPORT.md",
    session.results_dir / "comparison.json"
])
```

---

## Checkpointing and Resumability

### Design Principle
**"Build for one Claude, checkpoint for many"**

The system is designed to complete in ONE session but checkpoints after each hypothesis for safety.

### Automatic Checkpointing
After analyzing each hypothesis, PRISM automatically:
1. Saves results to JSON
2. Updates state.json
3. Writes RESUME.md with instructions
4. Estimates tokens used

### Resume Protocol
If analysis is interrupted:

```python
from prism_session import PRISMSession

# Load existing project
session = PRISMSession("project_name")  # Automatically loads state

# Continue analysis
session.resume()  # Or session.analyze_all()
```

The RESUME.md file tells you exactly where you left off:
- Which hypotheses are completed
- Which are pending
- Python code to continue

---

## Critical Implementation Rules

1. **ALWAYS use web_search** when user asks about real-world hypotheses
   - Search for recent studies, clinical trials, systematic reviews
   - Extract evidence from authoritative sources
   - Don't fabricate studies or data

2. **ALWAYS checkpoint after each hypothesis**
   - Use `session._checkpoint_hypothesis()` automatically called
   - Saves to `/mnt/user-data/outputs/prism_{project}/`

3. **ALWAYS apply optimizer's curse correction** when comparing >1 hypothesis
   - Set `n_compared` on each hypothesis
   - Or use `session.analyze_all(set_n_compared=True)`

4. **ALWAYS use hierarchical correlation**
   - Evidence is automatically clustered by (author, study_design, source)
   - within_rho=0.6, between_rho=0.2

5. **ALWAYS decompose uncertainty**
   - Statistical (from CI width)
   - Prior (from reference class uncertainty)
   - Model (from correlation assumptions)

6. **Token awareness:**
   - Typical analysis: 5-10K tokens per hypothesis
   - 8 hypotheses ≈ 40-80K tokens
   - Well within 190K budget for most analyses

---

## Study Design Strength

Evidence strength by study design (likelihood ratios):

| Study Type | LR+ | Interpretation |
|------------|-----|----------------|
| Meta-analysis | 5.4 | Very strong evidence |
| Systematic review | 5.4 | Very strong evidence |
| RCT | 2.3 | Strong evidence |
| Cohort | 1.9 | Moderate evidence |
| Case-control | 1.5 | Weak evidence |
| Observational | 1.2 | Minimal evidence |
| Expert opinion | 1.1 | Very weak evidence |

---

## Interpretation Guidelines

### Posterior Probability Scale

| Posterior | Interpretation | Action |
|-----------|----------------|--------|
| < 10% | Unlikely | Deprioritize |
| 10-30% | Possible | Gather more evidence |
| 30-70% | Uncertain | Key decision point |
| 70-90% | Probable | Consider acting |
| > 90% | Highly likely | Act with monitoring |

### Warning Signs

⚠️ **High model uncertainty (>25%)**: Evidence may be highly correlated  
⚠️ **P-hacking detected**: Effect sizes may be inflated  
⚠️ **High I² (>75%)**: Studies measuring different things  
⚠️ **Extreme posterior (>95%)**: Likely overconfident  
⚠️ **Wide credible intervals**: Need more evidence

---

## Example Output Format

When presenting results to users:

```
🏆 Best Hypothesis: h1_weight_loss
   Weight loss reduces knee OA pain and improves function
   Posterior (corrected): 88.8%

📋 Full Ranking:
   🥇 h1_weight_loss: 88.8%
   🥈 h2_exercise: 88.5%
   🥉 h3_combination: 67.2%
   ...

📄 Full report available: [link to FINAL_REPORT.md]
```

---

## Limitations

### What PRISM Cannot Do
1. Replace domain expertise - requires judgment for prior selection
2. Detect fraud - assumes evidence is honestly reported
3. Handle unknown unknowns - only evaluates provided evidence
4. Guarantee calibration - model assumptions may be wrong

### Appropriate Use
✅ Exploratory analysis and hypothesis generation  
✅ Research prioritization and resource allocation  
✅ Structured evidence synthesis  
✅ Treatment comparison and decision support  
✅ Teaching Bayesian reasoning

### Inappropriate Use
❌ Regulatory approval decisions (use established methods)  
❌ Legal proceedings (requires validated forensic tools)  
❌ Fully automated decision-making  
❌ Single-study evaluation

---

## Quick Reference

### Available Reference Classes
```python
'phase2_clinical'  # ~15% prior (early drug development)
'phase3_clinical'  # ~35% prior (late drug development)
'drug_approval'    # ~10% prior (FDA approval standard)
'replication'      # ~40% prior (scientific replication)
'general'          # ~50% prior (uninformative)
```

### Key Functions
```python
# Session management
session = PRISMSession("project_name")
h = session.add_hypothesis(id, title, domain, ref_class)
session.analyze_all()
session.generate_report()

# Evidence creation
e = Evidence(id, content, source, domain, study_design,
             sample_size, supports, p_value, effect_size, 
             effect_var, authors)
```

---

## References

1. Gelman et al. (2013) *Bayesian Data Analysis*
2. Patterson & Thompson (1971) *Biometrika* - REML
3. Hartung & Knapp (2001) *Statistics in Medicine*
4. Simonsohn et al. (2014) *J Exp Psych: General* - P-curve
5. Kahneman & Tversky (1979) - Reference class forecasting
6. Dr. Aneesh Joseph (2025) *PRISM v2.2 Scientific Guide*

---

**PRISM v2.2 - Rigorous hypothesis evaluation for evidence-based science**
