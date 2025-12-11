# PRISM v1.0 - Quick Start Guide

## Files Included

| File | Size | Description |
|------|------|-------------|
| `prism_v1.py` | ~114KB | **Main framework** - Complete PRISM 1.0 implementation |
| `prism_usage_examples.py` | ~20KB | **Examples** - 6 comprehensive usage scenarios |
| `PRISM_README.md` | This file | Quick start guide |

## What Is PRISM?

**P**rotocol for **R**igorous **I**nvestigation of **S**cientific **M**echanisms

A hardened decision analysis framework that addresses all 25+ vulnerabilities identified in the red team analysis of Enhanced Analytical Protocol v2.0.

## Key Improvements Over v2.0

1. **Safety Limits** - Hard caps on iterations, evidence, complexity
2. **Warning System** - Proactive alerts for dangerous configurations  
3. **Numerical Stability** - Clamping with tracking and warnings
4. **Quality-First Causal Inference** - Strong cohort > weak RCT
5. **Evidence Independence Checking** - Detects redundancy
6. **Realistic VOI** - Accounts for costs, time, quality limits
7. **Risk-Aware Utility** - CRRA utility functions
8. **Content-Based Fatal Flaw Detection** - Scans for legal/safety issues
9. **Anti-Gaming Measures** - Weight reasonableness checks
10. **Improved Bias Detection** - Doesn't penalize established truths
11. **Calibration Cold-Start Handling** - Explicit uncertainty when uncalibrated

## Quick Start

```python
from prism_v1 import (
    AnalysisElement, Evidence, EvidenceDomain, CausalLevel,
    run_analysis, explain_result
)

# Create hypothesis
h = AnalysisElement(
    name="My Decision",
    domain=EvidenceDomain.BUSINESS
)

# Set foundation
h.set_what("What you're deciding", 0.85)
h.set_why("Why it matters", 0.7)
h.set_how("How it will work", 0.75)
h.set_measure("How you'll measure success", 0.6)

# Set feasibility
h.set_feasibility(technical=0.8, economic=0.7, timeline=0.65)

# Add evidence (with full metadata!)
h.add_evidence(Evidence(
    id="evidence_1",
    content="Description of evidence",
    source="Source name",
    quality=0.7,
    date="2024-01",
    domain=EvidenceDomain.BUSINESS,
    study_design="case_study",
    causal_level=CausalLevel.ASSOCIATION,
    supports_hypothesis=True,
    sample_size=100  # Important for proper weighting!
))

# Add scenarios with risk aversion
h.set_risk_aversion(1.0)  # 0=neutral, 1=moderate, 2+=high
h.add_scenario("Success", probability=0.5, utility=1.0)
h.add_scenario("Failure", probability=0.5, utility=-0.3)

# Run analysis
results = run_analysis(h, rigor_level=2, max_iter=10)

# Get human-readable explanation
print(explain_result(results))
```

## Running the Demo

```bash
python prism_v1.py
```

## Running All Examples

```bash
python prism_usage_examples.py
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `AnalysisElement` | Main container for hypothesis analysis |
| `Evidence` | Evidence with quality-first assessment |
| `MechanismNode/Edge` | Causal mechanism mapping |
| `RiskAwareUtilityModel` | Decision theory with risk aversion |
| `ImprovedBiasDetector` | Cognitive bias detection |
| `WarningSystem` | Safety alerts |
| `CalibrationTracker` | Historical prediction tracking |

## Safety Features

### Red Flags (System Will Warn)
- Credence > 90% (suspiciously high)
- All evidence supports hypothesis (possible confirmation bias)
- Evidence < 3 pieces (insufficient)
- Low evidence independence
- Content flags (legal, safety, ethical issues)
- Weight gaming attempts

### What To Do With Warnings
1. **INFO** - Consider, but proceed
2. **WARNING** - Investigate before proceeding
3. **CRITICAL** - Must address before decision
4. **FATAL** - Stop and review with humans

## Important Notes

1. **Evidence Independence**: Always provide `authors`, `cites`, `underlying_data` when known
2. **Established Facts**: Use `h.set_established_hypothesis(True)` for well-proven hypotheses
3. **VOI Calculations**: Always specify `info_cost`, `signal_accuracy`, `time_cost`
4. **Risk Aversion**: Set appropriately for your context (individuals ~1, orgs ~2)

## Dependencies

- Python 3.8+
- NumPy

```bash
pip install numpy
```

---

**Author**: Dr. Aneesh Joseph (Architecture) + Claude (Implementation)  
**Version**: 1.0  
**Date**: December 2025
