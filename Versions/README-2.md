# PRISM v1.1: Protocol for Rigorous Investigation of Scientific Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production-green)](/Dr-AneeshJoseph/Prism/blob/main)
[![Claude AI](https://img.shields.io/badge/Optimized_For-Claude_Sonnet_4.5%2B-5A4FBB?logo=anthropic&logoColor=white)](https://www.anthropic.com)
[![Security](https://img.shields.io/badge/Security-A--Grade-brightgreen)](CHANGELOG.md)

---

> "Kill the weak hypotheses so the truth may survive."

PRISM is a hypothesis evolutionary engine designed specifically for high-fidelity reasoning models (Claude Sonnet 4.5 and above). It provides a hardened epistemic structure that forces models to rigorously weigh competing hypotheses in a research problem area, systematically "killing off" invalid paths using a layered analysis architecture.

**Promptware Architect:** Dr. Aneesh Joseph

---

## 🆕 What's New in v1.1

**Security Grade: B+ → A-** | **Risk Level: MEDIUM-HIGH → MEDIUM**

Version 1.1 addresses all critical vulnerabilities identified in the second-round red team analysis:

| Fix | Description | Impact |
|-----|-------------|--------|
| 🔒 **Semantic Independence** | Name normalization + entity extraction | Catches same study reported multiple ways |
| 🛡️ **100+ Content Patterns** | Euphemism detection for legal/safety/financial | No more "regulatory gray area" evasion |
| ✅ **Verified Establishment** | Requires evidence for "established" claims | Prevents bias check bypass |
| 📊 **Smart Warnings** | Deduplication + aggregation + blocking | Critical warnings no longer buried |
| 🎯 **Sample Size Validation** | Cross-checks claimed N vs content | Catches subgroup gaming |
| ⚖️ **Weight Enforcement** | Blocking mode (not just warnings) | Prevents dimension manipulation |
| 📈 **Risk Aversion Guidance** | Domain defaults + sensitivity display | Consistent risk handling |
| 🚫 **Fatal Flaw Blocking** | Analysis halts until resolved | No silent pass-through |

See [CHANGELOG.md](CHANGELOG.md) for complete details.

---

## 🎯 Core Philosophy

Standard analytical tools often suffer from "Epistemic Drift"—they try to make every idea sound plausible. PRISM acts as a **Sincerity Firewall**.

It operates on three principles:

1. **Hypothesis Selection:** Creates a hierarchy of truth, identifying the single best explanation
2. **Fatal Flaw Detection:** Employs a "Kill Switch" for safety/legal violations or feasibility failures
3. **Layered Filtration:** Analysis occurs in 5 distinct layers, moving from definition to decision

---

## 🏗️ The 5-Layer Architecture

PRISM filters noise through a structured cognitive loop:

| Layer | Function | Mechanism |
|-------|----------|-----------|
| **L0** | Characterization | Defines the "Epistemic Target." What strictly counts as a solution? |
| **L0.5** | Pre-flight Safety | Checks evidence sufficiency, independence, and fatal content |
| **L1** | Mechanism Mapping | Maps causal nodes. Quality-First Inference weights evidence properly |
| **L2** | Adversarial Testing | The "Red Team" layer. Scans for 7 cognitive biases and content flags |
| **L3** | Sensitivity Analysis | Tests numerical stability with ±10% and ±20% perturbations |
| **L4** | The Kill Switch | Gate Checks. Hypotheses with fatal flaws are **blocked** here |
| **L5** | Selection | Synthesis of surviving branches using Risk-Aware Utility (CRRA) |

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `prism_v1_1.py` | **Core Framework v1.1.** Complete logic engine with all security fixes |
| `prism_v1.py` | Legacy v1.0 (kept for backwards compatibility) |
| `prism_usage_examples.py` | Tutorials. 6 comprehensive scenarios |
| `prism_v1_1_fix_tests.py` | Security verification test suite |
| `CHANGELOG.md` | Detailed documentation of all v1.1 fixes |
| `THEORY_OF_OPERATION.md` | Deep dive into PRISM's design philosophy |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Dr-AneeshJoseph/Prism.git
cd Prism
pip install numpy
```

### Usage: The Hypothesis Battle

```python
from prism_v1_1 import (
    AnalysisElement, Evidence, EvidenceDomain, CausalLevel,
    run_analysis, explain_result, EstablishedHypothesisEvidence
)

# 1. Initialize the Hypothesis
h = AnalysisElement(name="Strategic Pivot", domain=EvidenceDomain.BUSINESS)

# 2. Set Foundation (The Logic Core)
h.set_what("Pivot to B2B Model", 0.85)
h.set_why("Market opportunity in enterprise segment", 0.8)
h.set_how("Leverage existing tech stack with new sales motion", 0.7)
h.set_feasibility(technical=0.8, economic=0.7, timeline=0.65)

# 3. Add Evidence (System checks independence, quality, and content)
h.add_evidence(Evidence(
    id="market_data",
    content="Competitor analysis shows 40% growth in B2B sector",
    source="Industry Report 2024",
    quality=0.8,
    date="2024-06",
    domain=EvidenceDomain.BUSINESS,
    study_design="benchmark",
    sample_size=500,
    causal_level=CausalLevel.ASSOCIATION,
    supports_hypothesis=True,
    authors=["Smith, J.", "Jones, M."]
))

# 4. Add scenarios for utility calculation
h.add_scenario("Strong adoption", 0.3, 1.5)
h.add_scenario("Moderate success", 0.4, 0.4)
h.add_scenario("Slow growth", 0.2, -0.1)
h.add_scenario("Pivot fails", 0.1, -0.8)

# 5. Run the 5-Layer Analysis
results = run_analysis(h, rigor_level=2, max_iter=10)

# 6. View the Results
print(explain_result(results))
```

### Verify Security Fixes

```bash
python prism_v1_1_fix_tests.py
```

Expected output:
```
✅ Passed: 8/8
🎉 ALL P0 AND P1 FIXES VERIFIED!
```

---

## 🛡️ Safety & Governance

PRISM enforces a "Constitution" of reasoning via the `SafetyLimits` class.

### The Warning System (v1.1 Enhanced)

The system proactively alerts or **blocks** if it detects:

| Issue | v1.0 Behavior | v1.1 Behavior |
|-------|---------------|---------------|
| 🔴 Fatal Content | Warning only | **BLOCKS analysis** |
| 🔴 Weight Violations | Warning only | **BLOCKS analysis** |
| 🟠 Low Independence | Individual warnings | Aggregated summary |
| 🟠 High Credence | Warning at 95% | Warning + explanation |
| 🟡 Calibration Cold Start | Silent | Explicit tracking |

### Blocking vs Warning

```python
# v1.1 blocking behavior
results = run_analysis(h)

if results.get('blocked'):
    print("Analysis BLOCKED:", results['blocking_reasons'])
    # Must resolve issues before proceeding
else:
    print("Decision:", results['recommendation'])
```

### Risk-Aware Utility (CRRA)

Unlike standard decision trees, PRISM models "Ruin." It uses Constant Relative Risk Aversion to penalize hypotheses with catastrophic failure risk.

**v1.1 Enhancement:** Domain-specific defaults

| Domain | Default γ | Rationale |
|--------|-----------|-----------|
| Medical | 2.5 | Higher risk aversion for health decisions |
| Business | 1.5 | Balanced approach |
| Technology | 1.0 | More risk tolerance for innovation |
| Policy | 2.0 | Conservative for public impact |

---

## 📊 Risk Assessment

### Current State (PRISM v1.1 - Grade A-)

**Safe For:**
- Business decisions up to $5M
- Clinical research (non-treatment)
- Policy analysis
- Strategic planning
- Team decision-making

**Use With Caution:**
- Medical treatment decisions (require expert review)
- Investments >$5M (require additional validation)
- Regulatory submissions (require human oversight)

**Not Recommended:**
- Fully autonomous life-critical systems
- Final medical approvals without physician review

---

## 🔧 Key v1.1 Features

### 1. Semantic Independence Checking

```python
# v1.0: These would be counted as independent
e1 = Evidence(..., authors=["Smith, J."])
e2 = Evidence(..., authors=["Dr. Jane Smith"])  # Same person!

# v1.1: Detected as related (60% independence penalty)
report = h.check_evidence_independence()
print(f"Effective evidence: {report['effective_evidence_count']}")
```

### 2. Established Hypothesis Verification

```python
# v1.0: Could bypass bias checks
h.set_established_hypothesis(True)  # No verification!

# v1.1: Requires evidence
evidence = EstablishedHypothesisEvidence(
    claim="Aspirin reduces inflammation",
    supporting_references=["Vane 1971"],
    meta_analyses_cited=3,
    expert_consensus=True
)
h.set_established_hypothesis(True, evidence)  # Verified ✓
```

### 3. Content Scanner with Euphemism Detection

```python
# v1.0: Missed euphemisms
"regulatory gray area"      # NOT detected
"adverse events of concern" # NOT detected

# v1.1: 100+ patterns including euphemisms
"regulatory gray area"      # DETECTED (legal, severity=0.7)
"adverse events of concern" # DETECTED (safety, severity=0.8)
```

### 4. Smart Warning Aggregation

```python
# v1.0: 45 individual warnings
# v1.1: Aggregated summary
print(h.warning_system.get_summary_header())
# Output: "🚨 CRITICAL: 2 | ⚠️ WARNING: 5 | ℹ️ INFO: 3"
```

---

## 🤝 Contributing

This project prioritizes epistemic integrity. Please review [Contributing.md](Contributing.md) before submitting pull requests.

**Important:** Changes to `SafetyLimits` are heavily scrutinized to maintain the framework's rigorous standards.

### Running Tests

```bash
# Run security verification
python prism_v1_1_fix_tests.py

# Run usage examples
python prism_usage_examples.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2025 Dr. Aneesh Joseph

---

## 📈 Version History

| Version | Grade | Key Changes |
|---------|-------|-------------|
| v1.1 | A- | All P0/P1 security fixes, blocking mode, semantic analysis |
| v1.0 | B+ | Initial release, fixed 18 vulnerabilities from v2.0 prototype |

---

## 🙏 Acknowledgments

- Red Team Analysis contributors for identifying vulnerabilities
- Claude AI team at Anthropic for the reasoning model
- All contributors and testers

---

**Current Version:** 1.1  
**Security Grade:** A-  
**Risk Level:** MEDIUM  
**Last Updated:** December 2025
