# PRISM v1.0: Protocol for Rigorous Investigation of Scientific Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Stable-green)]()

**PRISM** is a hardened decision analysis framework designed specifically for Claude AI to compare competing research Hypothesis.
it prevents epistemic failure modes in automated reasoning. It addresses vulnerabilities found in standard analytical protocols by enforcing strict safety limits, numerical stability, and risk-aware utility modeling.

> **Promptware Architect:** Dr. Aneesh Joseph  
> **Version:** 1.0 (Lucid Sovereign)

---

## 🛡️ Core Philosophy

PRISM is not just a decision tree; it is a governance layer for thought. It operates on the principle of **Epistemic Integrity over Speed**.

1.  **Safety First:** Hard caps on confidence intervals and likelihood ratios prevent "hallucinated certainty".
2.  **Quality Dominance:** A large, high-quality Cohort study outweighs a small, flawed RCT. We use a "Base Quality + Causal Boost" logic.
3.  **Echo Chamber Detection:** The system mathematically penalizes redundant evidence (shared citations, authors, or datasets).
4.  **Ruin Aversion:** Utility modeling uses CRRA (Constant Relative Risk Aversion) to correctly weigh catastrophic risks.

## 📂 Files Included

| File | Description |
|------|-------------|
| `prism_v1.py` | **Core Framework.** The complete logic engine containing `AnalysisElement`, `Evidence`, and `SafetyLimits`. |
| `prism_usage_examples.py` | **Tutorials.** 6 comprehensive scenarios including Hypothesis Comparison and VOI analysis. |

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* NumPy

### Installation

```bash
git clone [https://github.com/YourUsername/prism-v1.git](https://github.com/YourUsername/prism-v1.git)
cd prism-v1
pip install -r requirements.txt
