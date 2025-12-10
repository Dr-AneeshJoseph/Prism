PRISM v1.0: Protocol for Rigorous Investigation of Scientific Mechanisms
> "Kill the weak hypotheses so the truth may survive."
> 
PRISM is a hypothesis evolutionary engine designed specifically for high-fidelity reasoning models (Claude Sonnet 4.5 and above). It provides a hardened epistemic structure that forces models to rigorously weigh competing hypotheses in a research problem area, systematically "killing off" invalid paths using a layered analysis architecture.
Promptware Architect: Dr. Aneesh Joseph
🎯 Core Philosophy
Standard analytical tools often suffer from "Epistemic Drift"—they try to make every idea sound plausible. PRISM acts as a Sincerity Firewall.
It operates on three principles:
 * Hypothesis Selection: It creates a hierarchy of truth, identifying the single best explanation for a problem.
 * Fatal Flaw Detection: It employs a "Kill Switch." If a hypothesis triggers a fatal safety/legal warning or hits a feasibility floor, it is terminated immediately.
 * Layered Filtration: Analysis occurs in 5 distinct layers, moving from definition to decision.
🏗️ The 5-Layer Architecture
PRISM filters noise through a structured cognitive loop.
| Layer | Function | Mechanism |
|---|---|---|
| L0 | Characterization | Defines the "Epistemic Target." What strictly counts as a solution? |
| L1 | Mechanism Mapping | Maps causal nodes. Strong cohort studies outweigh weak RCTs (Quality-First Inference). |
| L2 | Adversarial Testing | The "Red Team" layer. Scans for 7 cognitive biases and content-based fatal flags. |
| L3 | Sensitivity Analysis | Tests numerical stability. "If we are wrong by 10%, does the hypothesis collapse?" |
| L4 | The Kill Switch | Gate Checks. Hypotheses with fatal flaws or safety violations are terminated here. |
| L5 | Selection | Synthesis of surviving branches using Risk-Aware Utility (CRRA). |
📂 Repository Contents
| File | Description |
|---|---|
| prism_v1.py | Core Framework. The complete logic engine containing AnalysisElement, Evidence, and SafetyLimits. |
| prism_usage_examples.py | Tutorials. 6 comprehensive scenarios including "Hypothesis Comparison" and "Evidence Independence". |
🚀 Quick Start
Installation
git clone https://github.com/DrAneeshJoseph/prism-v1.git
cd prism-v1
pip install numpy

Usage: The Hypothesis Battle
from prism_v1 import (
    AnalysisElement, Evidence, EvidenceDomain, CausalLevel,
    run_analysis, explain_result
)

# 1. Initialize the Hypothesis
h = AnalysisElement(name="Strategic Pivot", domain=EvidenceDomain.BUSINESS)

# 2. Set Foundation (The Logic Core)
h.set_what("Pivot to B2B Model", 0.85)
h.set_feasibility(technical=0.8, economic=0.7, timeline=0.65)

# 3. Add Evidence (The System Checks for Independence & Quality)
h.add_evidence(Evidence(
    id="market_data",
    content="Competitor analysis shows 40% growth in B2B sector",
    source="Industry Corp",
    quality=0.8,
    study_design="cohort",
    causal_level=CausalLevel.ASSOCIATION, # PRISM handles the weighting
    supports_hypothesis=True
))

# 4. Run the 5-Layer Analysis
# Rigor Level 2 activates the standard adversarial tester
results = run_analysis(h, rigor_level=2, max_iter=10)

# 5. View the Trace
print(explain_result(results))

🛡️ Safety & Governance
PRISM enforces a "Constitution" of reasoning via the SafetyLimits class.
The Warning System
The system will proactively alert or halt if it detects:
 * Echo Chambers: Low evidence independence (redundant citations/authors).
 * Hallucinated Certainty: Credence > 99% or infinite Log-Odds.
 * Cold Start: Calibration warnings when historical data is insufficient.
 * Content Flags: Keywords indicating legal, safety, or ethical violations.
Risk-Aware Utility (CRRA)
Unlike standard decision trees, PRISM models "Ruin." It uses Constant Relative Risk Aversion to penalize hypotheses that have a small chance of catastrophic failure, even if their "average" outcome is positive.
🤝 Contributing
This project prioritizes epistemic integrity. Please review CONTRIBUTING.md before submitting pull requests. Note that changes to SafetyLimits are heavily scrutinized to maintain the framework's rigorous standards.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
Copyright © 2025 Dr. Aneesh Joseph.
