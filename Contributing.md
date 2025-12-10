# Contributing to PRISM

Thank you for your interest in PRISM (Protocol for Rigorous Investigation of Scientific Mechanisms). This project is maintained by Dr. Aneesh Joseph with a focus on high-fidelity reasoning and epistemic safety.

## Core Directives

When contributing code, adhere to the **P0 Safety Fixes** established in v1.0:

1.  **Do Not Relax Safety Limits:** The `SafetyLimits` class defines the boundaries of valid reasoning (e.g., `MAX_LIKELIHOOD_RATIO`, `MAX_EVIDENCE_BITS`). Pull requests that remove these caps without rigorous mathematical justification will be rejected.
2.  **Traceability:** All new logic must emit events to the `AnalysisTracer`. Hidden logic is not permitted.
3.  **Numerical Stability:** Any Bayesian update logic must utilize the `_clamp_values` safeguards to prevent floating-point errors.

## Pull Request Process

1.  Ensure your code adheres to Python 3.8+ standards.
2.  If you add new evidence types, update the `EvidenceDomain` enum.
3.  Run `prism_usage_examples.py` to verify that the core logic remains stable.
4.  Update the documentation to reflect any changes in the API.

## Code Style

* Use Type Hints (`typing`) for all function signatures.
* Use Data Classes (`@dataclass`) for structural elements.
* Document all public methods with docstrings.
* 
