# PRISM v2.2
## Protocol for Rigorous Investigation of Scientific Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)](https://github.com/Dr-AneeshJoseph/Prism)
[![Claude AI](https://img.shields.io/badge/Optimized_For-Claude_Sonnet_4.5%2B-5A4FBB?logo=anthropic&logoColor=white)](https://www.anthropic.com)

---

**Author:** Dr. Aneesh Joseph  
**Status:** Production Ready  
**License:** MIT

---

## ⚠️ Disclaimer

**PRISM is a research and educational tool.** It is NOT intended for:
- Medical diagnosis or treatment decisions
- Legal advice or proceedings
- Financial investment decisions
- Regulatory approval submissions

PRISM provides **exploratory analysis** to aid human decision-making. Always consult qualified professionals for medical, legal, or financial decisions. The author assumes no liability for decisions made based on PRISM outputs.

---

## Overview

PRISM is a Bayesian framework for quantitative hypothesis evaluation that integrates multiple statistical methods to convert heterogeneous scientific evidence into calibrated probability estimates.

> "Kill the weak hypotheses so the truth may survive."

**Key Features:**
- Bayesian updating with reference class priors (with uncertainty quantification)
- Hierarchical correlation correction (addresses the "deadly product" problem)
- REML meta-analysis with Hartung-Knapp adjustment
- P-curve publication bias detection
- Optimizer's curse correction for multiple comparisons
- Uncertainty decomposition (statistical + prior + model)
- Kalman filtering for temporal evidence streams

---

## Installation

### Standard Python Usage

```bash
# Clone the repository
git clone https://github.com/Dr-AneeshJoseph/Prism.git
cd Prism

# Install dependencies
pip install -r requirements.txt

# Run example
python examples/example_osteoarthritis.py
```

### Requirements
- Python 3.8+
- NumPy
- SciPy

---

## Quick Start

```python
from prism_v2_2 import Hypothesis, Evidence, Domain

# Create hypothesis with reference class prior
h = Hypothesis(
    "Drug X reduces symptoms by >20%",
    Domain.MEDICAL,
    ref_class="phase2_clinical"  # 15% base rate from FDA data
)

# Add evidence
h.add_evidence(Evidence(
    id="rct_2024",
    content="RCT shows 25% reduction",
    source="NEJM 2024",
    domain=Domain.MEDICAL,
    study_design="rct",
    sample_size=200,
    supports=True,
    p_value=0.01,
    effect_size=-0.45,
    effect_var=0.0144,
    authors=["Smith"]
))

# Analyze
results = h.analyze()
print(f"Posterior: {results['posterior_bayes']:.1%}")
print(f"95% CI: [{results['ci_bayes'][0]:.1%}, {results['ci_bayes'][1]:.1%}]")
```

---

## Using PRISM with Claude AI

PRISM includes a Claude skill that enables AI-assisted hypothesis analysis with automatic evidence extraction from the web.

### Setup (One-Time)

1. Download `Claude/SKILL.md` from this repo
2. In a Claude chat, upload the file
3. Ask Claude: *"Install this as a user skill for PRISM"*
4. Done! Future Claude sessions will know how to use PRISM

### Usage with Claude

Simply ask:
```
"Use PRISM to find the best treatment for [condition]"
```

Claude will:
1. Search for relevant studies
2. Extract evidence automatically
3. Run PRISM analysis
4. Present ranked results with uncertainty

**See [`Claude/README_CLAUDE.md`](Claude/README_CLAUDE.md) for detailed instructions.**

**Mobile users: See [`Claude/MOBILE_QUICK_START.md`](Claude/MOBILE_QUICK_START.md) for quick setup.**

---

## Directory Structure

```
Prism/
├── prism_v2_2.py              # Core PRISM engine
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── THEORY_OF_OPERATION.md     # Scientific methodology
│
├── Claude/                    # Claude AI integration
│   ├── SKILL.md              # Skill file for Claude
│   ├── prism_session.py      # Session management
│   ├── README_CLAUDE.md      # Claude usage guide
│   └── MOBILE_QUICK_START.md # Mobile quick start
│
├── examples/                  # Example analyses
│   └── example_osteoarthritis.py
│
└── Versions/                  # Previous versions
    └── prism_v1.py
```

---

## Key Improvements in v2.2

### 1. Reference Class Priors with Uncertainty
Instead of point estimates, priors are now Beta distributions with credible intervals:
```python
# Phase 2 clinical: 15% [8%, 24%] based on FDA 2000-2020 data
h = Hypothesis("Drug efficacy", Domain.MEDICAL, ref_class="phase2_clinical")
```

### 2. Hierarchical Correlation Correction
Addresses the "deadly product" problem where naive LR multiplication gives overconfident results:
```python
# Automatically detects correlated evidence (same authors, similar methods)
# Applies design effect adjustment: LR^(1/√DEFF)
```

### 3. Optimizer's Curse Correction
When comparing multiple hypotheses, the "winner" is shrunk to account for selection bias:
```python
# Raw: 85% → Corrected (n=10 hypotheses): 71%
```

### 4. Enhanced Independence Detection
TF-IDF + author overlap + source overlap to detect redundant evidence.

---

## Available Reference Classes

| Reference Class | Prior | 95% CI | Source |
|----------------|-------|--------|--------|
| `phase2_clinical` | 15% | [8%, 24%] | FDA 2000-2020 |
| `phase3_clinical` | 35% | [26%, 45%] | FDA 2000-2020 |
| `drug_approval` | 10% | [5%, 18%] | FDA 2000-2020 |
| `startup_5yr` | 10% | [5%, 18%] | CB Insights |
| `replication` | 40% | [31%, 50%] | OSF 2015 |
| `general` | 50% | [32%, 68%] | Uninformative |

---

## Examples

### Medical Research
```bash
# Compare treatments - see examples/example_osteoarthritis.py
python examples/example_osteoarthritis.py
```

### With Claude AI
```
You: "Use PRISM to compare treatments for knee osteoarthritis"

Claude: [Searches literature, extracts evidence, runs analysis]

Result:
🏆 Combined weight loss + exercise: 84.2%
🥈 Exercise alone: 78.5%
🥉 Weight loss alone: 76.1%
...
```

---

## Citation

If you use PRISM in your research, please cite:

```bibtex
@software{joseph2025prism,
  author = {Joseph, Aneesh},
  title = {PRISM: Protocol for Rigorous Investigation of Scientific Mechanisms},
  year = {2025},
  version = {2.2},
  url = {https://github.com/Dr-AneeshJoseph/Prism}
}
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [Contributing.md](Contributing.md) for detailed guidelines.

---

## Version History

### v2.2 (Current)
- Reference class priors with Beta distribution uncertainty
- Hierarchical correlation correction
- Optimizer's curse adjustment
- P-curve publication bias detection
- Kalman filtering for temporal evidence
- Claude AI integration with skills

### v1.0
- Initial public release
- Basic 5-layer architecture
- Safety limits and kill switches

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Dr. Aneesh Joseph**  
GitHub: [@Dr-AneeshJoseph](https://github.com/Dr-AneeshJoseph)

---

## Acknowledgments

Claude AI integration developed in collaboration with Anthropic's Claude.

---

## Appropriate Use

### ✅ PRISM is appropriate for:
- Exploratory analysis and hypothesis generation
- Research prioritization and resource allocation
- Structured evidence synthesis
- Treatment comparison and decision support
- Teaching Bayesian reasoning

### ❌ PRISM is NOT appropriate for:
- Regulatory approval decisions (use established methods)
- Legal proceedings (requires validated forensic tools)
- Fully automated decision-making
- Single-study evaluation
- Medical diagnosis without professional oversight
