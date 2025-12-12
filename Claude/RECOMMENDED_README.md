# PRISM v2.3
## Protocol for Rigorous Investigation of Scientific Mechanisms

**Author:** Dr. Aneesh Joseph  
**Status:** Production Ready  
**License:** MIT (or your choice)

---

## Overview

PRISM is a Bayesian framework for quantitative hypothesis evaluation that integrates multiple statistical methods to convert heterogeneous scientific evidence into calibrated probability estimates.

**Key Features:**
- Bayesian updating with reference class priors
- Hierarchical correlation correction
- REML meta-analysis with Hartung-Knapp adjustment
- P-curve publication bias detection
- Optimizer's curse correction for multiple comparisons
- Uncertainty decomposition (statistical + prior + model)

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
python examples/example_medical.py
```

### Requirements
- Python 3.8+
- NumPy
- SciPy
- (see requirements.txt for full list)

---

## Quick Start

```python
from prism_v23 import Hypothesis, Evidence, Domain

# Create hypothesis
h = Hypothesis(
    "Drug X reduces symptoms by >20%",
    Domain.MEDICAL,
    reference_class="phase2_clinical"
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
```

---

## Using PRISM with Claude AI

PRISM includes a Claude skill that enables AI-assisted hypothesis analysis.

### Setup (One-Time)

1. Download `claude/SKILL.md` from this repo
2. In a Claude chat, upload the file
3. Ask Claude: "Install this as a user skill for PRISM"
4. Done! Future Claude sessions will know how to use PRISM

### Usage with Claude

Simply ask:
```
"Use PRISM to find the best treatment for [condition]"
```

Claude will:
1. Search for relevant studies
2. Extract evidence
3. Run PRISM analysis
4. Present ranked results

**See `claude/README_CLAUDE.md` for detailed instructions.**

---

## Directory Structure

```
Prism/
├── prism_v23.py              # Core PRISM engine
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── claude/                   # Claude AI integration
│   ├── SKILL.md             # Skill file for Claude
│   ├── prism_session.py     # Session management
│   └── README_CLAUDE.md     # Claude usage guide
│
├── examples/                 # Example analyses
│   ├── example_medical.py
│   ├── example_business.py
│   └── example_policy.py
│
├── docs/                     # Documentation
│   ├── scientific_guide.md  # Mathematical framework
│   ├── user_guide.md        # Usage instructions
│   └── api_reference.md     # API documentation
│
└── tests/                    # Unit tests
    └── test_prism.py
```

---

## Documentation

- **[Scientific Guide](docs/scientific_guide.md)** - Mathematical framework and methodology
- **[User Guide](docs/user_guide.md)** - Detailed usage instructions
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Claude Integration](claude/README_CLAUDE.md)** - Using PRISM with Claude AI

---

## Examples

### Medical Research
```python
# Compare diabetes treatments
python examples/example_medical.py
```

### Business Decisions
```python
# Evaluate business strategies
python examples/example_business.py
```

### Policy Analysis
```python
# Compare policy interventions
python examples/example_policy.py
```

---

## Citation

If you use PRISM in your research, please cite:

```bibtex
@software{joseph2025prism,
  author = {Joseph, Aneesh},
  title = {PRISM: Protocol for Rigorous Investigation of Scientific Mechanisms},
  year = {2025},
  version = {2.3},
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

---

## Version History

### v2.3 (Development)
- Added causal inference capabilities
- Improved meta-analysis methods
- Enhanced uncertainty quantification

### v2.2 (Current)
- Hierarchical correlation correction
- Optimizer's curse adjustment
- P-curve publication bias detection
- Kalman filtering for temporal evidence

### v2.1
- Initial public release

---

## License

MIT License (or your preferred license)

---

## Contact

**Dr. Aneesh Joseph**  
GitHub: [@Dr-AneeshJoseph](https://github.com/Dr-AneeshJoseph)

---

## Acknowledgments

Claude AI integration developed in collaboration with Anthropic's Claude.
