# Contributing to PRISM

Thank you for your interest in PRISM (Protocol for Rigorous Investigation of Scientific Mechanisms). This project is maintained by Dr. Aneesh Joseph with a focus on high-fidelity reasoning and epistemic safety.

---

## Core Principles

PRISM v2.2 prioritizes **epistemic integrity** over convenience. When contributing, keep these principles in mind:

1. **Calibration over Confidence:** Never trade accuracy for apparent precision
2. **Transparency:** All assumptions must be explicit and documented
3. **Safety:** Numerical safeguards prevent overconfident conclusions
4. **Reproducibility:** Results should be deterministic given the same inputs

---

## Safety-Critical Code

The following components are **safety-critical** and require extra scrutiny:

### Protected Constants (in `prism_v2_2.py`)

```python
class L:  # Limits
    MAX_ITER = 100
    MAX_EVIDENCE = 500
    MAX_LR = 100.0
    MIN_LR = 0.01
    MAX_LOG_ODDS = 10.0
    MIN_LOG_ODDS = -10.0
    CRED_CAP = 0.99
    CRED_FLOOR = 0.01
```

**Pull requests that relax these limits will be rejected** unless accompanied by:
- Mathematical proof of safety
- Extensive testing across edge cases
- Clear documentation of implications

### Protected Classes

1. **`RefClassPrior`** - Reference class priors with Beta distributions
2. **`HierarchicalCorr`** - Correlation correction parameters
3. **`OptimizerCurse`** - Selection bias correction

---

## Pull Request Process

### Before Submitting

1. ✅ Ensure code adheres to Python 3.8+ standards
2. ✅ Add type hints (`typing`) for all function signatures
3. ✅ Use dataclasses (`@dataclass`) for structural elements
4. ✅ Document all public methods with docstrings
5. ✅ Run the example to verify functionality:
   ```bash
   python examples/example_osteoarthritis.py
   ```

### PR Categories

| Category | Review Level | Examples |
|----------|--------------|----------|
| 🟢 Documentation | Standard | README updates, typos, comments |
| 🟡 Features | Thorough | New study types, new domains |
| 🔴 Safety-Critical | Intensive | Limit changes, prior modifications |

### Submission Checklist

- [ ] Code follows existing style conventions
- [ ] New features have corresponding tests
- [ ] Documentation is updated
- [ ] No changes to safety limits without justification
- [ ] Example runs without errors

---

## Code Style

### Python Conventions

```python
# Good: Type hints, docstrings, dataclasses
@dataclass
class Evidence:
    """Evidence with statistical characterization."""
    id: str
    content: str
    source: str
    domain: Domain
    study_design: str = "observational"
    sample_size: Optional[int] = None

# Good: Clear function signatures
def analyze(self, rigor_level: int = 2) -> Dict[str, Any]:
    """
    Run complete PRISM analysis.
    
    Args:
        rigor_level: Analysis depth (1-3)
        
    Returns:
        Dictionary with posterior estimates and diagnostics
    """
```

### Naming Conventions

- **Classes:** `PascalCase` (e.g., `Hypothesis`, `RefClassPrior`)
- **Functions:** `snake_case` (e.g., `add_evidence`, `analyze_all`)
- **Constants:** `UPPER_CASE` (e.g., `MAX_LR`, `CRED_CAP`)
- **Enums:** `PascalCase` with `UPPER_CASE` values

---

## Adding New Features

### New Study Design Types

To add a new study design:

1. Add to `STUDY_METRICS` in `prism_v2_2.py`:
   ```python
   STUDY_METRICS = {
       # ... existing types ...
       'new_design': DiagMetrics(sens_a, sens_b, spec_a, spec_b),
   }
   ```

2. Document the sensitivity/specificity basis
3. Add corresponding test case

### New Reference Classes

To add a new reference class prior:

1. Provide **empirical data** (success rate from historical sample)
2. Add to `REF_PRIORS`:
   ```python
   REF_PRIORS = {
       # ... existing priors ...
       'new_class': RefClassPrior.from_data('new_class', successes, total, 'Source'),
   }
   ```

3. Include citation for the source data

### New Domains

To add a new analysis domain:

1. Add to `Domain` enum:
   ```python
   class Domain(Enum):
       # ... existing domains ...
       NEW_DOMAIN = "new_domain"
   ```

2. Update any domain-specific logic if needed

---

## Testing

### Running Tests

```bash
# Run the built-in test
python prism_v2_2.py

# Run the example analysis
python examples/example_osteoarthritis.py
```

### What to Test

- Edge cases (empty evidence, single study, extreme values)
- Numerical stability (very large/small likelihood ratios)
- Consistency (same inputs → same outputs)

---

## Claude Integration

When modifying Claude-related files (`Claude/` directory):

1. **Keep SKILL.md synchronized** with `prism_v2_2.py`
2. **Update version numbers** in both files
3. **Test with Claude** before committing
4. **Update README_CLAUDE.md** with any new features

---

## Documentation

### Required Documentation

- All public functions need docstrings
- Complex algorithms need inline comments
- API changes need README updates
- Mathematical methods need THEORY_OF_OPERATION updates

### Mathematical Notation

Use LaTeX-style notation in documentation:
```markdown
$$\text{Posterior odds} = \text{Prior odds} \times LR$$
```

---

## Reporting Issues

### Bug Reports

Please include:
- Python version and OS
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Please describe:
- Use case and motivation
- Proposed solution
- Alternative approaches considered
- Impact on existing functionality

---

## Code of Conduct

1. **Be respectful** in all interactions
2. **Focus on the science** - debates should be evidence-based
3. **Acknowledge uncertainty** - PRISM is about honest assessment
4. **Credit appropriately** - cite sources and collaborators

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Contact

- **Author:** Dr. Aneesh Joseph
- **GitHub:** [@Dr-AneeshJoseph](https://github.com/Dr-AneeshJoseph)
- **Issues:** [GitHub Issues](https://github.com/Dr-AneeshJoseph/Prism/issues)

---

**Thank you for helping make PRISM better!**
