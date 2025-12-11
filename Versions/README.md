# Analytical Protocol

A personal tool for systematic thinking with Claude.

## What This Is

A framework for thinking through decisions and problems that:
- Forces **numerical confidence** (honest about uncertainty)
- **Iterates systematically** (keep finding flaws until solid)
- **Integrates evidence** properly (weighted by quality)
- **Computes quality scores** (know when you're done)
- **Incorporates proven techniques** (pre-mortem, red teaming, etc.)

Not claiming this is revolutionary or proven. Just that it helps me think more clearly.

---

## Files

1. **ANALYTICAL_PROTOCOL.md** - Full framework with all details
2. **analytical_protocol.py** - Python implementation for computational mode
3. **QUICK_REFERENCE.md** - One-page cheat sheet
4. **EXAMPLES.md** - Real analyses showing how to use it
5. This README

---

## Quick Start

### With Claude (Conversational)

In any Claude conversation:
```
"Claude, help me think through [MY DECISION] using the analytical protocol.
Use standard rigor, iterate until quality ≥ 0.7"
```

Claude knows this framework and will guide you through:
- Foundation questions (WHAT/WHY/HOW/MEASURE)
- Evidence assessment
- Adversarial testing (pre-mortem, assumptions, alternatives)
- Quality scoring
- When to stop

### With Python (Computational)

```python
from analytical_protocol import FoundationElement, run_analysis, Evidence

# Create your analysis
element = FoundationElement(name="Should I take this job?")
element.what = "Senior Engineer at StartupX, $150K, remote"
element.what_confidence = 0.9

element.why = "Better growth and compensation"
element.why_confidence = 0.6

# Add evidence
element.add_evidence(Evidence(
    content="Glassdoor: 3.8/5, growing fast",
    source="Glassdoor + TechCrunch",
    quality=0.5,
    date="2024"
))

# Run analysis
results = run_analysis(element, rigor_level=2)

print(f"Quality: {results['quality']:.2f}")
print(f"Ready to decide: {results['ready']}")
print(f"Gaps: {results['gaps']}")
```

---

## Core Concepts

### 1. Numerical Confidence (0-1 scale)

Instead of "pretty sure" say "0.75 confident"

Forces honesty about uncertainty:
- 0.9+ = Very confident (rare!)
- 0.7-0.9 = Reasonably confident
- 0.5-0.7 = Uncertain
- <0.5 = Very uncertain

### 2. Iterative Refinement

Don't do pre-mortem once. Do it until you stop finding problems.

The cycle:
1. Build foundation
2. Generate criticisms (pre-mortem, assumption tests, red team)
3. Address criticisms
4. Calculate quality score
5. If quality < target, go to step 2
6. When quality ≥ target AND completeness ≥ 0.5, stop

### 3. Evidence Integration

Not all evidence is equal:
- Meta-analysis (1.0) >> Expert opinion (0.3) >> Anecdote (0.15)
- Many weak sources ≠ one strong source (diminishing returns)
- Domain-specific hierarchies (medical ≠ business)

### 4. Quality Scoring

Tells you when analysis is "good enough":

```
Overall = 0.30×Completeness + 0.20×Confidence 
        + 0.20×Evidence + 0.20×Consistency + 0.10×Efficiency
```

Targets:
- Light rigor: 0.5
- Standard: 0.7
- Deep: 0.9

---

## What It Does Well

✅ Structures messy thinking  
✅ Forces explicit confidence estimates  
✅ Systematically finds gaps through iteration  
✅ Prevents evidence quality inflation  
✅ Incorporates proven techniques (CIA, Decision Analysis)  
✅ Tells you when to stop analyzing  

---

## Limitations (Be Honest)

❌ Cannot detect contradictory evidence (YOU must check)  
❌ Cannot account for unknown unknowns (add black swan check)  
❌ Cannot replace domain expertise  
❌ Cannot make decisions for you  
❌ Cannot guarantee you're right  

**This is a thinking aid, not a magic solution.**

---

## When To Use

**Use this for:**
- Important decisions (job, investment, strategy)
- Complex problems (many unknowns, high stakes)
- Analyses you'll share (transparent reasoning)
- Learning (forces systematic thinking)

**Don't use for:**
- Trivial decisions (analysis paralysis)
- Time-critical emergencies (use checklists)
- Purely creative work (too constraining)
- When gut feeling is sufficient

---

## Rigor Levels

Match effort to stakes:

**Light (target: 0.5, ~10-20min):**
- Low stakes
- Reversible
- Learning experiments

**Standard (target: 0.7, ~30-90min):**
- Typical decisions
- Moderate stakes
- Semi-reversible

**Deep (target: 0.9, ~2-4hr):**
- High stakes
- Irreversible
- Affects many people

Don't over-analyze low-stakes decisions.

---

## Best Practices

1. **Start with Claude conversationally** for most things
2. **Use Python for important decisions** (more rigorous)
3. **Actually look for contradictory evidence** (system can't)
4. **Update confidence honestly** as you learn
5. **Check cognitive biases** explicitly
6. **Get external review** on high-stakes decisions
7. **Track actual outcomes** vs. predictions (calibration)

---

## Borrowed Techniques

This framework synthesizes proven techniques:

**From CIA Structured Analytics:**
- Pre-mortem
- Key assumptions check
- Alternative hypotheses
- Red team analysis

**From Decision Analysis:**
- Expected value calculation
- Multi-criteria scoring
- Sensitivity analysis

**From Design Thinking:**
- Stakeholder empathy
- Rapid iteration

**From Six Sigma:**
- Measurable outcomes
- Iterative refinement

**The innovation:** Combining these with computational iteration and numerical confidence tracking.

---

## Not Claiming

I'm NOT claiming this is:
- ❌ Novel or revolutionary
- ❌ Proven by research
- ❌ Better than alternatives
- ❌ Suitable for everyone
- ❌ Perfect or complete

I AM saying:
- ✅ It helps ME think more clearly
- ✅ The techniques are borrowed from proven frameworks
- ✅ The computational approach forces rigor
- ✅ You might find it useful too

---

## Examples

See EXAMPLES.md for detailed walk-throughs:
- Job offer decision
- Product feature build
- Investment decision
- Business strategy
- Career pivot

---

## Requirements

**For conversational mode:**
- Just Claude (any version with this protocol document)

**For computational mode:**
- Python 3.8+
- numpy

```bash
pip install numpy
python analytical_protocol.py  # Run demo
```

---

## License

Personal tool. Use it however you want. No warranty.

If you improve it, I'd love to hear about it, but you're not obligated.

---

## FAQ

**Q: Is this proven to work?**
A: No. It's a personal tool I find helpful. Use your judgment.

**Q: Can I modify it?**
A: Yes. Adapt it to your needs. The framework is a starting point.

**Q: How is this different from other frameworks?**
A: Main difference is computational iteration + numerical confidence tracking. Most frameworks are mental checklists.

**Q: Do I need to use Python?**
A: No. Conversational mode with Claude works fine for most things.

**Q: What if I disagree with the quality scores?**
A: Trust your judgment. Scores are guides, not rules.

**Q: Can I use this for [domain]?**
A: Probably. It's domain-agnostic. You may need to customize evidence hierarchies.

---

## Contact

This is a personal project. No support provided. But if you find bugs or have suggestions, feel free to share.

---

**Built for personal use. Shared in case useful to others.**
