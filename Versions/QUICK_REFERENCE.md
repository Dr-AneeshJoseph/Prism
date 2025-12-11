# ANALYTICAL PROTOCOL - QUICK REFERENCE

## One-Page Cheat Sheet for Daily Use

---

### QUICK START

**Say to Claude:**
> "Analyze [MY PROBLEM] using the analytical protocol. Standard rigor, iterate until quality ≥ 0.7"

---

### THE ESSENTIALS

#### 1. FOUNDATION (Always)

**WHAT** = Clear definition (conf: __/1.0)  
**WHY** = Real justification (conf: __/1.0)  
**HOW** = Actual mechanism (conf: __/1.0)  
**MEASURE** = Specific metric (conf: __/1.0)

#### 2. EVIDENCE (For each claim)

Quality scale:
- 1.0 = Meta-analysis/rigorous multi-study
- 0.85 = Single RCT/controlled experiment
- 0.7 = Good observational study
- 0.5 = Case study/uncontrolled
- 0.3 = Expert opinion
- 0.15 = Anecdote/"everyone says"

#### 3. ADVERSARIAL (Iterate)

**Pre-mortem:** "This failed. Why?"  
**Assumptions:** "What if X is wrong?"  
**Alternatives:** "What else explains this?"  
**Red team:** "How would a critic attack this?"  
**Stakeholders:** "Who gets hurt?"

→ Keep going until quality ≥ target

#### 4. QUALITY CHECK

```
Overall = 0.30×Completeness + 0.20×Confidence 
        + 0.20×Evidence + 0.20×Consistency + 0.10×Efficiency
```

**Must have:** Completeness ≥ 0.5

#### 5. BLACK SWANS

List 3-5 external shocks that could invalidate everything

---

### CONFIDENCE SCALE

- **0.9+** = Very confident (rare!)
- **0.7-0.9** = Reasonably confident
- **0.5-0.7** = Uncertain
- **<0.5** = Very uncertain

Be honest. Overconfidence kills.

---

### RIGOR LEVELS

| Level | Quality Target | Time | Iterations |
|-------|---------------|------|------------|
| Light | 0.5 | 10-20min | ~3 |
| Standard | 0.7 | 30-90min | ~7 |
| Deep | 0.9 | 2-4hr | ~15 |

Match rigor to stakes.

---

### DECISION MATRIX

| Criteria | Weight | Option A | Option B |
|----------|--------|----------|----------|
| Effectiveness | 30% | __/10 | __/10 |
| Feasibility | 25% | __/10 | __/10 |
| Cost | 20% | __/10 | __/10 |
| Risk | 15% | __/10 | __/10 |
| Reversibility | 10% | __/10 | __/10 |

---

### RED FLAGS 🚩

- Can't define success → How will you know?
- No contradictory evidence → Look harder
- Everyone agrees → Where's dissent?
- Confidence >0.9 on future → Overconfident
- Quality <0.5 → Not ready
- Completeness <0.5 → Too incomplete

---

### BIAS CHECKS

- [ ] Confirmation (seeking supporting only?)
- [ ] Availability (recent = important?)
- [ ] Anchoring (stuck on first number?)
- [ ] Sunk cost (continuing because invested?)
- [ ] Overconfidence (would you bet on it?)
- [ ] Groupthink (no disagreement?)

---

### PYTHON QUICK START

```python
from analytical_protocol import *

element = FoundationElement(name="My Decision")
element.what = "Definition"
element.what_confidence = 0.8

element.add_evidence(Evidence(
    content="Study X",
    source="Source Y",
    quality=0.7,
    date="2024"
))

results = run_analysis(element, rigor_level=2)
print(f"Quality: {results['quality']:.2f}")
print(f"Ready: {results['ready']}")
```

---

### LIMITATIONS

**CANNOT:**
- ❌ Detect contradictory evidence (you must check)
- ❌ Account for unknown unknowns
- ❌ Replace domain expertise
- ❌ Guarantee correctness

**CAN:**
- ✅ Structure thinking
- ✅ Force honesty about uncertainty
- ✅ Find gaps systematically
- ✅ Tell you when "good enough"

---

### WHEN TO STOP

Stop iterating when:
- Quality ≥ target AND completeness ≥ 0.5
- OR: No new criticisms
- OR: Diminishing returns
- OR: Max iterations

Don't over-analyze.

---

### QUICK EXAMPLES

**Job offer:** WHAT=role details, WHY=better growth?, HOW=transition plan, MEASURE=happier in 6mo?

**Feature build:** WHAT=feature spec, WHY=user need?, HOW=implementation, MEASURE=usage rate?

**Investment:** WHAT=the asset, WHY=thesis, HOW=mechanics, MEASURE=return target?

---

**Keep this handy. Reference as needed. Adapt to your style.**
