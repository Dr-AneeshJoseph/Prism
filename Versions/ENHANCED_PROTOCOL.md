# ENHANCED ANALYTICAL PROTOCOL — VISUAL MAPPING SYSTEM

A computational thinking tool that visually maps all analysis paths, uses dual scoring (additive + multiplicative), detects fatal flaws, compares hypotheses, and shows what matters most via sensitivity analysis.

---

## WHAT'S NEW (vs Original Protocol)

| Feature | Original | Enhanced |
|---------|----------|----------|
| Scoring | Additive only | **Dual: Additive + Multiplicative** |
| Fatal Flaws | Not tracked | **Any dimension < 0.3 caps score** |
| Mechanism | Text description | **Visual causal graph** |
| Comparison | Single hypothesis | **Multi-hypothesis side-by-side** |
| Sensitivity | Not included | **Shows which assumptions matter most** |
| Visualization | Timeline only | **6 views: Overview, Mechanism, Sensitivity, Dimensions, Timeline, Criticisms** |

---

## QUICK START

```
"Claude, analyze [MY DECISION] using the enhanced analytical protocol.
Build a mechanism map, use standard rigor, and show me the visual trace."
```

For comparing options:
```
"Compare [OPTION A] vs [OPTION B] using the enhanced protocol.
Show me which has fatal flaws and why one ranks higher."
```

---

## THE FRAMEWORK

### LAYER 0: Characterize Problem

**Type:** Decision / Analysis / Design / Prediction / Diagnosis  
**Stakes:** Low / Medium / High / Critical  
**Rigor Level:** Light (0.5) | Standard (0.7) | Deep (0.85)

---

### LAYER 1: Foundation + Mechanism Map

#### Core Dimensions

| Element | What to Fill | Confidence | Fatal Below |
|---------|-------------|------------|-------------|
| **WHAT** | Clear definition | __/1.0 | 0.3 |
| **WHY** | Real justification | __/1.0 | 0.3 |
| **HOW** | Causal mechanism | __/1.0 | **0.25** (critical!) |
| **MEASURE** | Observable metric | __/1.0 | 0.3 |

#### Mechanism Map Nodes

Build a causal graph with these node types:

| Node Type | Color | Purpose |
|-----------|-------|---------|
| **CAUSE** | 🔴 Red | Root driver of the problem |
| **MECHANISM** | 🔵 Blue | Causal pathway / how it works |
| **OUTCOME** | 🟢 Green | Desired result |
| **BLOCKER** | 🟠 Orange | What could prevent success |
| **ASSUMPTION** | 🟡 Yellow | Untested belief (critical!) |
| **EVIDENCE** | 🟣 Purple | Supporting data |
| **INTERVENTION** | 🔷 Cyan | Where you act |

#### Mechanism Map Edges

| Relationship | Meaning |
|--------------|---------|
| **CAUSES** | A leads to B |
| **PREVENTS** | A blocks B |
| **ENABLES** | A makes B possible |
| **COMPENSATES** | A counteracts B |
| **REQUIRES** | A needs B to work |
| **SUPPORTS** | Evidence for claim |
| **CONTRADICTS** | Evidence against claim |

---

### LAYER 2: Adversarial Testing (Iterate)

#### Criticism Cycles

1. **Pre-mortem:** "It failed. Why?" — Check BLOCKERS in mechanism map
2. **Assumption test:** Check all ASSUMPTION nodes with confidence < 0.7
3. **Fatal flaw scan:** Any dimension below threshold?
4. **Evidence gaps:** < 2 sources? Weak quality?
5. **Weak links:** Mechanism nodes with confidence < 0.5

#### Severity Guide
- **0.9+** = Critical — must address or kill hypothesis
- **0.7-0.9** = Serious — needs mitigation
- **0.5-0.7** = Important — note as limitation
- **< 0.5** = Minor — acceptable risk

---

### LAYER 3: Sensitivity Analysis

**For each dimension, ask:**
- What if this improves by 0.1?
- What if this worsens by 0.1?
- How much does the final score change?

**Output:** Ranked list of dimensions by impact. Focus on high-impact dimensions.

---

### LAYER 4: Quality Scoring

#### Dual Scoring System

**Additive Score (traditional):**
```
Additive = Σ (dimension_value × weight) / Σ weights
```

**Multiplicative Score (fatal flaw detection):**
```
Multiplicative = geometric_mean(all dimension values)
               = exp(mean(log(values)))
```
*Heavily penalizes any low value — one bad score kills it*

**Combined Score:**
```
IF any dimension < fatal_threshold:
    Combined = min(0.3, Multiplicative)  # Capped!
ELSE:
    Combined = 0.6 × Additive + 0.4 × Multiplicative
```

#### Interpretation
- **0.7+** = Proceed
- **0.5-0.7** = Caution, address gaps
- **< 0.5** = Do not proceed
- **Fatal flaw present** = Capped at 0.3 regardless of other scores

---

### LAYER 5: Decision

**Output:**
- **PROCEED** if: Combined ≥ target AND no fatal flaws
- **DO NOT PROCEED** if: Fatal flaws OR Combined < target

**For each option, report:**
- Combined score
- Additive vs Multiplicative breakdown
- Fatal flaws (if any)
- Top 3 sensitivities (what to focus on)
- Mechanism map confidence

---

## MULTI-HYPOTHESIS COMPARISON

When comparing options:

| Metric | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Combined Score | 0.75 | 0.45 | 0.30 |
| Additive | 0.78 | 0.52 | 0.55 |
| Multiplicative | 0.70 | 0.38 | 0.18 |
| Fatal Flaws | None | evidence_quality | mechanism_validity |
| Rank | **#1** | #2 | #3 |

**Winner:** Highest combined score with no fatal flaws

---

## VISUALIZATION VIEWS

The system generates 6 interactive views:

1. **Overview** — Quality evolution chart, scoring method comparison, key stats
2. **Mechanism** — Interactive causal graph with nodes and edges
3. **Sensitivity** — Ranked impact of each dimension
4. **Dimensions** — All scoring dimensions with values and fatal flags
5. **Timeline** — Every event traced during analysis
6. **Criticisms** — All criticisms with severity and resolution status
7. **Comparison** (if multiple hypotheses) — Side-by-side ranking table

---

## EXAMPLE: Hire DS vs Buy Tool

### Hypothesis 1: Hire Data Scientist
```
WHAT: Full-time DS, $120K
WHY: Lack statistical expertise
HOW: Hire → Onboard → Deliver insights

Mechanism Map:
[Lack expertise] → [DS brings expertise] → [Better analysis] → [Improved decisions]
                                                               ↑
[Org resistance] ----PREVENTS-----------------------------────┘

Combined Score: 0.745 ✓
Fatal Flaws: None
Rank: #1
```

### Hypothesis 2: Buy Analytics Tool
```
WHAT: Enterprise platform, $50K/year
WHY: Automate without hiring
HOW: Purchase → Deploy → Train

Mechanism Map:
[Manual is slow] → [Automated dashboards] → [Faster decisions]
                                            ↑
[Tool doesn't fit] ----PREVENTS-------------┤
[Team won't adopt] ----PREVENTS-------------┘

Combined Score: 0.300 ✗
Fatal Flaws: evidence_quality (only vendor case studies)
Rank: #2
```

**Decision:** Proceed with Hire Data Scientist. Buy Analytics Tool has fatal flaw (weak evidence).

---

## CRITICAL LIMITATIONS

❌ **Cannot detect contradictory evidence** — You must check if evidence supports or contradicts

❌ **Cannot find unknown unknowns** — Mechanism map only shows what you put in

❌ **Multiplicative scoring is harsh** — One weak dimension tanks the score (by design)

❌ **Garbage in = garbage out** — Map quality depends on your domain knowledge

---

## FILES

| File | Purpose |
|------|---------|
| `enhanced_protocol.py` | Full Python implementation with visualization |
| `enhanced_analysis.jsx` | Generated React visualization |

**To use:** Upload `enhanced_protocol.py`, then:
```python
from enhanced_protocol import *

# Create analysis
h = AnalysisElement(name="My Decision")
h.set_what("Definition", 0.9)
h.set_why("Reason", 0.7)
h.set_how("Mechanism", 0.8)
h.set_measure("Metric", 0.7)

# Build mechanism map
h.add_mechanism_node(MechanismNode("c1", "Root Cause", NodeType.CAUSE, 0.9))
h.add_mechanism_node(MechanismNode("m1", "How it works", NodeType.MECHANISM, 0.7))
h.add_mechanism_node(MechanismNode("o1", "Desired result", NodeType.OUTCOME, 0.8))
h.add_mechanism_edge(MechanismEdge("c1", "m1", EdgeType.CAUSES, 0.8))
h.add_mechanism_edge(MechanismEdge("m1", "o1", EdgeType.ENABLES, 0.7))

# Add evidence
h.add_evidence(Evidence("e1", "Supporting study", "Source", 0.7, "2024"))

# Set feasibility
h.set_feasibility(technical=0.8, economic=0.7, timeline=0.8)

# Run analysis
results = run_analysis(h, rigor_level=2)
print(f"Combined: {results['combined_score']:.3f}")
print(f"Ready: {results['ready']}")
print(f"Fatal Flaws: {results['fatal_flaws']}")

# Generate visualization
viz = generate_visualization(results)
```

---

## KEY INSIGHT

**Original protocol:** "Is this good enough?" (additive scoring)

**Enhanced protocol:** "Is there anything fatally wrong?" (multiplicative) + "Is it good enough overall?" (additive) + "What matters most?" (sensitivity) + "How does it compare to alternatives?" (comparison)

**The mechanism map forces you to articulate HOW, not just WHAT and WHY.**

Most analyses fail because the mechanism is unclear or has untested assumptions. The visual map exposes these gaps immediately.

---

**Use this when the HOW matters as much as the WHAT.**
