# CLAUDE ANALYTICAL PROTOCOL (CAP) v3.0 - USER GUIDE

## What Changed in v3.0?

### Critical Fixes:
✅ **Removed false "Bayesian" claims** - Now uses honest heuristic weighting  
✅ **Fixed confidence mathematics** - Transparent, no fake rigor  
✅ **Renamed "adversarial robustness"** → "internal consistency" (accurate)  
✅ **Removed inflated claims** - No unsupported percentages  
✅ **Added domain-specific evidence hierarchies** - Medical, business, policy, etc.  
✅ **Improved documentation** - Clear about what's validated vs. proposed  
✅ **Added limitations section** - Honest about what CAP can't do

---

## What CAP Actually Is

**Honest Description:**
CAP is a structured approach to systematic thinking that combines:
- Explicit confidence tracking
- Iterative self-critique
- Evidence quality assessment
- Modular framework that adapts to problem type

**What it might do:**
- Help structure complex analyses
- Force explicit consideration of alternatives
- Track uncertainty transparently
- Provide consistent methodology

**What it definitely doesn't do:**
- Guarantee correct decisions
- Replace domain expertise
- Provide statistically validated probabilities
- Think truly adversarially (that requires external review)

---

## How to Use With Claude

### Option 1: Simple Request

```
"Claude, analyze [PROBLEM] using the Computational Analytical Protocol version 3."
```

Claude will:
1. Assess your problem (type, stakes, complexity)
2. Recommend appropriate rigor level
3. Build foundation with explicit confidence tracking
4. Run systematic self-critique cycles
5. Provide component scores and recommendation
6. **Note limitations clearly**

### Option 2: Specific Configuration

```
"Claude, use CAP v3 with:
- Rigor level: 2 (standard)
- Domain: medical
- Max iterations: 7
- Focus layers: foundation + implementation"
```

### Option 3: Just One Component

```
"Claude, use the adversarial testing component of CAP to critique this analysis: [X]"
```

---

## Core Concepts Explained

### 1. Confidence Scores (NOT Statistical Confidence)

**What they are:**
- Subjective estimates: "How certain am I about this?"
- Range: 0.0 (no idea) to 1.0 (very certain, though we cap at 0.95)
- Updated as evidence accumulates

**What they're NOT:**
- Statistically calibrated probabilities
- Validated confidence intervals
- Objective measures

**How to interpret:**
- 0.9-0.95: Very confident (but still could be wrong)
- 0.7-0.9: Reasonably confident
- 0.5-0.7: Uncertain, need more info
- Below 0.5: Very uncertain, definitely need more info

### 2. Evidence Quality Hierarchies

**Different domains need different hierarchies:**

**Medical/Scientific:**
- Systematic review: 1.0
- RCT: 0.85
- Cohort study: 0.65
- Case series: 0.35
- Expert opinion: 0.2

**Business:**
- Multi-company analysis: 0.9
- Internal controlled data: 0.8
- Industry benchmarks: 0.7
- Single case study: 0.5
- Expert opinion: 0.4

**Policy:**
- Randomized evaluation: 0.9
- Quasi-experimental: 0.75
- Before/after with controls: 0.6
- Expert judgment: 0.25

**Important notes:**
- These are **proposed** scales, not validated
- Quality within type varies enormously
- Context matters: one excellent case study > three poor RCTs
- Higher score ≠ more relevant to your decision

### 3. Uncertainty Types

**Epistemic (Reducible):**
- We don't know, but we *could* find out
- Example: "What's the market size?" → Can research
- **Action**: Do more research before deciding

**Aleatory (Irreducible):**
- Inherently random, can't reduce further
- Example: "Will this specific coin flip be heads?" → Random
- **Action**: Proceed with risk management

### 4. Internal Consistency (formerly "Adversarial Robustness")

**What it measures:**
- Fraction of self-generated criticisms that were addressed
- Formula: `resolved / total_criticisms`

**Critical limitation:**
- This is **self-critique**, not external validation
- AI can't truly think adversarially to itself
- High score means internal coherence, not correctness
- **Always get external review for high-stakes decisions**

### 5. Rigor Levels

**Level 1: Exploratory**
- Use for: Brainstorming, ideation, low-stakes decisions
- Confidence threshold: 0.5
- Iterations: ~3
- Layers: Foundation only

**Level 2: Standard**
- Use for: Business decisions, project planning, typical work
- Confidence threshold: 0.7
- Iterations: ~7
- Layers: Foundation + Implementation

**Level 3: Rigorous**
- Use for: Publications, high-stakes decisions, policy
- Confidence threshold: 0.9
- Iterations: 15+
- Layers: All applicable
- **Must include external review**

---

## Quality Scores Explained

### Component Scores:

**Completeness (0-1):**
- Are the core questions answered?
- Weighted by importance (WHAT/WHY most critical)

**Average Confidence (0-1):**
- Mean confidence across dimensions
- Remember: subjective estimates, not statistics

**Evidence Quality (0-1):**
- Average quality of supporting evidence
- Based on domain-appropriate hierarchy

**Internal Consistency (0-1):**
- Fraction of criticisms addressed
- **Limitation**: Self-generated, not external

**Iteration Efficiency (0-1):**
- Did we iterate appropriately?
- Penalizes both rushing and over-analyzing

### Overall Quality Score:

**Proposed formula (REQUIRING VALIDATION):**
```
Overall = 0.30×Completeness + 0.20×Confidence + 0.20×Evidence 
        + 0.20×Consistency + 0.10×Efficiency
```

**Interpretation (proposed guidelines):**
- 0.85-1.0: High quality, likely ready
- 0.70-0.85: Moderate quality, reasonable for standard work
- 0.50-0.70: Low quality, acceptable only for exploration
- Below 0.50: Poor quality, needs work

**CRITICAL CAVEAT:**
These weights are **initial guesses**, not empirically validated. Different domains may need different weights. Always examine component scores separately, not just overall.

---

## Example Conversation Flow

**You:** "Should we acquire Company X for $50M?"

**Claude (using CAP v3):**

> "I'll analyze this using the Computational Analytical Protocol v3. Based on the characteristics (high stakes decision, moderate complexity), I recommend standard rigor (Level 2).
>
> **FOUNDATION LAYER:**
>
> WHAT: Acquisition of Company X (B2B SaaS, 200 customers, $3M ARR, 25 employees)  
> Confidence: 0.90 (well-defined from your description)
>
> WHY: Strategic rationale stated as: customer demand, talent acquisition, market share growth  
> Confidence: 0.50 (need evidence to support these claims)
>
> Let me search for evidence on SaaS acquisition success rates...
>
> [Searches, finds 3 studies on SaaS acquisitions]
>
> After evidence integration: WHY confidence = 0.68
> Evidence quality: 0.72 (two industry analyses, one case study)
>
> **ADVERSARIAL TESTING (Cycle 1):**
>
> Criticism 1 (severity 0.8): "What if key employees leave post-acquisition? SaaS value is in people."  
> Response: Add retention packages, 2-year vesting
>
> Criticism 2 (severity 0.7): "Customer retention data missing. B2B SaaS acquisitions see 30-40% customer churn."  
> Response: CRITICAL GAP - need customer survey
>
> [... 5 more cycles ...]
>
> **FINAL ASSESSMENT:**
>
> Quality Scores:
> - Completeness: 0.78
> - Confidence: 0.65
> - Evidence Quality: 0.72
> - Internal Consistency: 0.75
> - Overall: 0.73
>
> **RECOMMENDATION: PROCEED WITH CAUTION ⚠️**
>
> The analysis reaches moderate quality but has one critical gap: no customer retention data. 
>
> **Action before deciding:**
> 1. Commission customer survey (cost: ~$10K, time: 2 weeks)
> 2. Get 3rd-party tech due diligence
> 3. Structure deal with performance earnouts
>
> **Limitations of this analysis:**
> - Confidence scores are my subjective estimates
> - Evidence quality based on proposed hierarchy (not validated)
> - Internal consistency reflects self-critique only
> - Should get independent M&A advisor review

---

## When NOT to Use CAP

### Don't use CAP for:

❌ **Time-critical emergencies** → Use checklists instead  
❌ **Simple problems with known solutions** → Use standard procedures  
❌ **Pure creative/artistic work** → Too constraining  
❌ **Purely mathematical problems** → Use formal proofs  
❌ **Decisions already made** → Don't retroactively justify  
❌ **When you need speed over thoroughness** → Trust expert judgment

### Use with extreme caution:

⚠️ **High-stakes irreversible decisions** → Add multiple external reviews  
⚠️ **Novel domains with no evidence** → Framework less helpful  
⚠️ **Strong expert disagreement** → Framework won't resolve  
⚠️ **Ethical dilemmas** → Values matter more than analysis  
⚠️ **When AI knowledge base is weak** → Garbage in, garbage out

---

## Known Limitations

### 1. Subjective Confidence
- Scores are estimates, not calibrated probabilities
- Prone to overconfidence bias
- Different analysts assign different scores

**Mitigation:** Always report ranges, get external calibration

### 2. Self-Generated Critique
- AI can't truly think adversarially to itself
- Criticisms limited by AI's knowledge
- No substitute for real external review

**Mitigation:** Require human peer review for important decisions

### 3. Evidence Hierarchies
- Oversimplify methodological complexity
- Don't account for quality variation within type
- May not transfer across domains

**Mitigation:** Use domain expertise to override when appropriate

### 4. Unvalidated Weights
- Quality score weights are educated guesses
- May need adjustment by domain
- No empirical validation yet

**Mitigation:** Report component scores separately, run sensitivity analyses

### 5. Computational Cost
- Multiple iterations take time
- May have diminishing returns after ~7 cycles
- Not practical for time-critical decisions

**Mitigation:** Set hard limits, use lower rigor for routine decisions

### 6. AI Knowledge Dependence
- Quality limited by AI's training data
- May miss domain-specific nuances
- Can hallucinate evidence

**Mitigation:** Always verify sources, involve domain experts

---

## Failure Modes & How to Avoid Them

### Analysis Paralysis
**Symptom:** Endless iteration, never deciding  
**Cause:** Perfectionism, low risk tolerance  
**Fix:** Set hard iteration limits, accept "good enough"

### False Precision
**Symptom:** Over-trusting numbers, false confidence  
**Cause:** Numbers feel objective  
**Fix:** Always show uncertainty ranges, question scores

### Confirmation Bias
**Symptom:** Superficial critique, predetermined conclusions  
**Cause:** Human nature + AI limitations  
**Fix:** Actively seek disconfirming evidence, external review

### Automation Bias
**Symptom:** Uncritical acceptance of systematic output  
**Cause:** "The AI did thorough analysis"  
**Fix:** Require human judgment, especially for high stakes

### Framework Rigidity
**Symptom:** Forcing all problems into same structure  
**Cause:** Tool becomes a hammer  
**Fix:** Adapt framework, skip inapplicable components

---

## Comparison to Other Frameworks

### vs. Six Sigma DMAIC
- **CAP better for:** Qualitative problems, rapid iteration
- **DMAIC better for:** Manufacturing, quantifiable processes

### vs. CIA Structured Analytics
- **CAP better for:** AI-assisted work, integrated framework
- **CIA SAT better for:** Human workshops, modular techniques

### vs. Design Thinking
- **CAP better for:** Analytical rigor, evidence-based decisions
- **DT better for:** Creative exploration, user empathy

### vs. Decision Analysis
- **CAP better for:** Ill-defined problems, partial information
- **DA better for:** Well-defined objectives, quantifiable utilities

**CAP's niche:** AI-assisted structured thinking for complex problems with incomplete information.

---

## Practical Tips

### Getting Started
1. Start with low-stakes decisions
2. Use Level 1 (exploratory) first
3. Focus on foundation layer only
4. See if it helps your thinking

### Building Skill
1. Track actual outcomes vs. predictions
2. Calibrate your confidence estimates
3. Learn which criticisms matter most
4. Adapt weights to your domain

### Advanced Usage
1. Create domain-specific evidence hierarchies
2. Develop templates for recurring problems
3. Build institutional knowledge base
4. Train team on consistent usage

---

## What We Don't Know Yet

### Unanswered Questions:

❓ Does CAP actually improve decisions? (Need empirical study)  
❓ What are optimal weights? (Need validation)  
❓ How much better than unstructured analysis? (Need comparison)  
❓ Does it work across domains? (Need testing)  
❓ What's the learning curve? (Need user studies)  
❓ When does it make things worse? (Need failure analysis)

**This is why v3.0 is a DRAFT requiring validation.**

---

## For Researchers: Validation Plan

### Proposed Study:
- 30 participants, 4 decisions each
- Compare CAP vs. unstructured analysis
- Measure: gap detection, decision quality, time, satisfaction
- 3-month outcome follow-up
- Pre-registered on OSF

### Hypotheses (to be tested):
- H1: CAP identifies more analytical gaps
- H2: CAP improves decision quality (unclear magnitude)
- H3: CAP takes longer (20-50% more time)
- H4: CAP improves confidence calibration

### Falsification criteria:
- Abandon if: No improvement in any measure
- Revise if: Improvement < cost in time
- Scale if: Clear benefits across domains

---

## Files in This Package

1. **CLAUDE_ANALYTICAL_PROTOCOL_v3.md** (Full document)
   - Complete theoretical foundation
   - Honest about limitations
   - Comparison to existing frameworks
   - Ready for review, not publication

2. **cap_implementation_v3.py** (Implementation)
   - Fixed mathematics
   - Proper validation
   - Type hints and tests
   - Honest heuristics

3. **THIS FILE** (User guide)
   - How to use CAP
   - What to expect
   - What can go wrong

4. **README_v3.md** (Overview)
   - Quick start
   - Honest assessment
   - Path forward

---

## Bottom Line

**CAP v3.0 is:**
- A structured thinking tool
- Potentially useful
- Requiring validation
- Worth trying on low-stakes decisions

**CAP v3.0 is NOT:**
- A proven methodology
- A magic solution
- A substitute for expertise
- Ready for high-stakes use without validation

**Recommendation:**
Use it, test it, report back. We need data to know if this actually helps.

---

## Contact & Feedback

**Created by:** [Your Name]  
**Version:** 3.0  
**Status:** Draft for validation  
**License:** MIT

**How to contribute:**
- Try it on real problems
- Report successes AND failures
- Suggest improvements
- Share validation data

**Critical feedback welcome.**

---

## Acknowledgments

This version benefited from critical analysis that identified:
- False mathematical claims
- Unsupported empirical assertions
- Misleading terminology
- Circular reasoning in validation

Thanks to honest critique for improving this framework substantially.

---

**Use CAP v3.0 skeptically and report what you learn.**
