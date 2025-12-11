# ANALYTICAL PROTOCOL
*A computational thinking tool for systematic analysis with Claude*

## What This Is

A structured approach to thinking through problems that:
- Uses **numerical confidence** (forces honesty: "I'm 0.65 confident" not "kinda sure")
- **Iterates systematically** (generate criticisms → fix → repeat until solid)
- **Tracks evidence quality** (one RCT ≠ ten blog posts)
- **Computes when you're done** (0.75 quality score vs 0.45)
- **Incorporates proven techniques** (pre-mortem, red teaming, assumption testing)

This is a **personal tool**, not a research framework. It helps ME think better. If it helps you too, great.

---

## How To Use

**Option 1: Conversational (Easiest)**
```
"Claude, analyze [MY DECISION] using the analytical protocol.
Track confidence numerically and iterate until quality ≥ 0.75"
```

**Option 2: Computational (Most Rigorous)**
```
"Claude, run the Python implementation on [MY PROBLEM].
Use standard rigor, generate at least 5 criticism cycles."
```

**Option 3: Manual (You Control Everything)**
Use the framework below as a checklist, filling in confidence scores yourself.

---

## The Framework

### LAYER 0: Problem Characterization

**What type of problem?**
- [ ] Decision (choose between options)
- [ ] Analysis (understand something)
- [ ] Design (create something new)
- [ ] Prediction (forecast future)
- [ ] Diagnosis (find root cause)

**Stakes:** Low / Medium / High / Critical

**Time available:** Hours / Days / Weeks / Months

**Your gut complexity:** 0.0 (trivial) to 1.0 (extremely complex)

**Your gut uncertainty:** 0.0 (know for sure) to 1.0 (totally uncertain)

→ This determines rigor level:
- **Light** (stakes low, time short): Quick analysis, 0.5 confidence threshold
- **Standard** (typical): Normal analysis, 0.7 confidence threshold  
- **Deep** (stakes high, irreversible): Rigorous analysis, 0.9 confidence threshold

---

### LAYER 1: Foundation (Always Do This)

#### WHAT exactly are we analyzing?

**Definition:** [Be specific and clear]

**Confidence in definition:** __ / 1.0
- 0.9+ = Crystal clear
- 0.7-0.9 = Pretty clear
- 0.5-0.7 = Somewhat fuzzy
- <0.5 = Confused

**Uncertainty type:**
- [ ] Epistemic (we don't know but could find out → go research it)
- [ ] Aleatory (inherently random → manage the risk)

---

#### WHY does this matter?

**Justification:** [The actual reason, not the stated reason]

**Evidence supporting this:**
1. [Evidence 1] - Quality: __ / 1.0 - Source: [where from]
2. [Evidence 2] - Quality: __ / 1.0 - Source: [where from]
3. [Evidence 3] - Quality: __ / 1.0 - Source: [where from]

**Evidence quality guide:**
- **1.0** = Meta-analysis of RCTs (medical) or multi-company rigorous analysis (business)
- **0.85** = Single RCT or controlled experiment
- **0.7** = Good observational study with controls
- **0.5** = Case study or analysis without controls
- **0.3** = Expert opinion
- **0.15** = Anecdote or "everyone says"

**Overall confidence in WHY:** __ / 1.0
- Starts at 0.5 (neutral)
- Updates based on evidence quality
- More evidence → higher confidence (but diminishing returns)

**Contradictory evidence:** [CRITICAL - list evidence AGAINST your view]
- If you can't find any, you're not looking hard enough

---

#### HOW would this work?

**Mechanism:** [The actual process or causal chain]

**Confidence in mechanism:** __ / 1.0

**Key assumptions:**
1. [Assumption 1] - If wrong: Critical / Major / Minor
2. [Assumption 2] - If wrong: Critical / Major / Minor
3. [Assumption 3] - If wrong: Critical / Major / Minor

---

#### MEASURE: How will we know?

**Success metric:** [Specific, observable, measurable]

**Confidence we can measure this:** __ / 1.0

---

### LAYER 2: Adversarial Testing (The Core Innovation)

This is where computational iteration matters. Don't do this once - do it until you can't find more problems.

#### CYCLE 1: Pre-Mortem

"It's 6 months from now. This failed completely."

**Why did it fail?**
1. [Failure reason 1] - Severity: __ / 1.0
2. [Failure reason 2] - Severity: __ / 1.0
3. [Failure reason 3] - Severity: __ / 1.0
4. [Failure reason 4] - Severity: __ / 1.0
5. [Failure reason 5] - Severity: __ / 1.0

**Severity guide:**
- 0.9-1.0 = Fundamental flaw, completely wrong
- 0.7-0.9 = Serious issue, must address
- 0.4-0.7 = Important limitation, note it
- 0.0-0.4 = Minor issue, acceptable

**For each severity ≥ 0.7:**
- How can we prevent this? [Action]
- How can we detect it early? [Indicator]
- Does this change our confidence? [Update numbers]

---

#### CYCLE 2: Assumption Test (from CIA Structured Analytics)

**For each CRITICAL assumption:**

**Assumption:** [State it clearly]

**Evidence FOR:**
- [Supporting evidence]

**Evidence AGAINST:**
- [Contradicting evidence]

**If this assumption is wrong:**
- Impact: [What breaks?]
- Can we test it?: [How?]
- Should we test it before proceeding?: Yes / No

**Alternative assumption:** [What if the opposite is true?]
- Does the analysis still work?
- What would change?

---

#### CYCLE 3: Alternative Explanations

**What else could explain the data/situation?**

1. **Alternative 1:** [Different explanation]
   - Evidence for: __ / 1.0
   - Evidence against: __ / 1.0
   - More likely than our explanation?: Yes / No

2. **Alternative 2:** [Different explanation]
   - Evidence for: __ / 1.0
   - Evidence against: __ / 1.0
   - More likely than our explanation?: Yes / No

**Devil's Advocate:** Steel-man the best alternative
- [Make the strongest possible case for it]
- What would convince us this is actually right?

---

#### CYCLE 4: Red Team (Adversarial Perspective)

**If someone wanted to prove us wrong, what would they say?**

**Criticism 1:** [Strong objection]
- Our response: [How we address it]
- Does this lower our confidence?: Yes / No / By how much: __

**Criticism 2:** [Strong objection]
- Our response: [How we address it]
- Does this lower our confidence?: Yes / No / By how much: __

**Criticism 3:** [Strong objection]
- Our response: [How we address it]
- Does this lower our confidence?: Yes / No / By how much: __

---

#### CYCLE 5: Stakeholder Reality Check

**WHO is affected?**
- [Stakeholder 1] - Impact: Positive / Negative / Mixed
- [Stakeholder 2] - Impact: Positive / Negative / Mixed
- [Stakeholder 3] - Impact: Positive / Negative / Mixed

**For each negatively impacted stakeholder:**
- Why are they hurt?
- Is this acceptable? (ethical check)
- Can they block this?
- What's their likely response?

**Incentive check:**
- Do people have incentive to do what we need?
- What's in it for them?
- What could misalign incentives?

---

#### CYCLE 6+: Keep Going Until...

**Stopping criteria:**
- Overall quality score ≥ target (0.5 light, 0.7 standard, 0.9 deep)
- OR: No criticisms above severity threshold
- OR: Diminishing returns (not finding new issues)
- OR: Max iterations (typically 5-10)

---

### LAYER 3: Black Swan Check (Unknown Unknowns)

**External shocks that could invalidate everything:**

**Economic:**
- [ ] Recession
- [ ] Inflation spike
- [ ] Market crash
- [ ] Currency crisis

**Regulatory:**
- [ ] New laws
- [ ] Enforcement change
- [ ] Compliance requirement

**Technology:**
- [ ] Disruption
- [ ] Obsolescence
- [ ] Competitor breakthrough

**Social:**
- [ ] Public opinion shift
- [ ] Scandal/reputation
- [ ] Cultural change

**Other:**
- [ ] Pandemic/health crisis
- [ ] Natural disaster
- [ ] War/geopolitical
- [ ] Key person leaves

**For top 3 risks:**
- Likelihood: __ / 1.0
- Impact if happens: __ / 1.0
- Can we detect early?: [Indicator]
- Mitigation: [What reduces risk]

---

### LAYER 4: Quality Assessment (Computational)

This is where numbers matter. Forces honesty.

#### Completeness Score

**Dimensions filled:**
- WHAT: Yes/No → Weight: 1.5 (critical)
- WHY: Yes/No → Weight: 1.5 (critical)
- HOW: Yes/No → Weight: 1.0 (important)
- MEASURE: Yes/No → Weight: 1.0 (important)

**Weighted by confidence:**
- Example: WHAT filled (1.5 weight) × 0.8 confidence = 1.2
- Sum all, divide by total weight (5.0)
- **Completeness = __ / 1.0**

#### Average Confidence

**Mean confidence across filled dimensions:**
- (what_conf + why_conf + how_conf + measure_conf) / 4
- **Average Confidence = __ / 1.0**

#### Evidence Quality

**Mean quality of all evidence pieces:**
- Sum all evidence quality scores / number of pieces
- **Evidence Quality = __ / 1.0**

#### Internal Consistency

**Criticisms addressed:**
- Total criticisms generated: __
- Criticisms resolved: __
- Resolution rate: __ / 1.0
- **Internal Consistency = __ / 1.0**

#### Iteration Efficiency

**Did we over/under analyze?**
- Iterations taken: __
- Expected for this rigor: __ (light=3, standard=7, deep=15)
- Too fast (< 0.5 × expected) = 0.7 score
- Good pace (0.5-1.5 × expected) = 1.0 score
- Too slow (> 1.5 × expected) = diminishing, min 0.5
- **Efficiency = __ / 1.0**

---

#### Overall Quality Score

**Weighted formula:**
```
Overall = 0.30 × Completeness
        + 0.20 × Average Confidence
        + 0.20 × Evidence Quality
        + 0.20 × Internal Consistency
        + 0.10 × Efficiency
```

**Overall Quality = __ / 1.0**

**Interpretation:**
- **0.85-1.0** = Excellent, very likely sound
- **0.70-0.85** = Good, reasonable to proceed
- **0.50-0.70** = Mediocre, proceed with caution
- **0.00-0.50** = Poor, needs more work

**Minimum completeness gate:**
- Even with high quality score, if completeness < 0.5 → NOT READY
- Must have at least WHAT and WHY solidly defined

---

### LAYER 5: Decision (If applicable)

#### If choosing between options:

**Decision Matrix:**

| Criteria | Weight | Option A | Option B | Option C |
|----------|--------|----------|----------|----------|
| Effectiveness (solves problem) | 30% | __/10 | __/10 | __/10 |
| Feasibility (can we do it) | 25% | __/10 | __/10 | __/10 |
| Cost (resources) | 20% | __/10 | __/10 | __/10 |
| Risk (downside) | 15% | __/10 | __/10 | __/10 |
| Reversibility (can undo) | 10% | __/10 | __/10 | __/10 |
| **Weighted Total** | | **__** | **__** | **__** |

**Expected Value (if quantifiable):**

For each option:
```
EV = (Upside × P(success)) - (Downside × P(failure))
```

**Option A EV:** __  
**Option B EV:** __  
**Option C EV:** __

---

#### The Decision:

**Chosen option:** [Which one and why]

**Confidence in decision:** __ / 1.0
- This is NOT the same as confidence in the analysis
- "Given what we know, how sure are we this is the right choice?"

**Key remaining uncertainties:**
1. [Uncertainty 1] - Impact if wrong: High/Med/Low
2. [Uncertainty 2] - Impact if wrong: High/Med/Low
3. [Uncertainty 3] - Impact if wrong: High/Med/Low

**Indicators to watch:**
- [Indicator 1]: If we see this, we're on track
- [Indicator 2]: If we see this, we need to pivot
- [Indicator 3]: If we see this, we need to stop

**Reversibility plan:**
- Can we undo this?: Yes / Partially / No
- If yes, under what conditions would we?
- Exit strategy: [What it is]

---

## Computational Mode (Using Python)

For most rigorous analysis, use the Python implementation which:
- **Automatically iterates** through adversarial cycles
- **Computes quality scores** objectively
- **Tracks confidence updates** from evidence
- **Applies diminishing returns** (prevents evidence inflation)
- **Determines convergence** (when to stop)

**To use:**
```python
from analytical_protocol import FoundationElement, run_analysis, Evidence, EvidenceDomain

# Create your analysis
element = FoundationElement(name="My Decision", domain=EvidenceDomain.BUSINESS)
element.what = "Specific definition"
element.what_confidence = 0.8
element.why = "Clear justification"
element.why_confidence = 0.6

# Add evidence
element.add_evidence(Evidence(
    content="Study showing X",
    source="Harvard Business Review 2024",
    strength=0.7,
    date="2024",
    domain=EvidenceDomain.BUSINESS,
    study_design="multi_company_analysis"
))

# Run analysis with iteration
results = run_analysis(element, rigor_level=2, max_iterations=10)

# Get scores
print(f"Quality: {results['quality_scores']['overall']:.2f}")
print(f"Ready: {results['ready_for_action']}")
```

The code handles:
- Evidence integration with proper weighting
- Diminishing returns for many weak sources
- Domain-specific quality hierarchies
- Automatic iteration until convergence
- Quality threshold checking

---

## Cognitive Bias Checks

Before finalizing, explicitly check for:

- [ ] **Confirmation bias** - Did I actively seek contradictory evidence?
- [ ] **Availability bias** - Am I overweighting recent/memorable examples?
- [ ] **Anchoring** - Am I stuck on the first number I heard?
- [ ] **Sunk cost** - Am I continuing because I already invested?
- [ ] **Overconfidence** - Is my confidence calibrated? (test: bet money on it?)
- [ ] **Groupthink** - Did I get dissenting opinions?
- [ ] **Planning fallacy** - Did I add buffer for things taking longer?
- [ ] **Optimism bias** - Am I assuming better than average outcome?

---

## Red Flags (Stop Signs)

🛑 **Can't define success measure** → How will you know if it worked?  
🛑 **No contradictory evidence found** → You didn't look hard enough  
🛑 **Everyone agrees too easily** → Where's the dissent?  
🛑 **Can't articulate key assumptions** → Fuzzy thinking  
🛑 **Confidence >0.9 on future prediction** → Overconfident  
🛑 **Zero risk identified** → Haven't thought it through  
🛑 **"We need to decide NOW"** → Artificial urgency, why?  
🛑 **Quality score <0.5** → Not ready, need more work  
🛑 **Completeness <0.5** → Even if quality ok, too incomplete  

---

## When To Use What Rigor

**Light (target quality 0.5):**
- Reversible decisions
- Low stakes
- Learning experiments
- Time: 10-20 minutes
- Iterations: ~3

**Standard (target quality 0.7):**
- Typical business decisions
- Moderate stakes
- Semi-reversible
- Time: 30-90 minutes
- Iterations: ~7

**Deep (target quality 0.9):**
- High stakes
- Irreversible
- Affects many people
- Time: 2-4 hours
- Iterations: ~15

**Don't over-analyze:** Using deep rigor on "which laptop to buy" is waste

---

## Limitations (Be Honest With Yourself)

**This framework CANNOT:**
- ❌ Detect when evidence contradicts your claim (YOU must check)
- ❌ Account for completely unanticipated events (black swans)
- ❌ Replace domain expertise or subject matter knowledge
- ❌ Make the decision for you (you still choose)
- ❌ Guarantee you're right (reduces error, doesn't eliminate)
- ❌ Tell you your values/priorities (that's on you)
- ❌ Work if you game it (garbage in, garbage out)

**This framework CAN:**
- ✅ Structure messy thinking
- ✅ Force explicit confidence estimates (honest uncertainty)
- ✅ Systematically find gaps through iteration
- ✅ Integrate evidence with proper weighting
- ✅ Prevent some cognitive biases
- ✅ Make reasoning transparent and reviewable
- ✅ Tell you when analysis is "good enough"

**Critical vulnerability:**
- If you add evidence that CONTRADICTS your claim, the system will treat it as supporting evidence and may INCREASE your confidence
- **YOU must manually check if evidence supports or contradicts**
- This is fundamental limitation of the approach

---

## Quick Start Examples

### Example 1: "Should I take this job offer?"

**Foundation:**
- WHAT: Senior Engineer at StartupX, $150K, equity, remote
- WHY: More growth, better pay, new challenges
- HOW: Apply → Interview → Negotiate → Decide
- MEASURE: Am I happier and growing in 6 months?

**Evidence:**
- Glassdoor reviews: 3.8/5 (quality: 0.5)
- Friend worked there: positive (quality: 0.3)
- Growth rate: 40% YoY (quality: 0.7)

**Pre-mortem failures:**
1. Company runs out of money (severity: 0.8)
2. Culture is actually toxic (severity: 0.7)
3. Role is boring (severity: 0.5)

**After 5 iterations:**
- Quality: 0.72
- Decision: Take it, but keep savings high (reversibility plan)
- Watch: Burn rate, team turnover, project excitement

### Example 2: "Should we build feature X?"

**Foundation:**
- WHAT: Social sharing for our app
- WHY: Users requesting it, competitors have it
- HOW: 2 sprints, API integration, UI design
- MEASURE: >20% of users share something weekly

**Evidence:**
- 15 users requested it (quality: 0.4 - small N)
- Competitor has it (quality: 0.3 - not evidence it works)
- No data showing sharing drives growth (quality: 0.0)

**After adversarial testing:**
- Assumption: "Users will actually share" - UNTESTED
- Alternative: "Users say they want it but won't use it"
- Quality: 0.55 (mediocre)
- Decision: Build MVP first, test assumption before full build

---

## Integration With Claude

**Claude can:**
- Guide you through the framework conversationally
- Run the Python implementation for computational rigor
- Generate pre-mortem scenarios
- Test assumptions adversarially
- Search for contradictory evidence
- Calculate quality scores
- Tell you when to stop iterating

**Claude cannot:**
- Know your domain as well as you do
- Have perfect information
- Eliminate all uncertainty
- Make value judgments for you
- Replace your decision-making

**Best practice:**
- Start conversational for quick things
- Use computational mode for important decisions
- Always verify evidence sources yourself
- Get external review on high-stakes choices

---

## The Core Insight

**Most thinking frameworks fail because they're either:**
1. Too vague ("think critically!") - no mechanism
2. Too rigid (must fill 47 boxes) - kills insight

**This framework works because:**
1. **Numerical confidence** forces honesty about uncertainty
2. **Computational iteration** systematically finds gaps
3. **Evidence weighting** prevents garbage accumulation
4. **Quality scoring** tells you when to stop
5. **Proven techniques** (pre-mortem, red team) incorporated
6. **Adaptable rigor** matches effort to stakes

**The secret:** Iteration with objective quality measurement

Do pre-mortem once? Miss stuff.  
Iterate until quality ≥ 0.7? Hard to miss major flaws.

---

## This Is A Personal Tool

Not claiming this is:
- ❌ Revolutionary
- ❌ Proven by research
- ❌ Better than alternatives
- ❌ Suitable for everyone

Just claiming:
- ✅ It helps ME think more clearly
- ✅ The techniques are borrowed from proven frameworks
- ✅ The computational approach is honest about limitations
- ✅ You might find it useful too

Use it if it helps. Ignore it if it doesn't.

---

**Files in this system:**
1. This protocol (framework + instructions)
2. Python implementation (computational mode)
3. Quick reference (one-page cheat sheet)
4. Examples (real analyses)

**Ready to use. Adapt as needed.**
