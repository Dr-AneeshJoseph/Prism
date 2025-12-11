# SYSTEMATIC THINKING PROTOCOL
*A practical framework for thinking through decisions and problems with Claude*

---

## HOW TO USE THIS

**In any Claude conversation, just say:**
> "Claude, help me think through [MY PROBLEM] systematically using the protocol."

That's it. Claude will guide you through structured thinking.

---

## THE CORE QUESTIONS

### 1. FOUNDATION (Always start here)

**WHAT exactly are we talking about?**
- Define it clearly and specifically
- What are the boundaries?
- What's in scope, what's out?

**WHY does this matter?**
- What problem does this solve?
- What happens if we do nothing?
- What's the real goal (not the stated goal)?

**HOW would this actually work?**
- What's the mechanism?
- What are the steps?
- What needs to happen?

**MEASURE: How will we know if it worked?**
- What specific outcome tells us we succeeded?
- What can we observe/measure?
- When will we know?

---

### 2. EVIDENCE CHECK (Do we actually know this?)

For each major claim, ask:
- What's our evidence for this?
- How strong is that evidence?
- What would contradict this? (actively look for it)
- Are we assuming or do we know?

**Evidence Quality (rough guide):**
- Multiple rigorous studies → Strong
- Single good study or data → Moderate  
- Expert opinion → Weak
- Gut feeling → Very weak
- "Everyone says" → Essentially nothing

**Confidence levels:**
- 0.9+ = Very confident (rare - save this for things you're certain about)
- 0.7-0.9 = Reasonably confident
- 0.5-0.7 = Uncertain, need more info
- <0.5 = Very uncertain, definitely need more

---

### 3. ADVERSARIAL TESTING (Devil's Advocate)

**Borrowed from CIA Structured Analytics - these questions are gold:**

**Pre-mortem:** Imagine this failed catastrophically. Why?
- What went wrong?
- What did we miss?
- What surprised us?

**Key Assumptions Check:**
- What are we assuming must be true?
- What if that assumption is wrong?
- Which assumption, if wrong, would kill this completely?

**Alternative Hypotheses:**
- What's a completely different explanation?
- How would a skeptic see this?
- What would convince us we're wrong?

**Red Team Questions:**
- What are we not thinking about?
- Where are we being overconfident?
- What contradicts our view? (go find it, don't ignore it)

**Indicators:** What would we see if we're on the wrong track?

---

### 4. STAKEHOLDER REALITY CHECK

**Borrowed from Design Thinking - empathy matters:**

WHO is affected by this?
- List everyone (not just obvious stakeholders)
- Who benefits? Who gets hurt?
- Whose support do we need?
- Who can kill this?

For each key stakeholder:
- What do they want?
- What do they fear?
- What's their incentive?
- Will they actually do what we need?

---

### 5. WHAT COULD GO WRONG (Black Swan Check)

**External shocks:**
- Economic: recession, inflation, market crash
- Regulatory: new laws, enforcement changes
- Technology: disruption, obsolescence
- Competitive: someone else does it better
- Social: public opinion shifts, scandal

**Internal failures:**
- Key person leaves
- Funding cut
- Technology doesn't work
- Takes 3x longer than planned
- Political opposition inside org

**Unknown unknowns:**
- "What could happen that we're not even thinking about?"
- If we had to list 5 surprise scenarios, what would they be?

---

### 6. DECISION CRITERIA (How do we choose?)

**Borrowed from Decision Analysis:**

If we have multiple options, score each on:
1. **Effectiveness:** Does it solve the problem? (0-10)
2. **Feasibility:** Can we actually do this? (0-10)
3. **Cost:** Resources required vs. available (0-10, higher = cheaper)
4. **Risk:** What's the downside? (0-10, higher = safer)
5. **Reversibility:** Can we undo this if wrong? (0-10, higher = more reversible)

**Simple Expected Value:**
- If success: What do we gain? (multiply by probability of success)
- If failure: What do we lose? (multiply by probability of failure)
- Net = (Gain × P(success)) - (Loss × P(failure))

---

## PROCESS FLOW

### For DECISIONS:

1. **Define** (WHAT/WHY/HOW/MEASURE)
2. **Gather evidence** (what do we actually know?)
3. **Test assumptions** (pre-mortem, alternatives, red team)
4. **Check stakeholders** (who affects, who's affected?)
5. **Scenario plan** (what could go wrong?)
6. **Decide** (use criteria, calculate if possible)
7. **Set indicators** (how will we know we need to change course?)

### For ANALYSIS:

1. **Frame the question** (WHAT are we analyzing and WHY?)
2. **Gather evidence** (actively seek contradictory evidence)
3. **Alternative explanations** (what else could explain this?)
4. **Test reasoning** (where are logical holes?)
5. **Confidence assessment** (how sure are we, really?)
6. **Limitations** (what don't we know? what can't we know?)

### For COMPLEX PROBLEMS:

1. **Decompose** (break into smaller pieces)
2. **Analyze each piece** (using above)
3. **Dependencies** (what depends on what?)
4. **Critical path** (what must happen in what order?)
5. **Integration** (how do pieces fit together?)
6. **Sensitivity** (which pieces matter most?)

---

## RED FLAGS (When to stop and reconsider)

🚩 **"Everyone agrees"** → Find a dissenter, hear them out
🚩 **"It's obvious"** → It probably isn't, dig deeper  
🚩 **"We've always done it this way"** → Why? Is it still valid?
🚩 **"Trust me"** → Never sufficient for important decisions
🚩 **"We need to decide NOW"** → Artificial urgency, why?
🚩 **"Too complicated to explain"** → Red flag for fuzzy thinking
🚩 **"Everyone else is doing it"** → Irrelevant
🚩 **Can't articulate the downside** → Haven't thought it through
🚩 **No measurement defined** → How will you know if it worked?
🚩 **Confirmation bias visible** → Only looking at supporting evidence

---

## CONFIDENCE CALIBRATION

**Be honest about uncertainty:**

- **Aleatory (random):** Stock prices, weather, dice rolls
  - Can't reduce uncertainty, only manage risk
  - Example: "Will this marketing campaign work?" → Inherently uncertain

- **Epistemic (knowledge gap):** Can reduce by research
  - Should reduce before deciding
  - Example: "What's the market size?" → Can research

**When to proceed despite uncertainty:**
- Low stakes (doesn't matter much if wrong)
- Reversible (can undo easily)
- Learning value (cheap experiment)
- Aleatory uncertainty (can't reduce further)

**When NOT to proceed:**
- High stakes + high uncertainty + irreversible
- Epistemic uncertainty we could reduce
- Contradictory evidence we're ignoring
- Key assumptions untested

---

## COGNITIVE BIAS CHECKLIST

Before finalizing, check for:

- **Confirmation bias:** Am I only looking at supporting evidence?
- **Availability bias:** Am I overweighting recent/memorable examples?
- **Anchoring:** Am I stuck on the first number/idea I heard?
- **Sunk cost:** Am I continuing because I've already invested?
- **Overconfidence:** Am I more certain than I should be?
- **Groupthink:** Is everyone agreeing too easily?
- **Recency bias:** Overweighting what just happened?
- **Optimism bias:** Assuming things will go better than average?

---

## PRACTICAL TEMPLATES

### DECISION TEMPLATE

**Decision:** [State clearly]

**Options:**
1. [Option A]
2. [Option B]
3. Do nothing

**Evaluation:**
| Criteria | Weight | Option A | Option B | Do Nothing |
|----------|--------|----------|----------|------------|
| Effectiveness | 30% | 8/10 | 6/10 | 2/10 |
| Feasibility | 25% | 7/10 | 9/10 | 10/10 |
| Cost | 20% | 5/10 | 8/10 | 10/10 |
| Risk | 15% | 6/10 | 7/10 | 4/10 |
| Reversibility | 10% | 4/10 | 8/10 | 10/10 |
| **Weighted Total** | | **6.7** | **7.4** | **6.0** |

**Key Uncertainties:**
- [What we don't know that matters]

**What could go wrong:**
- [Pre-mortem results]

**Decision:** [Choose, with caveats]

**Indicators to watch:** [What tells us we need to pivot]

---

### PRE-MORTEM TEMPLATE

"It's [6 months/1 year] from now. This failed badly."

**Why did it fail? (List 5-10 reasons)**
1. 
2. 
3. 
4. 
5. 

**For each failure mode:**
- How likely?
- How bad?
- Could we detect it early?
- How could we prevent/mitigate?

---

### ASSUMPTION TEST TEMPLATE

**Key Assumptions:**
1. [Assumption 1]
   - Evidence for: 
   - Evidence against:
   - If wrong, impact: [Critical/Major/Minor]
   - Can we test this?: [How?]

2. [Assumption 2]
   - Evidence for:
   - Evidence against:
   - If wrong, impact:
   - Can we test this?:

**Assumptions to test before proceeding:** [List critical ones]

---

## WHEN TO USE WHAT LEVEL OF RIGOR

**Light (5-10 minutes):**
- Low stakes decisions
- Reversible choices
- Learning experiments
- Use: WHAT/WHY/HOW + quick pre-mortem

**Standard (30-60 minutes):**
- Moderate stakes
- Business decisions
- Personal major decisions
- Use: Full foundation + adversarial testing + stakeholder check

**Deep (2-4 hours):**
- High stakes
- Irreversible
- Affects many people
- Use: Everything + external review + formal scoring

**Know when to stop:** Don't over-analyze low-stakes decisions

---

## LIMITATIONS (Be honest)

**This framework CANNOT:**
- Detect when evidence contradicts your claim (you must do this)
- Account for completely unanticipated events
- Replace domain expertise
- Make the decision for you
- Guarantee correctness
- Tell you your values/priorities

**This framework CAN:**
- Structure your thinking
- Force consideration of alternatives
- Check for major gaps
- Make reasoning transparent
- Reduce some cognitive biases
- Help communicate reasoning

**Use it as a thinking aid, not a crutch.**

---

## QUICK START EXAMPLES

### Example 1: Job Offer

"Claude, I got a job offer. Help me think through it systematically."

Claude will guide you through:
- What exactly is the offer? (specifics)
- Why is this good/bad? (compared to current)
- Evidence about company, role, growth
- Pre-mortem: why might this fail?
- Stakeholder: family, current employer
- Alternative: stay, negotiate, keep looking
- Decision criteria scoring

### Example 2: Product Launch

"Claude, should we launch this product? Think through it systematically."

Claude will guide you through:
- What exactly is the product?
- Why do customers need this?
- Evidence of demand
- Pre-mortem: launch failures
- Stakeholder: customers, team, competitors
- What could go wrong: tech, market, execution
- Go/no-go decision with indicators

### Example 3: Investment Decision

"Claude, I'm considering investing $50K in X. Systematic analysis please."

Claude will guide you through:
- What exactly is the investment?
- Why is this attractive? (thesis)
- Evidence for thesis
- Alternative explanations for data
- Assumptions: market, technology, competition
- What could go wrong: macro, micro, company-specific
- Expected value calculation
- Decision with risk management

---

## FINAL REMINDERS

1. **Garbage in, garbage out:** Framework won't save bad inputs
2. **You decide, not the process:** This aids thinking, doesn't replace it
3. **Adapt to your needs:** Skip irrelevant parts, add what matters
4. **Update as you learn:** If something happens you didn't anticipate, update your mental model
5. **Get external input:** For important decisions, run this by someone who disagrees
6. **Write it down:** Thinking is clearer when written
7. **Review after:** Did this help? What worked? What didn't?

---

## HOW CLAUDE USES THIS

When you ask Claude to "think systematically," Claude will:

1. Start with foundation questions (WHAT/WHY/HOW/MEASURE)
2. Assess what evidence we have vs. need
3. Run adversarial tests (pre-mortem, alternatives, assumptions)
4. Check for blind spots (stakeholders, black swans, biases)
5. Help you decide using appropriate criteria
6. Note key uncertainties and limitations

Claude will adapt the depth to your problem - simple questions get simple treatment, complex decisions get deep analysis.

---

## THIS IS A THINKING TOOL, NOT A RULEBOOK

Use what helps. Skip what doesn't. Adapt as needed.

The goal is **clearer thinking**, not **perfect process**.

---

**Ready to use. No setup required. Just reference this in any Claude conversation.**
