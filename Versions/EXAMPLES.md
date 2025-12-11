# EXAMPLES - Real Analyses Using the Protocol

These are realistic examples showing how to actually use the protocol.

---

## EXAMPLE 1: Job Offer Decision

### Context
Got offer for Senior Engineer at a startup. Current job is stable but boring. Need to decide.

### Foundation

**WHAT:** Senior Engineer at TechStartupX
- $150K base + equity (0.5%)
- Remote, team of 12 engineers
- Series B funded ($20M), 18mo runway
- Building B2B SaaS platform

**Confidence in WHAT:** 0.9 (details are clear)

**WHY:** Better growth and compensation
- Current role: stagnant, limited learning
- New role: technical lead, greenfield projects
- Compensation: +$30K + equity upside
- Team: respected engineers, good culture fit

**Confidence in WHY:** 0.6 (need to validate assumptions)

**HOW:** Transition plan
- Give 2 weeks notice
- Complete handoff docs
- Start March 1st
- First 90 days: ramp up, deliver quick win

**Confidence in HOW:** 0.8 (straightforward)

**MEASURE:** Am I happier and growing in 6 months?
- Learning: shipping significant features
- Growth: mastering new tech stack
- Happiness: excited about Monday mornings
- Financial: equity increasing in value

**Confidence in MEASURE:** 0.7 (somewhat subjective but trackable)

### Evidence

**Evidence 1:** Glassdoor reviews (3.8/5)
- Quality: 0.5 (case-series level, self-selected)
- Positive: good culture, smart people
- Negative: some process issues, high pressure

**Evidence 2:** Friend worked there
- Quality: 0.3 (single anecdote)
- Says: "Best team I've worked with, but chaotic"

**Evidence 3:** Funding and growth metrics
- Quality: 0.7 (public data, verified)
- Series B from reputable VC
- 40% YoY revenue growth
- 18-month runway (verified)

**WHY confidence after evidence:** 0.72 (up from 0.6)

### Adversarial Testing - Cycle 1: Pre-mortem

"It's 1 year from now. Taking this job was a mistake."

**Why did it fail?**

1. **Company ran out of money** (Severity: 0.85)
   - Startup risk, burn rate uncertain
   - Market conditions could worsen
   - Response: Check runway carefully, ask about burn rate
   - Mitigation: Keep 6 months expenses saved

2. **Culture was actually toxic** (Severity: 0.7)
   - Glassdoor reviews are filtered
   - Friend might be outlier
   - Response: Talk to more employees before accepting
   - Mitigation: Ask for 1-day trial/shadow

3. **Role became boring quickly** (Severity: 0.5)
   - Greenfield might mean messy legacy soon
   - Tech stack might not be interesting
   - Response: Ask about tech debt, architectural decisions
   - Mitigation: Reversible - can leave after 1 year

4. **Equity became worthless** (Severity: 0.6)
   - Most startups fail
   - 0.5% might be nothing
   - Response: Don't count on equity
   - Mitigation: Base salary alone should be acceptable

5. **Remote isolation** (Severity: 0.4)
   - Miss office social interaction
   - Response: Join coworking space
   - Mitigation: Low severity, manageable

### Adversarial Testing - Cycle 2: Assumptions

**Key Assumption 1:** "The company will survive 18+ months"
- Evidence FOR: Just raised $20M, strong revenue growth
- Evidence AGAINST: Burn rate unknown, market competitive
- If wrong: Unemployed in 18 months (critical)
- Can test: Ask about burn rate, path to profitability
- **Action: MUST test before accepting**

**Key Assumption 2:** "I'll actually learn and grow"
- Evidence FOR: Greenfield project, smart team
- Evidence AGAINST: Friend said "chaotic" - might be firefighting
- If wrong: Waste time, hurt resume (major)
- Can test: Ask what % time is new features vs. maintenance
- **Action: Test in interview**

**Key Assumption 3:** "Culture fit is real"
- Evidence FOR: Friend's experience, Glassdoor
- Evidence AGAINST: Small sample, possibly cherry-picked
- If wrong: Miserable at work (major)
- Can test: Talk to more engineers, shadow day
- **Action: Test before accepting**

### Adversarial Testing - Cycle 3: Alternative Explanation

**Alternative hypothesis:** "I'm running away from boredom, not toward growth"

Evidence FOR this alternative:
- Current dissatisfaction driving decision
- New role might have same issues eventually
- "Grass is greener" bias

Evidence AGAINST:
- Specific growth opportunities identified
- Technical challenges are real
- Compensation is objectively better

**Conclusion:** This IS partially running away, but to something genuinely better. Acceptable.

### Adversarial Testing - Cycle 4: Black Swans

**External shocks:**
1. Economic recession → startups struggle (likelihood: 0.3, impact: 0.9)
2. Major competitor launches better product (likelihood: 0.2, impact: 0.7)
3. Key founder leaves (likelihood: 0.1, impact: 0.8)

**Mitigation:** Keep savings high, maintain industry connections, stay marketable

### Quality Assessment

**Completeness:** 
- WHAT: filled × 0.9 conf × 1.5 weight = 1.35
- WHY: filled × 0.72 conf × 1.5 weight = 1.08
- HOW: filled × 0.8 conf × 1.0 weight = 0.8
- MEASURE: filled × 0.7 conf × 1.0 weight = 0.7
- Total: 3.93 / 5.0 = **0.79**

**Average confidence:** (0.9 + 0.72 + 0.8 + 0.7) / 4 = **0.78**

**Evidence quality:** (0.5 + 0.3 + 0.7) / 3 = **0.5**

**Internal consistency:** 3 critical issues identified, all addressed = **1.0**

**Efficiency:** 4 cycles, target was ~5 = **1.0**

**Overall quality:** 0.30×0.79 + 0.20×0.78 + 0.20×0.5 + 0.20×1.0 + 0.10×1.0 = **0.75**

### Decision

**Choice:** Accept the offer, BUT with conditions:

**Conditions:**
1. Verify 18-month runway (ask for details)
2. Talk to 2 more engineers
3. Ask for 1-day trial/shadow
4. Negotiate 4 weeks instead of 2 weeks notice (reversibility)

**Confidence in decision:** 0.75

**Indicators to watch (first 90 days):**
- ✅ On track: Shipping code, learning new tech, enjoying work
- ⚠️ Warning: Constant firefighting, no learning, cultural issues
- 🛑 Abort: Company financials deteriorate, key people leaving

**Reversibility:** High (can leave after 6-12 months)

**Outcome check:** Review in 6 months, did we hit MEASURE criteria?

---

## EXAMPLE 2: Product Feature Decision

### Context
Product manager at SaaS company. Users requesting social sharing feature. Should we build it?

### Foundation

**WHAT:** Social sharing functionality
- Users can share content to Twitter, LinkedIn, Facebook
- One-click share with auto-generated preview
- Track shares via analytics
- Estimated: 2 engineering sprints (4 weeks)

**Confidence in WHAT:** 0.85

**WHY:** User requests + competitive pressure
- 15 users requested it in last quarter
- Top 3 competitors all have it
- Could drive viral growth
- **But: No data showing sharing drives growth for us**

**Confidence in WHY:** 0.4 (weak justification)

**HOW:** Standard integration
- OAuth with platforms
- Use their sharing APIs
- Design UI for share buttons
- Add analytics tracking

**Confidence in HOW:** 0.9 (straightforward technically)

**MEASURE:** >20% of active users share something weekly

**Confidence in MEASURE:** 0.8 (clear metric)

### Evidence

**Evidence 1:** 15 user requests
- Quality: 0.3 (users say they want it, but will they use it?)
- Note: Sample of 15 out of 5000 users (0.3%)

**Evidence 2:** Competitors have it
- Quality: 0.2 (not evidence it works, just that others did it)

**Evidence 3:** No data on whether sharing drives growth
- Quality: 0.0 (absence of evidence)
- **This is the critical gap**

**WHY confidence after evidence:** 0.35 (actually went DOWN - good!)

### Adversarial Testing - Cycle 1: Pre-mortem

"We built this feature. 6 months later, usage is <5%."

**Why did it fail?**

1. **Users SAY they want it but don't USE it** (Severity: 0.9)
   - Classic product trap
   - No evidence of actual need
   - Response: This is the core issue

2. **Sharing doesn't fit our use case** (Severity: 0.8)
   - B2B tool, not consumer
   - Users don't want to share work publicly
   - Response: Good point, need to think about context

3. **Engineering effort wasted** (Severity: 0.7)
   - 4 weeks on low-usage feature
   - Could have built something impactful
   - Opportunity cost

### Adversarial Testing - Cycle 2: Assumptions

**Key Assumption 1:** "Users will actually share"
- Evidence FOR: They requested it
- Evidence AGAINST: Requests ≠ usage, no behavior data
- If wrong: Wasted 4 weeks (critical)
- Can test: YES - build fake share button, track click-through
- **Action: MUST test assumption first**

**Key Assumption 2:** "Sharing will drive growth"
- Evidence FOR: None
- Evidence AGAINST: None (no data)
- If wrong: Feature doesn't move metrics (critical)
- Can test: Survey users on sharing behavior, look at competitor data
- **Action: Test before building**

**Key Assumption 3:** "Competitors have it because it works"
- Evidence FOR: None (cargo cult thinking)
- Evidence AGAINST: They might have same problem
- If wrong: Following failed strategy
- Can test: Talk to competitor users
- **Action: Don't assume**

### Alternative Hypothesis

**Alternative:** "Users are requesting this because they don't know what they actually need"

Evidence FOR:
- Small sample (0.3% of users)
- No behavior data showing they'd use it
- B2B context doesn't encourage sharing

**Better alternative:** "Users want OUTCOMES (reach, visibility) not FEATURES (share button)"

Could we achieve those outcomes differently?
- Better SEO
- Email sharing (more appropriate for B2B)
- Team invites

### Quality Assessment

**Completeness:** 0.73 (good)
**Average confidence:** 0.6 (moderate)
**Evidence quality:** 0.17 (very low - this is the issue)
**Internal consistency:** 1.0 (issues addressed)
**Efficiency:** 1.0

**Overall quality:** 0.30×0.73 + 0.20×0.6 + 0.20×0.17 + 0.20×1.0 + 0.10×1.0 = **0.55** (MEDIOCRE)

### Decision

**DO NOT build full feature yet**

**Instead:**
1. Add fake share button, track clicks (2 dev days)
2. Survey users: "Would you share? How often?" (1 day)
3. Run for 2 weeks, gather data
4. If >20% click fake button → build real feature
5. If <20% → decline feature, investigate real need

**Why this is better:**
- Tests assumption cheaply
- Reduces risk (2 days vs 4 weeks)
- Gets real behavior data
- Reversible

**Confidence in decision:** 0.8

**This is what the protocol does well:** Found the untested assumption that would have led to waste.

---

## EXAMPLE 3: Investment Decision (Brief)

### Foundation

**WHAT:** $50K into friend's startup (pre-seed round)

**WHY:** 
- Friend is smart, good team
- Market is big ($10B)
- Product has traction (100 users)
- **Confidence: 0.5** (medium uncertainty)

### Critical Pre-mortem Finding

**Failure:** "Lost entire $50K"

**Why:** Startup failed (as 90% do)

**Key realization:** Am I ok losing $50K completely?
- If yes → acceptable risk
- If no → don't invest

### Alternative Hypothesis

"I'm investing because he's my friend, not because it's a good investment"

Evidence: I wouldn't invest in stranger with same metrics

**Decision:** Invest $10K (amount I'm OK losing completely), not $50K

**Quality score:** 0.68 → proceed with reduced amount

---

## KEY TAKEAWAYS FROM EXAMPLES

### What the Protocol Does Well:

1. **Forces honesty about evidence quality**
   - Example 2: Realized evidence was weak (0.17)
   - Led to "test first" decision instead of build

2. **Finds untested assumptions**
   - Example 2: "Users will actually use it" was untested
   - Prevented 4-week waste

3. **Pre-mortem reveals real risks**
   - Example 1: Company financial risk was critical
   - Led to specific mitigation (check runway)

4. **Quality scores guide decisions**
   - 0.75 → proceed with conditions
   - 0.55 → test first, don't commit
   - 0.68 → proceed with reduced risk

5. **Iteration finds gaps**
   - First pass often looks good
   - Cycles 2-4 reveal hidden issues

### Common Patterns:

- **Low evidence quality → test assumptions first**
- **High severity pre-mortem items → must mitigate**
- **Alternative hypotheses reveal bias**
- **Quality <0.7 → more work needed**

### When Quality Score Saved Me:

- Example 2: Would have built feature without testing
- Protocol forced evidence check
- 0.55 quality → realized need to validate first
- Saved 4 weeks of engineering time

---

**Use these as templates for your own analyses.**
