"""
USER SAFETY GUIDE
=================
Essential guidelines for using the Enhanced Analytical Protocol safely.

READ THIS BEFORE MAKING HIGH-STAKES DECISIONS
"""

# =============================================================================
# RED FLAGS: When NOT to Trust the System
# =============================================================================

RED_FLAGS = """
🚨 STOP and reconsider if you see ANY of these:

1. EXTREME CONFIDENCE (>95%)
   - System rarely has enough information for 95%+ confidence
   - High confidence often indicates data problems, not truth
   - Action: Actively seek contradicting evidence

2. ALL EVIDENCE SUPPORTS YOUR HYPOTHESIS
   - Real-world: There's almost always SOME contradicting evidence
   - If you have none, you haven't looked hard enough
   - Action: Play devil's advocate. What could go wrong?

3. VERY FEW EVIDENCE PIECES (<3)
   - System needs at least 3-5 pieces for reliable decisions
   - 1-2 pieces is essentially "trust me bro"
   - Action: Gather more evidence or acknowledge uncertainty

4. EVIDENCE FROM SINGLE SOURCE/TIMEFRAME
   - Clustering suggests redundancy, not independent confirmation
   - Publication bias: Positive results get published multiple times
   - Action: Seek diverse sources across time

5. WEAK EVIDENCE BUT HIGH SCORES
   - If evidence quality is low (0.3-0.5) but scores are high (>0.8)
   - System may have been gamed via weights or priors
   - Action: Review dimension weights and evidence quality

6. VOI > 0.5 (or >50% of decision utility)
   - Unrealistically high Value of Information
   - System assumes perfect information is free and instant
   - Action: Estimate realistic costs and time delays

7. NO FATAL FLAWS DESPITE OBVIOUS PROBLEMS
   - If you know there's a legal, ethical, or safety issue
   - But system shows no fatal flaws
   - Action: TRUST YOUR JUDGMENT, not the numbers

8. COMPLEX MECHANISM MAP (>50 nodes)
   - Analysis becomes unreliable with complexity
   - More nodes ≠ more understanding
   - Action: Simplify or use different approach
"""

# =============================================================================
# SAFE USAGE CHECKLIST
# =============================================================================

SAFE_USAGE_CHECKLIST = """
✓ BEFORE TRUSTING A DECISION:

EVIDENCE QUALITY:
[ ] Do I have at least 3-5 independent pieces of evidence?
[ ] Is evidence from diverse sources (not just one)?
[ ] Have I actively sought contradicting evidence?
[ ] Is evidence quality accurately rated (not inflated)?
[ ] Are study sizes adequate for conclusions drawn?

SYSTEM INPUTS:
[ ] Are dimension weights justified (not gamed)?
[ ] Is initial credence reasonable (not anchored at 0.5)?
[ ] Have I checked for evidence redundancy?
[ ] Are causal levels appropriate for study designs?
[ ] Have I considered confounding factors?

OUTPUTS:
[ ] Is confidence level reasonable (<90% for most decisions)?
[ ] Does recommendation match my domain expertise?
[ ] Have I reviewed the sensitivity analysis?
[ ] Are there any system warnings I'm ignoring?
[ ] Can I explain WHY the system reached this conclusion?

REALITY CHECK:
[ ] Would I make this decision without the system?
[ ] Does this pass the "common sense" test?
[ ] Have I consulted domain experts?
[ ] What's the worst-case scenario if I'm wrong?
[ ] Am I being overconfident due to mathematical rigor?
"""

# =============================================================================
# DECISION THRESHOLDS
# =============================================================================

DECISION_THRESHOLDS = """
HOW TO INTERPRET SCORES:

BAYESIAN SCORE / CREDENCE:
0.90 - 1.00  : HIGH CONFIDENCE (but verify it's justified!)
0.70 - 0.89  : MODERATE CONFIDENCE (reasonable for most decisions)
0.50 - 0.69  : UNCERTAIN (gather more info or accept risk)
0.30 - 0.49  : LIKELY FALSE (but consider cost of Type II error)
0.00 - 0.29  : LOW CONFIDENCE (strong evidence against)

CALIBRATED SCORE (after bias adjustment):
This is your MOST RELIABLE score IF:
- You have 20+ historical predictions
- Your past predictions were honest
- You tracked actual outcomes

VALUE OF INFORMATION:
<0.10 : Not worth gathering more info, decide now
0.10-0.30 : Consider gathering info if cheap/fast
0.30-0.50 : Probably worth gathering more info
>0.50 : Likely overestimate, check assumptions

FATAL FLAWS:
If ANY dimension is marked fatal: REJECT
No exceptions, no matter what other scores say.
"""

# =============================================================================
# COMMON MISTAKES
# =============================================================================

COMMON_MISTAKES = """
⚠️  TOP 10 MISTAKES USERS MAKE:

1. TRUSTING PRECISE NUMBERS
   Wrong: "System says 0.847, so I'm 84.7% sure"
   Right: "System suggests ~0.8-0.9 range"

2. IGNORING DOMAIN EXPERTISE
   Wrong: "System says approve, so approve"
   Right: "System suggests approve, but my experience says no..."

3. GAMING THE SYSTEM
   Wrong: Setting weights to get desired outcome
   Right: Setting weights based on true importance

4. COUNTING REDUNDANT EVIDENCE
   Wrong: Adding 5 sources citing same study
   Right: Noting redundancy, counting as 1-2 pieces

5. OVERWEIGHTING RCT DESIGN
   Wrong: "It's an RCT so it's automatically better"
   Right: "RCT with n=50 and 50% dropout is worse than good cohort"

6. IGNORING PRIORS
   Wrong: Starting at 50/50 for everything
   Right: Starting at appropriate prior (base rate)

7. STOPPING AT CONFIRMATION
   Wrong: "Found supporting evidence, we're done"
   Right: "Found support, now actively seek contradictions"

8. ASSUMING INDEPENDENCE
   Wrong: "3 studies = 3 independent data points"
   Right: "Do these 3 studies share data/authors/funding?"

9. MISUSING VOI
   Wrong: "VOI is 0.6, so let's spend $10M studying this"
   Right: "VOI suggests gathering info IF cost/time permits"

10. IGNORING BLACK SWANS
    Wrong: "No evidence of risk = safe"
    Right: "Absence of evidence ≠ evidence of absence"
"""

# =============================================================================
# WHEN TO OVERRIDE THE SYSTEM
# =============================================================================

OVERRIDE_GUIDELINES = """
YOU SHOULD OVERRIDE THE SYSTEM WHEN:

LEGAL/ETHICAL:
- System approves but action violates laws
- Ethical concerns not captured in utility model
- Potential for harm to vulnerable populations

DOMAIN EXPERTISE:
- Conflicts with established domain knowledge
- Ignores context system cannot capture
- Misses critical factors in your industry

INCOMPLETE MODELING:
- Political/social factors not in model
- Regulatory changes on horizon
- Competitive dynamics misunderstood
- Second-order effects not captured

TIMING:
- System says "gather more info" but decision is time-critical
- Market window closing
- First-mover advantage matters

BLACK SWAN RISKS:
- Low probability, catastrophic outcomes
- Existential risks to organization
- Reputational damage scenarios
- Irreversible decisions

REMEMBER: You have the ultimate responsibility.
The system is a tool to SUPPORT your decision-making,
not to REPLACE your judgment.
"""

# =============================================================================
# SAFE USAGE PATTERNS
# =============================================================================

SAFE_PATTERNS = """
✓ RECOMMENDED USAGE PATTERNS:

PATTERN 1: EXPLORATION MODE
- Use system to STRUCTURE your thinking
- Identify gaps in your analysis
- Generate hypotheses to test
- Find blind spots you missed
→ DON'T use scores as final decision

PATTERN 2: SENSITIVITY ANALYSIS MODE
- Test how robust your decision is
- Find critical assumptions
- Identify information that would change decision
- Understand your vulnerabilities
→ DON'T rely on base case score alone

PATTERN 3: TEAM ALIGNMENT MODE
- Make reasoning explicit and sharable
- Document decision rationale
- Facilitate structured discussion
- Create audit trail
→ DON'T let numbers replace debate

PATTERN 4: BIAS CHECK MODE
- Challenge your initial intuition
- Force consideration of alternatives
- Seek contradicting evidence
- Test for overconfidence
→ DON'T assume system catches all biases

PATTERN 5: LEARNING MODE
- Track predictions vs outcomes
- Build calibration over time
- Learn from mistakes
- Improve estimation skills
→ DON'T use until you have 20+ predictions logged
"""

# =============================================================================
# EMERGENCY CHECKLIST
# =============================================================================

EMERGENCY_CHECKLIST = """
🚨 HIGH-STAKES DECISION EMERGENCY CHECKLIST:

If decision has major consequences ($1M+, lives, reputation):

[ ] Have I slept on this decision?
[ ] Have I consulted 3+ domain experts?
[ ] Have I actively sought dissenting opinions?
[ ] Have I considered 2nd and 3rd order effects?
[ ] Have I stress-tested against worst-case scenarios?
[ ] Have I checked for motivated reasoning?
[ ] Can I explain this decision to a skeptical board?
[ ] Would I bet my job on this being correct?
[ ] Have I documented my reasoning for future review?
[ ] Am I prepared for this to be wrong?

If you answered NO to any of these: SLOW DOWN.
"""

# =============================================================================
# QUICK REFERENCE CARD
# =============================================================================

QUICK_REFERENCE = """
═══════════════════════════════════════════════════════════
                    QUICK REFERENCE CARD
═══════════════════════════════════════════════════════════

STOP SIGNS 🛑
- Credence > 95%
- All evidence supports hypothesis
- Evidence < 3 pieces
- VOI > 0.5
- No contradicting evidence
- Complex mechanism (>50 nodes)

MINIMUM REQUIREMENTS ✓
- 3-5 independent evidence pieces
- Evidence from diverse sources
- Actively sought contradictions
- Realistic dimension weights
- Domain expert review

SCORE INTERPRETATION 📊
0.90+  : High confidence (verify!)
0.70-0.89 : Moderate (good for most decisions)
0.50-0.69 : Uncertain (gather info or accept risk)
<0.50  : Low confidence

OVERRIDE IF 🚫
- Legal/ethical issues
- Domain expertise conflicts
- Missing critical context
- Time-critical decision
- Black swan risks

REMEMBER 💡
More complexity ≠ More correctness
Trust your judgment + the system
Document your reasoning
Track outcomes to calibrate
═══════════════════════════════════════════════════════════
"""

# =============================================================================
# SELF-ASSESSMENT
# =============================================================================

SELF_ASSESSMENT = """
SELF-ASSESSMENT: Are you using the system safely?

Answer honestly:

1. Do you check if evidence is independent?
   [ ] Always  [ ] Sometimes  [ ] Rarely

2. Do you actively seek contradicting evidence?
   [ ] Always  [ ] Sometimes  [ ] Rarely

3. Do you override the system when appropriate?
   [ ] Always  [ ] Sometimes  [ ] Rarely

4. Do you treat scores as ranges, not precise?
   [ ] Always  [ ] Sometimes  [ ] Rarely

5. Do you consult domain experts?
   [ ] Always  [ ] Sometimes  [ ] Rarely

6. Do you track predictions vs outcomes?
   [ ] Always  [ ] Sometimes  [ ] Rarely

SCORING:
All "Always": ✓ You're using the system safely
Mix of A/S: ⚠️  Room for improvement
Any "Rarely": 🚨 High risk of misuse

IF YOU HAVE MORE THAN 2 "RARELY" RESPONSES:
Stop making high-stakes decisions with the system until
you improve your practices.
"""

# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_safety_guide():
    """Print complete safety guide"""
    print("\n" + "=" * 70)
    print("ENHANCED ANALYTICAL PROTOCOL - USER SAFETY GUIDE")
    print("=" * 70)
    
    print(RED_FLAGS)
    print("\n" + "=" * 70)
    print(SAFE_USAGE_CHECKLIST)
    print("\n" + "=" * 70)
    print(DECISION_THRESHOLDS)
    print("\n" + "=" * 70)
    print(COMMON_MISTAKES)
    print("\n" + "=" * 70)
    print(OVERRIDE_GUIDELINES)
    print("\n" + "=" * 70)
    print(SAFE_PATTERNS)
    print("\n" + "=" * 70)
    print(EMERGENCY_CHECKLIST)
    print("\n" + "=" * 70)
    print(QUICK_REFERENCE)
    print("\n" + "=" * 70)
    print(SELF_ASSESSMENT)


def print_quick_reference():
    """Print just the quick reference card"""
    print(QUICK_REFERENCE)


if __name__ == "__main__":
    print_safety_guide()
    
    print("\n\n" + "=" * 70)
    print("REMEMBER THE GOLDEN RULE:")
    print("=" * 70)
    print("""
    The system is a TOOL, not an ORACLE.
    
    It helps you think more clearly.
    It doesn't think FOR you.
    
    Your judgment + System's structure = Good decisions
    System alone = Overconfident mistakes
    Your judgment alone = Unstructured thinking
    
    Use both. Stay humble. Track outcomes. Learn.
    """)
    print("=" * 70)
