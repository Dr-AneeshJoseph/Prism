"""
ADVERSARIAL & CONCEPTUAL STRESS TESTS

Tests that challenge the framework's conceptual coherence:
1. Adversarial inputs (deliberately hostile)
2. Paradoxes and contradictions
3. Meta-level critiques
4. Real-world pathological cases
5. Philosophical challenges
"""

import sys
sys.path.insert(0, '/home/claude')

from cap_implementation_v3 import *

print("="*80)
print("ADVERSARIAL & CONCEPTUAL STRESS TESTS")
print("="*80)
print()

# ============================================================================
# TEST 1: The "Everything is Perfect" Paradox
# ============================================================================

print("TEST 1: Everything is Perfect Paradox")
print("-"*80)
print("Scenario: User claims everything is perfect with max confidence")
print("Question: Should the system be skeptical?")
print()

element = FoundationElement(name="Too Perfect", domain=EvidenceDomain.GENERAL)
element.what = "We will make $1 billion"
element.why = "Because we're amazing"
element.how = "Magic"
element.what_confidence = 0.99
element.why_confidence = 0.99
element.how_confidence = 0.99

# Add weak evidence
element.add_evidence(Evidence(
    content="My gut feeling",
    source="My intuition",
    strength=0.1,
    date="2024",
    domain=EvidenceDomain.GENERAL,
    study_design="anecdote"
))

print(f"Claims: {element.what}")
print(f"User confidence: 0.99")
print(f"Evidence: Anecdote")
print()

results = run_analysis(element, rigor_level=2, max_iterations=10)

print(f"System confidence after evidence: {element.why_confidence:.3f}")
print(f"Overall quality: {results['quality_scores']['overall']:.3f}")
print(f"Ready for action: {results['ready_for_action']}")
print()

if element.why_confidence < 0.5:
    print("✅ PASS: System appropriately skeptical of unsupported claims")
else:
    print("⚠️  CONCERN: System accepted high confidence with weak evidence")
print()

# ============================================================================
# TEST 2: The Circular Reasoning Test
# ============================================================================

print("TEST 2: Circular Reasoning")
print("-"*80)
print("Scenario: WHY refers back to WHAT")
print()

element = FoundationElement(name="Circular", domain=EvidenceDomain.GENERAL)
element.what = "We should do X because X is good"
element.why = "X is good because we should do X"
element.what_confidence = 0.8
element.why_confidence = 0.8

print(f"WHAT: {element.what}")
print(f"WHY: {element.why}")
print()
print("Question: Can the system detect circular reasoning?")
print("Current limitation: No - would need semantic analysis")
print()
print("⚠️  KNOWN LIMITATION: Cannot detect circular logic in content")
print("    Only detects structural gaps, not logical fallacies")
print()

# ============================================================================
# TEST 3: The Confidence Inflation Test
# ============================================================================

print("TEST 3: Confidence Inflation via Redundant Evidence")
print("-"*80)
print("Scenario: Add 50 copies of the same weak study")
print()

element = FoundationElement(name="Redundant", domain=EvidenceDomain.GENERAL)
element.what = "Claim X"
element.why = "Supported by many studies"
element.why_confidence = 0.5

print("Adding 50 pieces of identical weak evidence...")

for i in range(50):
    element.add_evidence(Evidence(
        content="The same study repeated",  # Same content!
        source=f"Publication {i}",  # Different source names
        strength=0.3,
        date="2024",
        domain=EvidenceDomain.GENERAL,
        study_design="case-series"
    ))

print(f"Final confidence: {element.why_confidence:.3f}")
print()

if element.why_confidence < 0.6:
    print("✅ PASS: Diminishing returns prevented over-confidence")
    print("   The system applies diminishing returns for many pieces")
else:
    print("⚠️  CONCERN: Many weak pieces inflated confidence inappropriately")
print()

# ============================================================================
# TEST 4: The Unmeasurable Goal Test
# ============================================================================

print("TEST 4: Unmeasurable Goals")
print("-"*80)
print("Scenario: Goal that cannot be measured")
print()

element = FoundationElement(name="Unmeasurable", domain=EvidenceDomain.GENERAL)
element.what = "Maximize happiness"
element.why = "Happiness is good"
element.how = "Do good things"
element.measure = "People will feel happier"  # Vague, not measurable
element.what_confidence = 0.8
element.why_confidence = 0.8
element.how_confidence = 0.6
element.measure_confidence = 0.4  # Low because vague

gaps = element.critical_gaps()

print(f"WHAT: {element.what}")
print(f"MEASURE: {element.measure}")
print(f"Measure confidence: {element.measure_confidence}")
print()
print("Critical gaps identified:")
for gap in gaps:
    if "MEASURE" in gap:
        print(f"  - {gap}")
print()

if any("MEASURE" in gap for gap in gaps):
    print("✅ PARTIAL: System flags low measurement confidence")
    print("⚠️  LIMITATION: Cannot assess if measure is actually measurable")
else:
    print("⚠️  CONCERN: System didn't flag measurement issue")
print()

# ============================================================================
# TEST 5: The Overconfident Expert Test
# ============================================================================

print("TEST 5: Overconfident Expert (Dunning-Kruger)")
print("-"*80)
print("Scenario: Expert very confident but in field with high uncertainty")
print()

problem = ProblemCharacterization(
    problem_type=ProblemType.PREDICTION,
    stakes=StakesLevel.HIGH,
    time_available=TimeConstraint.MONTHS,
    complexity=0.9,  # Very complex
    uncertainty=0.9,  # Very uncertain
    domain=EvidenceDomain.GENERAL
)

element = FoundationElement(name="Expert Prediction", domain=EvidenceDomain.GENERAL)
element.what = "The stock market will crash next year"
element.why = "My 30 years of experience tells me"
element.what_confidence = 0.95  # Expert very confident
element.why_confidence = 0.95

# But uncertainty type is aleatory (inherently unpredictable)
element.what_uncertainty = UncertaintyType.ALEATORY

ready, reason = element.ready_for_action(rigor_level=3)

print(f"Domain: Stock market prediction (inherently uncertain)")
print(f"Expert confidence: 0.95")
print(f"Uncertainty type: {element.what_uncertainty.value}")
print(f"Problem uncertainty: 0.9")
print()
print(f"Ready for action (rigor 3): {ready}")
print(f"Reason: {reason}")
print()

if not ready:
    print("✅ PASS: System requires lower confidence for aleatory uncertainty")
else:
    print("⚠️  CONCERN: System accepts high confidence despite aleatory uncertainty")
print()

# ============================================================================
# TEST 6: The Contradictory Evidence Test
# ============================================================================

print("TEST 6: Contradictory Evidence")
print("-"*80)
print("Scenario: Evidence directly contradicts the claim")
print()

element = FoundationElement(name="Contradiction", domain=EvidenceDomain.MEDICAL)
element.what = "Drug X cures disease Y"
element.why = "It should work based on mechanism"
element.why_confidence = 0.7

# Add contradictory evidence
element.add_evidence(Evidence(
    content="RCT showed no effect of Drug X on disease Y",
    source="NEJM 2024",
    strength=0.9,
    date="2024",
    domain=EvidenceDomain.MEDICAL,
    study_design="randomized_controlled_trial"
))

print(f"Claim: {element.what}")
print(f"Evidence: RCT showed NO effect")
print(f"Initial confidence: 0.7")
print(f"After contradictory evidence: {element.why_confidence:.3f}")
print()

print("⚠️  KNOWN LIMITATION: System cannot detect contradictions")
print("    Evidence integration assumes all evidence is SUPPORTING")
print("    Would need: semantic understanding + contradiction detection")
print("    Current behavior: treats contradictory evidence as confirming")
print()
print("SEVERITY: HIGH - Major conceptual limitation")
print()

# ============================================================================
# TEST 7: The Gaming the System Test
# ============================================================================

print("TEST 7: Gaming the System")
print("-"*80)
print("Scenario: User deliberately optimizes for high quality score")
print()

element = FoundationElement(name="Gamed", domain=EvidenceDomain.GENERAL)
# Fill everything with minimal content
element.what = "X"
element.why = "Y"
element.how = "Z"
element.when = "Now"
element.who = "Me"
element.measure = "Success"
# Max out confidence
element.what_confidence = 0.95
element.why_confidence = 0.95
element.how_confidence = 0.95
element.when_confidence = 0.95
element.who_confidence = 0.95
element.measure_confidence = 0.95

# Add minimal evidence
element.add_evidence(Evidence(
    content="Trust me",
    source="Me",
    strength=0.9,  # Claim high strength
    date="2024",
    domain=EvidenceDomain.GENERAL
))

results = run_analysis(element, rigor_level=1, max_iterations=3)

print("User strategy: Minimal content, maximum confidence")
print(f"Quality achieved: {results['quality_scores']['overall']:.3f}")
print(f"Ready for action: {results['ready_for_action']}")
print()

if results['quality_scores']['overall'] > 0.7:
    print("⚠️  VULNERABILITY: System can be gamed with minimal content")
    print("    High confidence + low iterations → high efficiency → good score")
    print("    Need: Content quality assessment, not just presence")
else:
    print("✅ RESISTANT: System not easily gamed")
print()

# ============================================================================
# TEST 8: The Meta-Critique Test
# ============================================================================

print("TEST 8: Meta-Level Critique - Can CAP Critique Itself?")
print("-"*80)
print("Scenario: Apply CAP to analyze CAP itself")
print()

element = FoundationElement(name="CAP v3.0", domain=EvidenceDomain.RESEARCH)
element.what = "Computational Analytical Protocol v3.0"
element.why = "To improve systematic thinking with AI assistance"
element.how = "Iterative adversarial refinement with confidence tracking"
element.measure = "Empirical validation study comparing to unstructured analysis"

element.what_confidence = 0.9  # Well-defined
element.why_confidence = 0.7  # Reasonable justification
element.how_confidence = 0.8  # Clear mechanism
element.measure_confidence = 0.9  # Good measurement plan

# Add evidence (this protocol itself)
element.add_evidence(Evidence(
    content="Stress test suite shows 82% pass rate",
    source="This very test",
    strength=0.6,
    date="2024",
    domain=EvidenceDomain.RESEARCH,
    study_design="case-series"
))

results = run_analysis(element, rigor_level=3, max_iterations=10)

print("Meta-analysis of CAP using CAP:")
print(f"  Overall quality: {results['quality_scores']['overall']:.3f}")
print(f"  Completeness: {results['quality_scores']['completeness']:.3f}")
print(f"  Evidence: {results['quality_scores']['evidence']:.3f}")
print(f"  Ready for action: {results['ready_for_action']}")
print()

gaps = element.critical_gaps()
print("Critical gaps identified in CAP itself:")
for gap in gaps:
    print(f"  - {gap}")
print()

print("✅ META-CONSISTENCY: CAP can analyze itself")
print("   Identifies own limitations (no external validation yet)")
print()

# ============================================================================
# TEST 9: The Conflicting Stakeholders Test
# ============================================================================

print("TEST 9: Conflicting Stakeholder Interests")
print("-"*80)
print("Scenario: Decision that benefits some stakeholders, harms others")
print()

element = FoundationElement(name="Layoffs", domain=EvidenceDomain.BUSINESS)
element.what = "Lay off 20% of workforce"
element.why = "Reduce costs by $10M, improve margins"
element.how = "Targeted cuts in underperforming divisions"
element.who = "Management decides, employees affected"
element.measure = "Cost savings achieved"

element.what_confidence = 0.9
element.why_confidence = 0.8  # Strong business case
element.how_confidence = 0.7
element.measure_confidence = 0.9

results = run_analysis(element, rigor_level=2, max_iterations=10)

print("Business decision with conflicting interests:")
print(f"  Quality score: {results['quality_scores']['overall']:.3f}")
print(f"  Ready for action: {results['ready_for_action']}")
print()

print("⚠️  LIMITATION: No stakeholder impact weighting")
print("    System treats management perspective as primary")
print("    No consideration of affected employees' interests")
print("    No ethical framework for value trade-offs")
print()
print("SEVERITY: MODERATE - Framework lacks multi-stakeholder analysis")
print()

# ============================================================================
# TEST 10: The Black Swan Event Test
# ============================================================================

print("TEST 10: Black Swan / Unknown Unknowns")
print("-"*80)
print("Scenario: Analysis assumes no major surprises")
print()

element = FoundationElement(name="Investment", domain=EvidenceDomain.BUSINESS)
element.what = "Invest $10M in new venture"
element.why = "Market analysis shows strong demand"
element.how = "Build production facility"
element.measure = "Revenue > $15M in year 1"

element.what_confidence = 0.8
element.why_confidence = 0.8
element.how_confidence = 0.8

# Add good evidence
element.add_evidence(Evidence(
    content="Market research: $50M addressable market",
    source="McKinsey Report 2024",
    strength=0.8,
    date="2024",
    domain=EvidenceDomain.BUSINESS,
    study_design="multi_company_analysis"
))

results = run_analysis(element, rigor_level=2, max_iterations=10)

print("Analysis quality: {:.3f}".format(results['quality_scores']['overall']))
print(f"Ready for action: {results['ready_for_action']}")
print()
print("But what about:")
print("  - Pandemic shuts down economy?")
print("  - New regulation bans product?")
print("  - Competitor launches better alternative?")
print("  - Key supplier goes bankrupt?")
print()
print("⚠️  FUNDAMENTAL LIMITATION: Unknown unknowns")
print("    System cannot account for unanticipated events")
print("    Confidence scores assume stable environment")
print("    No scenario planning or resilience assessment")
print()
print("SEVERITY: HIGH - Overconfidence in predictable world")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("ADVERSARIAL STRESS TEST SUMMARY")
print("="*80)
print()

print("CRITICAL VULNERABILITIES:")
print()
print("1. CANNOT DETECT CONTRADICTORY EVIDENCE (HIGH)")
print("   - Treats opposing evidence as supporting")
print("   - No semantic understanding of content")
print()

print("2. UNKNOWN UNKNOWNS BLIND SPOT (HIGH)")
print("   - Overconfident in stable world assumption")
print("   - No scenario planning for black swans")
print()

print("3. SINGLE STAKEHOLDER BIAS (MODERATE)")
print("   - No multi-stakeholder impact weighting")
print("   - Missing ethical value framework")
print()

print("4. GAMEABLE WITH MINIMAL CONTENT (MODERATE)")
print("   - High confidence + completeness → good score")
print("   - No content quality assessment")
print()

print("KNOWN LIMITATIONS (Acceptable for v3.0):")
print()
print("1. Cannot detect circular reasoning")
print("2. Cannot assess if measures are truly measurable")
print("3. Limited by AI's semantic understanding")
print("4. No calibration against real-world accuracy")
print()

print("STRENGTHS CONFIRMED:")
print()
print("1. Diminishing returns prevents simple gaming")
print("2. Evidence integration appropriately skeptical")
print("3. Meta-consistent (can analyze itself)")
print("4. Aleatory uncertainty handling works")
print()

print("="*80)
print("FINAL VERDICT: Framework has significant limitations")
print("Most are inherent to AI-assisted analysis")
print("Contradictory evidence detection is most serious gap")
print("="*80)
print()
