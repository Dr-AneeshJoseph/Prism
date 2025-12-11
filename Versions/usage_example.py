"""
USAGE EXAMPLE: Enhanced Analytical Protocol v2.0
================================================
Copy enhanced_protocol_v2.py to the same directory, then run this file.
"""

from enhanced_protocol_v2 import (
    AnalysisElement, MechanismNode, MechanismEdge, Evidence,
    EvidenceDomain, NodeType, EdgeType, CausalLevel,
    run_analysis, HypothesisComparator
)


def example_business_decision():
    """Example: Should we hire a data scientist?"""
    
    # 1. Create analysis element
    h = AnalysisElement(
        name="Hire Data Scientist",
        domain=EvidenceDomain.BUSINESS
    )
    
    # 2. Set foundation (WHAT/WHY/HOW/MEASURE)
    h.set_what("Hire full-time data scientist at $120K to improve analytics", 0.9)
    h.set_why("We lack statistical expertise for A/B testing", 0.75)
    h.set_how("Post job → Interview → Hire → Onboard → Deliver insights", 0.8)
    h.set_measure("A/B test success rate improves by 20%", 0.7)
    
    # 3. Set feasibility and risk
    h.set_feasibility(technical=0.85, economic=0.7, timeline=0.8)
    h.set_risk(execution_risk=0.3, external_risk=0.2)
    
    # 4. Build causal mechanism map
    # Add nodes
    n1 = MechanismNode("cause1", "Lack of stats expertise", NodeType.CAUSE)
    n1.confidence = 0.9
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("mech1", "DS provides expertise", NodeType.MECHANISM)
    n2.confidence = 0.85
    h.add_mechanism_node(n2)
    
    n3 = MechanismNode("outcome1", "Better decisions", NodeType.OUTCOME)
    n3.confidence = 0.7
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("blocker1", "Org resistance", NodeType.BLOCKER)
    n4.confidence = 0.4  # 40% likely to block
    h.add_mechanism_node(n4)
    
    n5 = MechanismNode("assume1", "DS can integrate with team", NodeType.ASSUMPTION)
    n5.confidence = 0.6  # Untested assumption
    h.add_mechanism_node(n5)
    
    # Add edges WITH CAUSAL LEVELS (critical for proper analysis)
    h.add_mechanism_edge(MechanismEdge(
        "cause1", "mech1", EdgeType.CAUSES, 0.9,
        causal_level=CausalLevel.COUNTERFACTUAL,  # Theoretical mechanism
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "mech1", "outcome1", EdgeType.ENABLES, 0.7,
        causal_level=CausalLevel.ASSOCIATION,  # Only have correlational data
        confounding_risk=0.5
    ))
    h.add_mechanism_edge(MechanismEdge(
        "blocker1", "outcome1", EdgeType.PREVENTS, 0.4,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.3
    ))
    
    # 5. Add evidence WITH CAUSAL LEVELS
    h.add_evidence(Evidence(
        id="ev1",
        content="HBR study: Companies with DS report 15% better outcomes",
        source="Harvard Business Review 2023",
        quality=0.7,
        date="2023",
        domain=EvidenceDomain.BUSINESS,
        study_design="multi_company_analysis",
        causal_level=CausalLevel.ASSOCIATION,  # Observational study!
        supports_hypothesis=True
    ))
    
    h.add_evidence(Evidence(
        id="ev2",
        content="Competitor hired DS, saw metric improvements",
        source="Industry contact",
        quality=0.3,
        date="2024",
        domain=EvidenceDomain.BUSINESS,
        study_design="anecdote",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    # 6. Define outcome scenarios (for decision theory)
    h.add_scenario("Success: DS integrates well, major improvements", 0.5, 1.0)
    h.add_scenario("Partial: Some improvement, not transformative", 0.3, 0.4)
    h.add_scenario("Failure: DS doesn't fit, leaves within year", 0.2, -0.3)
    
    # 7. Run analysis
    results = run_analysis(h, rigor_level=2, max_iter=10)
    
    # 8. Print results
    print("=" * 60)
    print(f"ANALYSIS: {results['name']}")
    print("=" * 60)
    
    print(f"\n📊 SCORES:")
    print(f"   Bayesian:    {results['bayesian_score']:.3f}")
    print(f"   Calibrated:  {results['calibrated_score']:.3f}")
    print(f"   Debiased:    {results['debiased_score']:.3f}")
    
    print(f"\n🎯 EPISTEMIC STATE:")
    print(f"   Credence:    {results['credence']:.3f}")
    ci = results['confidence_interval']
    print(f"   95% CI:      [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    print(f"\n💰 DECISION THEORY:")
    print(f"   Expected Utility:     {results['expected_utility']:.3f}")
    print(f"   Certainty Equivalent: {results['certainty_equivalent']:.3f}")
    print(f"   Value of Information: {results['value_of_information']:.3f}")
    
    print(f"\n⚠️  BIASES DETECTED:")
    if results['biases_detected']:
        for bias in results['biases_detected']:
            print(f"   • {bias['type']}: {bias['evidence']}")
    else:
        print("   None")
    
    print(f"\n🔗 CAUSAL ANALYSIS:")
    print(f"   Mechanism Confidence:    {results['mechanism_confidence']:.3f}")
    print(f"   Average Causal Strength: {results['average_causal_strength']:.3f}")
    
    if results['fatal_flaws']:
        print(f"\n💀 FATAL FLAWS:")
        for flaw in results['fatal_flaws']:
            print(f"   • {flaw['name']}: {flaw['value']:.2f}")
    
    print(f"\n" + "=" * 60)
    print(f"📋 RECOMMENDATION: {results['recommendation']}")
    print("=" * 60)
    
    return results


def example_compare_hypotheses():
    """Example: Compare two alternative hypotheses"""
    
    # Hypothesis 1: Hire Data Scientist
    h1 = AnalysisElement(name="Hire Data Scientist", domain=EvidenceDomain.BUSINESS)
    h1.set_what("Hire DS at $120K", 0.9)
    h1.set_why("Need expertise", 0.7)
    h1.set_how("Standard hiring", 0.8)
    h1.set_measure("20% improvement", 0.7)
    h1.set_feasibility(0.85, 0.7, 0.8)
    h1.add_scenario("Success", 0.5, 1.0)
    h1.add_scenario("Failure", 0.5, -0.3)
    
    # Hypothesis 2: Buy Analytics Tool
    h2 = AnalysisElement(name="Buy Analytics Tool", domain=EvidenceDomain.BUSINESS)
    h2.set_what("Purchase analytics platform at $50K/year", 0.95)
    h2.set_why("Automate analysis", 0.5)
    h2.set_how("Evaluate → Purchase → Deploy", 0.85)
    h2.set_measure("80% dashboard adoption", 0.6)
    h2.set_feasibility(0.8, 0.85, 0.9)
    h2.add_scenario("Success", 0.4, 0.8)
    h2.add_scenario("Failure", 0.6, -0.1)
    
    # Run analyses
    r1 = run_analysis(h1, rigor_level=2, max_iter=5)
    r2 = run_analysis(h2, rigor_level=2, max_iter=5)
    
    # Compare
    comparator = HypothesisComparator()
    comparator.add_hypothesis(h1)
    comparator.add_hypothesis(h2)
    comparison = comparator.compare()
    
    print("\n" + "=" * 60)
    print("HYPOTHESIS COMPARISON")
    print("=" * 60)
    
    for hyp in comparison['hypotheses']:
        rank = comparison['rankings'][hyp['name']]
        print(f"\n#{rank} {hyp['name']}")
        print(f"   Score:    {hyp['combined_score']:.3f}")
        print(f"   Credence: {hyp['credence']:.3f}")
        print(f"   EU:       {hyp['expected_utility']:.3f}")
        print(f"   VOI:      {hyp['value_of_information']:.3f}")
    
    return comparison


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Hypothesis Analysis")
    print("=" * 60)
    example_business_decision()
    
    print("\n\n")
    example_compare_hypotheses()
