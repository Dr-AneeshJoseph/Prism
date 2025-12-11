"""
PRISM v1.0 - Usage Examples
============================

Comprehensive examples demonstrating proper usage of PRISM 1.0 with all
safety features enabled.

These examples show:
1. Basic hypothesis evaluation
2. Comparing multiple hypotheses
3. Handling established facts (avoiding bias detector paradox)
4. Working with evidence independence
5. Interpreting warnings properly
6. Using VOI for decision timing
"""

from prism_v1 import (
    AnalysisElement, Evidence, EvidenceDomain, CausalLevel,
    MechanismNode, MechanismEdge, NodeType, EdgeType,
    run_analysis, explain_result, HypothesisComparator,
    WarningLevel
)


def example_1_basic_analysis():
    """
    Example 1: Basic hypothesis analysis
    
    Demonstrates proper setup of a hypothesis with all components.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Hypothesis Analysis")
    print("=" * 70)
    
    # Create hypothesis
    h = AnalysisElement(
        name="Launch New Product Line",
        domain=EvidenceDomain.BUSINESS
    )
    
    # Set foundation with realistic confidence levels
    h.set_what("Launch premium widget line in Q2 2025", 0.85)
    h.set_why("Market research shows 30% unmet demand in premium segment", 0.7)
    h.set_how("Design → Prototype → Test → Manufacturing → Launch", 0.75)
    h.set_measure("Capture 10% market share in 12 months", 0.6)
    
    # Set feasibility
    h.set_feasibility(
        technical=0.8,   # We have the capabilities
        economic=0.7,    # ROI looks positive but uncertain
        timeline=0.65    # Timeline is ambitious
    )
    
    # Set risk levels
    h.set_risk(
        execution_risk=0.3,   # Some execution challenges expected
        external_risk=0.25    # Market/competitive risk
    )
    
    # Add evidence with FULL METADATA for independence checking
    h.add_evidence(Evidence(
        id="market_research",
        content="Market study shows 30% of consumers would pay premium for our category",
        source="Nielsen Research",
        quality=0.75,
        date="2024-06",
        domain=EvidenceDomain.BUSINESS,
        study_design="multi_company_analysis",
        sample_size=2500,
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True,
        authors=["Nielsen Research Team"],
        funding_source="Company internal"
    ))
    
    h.add_evidence(Evidence(
        id="competitor_analysis",
        content="Competitor launched premium line last year, achieved 8% share",
        source="Industry Report",
        quality=0.6,
        date="2024-09",
        domain=EvidenceDomain.BUSINESS,
        study_design="case_study",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    # IMPORTANT: Add contradicting evidence
    h.add_evidence(Evidence(
        id="internal_concern",
        content="Manufacturing team reports capacity constraints may delay launch",
        source="Internal memo",
        quality=0.7,
        date="2024-11",
        domain=EvidenceDomain.BUSINESS,
        study_design="expert_opinion",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=False  # Contradicting!
    ))
    
    h.add_evidence(Evidence(
        id="economic_outlook",
        content="Economic forecast suggests consumer spending may decline 5% next year",
        source="Federal Reserve",
        quality=0.8,
        date="2024-10",
        domain=EvidenceDomain.BUSINESS,
        study_design="benchmark",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=False
    ))
    
    # Add outcome scenarios with RISK AVERSION
    h.set_risk_aversion(1.5)  # Moderately risk-averse organization
    h.add_scenario("Strong Success: 15%+ market share", 0.25, 1.5)
    h.add_scenario("Moderate Success: 8-15% share", 0.35, 0.6)
    h.add_scenario("Break Even: 5-8% share", 0.25, 0.1)
    h.add_scenario("Failure: <5% share", 0.15, -0.5)
    
    # Run analysis
    results = run_analysis(h, rigor_level=2, max_iter=10)
    
    # Print results
    print(explain_result(results))
    
    # Show warnings
    print("\n📢 ALL WARNINGS:")
    h.warning_system.print_warnings(WarningLevel.INFO)
    
    return results


def example_2_compare_hypotheses():
    """
    Example 2: Compare alternative hypotheses
    
    Shows how to evaluate competing strategies.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Compare Alternative Hypotheses")
    print("=" * 70)
    
    # Hypothesis 1: Build in-house
    h1 = AnalysisElement(name="Build In-House", domain=EvidenceDomain.TECHNOLOGY)
    h1.set_what("Develop ML platform internally", 0.85)
    h1.set_why("Full control and customization", 0.7)
    h1.set_how("Hire team → Design → Build → Deploy", 0.6)
    h1.set_measure("Platform operational in 18 months", 0.5)
    h1.set_feasibility(0.6, 0.5, 0.4)  # Challenging
    h1.set_risk_aversion(1.0)
    h1.add_scenario("Success", 0.4, 1.2)
    h1.add_scenario("Partial", 0.35, 0.3)
    h1.add_scenario("Failure", 0.25, -0.6)
    
    h1.add_evidence(Evidence(
        "h1_ev1", "Similar projects took 2-3 years at other companies",
        "Tech industry survey", 0.7, "2024", EvidenceDomain.TECHNOLOGY,
        "benchmark", CausalLevel.ASSOCIATION, False, sample_size=50
    ))
    h1.add_evidence(Evidence(
        "h1_ev2", "We have some ML expertise on staff",
        "HR records", 0.8, "2024", EvidenceDomain.TECHNOLOGY,
        "expert_opinion", CausalLevel.ASSOCIATION, True
    ))
    
    # Hypothesis 2: Buy vendor solution
    h2 = AnalysisElement(name="Buy Vendor Solution", domain=EvidenceDomain.TECHNOLOGY)
    h2.set_what("Purchase enterprise ML platform", 0.95)
    h2.set_why("Faster time to value", 0.8)
    h2.set_how("Evaluate → Select → Implement → Train", 0.85)
    h2.set_measure("Platform operational in 6 months", 0.75)
    h2.set_feasibility(0.85, 0.7, 0.8)  # More feasible
    h2.set_risk_aversion(1.0)
    h2.add_scenario("Success", 0.55, 0.8)
    h2.add_scenario("Partial", 0.30, 0.3)
    h2.add_scenario("Failure", 0.15, -0.3)
    
    h2.add_evidence(Evidence(
        "h2_ev1", "Vendor has 90% customer satisfaction rate",
        "Gartner Report", 0.75, "2024", EvidenceDomain.TECHNOLOGY,
        "multi_company_analysis", CausalLevel.ASSOCIATION, True, sample_size=200
    ))
    h2.add_evidence(Evidence(
        "h2_ev2", "Typical implementation takes 4-8 months",
        "Vendor case studies", 0.5, "2024", EvidenceDomain.TECHNOLOGY,
        "case_study", CausalLevel.ASSOCIATION, True
    ))
    h2.add_evidence(Evidence(
        "h2_ev3", "Long-term costs may exceed build option after 3 years",
        "Financial analysis", 0.7, "2024", EvidenceDomain.TECHNOLOGY,
        "benchmark", CausalLevel.ASSOCIATION, False
    ))
    
    # Run analyses
    r1 = run_analysis(h1, rigor_level=2, max_iter=5)
    r2 = run_analysis(h2, rigor_level=2, max_iter=5)
    
    # Compare
    comparator = HypothesisComparator()
    comparator.add_hypothesis(h1)
    comparator.add_hypothesis(h2)
    comparison = comparator.compare()
    
    print("\nCOMPARISON RESULTS:")
    print("-" * 60)
    
    for hyp in comparison['hypotheses']:
        rank = comparison['rankings'].get(hyp['name'], '?')
        print(f"\n#{rank} {hyp['name']}")
        print(f"   Combined Score: {hyp['combined_score']:.3f}")
        print(f"   Credence: {hyp['credence']:.1%}")
        print(f"   Expected Utility: {hyp['expected_utility']:.3f}")
        print(f"   Certainty Equivalent: {hyp['certainty_equivalent']:.3f}")
        print(f"   Fatal Flaws: {hyp['fatal_flaws'] or 'None'}")
        print(f"   Critical Warnings: {'Yes' if hyp['has_critical_warnings'] else 'No'}")
    
    if comparison['best_choice']:
        print(f"\n✅ RECOMMENDED: {comparison['best_choice']}")
    
    return comparison


def example_3_established_hypothesis():
    """
    Example 3: Handling established facts
    
    Shows how to properly mark established hypotheses to avoid
    the bias detector paradox (flagging well-proven facts as biased).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Established Hypothesis (Avoiding Bias Detector Paradox)")
    print("=" * 70)
    
    # A well-established scientific fact
    h = AnalysisElement(
        name="Smoking Causes Lung Cancer",
        domain=EvidenceDomain.MEDICAL
    )
    
    # CRITICAL: Mark as established hypothesis
    h.set_established_hypothesis(True)
    
    h.set_what("Tobacco smoking is a primary cause of lung cancer", 0.99)
    h.set_why("Decades of consistent evidence across multiple study types", 0.98)
    h.set_how("Carcinogens in smoke damage DNA → mutations → cancer", 0.95)
    h.set_measure("Relative risk >15 in heavy smokers", 0.95)
    
    h.set_feasibility(0.99, 0.95, 0.95)  # Well established
    
    # Add extensive supporting evidence (as expected for established fact)
    for i in range(8):
        h.add_evidence(Evidence(
            f"study_{i}",
            f"Large cohort study {i+1} shows strong association (RR > 10)",
            f"Major Medical Journal {i+1}",
            0.9,
            f"20{10+i}",
            EvidenceDomain.MEDICAL,
            "cohort",
            CausalLevel.ASSOCIATION,
            True,  # All support - expected for established fact!
            sample_size=50000 + i*10000
        ))
    
    # No contradicting evidence - which is EXPECTED and APPROPRIATE here
    
    results = run_analysis(h, rigor_level=2, max_iter=5)
    
    print(explain_result(results))
    
    # Note: Because we marked it as established, the bias detector
    # should NOT flag it for confirmation bias
    print("\n📝 NOTE: Confirmation bias check result:")
    for bias in results['biases_detected']:
        if bias['type'] == 'confirmation':
            print(f"   Confirmation bias detected: {bias['detected']}")
            print(f"   Evidence: {bias['evidence']}")
    
    if not any(b['type'] == 'confirmation' and b['detected'] for b in results['biases_detected']):
        print("   ✅ Confirmation bias NOT flagged (correct for established fact)")
    
    return results


def example_4_evidence_independence():
    """
    Example 4: Evidence independence checking
    
    Demonstrates how PRISM detects and handles redundant evidence.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Evidence Independence Checking")
    print("=" * 70)
    
    h = AnalysisElement(
        name="New Marketing Campaign",
        domain=EvidenceDomain.BUSINESS
    )
    
    h.set_what("Launch social media campaign for Q1", 0.85)
    h.set_why("Increase brand awareness among millennials", 0.7)
    h.set_how("Content creation → Platform deployment → Analytics", 0.8)
    h.set_measure("20% increase in brand mentions", 0.6)
    h.set_feasibility(0.85, 0.8, 0.9)
    
    # Add REDUNDANT evidence (same underlying study, different presentations)
    # This should trigger independence warnings
    
    h.add_evidence(Evidence(
        "study_original",
        "Social media campaigns increase brand awareness by 25%",
        source="Marketing Research Institute",
        quality=0.7,
        date="2024-03",
        domain=EvidenceDomain.BUSINESS,
        study_design="controlled_experiment",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True,
        authors=["Dr. Smith", "Dr. Jones"],
        underlying_data="MRI_2024_dataset"
    ))
    
    h.add_evidence(Evidence(
        "study_reanalysis",
        "Re-analysis confirms 25% awareness boost from social campaigns",
        source="Marketing Research Institute",  # Same source!
        quality=0.7,
        date="2024-06",
        domain=EvidenceDomain.BUSINESS,
        study_design="controlled_experiment",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True,
        authors=["Dr. Smith"],  # Same author!
        underlying_data="MRI_2024_dataset"  # Same data!
    ))
    
    h.add_evidence(Evidence(
        "press_coverage",
        "According to MRI study, social campaigns boost awareness 25%",
        source="Marketing Weekly",
        quality=0.5,
        date="2024-04",
        domain=EvidenceDomain.BUSINESS,
        study_design="expert_opinion",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True,
        cites=["study_original"]  # Cites the original!
    ))
    
    # Add truly independent evidence
    h.add_evidence(Evidence(
        "competitor_case",
        "Competitor's similar campaign achieved 18% awareness increase",
        source="Industry Analysis Firm",
        quality=0.6,
        date="2024-08",
        domain=EvidenceDomain.BUSINESS,
        study_design="case_study",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
        # No shared authors, data, or citations
    ))
    
    # Run analysis
    results = run_analysis(h, rigor_level=2, max_iter=5)
    
    print(explain_result(results))
    
    # Show independence report
    print("\n📊 EVIDENCE INDEPENDENCE REPORT:")
    if results['independence_report']:
        print(f"   Overall Independence: {results['independence_report']['overall_independence']:.0%}")
        print(f"   Issues Found: {len(results['independence_report']['issues'])}")
        for issue in results['independence_report']['issues']:
            print(f"   • {issue['evidence_1']} ↔ {issue['evidence_2']}: {issue['independence']:.0%}")
            for prob in issue['issues']:
                print(f"     - {prob}")
    
    print(f"\n   Total Evidence Bits: {results['total_evidence_bits']:.2f}")
    print(f"   Effective Evidence Bits: {results['effective_evidence_bits']:.2f}")
    print(f"   Discount Applied: {(1 - results['effective_evidence_bits']/results['total_evidence_bits'])*100:.0f}%")
    
    return results


def example_5_fatal_content_detection():
    """
    Example 5: Content-based fatal flaw detection
    
    Shows how PRISM scans evidence content for critical issues
    that numeric scores might miss.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Content-Based Fatal Flaw Detection")
    print("=" * 70)
    
    h = AnalysisElement(
        name="Launch Product in New Market",
        domain=EvidenceDomain.BUSINESS
    )
    
    h.set_what("Enter European market with current product", 0.9)
    h.set_why("Diversify revenue streams", 0.8)
    h.set_how("Establish EU entity → Localize → Launch", 0.75)
    h.set_measure("€10M revenue in Year 1", 0.6)
    
    # All numeric dimensions look good!
    h.set_feasibility(0.85, 0.8, 0.75)
    h.set_risk(0.2, 0.3)
    
    # Add supporting evidence
    h.add_evidence(Evidence(
        "market_size",
        "EU market size is €500M with 15% annual growth",
        "Market Research Firm",
        0.75,
        "2024",
        EvidenceDomain.BUSINESS,
        "benchmark",
        CausalLevel.ASSOCIATION,
        True
    ))
    
    h.add_evidence(Evidence(
        "demand_signal",
        "We receive 100+ inquiries/month from EU customers",
        "Sales Team",
        0.7,
        "2024",
        EvidenceDomain.BUSINESS,
        "case_study",
        CausalLevel.ASSOCIATION,
        True
    ))
    
    # BUT: Evidence with FATAL content that scores wouldn't catch!
    h.add_evidence(Evidence(
        "legal_review",
        "Legal analysis: Product violates EU GDPR requirements and is prohibited "
        "under current regulations. Would require 18-month certification process.",
        "Legal Department",
        0.95,  # High quality evidence
        "2024",
        EvidenceDomain.BUSINESS,
        "expert_opinion",
        CausalLevel.COUNTERFACTUAL,
        False
    ))
    
    # Run analysis
    results = run_analysis(h, rigor_level=2, max_iter=5)
    
    print(explain_result(results))
    
    # Highlight content-based detection
    if results['content_fatal_flags']:
        print("\n🚨 CONTENT-BASED FATAL FLAGS DETECTED:")
        for flag in results['content_fatal_flags']:
            print(f"   Category: {flag['category']}")
            print(f"   Evidence: {flag['evidence_id']}")
            print(f"   Content: {flag['content_snippet'][:80]}...")
        print("\n   ⚠️  These issues were detected from evidence CONTENT,")
        print("      not from numeric scores. Human review required!")
    
    return results


def example_6_realistic_voi():
    """
    Example 6: Realistic Value of Information
    
    Shows how to properly use VOI with costs and constraints.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Realistic Value of Information")
    print("=" * 70)
    
    h = AnalysisElement(
        name="Major Capital Investment",
        domain=EvidenceDomain.BUSINESS
    )
    
    h.set_what("Build new manufacturing facility", 0.85)
    h.set_why("Meet projected demand increase", 0.65)  # Uncertain
    h.set_how("Site selection → Construction → Equipment → Operations", 0.8)
    h.set_measure("20% capacity increase, 15% ROI", 0.55)  # Uncertain
    h.set_feasibility(0.8, 0.6, 0.7)
    
    h.add_evidence(Evidence(
        "demand_forecast",
        "Demand expected to grow 25% over 3 years",
        "Internal forecast",
        0.5,  # Forecasts are uncertain
        "2024",
        EvidenceDomain.BUSINESS,
        "benchmark",
        CausalLevel.ASSOCIATION,
        True
    ))
    
    # High-stakes scenarios
    h.set_risk_aversion(2.0)  # High risk aversion for major investment
    h.add_scenario("Strong demand: High ROI", 0.30, 2.0)
    h.add_scenario("Moderate demand: Break even", 0.40, 0.2)
    h.add_scenario("Weak demand: Major loss", 0.30, -1.5)
    
    results = run_analysis(h, rigor_level=2, max_iter=5)
    
    print(explain_result(results))
    
    # Calculate VOI with realistic parameters
    print("\n💰 VALUE OF INFORMATION ANALYSIS:")
    
    # Scenario 1: Perfect, free, instant (unrealistic)
    voi_perfect = h.value_of_information(
        info_cost=0.0,
        signal_accuracy=1.0,
        time_cost=0.0
    )
    print(f"\n   Unrealistic VOI (perfect, free, instant):")
    print(f"   Raw VOI: {voi_perfect['raw_voi']:.3f}")
    print(f"   → This is what naive VOI calculations give you")
    
    # Scenario 2: Realistic market study
    voi_study = h.value_of_information(
        info_cost=0.1,      # Study costs ~10% of potential gain
        signal_accuracy=0.7, # Studies aren't perfect
        time_cost=0.05      # 6-month delay has opportunity cost
    )
    print(f"\n   Realistic VOI (market study, $100K, 6 months, 70% accurate):")
    print(f"   Raw VOI: {voi_study['raw_voi']:.3f}")
    print(f"   After accuracy adjustment: {voi_study['realistic_voi']:.3f}")
    print(f"   After costs: {voi_study['net_voi']:.3f}")
    print(f"   Recommendation: {voi_study['recommendation']}")
    
    # Scenario 3: Expensive pilot project
    voi_pilot = h.value_of_information(
        info_cost=0.3,      # Pilot costs 30% of potential gain
        signal_accuracy=0.9, # Pilots are more accurate
        time_cost=0.15      # 12-month delay
    )
    print(f"\n   Realistic VOI (pilot project, $300K, 12 months, 90% accurate):")
    print(f"   Raw VOI: {voi_pilot['raw_voi']:.3f}")
    print(f"   After accuracy adjustment: {voi_pilot['realistic_voi']:.3f}")
    print(f"   After costs: {voi_pilot['net_voi']:.3f}")
    print(f"   Recommendation: {voi_pilot['recommendation']}")
    
    return results


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PRISM v1.0 - Comprehensive Usage Examples")
    print("=" * 70)
    
    examples = [
        ("Basic Analysis", example_1_basic_analysis),
        ("Compare Hypotheses", example_2_compare_hypotheses),
        ("Established Hypothesis", example_3_established_hypothesis),
        ("Evidence Independence", example_4_evidence_independence),
        ("Fatal Content Detection", example_5_fatal_content_detection),
        ("Realistic VOI", example_6_realistic_voi),
    ]
    
    for name, func in examples:
        try:
            print(f"\n\n{'#' * 70}")
            print(f"# Running: {name}")
            print(f"{'#' * 70}")
            func()
            print(f"\n✅ {name} completed successfully")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
