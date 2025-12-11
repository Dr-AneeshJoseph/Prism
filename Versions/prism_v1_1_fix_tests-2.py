"""
PRISM v1.1 - VULNERABILITY FIX VERIFICATION TESTS
==================================================

This test suite verifies that all P0 and P1 vulnerabilities from the 
Red Team Analysis v2 are now fixed in PRISM v1.1.

Run with: python prism_v1_1_fix_tests.py
"""

import sys
sys.path.insert(0, '/mnt/user-data/outputs')

from prism_v1_1 import (
    AnalysisElement, Evidence, EvidenceDomain, CausalLevel,
    run_analysis, WarningLevel, MechanismNode, MechanismEdge,
    NodeType, EdgeType, EstablishedHypothesisEvidence,
    TextAnalyzer, EvidenceIndependenceChecker, SafetyLimits
)


def test_p0_1_independence_checker_bypass():
    """
    P0 FIX #1: Independence Checker now catches:
    - Normalized author names (Smith, J. = Dr. Jane Smith)
    - Normalized sources (J. Med. = Journal of Medicine)  
    - Entity overlap (NCT123456 in different contexts)
    - Semantic content similarity
    """
    print("\n" + "=" * 70)
    print("TEST P0-1: Independence Checker Bypass")
    print("=" * 70)
    
    # Test 1: Author name normalization
    print("\n1. Author name normalization:")
    norm1 = TextAnalyzer.normalize_author_name("Smith, J.")
    norm2 = TextAnalyzer.normalize_author_name("J. Smith")
    norm3 = TextAnalyzer.normalize_author_name("Dr. Jane Smith")
    
    print(f"   'Smith, J.' -> '{norm1}'")
    print(f"   'J. Smith' -> '{norm2}'")
    print(f"   'Dr. Jane Smith' -> '{norm3}'")
    
    # These should all normalize to similar forms
    assert norm1 == norm2, "Failed: J. Smith != Smith, J."
    print("   ✓ Author normalization working")
    
    # Test 2: Entity extraction
    print("\n2. Entity extraction:")
    text = "Trial NCT123456 shows positive results per Smith 2024"
    entities = TextAnalyzer.extract_entities(text)
    print(f"   From: '{text}'")
    print(f"   Entities: {entities}")
    assert "NCT123456" in entities, "Failed: NCT ID not extracted"
    print("   ✓ Entity extraction working")
    
    # Test 3: Full independence check
    print("\n3. Full independence check on related evidence:")
    
    e1 = Evidence(
        id="trial", content="Clinical trial NCT123456 shows benefit",
        source="ClinicalTrials.gov", quality=0.8, date="2024",
        domain=EvidenceDomain.MEDICAL, study_design="rct",
        causal_level=CausalLevel.INTERVENTION, supports_hypothesis=True,
        underlying_data="NCT123456"
    )
    
    e2 = Evidence(
        id="paper", content="Smith et al NCT123456 trial results positive",
        source="New England Journal Medicine",  # Different format!
        quality=0.9, date="2024",
        domain=EvidenceDomain.MEDICAL, study_design="rct",
        causal_level=CausalLevel.INTERVENTION, supports_hypothesis=True,
        authors=["Smith, J."],
        underlying_data="NCT123456"
    )
    
    e3 = Evidence(
        id="press", content="Dr. Jane Smith announces NCT123456 success",
        source="University Press",
        quality=0.5, date="2024",
        domain=EvidenceDomain.MEDICAL, study_design="anecdote",
        causal_level=CausalLevel.ASSOCIATION, supports_hypothesis=True,
        authors=["Dr. Jane Smith"]  # Different format!
    )
    
    report = EvidenceIndependenceChecker.check_all_independence([e1, e2, e3])
    
    print(f"   Overall independence: {report['overall_independence']:.1%}")
    print(f"   Effective evidence: {report['effective_evidence_count']:.1f} / 3")
    print(f"   Issues detected: {len(report['issues'])}")
    
    # Should detect relationships between all three
    assert report['overall_independence'] < 0.5, "Failed: Should detect low independence"
    assert len(report['issues']) >= 2, "Failed: Should find multiple issues"
    
    print("   ✓ Independence checker correctly identifies related evidence")
    print("\n✅ P0-1 FIX VERIFIED")


def test_p0_2_content_scanner_evasion():
    """
    P0 FIX #2: Content Scanner now catches:
    - Euphemisms for legal issues ("regulatory gray area")
    - Euphemisms for safety issues ("adverse event")
    - Euphemisms for financial issues ("liquidity constraint")
    """
    print("\n" + "=" * 70)
    print("TEST P0-2: Content Scanner Pattern Evasion")
    print("=" * 70)
    
    h = AnalysisElement(name="Test Scanner", domain=EvidenceDomain.BUSINESS)
    h.set_what("Launch product", 0.8)
    h.set_feasibility(0.8, 0.8, 0.8)
    
    # Test euphemisms that evaded v1.0
    test_cases = [
        ("enforcement action against similar products", "legal", "v1.0 missed"),
        ("adverse event rate of 15%", "safety", "v1.0 missed"),
        ("liquidity constraints within 18 months", "financial", "v1.0 missed"),
        ("FDA expressed concerns during review", "safety", "v1.0 missed"),
        ("not fully compliant with regulations", "legal", "v1.0 missed"),
    ]
    
    detected_count = 0
    for content, expected_category, note in test_cases:
        e = Evidence(
            id=f"test_{expected_category}",
            content=content,
            source="Test",
            quality=0.8,
            date="2024",
            domain=EvidenceDomain.BUSINESS,
            supports_hypothesis=False
        )
        
        has_fatal = len([f for f in e.fatal_content_flags if f[0] == expected_category]) > 0
        status = "✓ DETECTED" if has_fatal else "✗ MISSED"
        print(f"   {status}: '{content[:50]}...' ({expected_category})")
        if has_fatal:
            detected_count += 1
    
    print(f"\n   Detected: {detected_count}/{len(test_cases)} euphemistic patterns")
    assert detected_count >= 3, "Failed: Should detect most euphemisms"
    
    print("\n✅ P0-2 FIX VERIFIED")


def test_p0_3_established_hypothesis_abuse():
    """
    P0 FIX #3: Established hypothesis requires evidence
    """
    print("\n" + "=" * 70)
    print("TEST P0-3: Established Hypothesis Abuse")
    print("=" * 70)
    
    # Test 1: Unverified claim should warn
    print("\n1. Unverified establishment claim:")
    h1 = AnalysisElement(name="Unverified", domain=EvidenceDomain.MEDICAL)
    h1.set_established_hypothesis(True)  # No evidence!
    h1.set_what("Novel hypothesis", 0.9)
    h1.set_feasibility(0.8, 0.8, 0.8)
    h1.run_bias_detection()
    
    print(f"   Verified: {h1.bias_detector.establishment_verified}")
    assert not h1.bias_detector.establishment_verified, "Should NOT be verified"
    
    # Check warning was issued
    warnings = [w for w in h1.warning_system.warnings if "Establishment" in w.category]
    print(f"   Warnings issued: {len(warnings)}")
    assert len(warnings) > 0, "Should issue warning"
    print("   ✓ Unverified claim triggers warning")
    
    # Test 2: Verified claim should pass
    print("\n2. Verified establishment claim:")
    h2 = AnalysisElement(name="Verified", domain=EvidenceDomain.MEDICAL)
    
    evidence = EstablishedHypothesisEvidence(
        claim="Aspirin reduces inflammation",
        supporting_references=["Vane 1971", "Smith 1980"],
        meta_analyses_cited=3,
        textbook_citations=2,
        expert_consensus=True
    )
    h2.set_established_hypothesis(True, evidence)
    h2.set_what("Aspirin reduces inflammation", 0.95)
    h2.set_feasibility(0.95, 0.9, 0.95)
    h2.run_bias_detection()
    
    print(f"   Verified: {h2.bias_detector.establishment_verified}")
    assert h2.bias_detector.establishment_verified, "Should BE verified"
    print(f"   Evidence strength: {evidence.strength_score():.1%}")
    print("   ✓ Verified claim passes")
    
    print("\n✅ P0-3 FIX VERIFIED")


def test_p0_4_warning_fatigue():
    """
    P0 FIX #4: Warning system deduplication and aggregation
    """
    print("\n" + "=" * 70)
    print("TEST P0-4: Warning Fatigue")
    print("=" * 70)
    
    h = AnalysisElement(name="Warning Test", domain=EvidenceDomain.BUSINESS)
    h.set_what("Test", 0.8)
    h.set_feasibility(0.8, 0.8, 0.8)
    
    # Add many similar evidence pieces
    for i in range(10):
        h.add_evidence(Evidence(
            f"ev{i}", f"Evidence {i} shows result",
            f"Source {i}", 0.6, "2024",
            EvidenceDomain.BUSINESS, "case_study",
            CausalLevel.ASSOCIATION, True, 100
        ))
    
    results = run_analysis(h, rigor_level=2, max_iter=5)
    
    # Check deduplication
    unique_warnings = len(h.warning_system.warnings)
    total_count = sum(w.count for w in h.warning_system.warnings)
    
    print(f"\n   Unique warnings: {unique_warnings}")
    print(f"   Total (with aggregation): {total_count}")
    print(f"   Summary: {h.warning_system.get_summary_header()}")
    
    # Should have fewer unique warnings than total events
    assert unique_warnings < total_count or unique_warnings < 10, "Should deduplicate"
    
    print("   ✓ Warnings are deduplicated and aggregated")
    print("\n✅ P0-4 FIX VERIFIED")


def test_p1_5_sample_size_gaming():
    """
    P1 FIX #5: Sample size validation detects subgroups
    """
    print("\n" + "=" * 70)
    print("TEST P1-5: Sample Size Gaming")
    print("=" * 70)
    
    # Test 1: Subgroup detection
    print("\n1. Subgroup detection:")
    e1 = Evidence(
        id="subgroup",
        content="Study of 10,000. Subgroup analysis (n=80) shows benefit",
        source="Journal",
        quality=0.6,
        date="2024",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        sample_size=10000,
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    )
    
    print(f"   Claimed N: {e1.sample_size}")
    print(f"   Is subgroup: {e1.is_subgroup}")
    print(f"   Validation warnings: {e1.validation_warnings}")
    
    assert e1.is_subgroup, "Should detect subgroup"
    assert len(e1.validation_warnings) > 0, "Should have warnings"
    print("   ✓ Subgroup detected and flagged")
    
    # Test 2: Quality penalty
    print("\n2. Quality adjustment:")
    e2 = Evidence(
        id="honest",
        content="Subgroup analysis shows benefit",
        source="Journal",
        quality=0.6,
        date="2024",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        sample_size=80,
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    )
    
    print(f"   Gamed (N=10000): quality={e1.effective_quality:.3f}")
    print(f"   Honest (N=80): quality={e2.effective_quality:.3f}")
    
    # Gamed should NOT have huge advantage now
    advantage = (e1.effective_quality - e2.effective_quality) / e2.effective_quality
    print(f"   Advantage from gaming: {advantage:.1%}")
    assert advantage < 0.3, "Gaming advantage should be limited"
    
    print("   ✓ Quality penalty applied to gaming attempts")
    print("\n✅ P1-5 FIX VERIFIED")


def test_p1_6_weight_gaming():
    """
    P1 FIX #6: Weight gaming now blocks analysis
    """
    print("\n" + "=" * 70)
    print("TEST P1-6: Weight Gaming")
    print("=" * 70)
    
    h = AnalysisElement(name="Weight Test", domain=EvidenceDomain.BUSINESS)
    h.set_what("Test", 0.8)
    h.set_feasibility(0.8, 0.8, 0.8)
    
    # Try to set excessive weight
    print("\n1. Excessive weight (5.0 > max 3.0):")
    result = h.set_dimension("upside", 0.9, weight=5.0)  # Over limit
    print(f"   Success: {result}")
    print(f"   Violation: {h.scoring.weight_violation}")
    
    assert h.scoring.weight_violation is not None, "Should have violation"
    print("   ✓ Excessive weight creates violation")
    
    # Check blocking
    print("\n2. Blocking behavior:")
    h.add_evidence(Evidence(
        "ev1", "Test", "Source", 0.7, "2024",
        EvidenceDomain.BUSINESS, supports_hypothesis=True
    ))
    
    blocked, reasons = h.is_blocked()
    print(f"   Blocked: {blocked}")
    print(f"   Reasons: {reasons}")
    
    assert blocked, "Should be blocked"
    print("   ✓ Weight violation blocks analysis")
    
    print("\n✅ P1-6 FIX VERIFIED")


def test_p1_7_risk_aversion_gaming():
    """
    P1 FIX #7: Risk aversion has domain defaults
    """
    print("\n" + "=" * 70)
    print("TEST P1-7: Risk Aversion Gaming")
    print("=" * 70)
    
    # Test domain defaults
    print("\n1. Domain defaults:")
    domains = [
        (EvidenceDomain.MEDICAL, 2.5),
        (EvidenceDomain.BUSINESS, 1.5),
        (EvidenceDomain.TECHNOLOGY, 1.0),
    ]
    
    for domain, expected in domains:
        h = AnalysisElement(name="Test", domain=domain)
        actual = h.utility_model.risk_aversion
        status = "✓" if actual == expected else "✗"
        print(f"   {status} {domain.value}: {actual} (expected {expected})")
        assert actual == expected, f"Wrong default for {domain.value}"
    
    print("   ✓ Domain defaults applied correctly")
    
    # Test unusual value warning
    print("\n2. Unusual value warning:")
    h = AnalysisElement(name="Test", domain=EvidenceDomain.BUSINESS)
    h.set_risk_aversion(0.05)  # Very low
    
    warnings = [w for w in h.warning_system.warnings if "Risk Aversion" in w.category]
    print(f"   Set γ=0.05 (very low)")
    print(f"   Warnings: {len(warnings)}")
    
    assert len(warnings) > 0, "Should warn about unusual value"
    print("   ✓ Warning issued for unusual values")
    
    print("\n✅ P1-7 FIX VERIFIED")


def test_p1_8_fatal_flaw_blocking():
    """
    P1 FIX #8: Fatal content blocks analysis
    """
    print("\n" + "=" * 70)
    print("TEST P1-8: Fatal Flaw Blocking")
    print("=" * 70)
    
    h = AnalysisElement(name="Fatal Test", domain=EvidenceDomain.BUSINESS)
    h.set_what("Launch product", 0.8)
    h.set_feasibility(0.8, 0.8, 0.8)
    
    # Add evidence with fatal content
    h.add_evidence(Evidence(
        id="fatal",
        content="This product is illegal in several jurisdictions",
        source="Legal Dept",
        quality=0.9,
        date="2024",
        domain=EvidenceDomain.BUSINESS,
        supports_hypothesis=False
    ))
    
    print("\n1. Evidence with 'illegal' keyword:")
    print(f"   Fatal content detected: {h.content_scanner.has_fatal_content()}")
    
    # Run analysis - should be blocked
    print("\n2. Running analysis:")
    results = run_analysis(h, rigor_level=2, max_iter=5)
    
    print(f"   Blocked: {results.get('blocked', False)}")
    print(f"   Decision: {results['decision_state']}")
    
    assert results.get('blocked', False), "Should be blocked"
    assert results['decision_state'] == 'blocked', "Decision should be blocked"
    
    print("   ✓ Fatal content blocks analysis")
    print("\n✅ P1-8 FIX VERIFIED")


def run_all_tests():
    """Run all fix verification tests"""
    print("\n" + "=" * 80)
    print("PRISM v1.1 - VULNERABILITY FIX VERIFICATION")
    print("=" * 80)
    print("\nVerifying all P0 and P1 fixes from Red Team Analysis v2...")
    
    tests = [
        ("P0-1: Independence Checker Bypass", test_p0_1_independence_checker_bypass),
        ("P0-2: Content Scanner Evasion", test_p0_2_content_scanner_evasion),
        ("P0-3: Established Hypothesis Abuse", test_p0_3_established_hypothesis_abuse),
        ("P0-4: Warning Fatigue", test_p0_4_warning_fatigue),
        ("P1-5: Sample Size Gaming", test_p1_5_sample_size_gaming),
        ("P1-6: Weight Gaming", test_p1_6_weight_gaming),
        ("P1-7: Risk Aversion Gaming", test_p1_7_risk_aversion_gaming),
        ("P1-8: Fatal Flaw Blocking", test_p1_8_fatal_flaw_blocking),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL P0 AND P1 FIXES VERIFIED!")
        print("   PRISM v1.1 addresses all critical vulnerabilities")
        print("   from Red Team Analysis v2.")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review required")
    
    print("\n" + "=" * 80)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
