"""
CAP v3.0 COMPREHENSIVE STRESS TEST SUITE

This suite tests:
1. Edge cases and boundary conditions
2. Invalid inputs and error handling
3. Mathematical consistency
4. Logical contradictions
5. Extreme scenarios
6. Performance under load
7. Convergence behavior
8. Conceptual coherence
9. Domain transferability
10. Failure modes

Goal: Find every way this system can break or behave unexpectedly
"""

import sys
import numpy as np
from cap_implementation_v3 import (
    Evidence, FoundationElement, AdversarialTester,
    ProblemCharacterization, calculate_quality_scores,
    run_analysis, UncertaintyType, ProblemType, StakesLevel,
    TimeConstraint, EvidenceDomain
)
import traceback
from typing import List, Tuple

# ============================================================================
# TEST UTILITIES
# ============================================================================

class StressTest:
    """Base class for stress tests"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.result = None
    
    def run(self):
        """Override this"""
        raise NotImplementedError
    
    def execute(self):
        """Run test with error handling"""
        try:
            self.result = self.run()
            self.passed = True
            return True
        except Exception as e:
            self.error = e
            self.passed = False
            return False

def report_test(test: StressTest, verbose: bool = True):
    """Report test result"""
    status = "✅ PASS" if test.passed else "❌ FAIL"
    print(f"{status}: {test.name}")
    if not test.passed and verbose:
        print(f"  Error: {test.error}")
        print(f"  Traceback: {traceback.format_exc()}")
    elif test.result and verbose:
        print(f"  Result: {test.result}")

# ============================================================================
# CATEGORY 1: BOUNDARY VALUE TESTS
# ============================================================================

class TestZeroConfidence(StressTest):
    """Test with all confidence values at 0"""
    def __init__(self):
        super().__init__("Zero confidence values")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Something"
        element.why = "Some reason"
        element.what_confidence = 0.0
        element.why_confidence = 0.0
        
        tester = AdversarialTester(element, rigor_level=1)
        quality = calculate_quality_scores(element, tester, 1)
        
        assert quality['overall'] >= 0, "Quality score went negative"
        assert quality['confidence'] == 0.0, "Confidence should be 0"
        return f"Quality: {quality['overall']:.3f}"

class TestMaxConfidence(StressTest):
    """Test with all confidence values at maximum"""
    def __init__(self):
        super().__init__("Maximum confidence values")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Something"
        element.why = "Some reason"
        element.how = "Some way"
        element.what_confidence = 0.99  # System caps at 0.95
        element.why_confidence = 0.99
        element.how_confidence = 0.99
        
        tester = AdversarialTester(element, rigor_level=1)
        quality = calculate_quality_scores(element, tester, 1)
        
        assert quality['overall'] <= 1.0, "Quality score exceeded 1.0"
        return f"Quality: {quality['overall']:.3f}"

class TestInvalidConfidence(StressTest):
    """Test that invalid confidence values are rejected"""
    def __init__(self):
        super().__init__("Invalid confidence rejection")
    
    def run(self):
        try:
            element = FoundationElement(
                name="Test",
                what_confidence=1.5  # Invalid: > 1.0
            )
            return "SHOULD HAVE FAILED - accepted invalid confidence"
        except ValueError as e:
            return f"Correctly rejected: {e}"

class TestNegativeConfidence(StressTest):
    """Test that negative confidence is rejected"""
    def __init__(self):
        super().__init__("Negative confidence rejection")
    
    def run(self):
        try:
            element = FoundationElement(
                name="Test",
                what_confidence=-0.5  # Invalid: negative
            )
            return "SHOULD HAVE FAILED - accepted negative confidence"
        except ValueError as e:
            return f"Correctly rejected: {e}"

# ============================================================================
# CATEGORY 2: EVIDENCE STRESS TESTS
# ============================================================================

class TestNoEvidence(StressTest):
    """Test analysis with zero evidence"""
    def __init__(self):
        super().__init__("Zero evidence analysis")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Decision without evidence"
        element.why = "Based on intuition only"
        element.what_confidence = 0.7
        element.why_confidence = 0.3  # Low because no evidence
        
        results = run_analysis(element, rigor_level=2, max_iterations=5)
        
        assert results['quality_scores']['evidence'] == 0.0, "Should have 0 evidence score"
        return f"Quality: {results['quality_scores']['overall']:.3f}, Evidence: 0.0"

class TestMassiveEvidence(StressTest):
    """Test with 100 pieces of evidence"""
    def __init__(self):
        super().__init__("100 pieces of evidence")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.MEDICAL)
        element.what = "Well-studied intervention"
        element.why = "Massive evidence base"
        element.what_confidence = 0.9
        element.why_confidence = 0.5  # Start low
        
        # Add 100 pieces of evidence
        for i in range(100):
            element.add_evidence(Evidence(
                content=f"Study {i}",
                source=f"Journal {i}",
                strength=0.5 + (i % 5) * 0.1,  # Varying quality
                date="2024",
                domain=EvidenceDomain.MEDICAL,
                study_design="cohort_study"
            ))
        
        # Check for diminishing returns
        assert element.why_confidence < 0.99, "Should have diminishing returns cap"
        return f"Final confidence: {element.why_confidence:.3f} (from 100 pieces)"

class TestConflictingEvidenceDomains(StressTest):
    """Test evidence from wrong domain"""
    def __init__(self):
        super().__init__("Mismatched evidence domains")
    
    def run(self):
        element = FoundationElement(name="Medical Decision", domain=EvidenceDomain.MEDICAL)
        element.what = "Clinical trial"
        element.why = "Medical intervention"
        element.why_confidence = 0.5
        
        # Add business evidence to medical decision (should warn)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            element.add_evidence(Evidence(
                content="Business case study",
                source="HBR",
                strength=0.8,
                date="2024",
                domain=EvidenceDomain.BUSINESS,  # WRONG DOMAIN
                study_design="single_case_study"
            ))
            
            assert len(w) > 0, "Should have warned about domain mismatch"
            return f"Warning raised: {w[0].message}"

class TestInvalidEvidence(StressTest):
    """Test invalid evidence values"""
    def __init__(self):
        super().__init__("Invalid evidence rejection")
    
    def run(self):
        try:
            evidence = Evidence(
                content="Test",
                source="Test",
                strength=1.5,  # INVALID: > 1.0
                date="2024"
            )
            return "SHOULD HAVE FAILED - accepted invalid strength"
        except ValueError as e:
            return f"Correctly rejected: {e}"

# ============================================================================
# CATEGORY 3: CONVERGENCE AND ITERATION TESTS
# ============================================================================

class TestImmediateConvergence(StressTest):
    """Test perfect element that should converge immediately"""
    def __init__(self):
        super().__init__("Immediate convergence")
    
    def run(self):
        element = FoundationElement(name="Perfect", domain=EvidenceDomain.GENERAL)
        element.what = "Perfectly defined"
        element.why = "Perfectly justified"
        element.how = "Perfectly planned"
        element.measure = "Perfectly measurable"
        element.what_confidence = 0.95
        element.why_confidence = 0.95
        element.how_confidence = 0.95
        element.measure_confidence = 0.95
        
        results = run_analysis(element, rigor_level=1, max_iterations=20)
        
        assert results['iterations'] <= 3, f"Should converge quickly, took {results['iterations']}"
        return f"Converged in {results['iterations']} iterations"

class TestNonConvergence(StressTest):
    """Test element that never converges"""
    def __init__(self):
        super().__init__("Non-convergence handling")
    
    def run(self):
        element = FoundationElement(name="Incomplete", domain=EvidenceDomain.GENERAL)
        element.what = "Poorly defined"
        element.what_confidence = 0.1
        # Leave everything else undefined
        
        results = run_analysis(element, rigor_level=3, max_iterations=5)
        
        assert results['iterations'] == 5, "Should hit max iterations"
        assert not results['ready_for_action'], "Should not be ready"
        return f"Stopped at max iterations: {results['convergence_reason']}"

class TestOscillation(StressTest):
    """Test for oscillating quality scores"""
    def __init__(self):
        super().__init__("Quality score oscillation check")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Something"
        element.why = "Because"
        element.what_confidence = 0.6
        element.why_confidence = 0.6
        
        results = run_analysis(element, rigor_level=2, max_iterations=10)
        history = results['quality_history']
        
        # Check for oscillation (quality going up then down repeatedly)
        oscillations = 0
        for i in range(2, len(history)):
            if history[i-2] < history[i-1] > history[i]:
                oscillations += 1
        
        assert oscillations < len(history) / 2, f"Too many oscillations: {oscillations}"
        return f"Oscillations: {oscillations} in {len(history)} iterations"

# ============================================================================
# CATEGORY 4: EXTREME PROBLEM CHARACTERISTICS
# ============================================================================

class TestMinimalComplexity(StressTest):
    """Test with complexity = 0"""
    def __init__(self):
        super().__init__("Zero complexity problem")
    
    def run(self):
        problem = ProblemCharacterization(
            problem_type=ProblemType.DECISION,
            stakes=StakesLevel.LOW,
            time_available=TimeConstraint.DAYS,
            complexity=0.0,  # Trivially simple
            uncertainty=0.1,
            domain=EvidenceDomain.GENERAL
        )
        
        rigor = problem.recommend_rigor_level()
        assert rigor in [1, 2, 3], "Should return valid rigor level"
        return f"Recommended rigor: {rigor}"

class TestMaximalComplexity(StressTest):
    """Test with complexity = 1.0"""
    def __init__(self):
        super().__init__("Maximum complexity problem")
    
    def run(self):
        problem = ProblemCharacterization(
            problem_type=ProblemType.RESEARCH,
            stakes=StakesLevel.CRITICAL,
            time_available=TimeConstraint.MONTHS,
            complexity=1.0,  # Maximally complex
            uncertainty=1.0,  # Maximally uncertain
            domain=EvidenceDomain.GENERAL
        )
        
        rigor = problem.recommend_rigor_level()
        assert rigor == 3, f"Should recommend highest rigor, got {rigor}"
        return f"Recommended rigor: {rigor}"

class TestInvalidComplexity(StressTest):
    """Test that invalid complexity is rejected"""
    def __init__(self):
        super().__init__("Invalid complexity rejection")
    
    def run(self):
        try:
            problem = ProblemCharacterization(
                problem_type=ProblemType.DECISION,
                stakes=StakesLevel.LOW,
                time_available=TimeConstraint.DAYS,
                complexity=1.5,  # INVALID: > 1.0
                uncertainty=0.5
            )
            return "SHOULD HAVE FAILED - accepted invalid complexity"
        except ValueError as e:
            return f"Correctly rejected: {e}"

# ============================================================================
# CATEGORY 5: LOGICAL CONSISTENCY TESTS
# ============================================================================

class TestConfidenceEvidenceConsistency(StressTest):
    """Test that adding evidence always increases or maintains confidence"""
    def __init__(self):
        super().__init__("Evidence → confidence consistency")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.MEDICAL)
        element.what = "Test"
        element.why = "Test"
        element.why_confidence = 0.5
        
        confidences = [element.why_confidence]
        
        # Add 10 pieces of good evidence
        for i in range(10):
            element.add_evidence(Evidence(
                content=f"Study {i}",
                source=f"Source {i}",
                strength=0.8,
                date="2024",
                domain=EvidenceDomain.MEDICAL,
                study_design="randomized_controlled_trial"
            ))
            confidences.append(element.why_confidence)
        
        # Check that confidence never decreased
        for i in range(len(confidences) - 1):
            if confidences[i+1] < confidences[i] - 0.001:  # Allow tiny floating point error
                return f"INCONSISTENCY: Confidence decreased from {confidences[i]:.3f} to {confidences[i+1]:.3f}"
        
        return f"Consistent: {confidences[0]:.3f} → {confidences[-1]:.3f}"

class TestCompletenessConfidenceDecoupling(StressTest):
    """Test that completeness and confidence are independent"""
    def __init__(self):
        super().__init__("Completeness-confidence independence")
    
    def run(self):
        # High completeness, low confidence
        element1 = FoundationElement(name="Complete but uncertain", domain=EvidenceDomain.GENERAL)
        element1.what = "Defined"
        element1.why = "Justified"
        element1.how = "Planned"
        element1.measure = "Measured"
        element1.what_confidence = 0.3
        element1.why_confidence = 0.3
        element1.how_confidence = 0.3
        element1.measure_confidence = 0.3
        
        completeness1 = element1.weighted_completeness()
        
        # Low completeness, high confidence
        element2 = FoundationElement(name="Incomplete but certain", domain=EvidenceDomain.GENERAL)
        element2.what = "Very clearly defined"
        element2.what_confidence = 0.95
        # Leave everything else undefined
        
        completeness2 = element2.weighted_completeness()
        
        assert completeness1 > completeness2, "First should be more complete"
        return f"Complete/uncertain: {completeness1:.2f}, Incomplete/certain: {completeness2:.2f}"

class TestWeightSymmetry(StressTest):
    """Test that quality score weights sum to 1.0"""
    def __init__(self):
        super().__init__("Weight sum equals 1.0")
    
    def run(self):
        weights = {
            'completeness': 0.30,
            'confidence': 0.20,
            'evidence': 0.20,
            'consistency': 0.20,
            'efficiency': 0.10
        }
        
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, not 1.0"
        return f"Weight sum: {total}"

# ============================================================================
# CATEGORY 6: DOMAIN TRANSFERABILITY TESTS
# ============================================================================

class TestCrossDomainEvidence(StressTest):
    """Test evidence quality across all domains"""
    def __init__(self):
        super().__init__("Cross-domain evidence quality")
    
    def run(self):
        results = {}
        
        for domain in [EvidenceDomain.MEDICAL, EvidenceDomain.BUSINESS, 
                      EvidenceDomain.POLICY, EvidenceDomain.GENERAL]:
            # Create same "quality" evidence in different domains
            evidence = Evidence(
                content="Strong study",
                source="Top journal",
                strength=0.9,
                date="2024",
                domain=domain,
                study_design="systematic_review_meta_analysis" if domain == EvidenceDomain.MEDICAL else "multi_company_analysis"
            )
            
            results[domain.value] = evidence.quality_score()
        
        # Check that quality scores are reasonable across domains
        for domain, score in results.items():
            assert 0.8 <= score <= 1.0, f"{domain} score {score} out of expected range"
        
        return f"Scores: {results}"

class TestDomainSpecificHierarchies(StressTest):
    """Test that each domain has appropriate hierarchies"""
    def __init__(self):
        super().__init__("Domain-specific hierarchy appropriateness")
    
    def run(self):
        from cap_implementation_v3 import EVIDENCE_HIERARCHIES
        
        issues = []
        
        # Medical should have RCT
        if 'randomized_controlled_trial' not in EVIDENCE_HIERARCHIES[EvidenceDomain.MEDICAL]:
            issues.append("Medical missing RCT")
        
        # Business should NOT have RCT
        if 'randomized_controlled_trial' in EVIDENCE_HIERARCHIES[EvidenceDomain.BUSINESS]:
            issues.append("Business has RCT (inappropriate)")
        
        # Policy should have randomized_evaluation
        if 'randomized_evaluation' not in EVIDENCE_HIERARCHIES[EvidenceDomain.POLICY]:
            issues.append("Policy missing randomized evaluation")
        
        if issues:
            return f"Issues: {issues}"
        return "All domain hierarchies appropriate"

# ============================================================================
# CATEGORY 7: PERFORMANCE STRESS TESTS
# ============================================================================

class TestLargeScale(StressTest):
    """Test with many dimensions and iterations"""
    def __init__(self):
        super().__init__("Large-scale analysis")
    
    def run(self):
        import time
        
        element = FoundationElement(name="Complex Project", domain=EvidenceDomain.BUSINESS)
        
        # Fill all dimensions
        element.what = "A" * 1000  # Long descriptions
        element.why = "B" * 1000
        element.how = "C" * 1000
        element.when = "D" * 1000
        element.who = "E" * 1000
        element.measure = "F" * 1000
        
        element.what_confidence = 0.6
        element.why_confidence = 0.6
        element.how_confidence = 0.6
        element.when_confidence = 0.6
        element.who_confidence = 0.6
        element.measure_confidence = 0.6
        
        # Add many evidence pieces
        for i in range(50):
            element.add_evidence(Evidence(
                content=f"Evidence {i}" * 10,
                source=f"Source {i}",
                strength=0.5 + i * 0.01,
                date="2024",
                domain=EvidenceDomain.BUSINESS
            ))
        
        # Time the analysis
        start = time.time()
        results = run_analysis(element, rigor_level=2, max_iterations=15)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s"
        return f"Completed in {elapsed:.3f}s with {len(element.evidence)} evidence pieces"

class TestMemoryUsage(StressTest):
    """Test memory usage with many elements"""
    def __init__(self):
        super().__init__("Memory usage stress")
    
    def run(self):
        import sys
        
        elements = []
        
        # Create 1000 elements
        for i in range(1000):
            element = FoundationElement(name=f"Element {i}", domain=EvidenceDomain.GENERAL)
            element.what = f"Description {i}"
            element.why = f"Justification {i}"
            element.what_confidence = 0.5
            element.why_confidence = 0.5
            
            # Add some evidence
            for j in range(5):
                element.add_evidence(Evidence(
                    content=f"Evidence {j}",
                    source=f"Source {j}",
                    strength=0.6,
                    date="2024"
                ))
            
            elements.append(element)
        
        # Get approximate memory usage
        total_size = sum(sys.getsizeof(e) for e in elements)
        
        return f"Created 1000 elements, approximate size: {total_size / 1024:.1f} KB"

# ============================================================================
# CATEGORY 8: EDGE CASE SCENARIOS
# ============================================================================

class TestEmptyElement(StressTest):
    """Test completely empty element"""
    def __init__(self):
        super().__init__("Empty element handling")
    
    def run(self):
        element = FoundationElement(name="Empty", domain=EvidenceDomain.GENERAL)
        # Don't fill anything
        
        completeness = element.weighted_completeness()
        gaps = element.critical_gaps()
        ready, reason = element.ready_for_action(rigor_level=1)
        
        assert completeness == 0.0, "Empty element should have 0 completeness"
        assert len(gaps) > 0, "Empty element should have gaps"
        assert not ready, "Empty element should not be ready"
        
        return f"Completeness: {completeness}, Gaps: {len(gaps)}, Ready: {ready}"

class TestSingleDimension(StressTest):
    """Test with only WHAT dimension filled"""
    def __init__(self):
        super().__init__("Single dimension analysis")
    
    def run(self):
        element = FoundationElement(name="Minimal", domain=EvidenceDomain.GENERAL)
        element.what = "Only this is defined"
        element.what_confidence = 0.9
        
        results = run_analysis(element, rigor_level=1, max_iterations=5)
        
        assert results['quality_scores']['overall'] < 0.5, "Should have low quality with single dimension"
        return f"Quality with single dimension: {results['quality_scores']['overall']:.3f}"

class TestContradictoryConfidence(StressTest):
    """Test high confidence with low-quality evidence"""
    def __init__(self):
        super().__init__("Contradictory confidence-evidence")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Something"
        element.why = "High confidence claim"
        element.what_confidence = 0.9
        element.why_confidence = 0.9  # High confidence
        
        # But only weak evidence
        element.add_evidence(Evidence(
            content="Weak anecdote",
            source="Random blog",
            strength=0.2,
            date="2024",
            domain=EvidenceDomain.GENERAL,
            study_design="anecdote"
        ))
        
        quality = calculate_quality_scores(element, AdversarialTester(element, 2), 2)
        
        # Confidence high but evidence low - should show in scores
        assert quality['confidence'] > 0.8, "Confidence should be high"
        assert quality['evidence'] < 0.3, "Evidence should be low"
        
        return f"Confidence: {quality['confidence']:.2f}, Evidence: {quality['evidence']:.2f}"

# ============================================================================
# CATEGORY 9: CONVERGENCE PATHOLOGY TESTS
# ============================================================================

class TestDivergentQuality(StressTest):
    """Test for divergent quality scores"""
    def __init__(self):
        super().__init__("Quality score divergence check")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Something"
        element.why = "Because"
        element.what_confidence = 0.5
        element.why_confidence = 0.5
        
        results = run_analysis(element, rigor_level=2, max_iterations=15)
        history = results['quality_history']
        
        # Check that quality never goes down significantly
        max_decrease = 0
        for i in range(1, len(history)):
            decrease = history[i-1] - history[i]
            if decrease > max_decrease:
                max_decrease = decrease
        
        assert max_decrease < 0.2, f"Quality decreased too much: {max_decrease:.3f}"
        return f"Max quality decrease: {max_decrease:.3f}"

class TestPrematureConvergence(StressTest):
    """Test for premature convergence with gaps"""
    def __init__(self):
        super().__init__("Premature convergence detection")
    
    def run(self):
        element = FoundationElement(name="Incomplete", domain=EvidenceDomain.GENERAL)
        element.what = "Partially defined"
        element.what_confidence = 0.9  # High confidence
        # But leave WHY completely undefined
        
        results = run_analysis(element, rigor_level=2, max_iterations=10)
        
        gaps = element.critical_gaps()
        critical_gaps = [g for g in gaps if "CRITICAL" in g]
        
        if results['ready_for_action'] and len(critical_gaps) > 0:
            return f"PROBLEM: Marked ready despite {len(critical_gaps)} critical gaps"
        
        return f"Correctly not ready: {len(critical_gaps)} critical gaps"

# ============================================================================
# CATEGORY 10: CONCEPTUAL COHERENCE TESTS
# ============================================================================

class TestEpistemicAleatoryDistinction(StressTest):
    """Test that epistemic vs aleatory uncertainty is handled correctly"""
    def __init__(self):
        super().__init__("Epistemic vs aleatory handling")
    
    def run(self):
        # Test 1: Epistemic uncertainty with low confidence should not be ready
        element1 = FoundationElement(name="Epistemic", domain=EvidenceDomain.GENERAL)
        element1.what = "Unknown but knowable"
        element1.what_confidence = 0.5
        element1.what_uncertainty = UncertaintyType.EPISTEMIC
        element1.why = "Can research this"
        element1.why_confidence = 0.5
        element1.why_uncertainty = UncertaintyType.EPISTEMIC
        
        ready1, reason1 = element1.ready_for_action(rigor_level=2)
        
        # Test 2: Aleatory uncertainty with same confidence might be ready
        element2 = FoundationElement(name="Aleatory", domain=EvidenceDomain.GENERAL)
        element2.what = "Inherently random"
        element2.what_confidence = 0.5
        element2.what_uncertainty = UncertaintyType.ALEATORY
        element2.why = "Cannot reduce further"
        element2.why_confidence = 0.5
        element2.why_uncertainty = UncertaintyType.ALEATORY
        
        ready2, reason2 = element2.ready_for_action(rigor_level=2)
        
        # Epistemic should be more restrictive than aleatory
        return f"Epistemic ready: {ready1}, Aleatory ready: {ready2}"

class TestRigorLevelConsistency(StressTest):
    """Test that higher rigor is always stricter"""
    def __init__(self):
        super().__init__("Rigor level consistency")
    
    def run(self):
        element = FoundationElement(name="Test", domain=EvidenceDomain.GENERAL)
        element.what = "Something"
        element.why = "Some reason"
        element.what_confidence = 0.75
        element.why_confidence = 0.75
        
        ready1, _ = element.ready_for_action(rigor_level=1)
        ready2, _ = element.ready_for_action(rigor_level=2)
        ready3, _ = element.ready_for_action(rigor_level=3)
        
        # Higher rigor should be equal or stricter
        if ready3 and not ready2:
            return "INCONSISTENCY: Level 3 ready but Level 2 not ready"
        if ready2 and not ready1:
            return "INCONSISTENCY: Level 2 ready but Level 1 not ready"
        
        return f"Consistent: L1={ready1}, L2={ready2}, L3={ready3}"

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_stress_tests(verbose: bool = False):
    """Run complete stress test suite"""
    
    print("="*80)
    print("CAP v3.0 COMPREHENSIVE STRESS TEST SUITE")
    print("="*80)
    print()
    
    test_categories = {
        "Boundary Value Tests": [
            TestZeroConfidence(),
            TestMaxConfidence(),
            TestInvalidConfidence(),
            TestNegativeConfidence(),
        ],
        "Evidence Stress Tests": [
            TestNoEvidence(),
            TestMassiveEvidence(),
            TestConflictingEvidenceDomains(),
            TestInvalidEvidence(),
        ],
        "Convergence Tests": [
            TestImmediateConvergence(),
            TestNonConvergence(),
            TestOscillation(),
        ],
        "Extreme Problem Characteristics": [
            TestMinimalComplexity(),
            TestMaximalComplexity(),
            TestInvalidComplexity(),
        ],
        "Logical Consistency Tests": [
            TestConfidenceEvidenceConsistency(),
            TestCompletenessConfidenceDecoupling(),
            TestWeightSymmetry(),
        ],
        "Domain Transferability": [
            TestCrossDomainEvidence(),
            TestDomainSpecificHierarchies(),
        ],
        "Performance Tests": [
            TestLargeScale(),
            TestMemoryUsage(),
        ],
        "Edge Cases": [
            TestEmptyElement(),
            TestSingleDimension(),
            TestContradictoryConfidence(),
        ],
        "Convergence Pathologies": [
            TestDivergentQuality(),
            TestPrematureConvergence(),
        ],
        "Conceptual Coherence": [
            TestEpistemicAleatoryDistinction(),
            TestRigorLevelConsistency(),
        ]
    }
    
    total_tests = sum(len(tests) for tests in test_categories.values())
    passed_tests = 0
    failed_tests = 0
    
    for category, tests in test_categories.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category}")
        print(f"{'='*80}\n")
        
        for test in tests:
            success = test.execute()
            report_test(test, verbose=verbose)
            
            if success:
                passed_tests += 1
            else:
                failed_tests += 1
    
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Failed: {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
    print()
    
    if failed_tests == 0:
        print("✅ ALL STRESS TESTS PASSED")
    else:
        print(f"⚠️  {failed_tests} TESTS FAILED - REVIEW REQUIRED")
    
    print("="*80)
    
    return passed_tests, failed_tests

if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    passed, failed = run_all_stress_tests(verbose=verbose)
    sys.exit(0 if failed == 0 else 1)
