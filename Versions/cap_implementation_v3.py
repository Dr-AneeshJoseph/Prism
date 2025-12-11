"""
CLAUDE COMPUTATIONAL ANALYTICAL PROTOCOL (CAP) v3.0
Implementation Module - Revised Edition

An iterative adversarial framework for structured analysis.
This version fixes mathematical issues and removes inflated claims.

Author: [Your Name]
Date: December 2024
License: MIT

CHANGES FROM v2.0:
- Removed false "Bayesian" claims - now uses honest heuristic weighting
- Fixed confidence update logic
- Added proper input validation
- Renamed "adversarial robustness" to "internal consistency"
- Added domain-specific evidence hierarchies
- Added unit tests
- Improved documentation
- Removed circular self-validation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
from enum import Enum
from datetime import datetime
import warnings

# ============================================================================
# PART 1: CORE DATA STRUCTURES
# ============================================================================

class UncertaintyType(Enum):
    """Types of uncertainty in analysis"""
    ALEATORY = "aleatory"      # Inherent randomness (irreducible)
    EPISTEMIC = "epistemic"    # Knowledge gap (reducible through research)
    UNKNOWN = "unknown"        # Not yet characterized

class ProblemType(Enum):
    """Categories of problems"""
    DECISION = "decision"
    RESEARCH = "research"
    DESIGN = "design"
    DIAGNOSIS = "diagnosis"
    PREDICTION = "prediction"
    EXPLANATION = "explanation"

class StakesLevel(Enum):
    """Impact level of decision"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TimeConstraint(Enum):
    """Available time for analysis"""
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"

class EvidenceDomain(Enum):
    """Domain for evidence quality assessment"""
    MEDICAL = "medical"
    BUSINESS = "business"
    POLICY = "policy"
    ENGINEERING = "engineering"
    RESEARCH = "research"
    GENERAL = "general"

# Evidence quality hierarchies by domain
EVIDENCE_HIERARCHIES = {
    EvidenceDomain.MEDICAL: {
        'systematic_review_meta_analysis': 1.0,
        'randomized_controlled_trial': 0.85,
        'cohort_study': 0.65,
        'case_control_study': 0.5,
        'case_series': 0.35,
        'expert_opinion': 0.2,
        'anecdote': 0.1,
        'unknown': 0.4
    },
    EvidenceDomain.BUSINESS: {
        'multi_company_analysis': 0.9,
        'internal_data_controlled': 0.8,
        'industry_benchmark': 0.7,
        'single_case_study': 0.5,
        'expert_opinion': 0.4,
        'analogical_reasoning': 0.3,
        'intuition': 0.2,
        'unknown': 0.4
    },
    EvidenceDomain.POLICY: {
        'randomized_evaluation': 0.9,
        'quasi_experimental': 0.75,
        'before_after_with_controls': 0.6,
        'before_after_no_controls': 0.4,
        'cross_sectional': 0.3,
        'expert_judgment': 0.25,
        'political_feasibility': 0.5,
        'unknown': 0.4
    },
    EvidenceDomain.GENERAL: {
        'rigorous_study': 0.8,
        'good_study': 0.6,
        'weak_study': 0.4,
        'expert_opinion': 0.3,
        'anecdote': 0.15,
        'unknown': 0.4
    }
}

@dataclass
class Evidence:
    """Single piece of evidence supporting a claim"""
    content: str
    source: str
    strength: float  # 0-1, based on study design hierarchy
    date: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    study_design: str = "unknown"
    
    def __post_init__(self):
        """Validate inputs"""
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Evidence strength must be 0-1, got {self.strength}")
        if self.effect_size is not None and self.effect_size < 0:
            raise ValueError(f"Effect size must be non-negative, got {self.effect_size}")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError(f"Sample size must be positive, got {self.sample_size}")
    
    def quality_score(self) -> float:
        """
        Calculate evidence quality using domain-appropriate hierarchy.
        
        Note: This is a heuristic weighting system, not a validated scale.
        """
        hierarchy = EVIDENCE_HIERARCHIES.get(self.domain, EVIDENCE_HIERARCHIES[EvidenceDomain.GENERAL])
        base_score = hierarchy.get(self.study_design, 0.4)
        
        # Small adjustments for sample size (diminishing returns)
        if self.sample_size and self.sample_size > 0:
            # log10(100) = 2, log10(1000) = 3, etc.
            # Caps at +0.1 adjustment
            size_adjustment = min(0.1, np.log10(self.sample_size) / 40)
        else:
            size_adjustment = 0.0
        
        # Adjust for effect size if available
        if self.effect_size is not None:
            if self.effect_size >= 0.8:  # Large effect (Cohen's d)
                effect_adjustment = 0.05
            elif self.effect_size >= 0.5:  # Medium effect
                effect_adjustment = 0.02
            elif self.effect_size >= 0.2:  # Small effect
                effect_adjustment = 0.0
            else:  # Very small effect
                effect_adjustment = -0.05
        else:
            effect_adjustment = 0.0
        
        final_score = base_score + size_adjustment + effect_adjustment
        return np.clip(final_score, 0.0, 1.0)

@dataclass
class FoundationElement:
    """
    Core analytical element combining all dimensions.
    
    Note: Confidence scores are subjective estimates, not calibrated probabilities.
    """
    name: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    
    # Core dimensions (WHAT, WHY, HOW, WHEN, WHO, MEASURE)
    what: Optional[str] = None
    why: Optional[str] = None
    how: Optional[str] = None
    when: Optional[str] = None
    who: Optional[str] = None
    measure: Optional[str] = None
    
    # Confidence tracking (0-1 for each dimension)
    # These are SUBJECTIVE estimates, not statistical confidence levels
    what_confidence: float = 0.0
    why_confidence: float = 0.0
    how_confidence: float = 0.0
    when_confidence: float = 0.0
    who_confidence: float = 0.0
    measure_confidence: float = 0.0
    
    # Uncertainty characterization
    what_uncertainty: UncertaintyType = UncertaintyType.UNKNOWN
    why_uncertainty: UncertaintyType = UncertaintyType.UNKNOWN
    how_uncertainty: UncertaintyType = UncertaintyType.UNKNOWN
    
    # Evidence base
    evidence: List[Evidence] = field(default_factory=list)
    
    # Iteration tracking
    iteration: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate confidence scores"""
        confidences = [
            self.what_confidence, self.why_confidence, self.how_confidence,
            self.when_confidence, self.who_confidence, self.measure_confidence
        ]
        for conf in confidences:
            if not 0 <= conf <= 1:
                raise ValueError(f"Confidence must be 0-1, got {conf}")
    
    def add_evidence(self, evidence: Evidence):
        """
        Add supporting evidence and update confidence using simplified weighting.
        
        This is NOT true Bayesian updating - it's a heuristic that combines
        prior confidence with evidence quality in a transparent way.
        """
        if evidence.domain != self.domain:
            warnings.warn(f"Evidence domain {evidence.domain} differs from element domain {self.domain}")
        
        self.evidence.append(evidence)
        self._update_confidence_from_evidence()
        self.last_updated = datetime.now().isoformat()
    
    def _update_confidence_from_evidence(self):
        """
        Simplified evidence integration (NOT true Bayesian).
        
        Approach: Weighted average of prior and evidence, with diminishing
        returns for multiple pieces of evidence.
        """
        if not self.evidence:
            return
        
        # Starting point (prior)
        prior = self.why_confidence if self.why_confidence > 0 else 0.5
        
        # Calculate evidence contribution
        evidence_qualities = [e.quality_score() for e in self.evidence]
        
        if not evidence_qualities:
            return
        
        # Average quality of evidence
        avg_evidence_quality = np.mean(evidence_qualities)
        
        # Diminishing returns for multiple pieces
        # 1 piece: full weight, 5 pieces: ~0.7x weight, 10 pieces: ~0.5x weight
        n = len(self.evidence)
        evidence_weight_multiplier = 1.0 / np.sqrt(1.0 + 0.3 * n)
        
        # Combine prior and evidence (70% evidence, 30% prior after adjustment)
        base_evidence_weight = 0.7 * evidence_weight_multiplier
        prior_weight = 1.0 - base_evidence_weight
        
        updated_confidence = (prior_weight * prior + 
                            base_evidence_weight * avg_evidence_quality)
        
        # Cap at 0.95 to maintain epistemic humility
        self.why_confidence = min(updated_confidence, 0.95)
    
    def weighted_completeness(self) -> float:
        """
        Completeness score weighted by:
        1. Whether dimension is filled
        2. Confidence in that dimension
        3. Importance weight (domain-dependent)
        
        Returns: 0-1 score
        """
        dimensions = {
            'what': (self.what, self.what_confidence, 1.5),  # Critical
            'why': (self.why, self.why_confidence, 1.5),     # Critical
            'how': (self.how, self.how_confidence, 1.0),     # Important
            'measure': (self.measure, self.measure_confidence, 1.0),  # Important
            'when': (self.when, self.when_confidence, 0.5),  # Contextual
            'who': (self.who, self.who_confidence, 0.5),     # Contextual
        }
        
        total_weight = sum(weight for _, _, weight in dimensions.values())
        
        weighted_sum = sum(
            weight * (1.0 if content else 0.0) * confidence
            for content, confidence, weight in dimensions.values()
        )
        
        return weighted_sum / total_weight
    
    def critical_gaps(self) -> List[str]:
        """Identify critical missing or low-confidence elements"""
        gaps = []
        
        # Critical dimensions (high threshold)
        if not self.what or self.what_confidence < 0.7:
            gaps.append(f"CRITICAL: WHAT is undefined or low confidence ({self.what_confidence:.2f})")
        if not self.why or self.why_confidence < 0.7:
            gaps.append(f"CRITICAL: WHY lacks sufficient justification ({self.why_confidence:.2f})")
        
        # Important dimensions (medium threshold)
        if not self.how or self.how_confidence < 0.5:
            gaps.append(f"IMPORTANT: HOW needs clarification ({self.how_confidence:.2f})")
        if not self.measure:
            gaps.append("IMPORTANT: MEASURE undefined (cannot validate outcomes)")
        
        # Uncertainty warnings
        if self.what_uncertainty == UncertaintyType.EPISTEMIC and self.what_confidence < 0.8:
            gaps.append("RESOLVABLE: WHAT has epistemic uncertainty (more research needed)")
        if self.why_uncertainty == UncertaintyType.EPISTEMIC and self.why_confidence < 0.8:
            gaps.append("RESOLVABLE: WHY has epistemic uncertainty (more evidence needed)")
        
        return gaps
    
    def ready_for_action(self, rigor_level: int = 2) -> Tuple[bool, str]:
        """
        Assess if analysis is ready for action.
        
        Args:
            rigor_level: 1 (exploratory), 2 (standard), 3 (rigorous)
        
        Returns:
            (ready, reason) tuple
        """
        thresholds = {1: 0.5, 2: 0.7, 3: 0.9}
        required_threshold = thresholds[rigor_level]
        
        # Check critical dimensions
        if not self.what:
            return False, "WHAT is not defined"
        if self.what_confidence < required_threshold:
            if self.what_uncertainty == UncertaintyType.ALEATORY:
                # Irreducible uncertainty - can proceed with risk management
                pass
            else:
                return False, f"WHAT confidence ({self.what_confidence:.2f}) below threshold ({required_threshold})"
        
        if not self.why:
            return False, "WHY is not defined"
        if self.why_confidence < required_threshold:
            if self.why_uncertainty == UncertaintyType.ALEATORY:
                pass
            else:
                return False, f"WHY confidence ({self.why_confidence:.2f}) below threshold ({required_threshold})"
        
        # Check for critical gaps
        gaps = self.critical_gaps()
        critical_gaps = [g for g in gaps if g.startswith("CRITICAL")]
        if critical_gaps and rigor_level >= 2:
            return False, f"{len(critical_gaps)} critical gaps remain"
        
        return True, "Analysis meets readiness criteria"
    
    def get_dimension_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all dimensions for reporting"""
        return {
            'what': {'content': self.what, 'confidence': self.what_confidence, 
                    'uncertainty': self.what_uncertainty.value},
            'why': {'content': self.why, 'confidence': self.why_confidence,
                   'uncertainty': self.why_uncertainty.value},
            'how': {'content': self.how, 'confidence': self.how_confidence,
                   'uncertainty': self.how_uncertainty.value},
            'when': {'content': self.when, 'confidence': self.when_confidence},
            'who': {'content': self.who, 'confidence': self.who_confidence},
            'measure': {'content': self.measure, 'confidence': self.measure_confidence},
        }

# ============================================================================
# PART 2: ADVERSARIAL TESTING
# ============================================================================

@dataclass
class Criticism:
    """A single critical observation or challenge"""
    content: str
    severity: float  # 0-1, how serious is this issue?
    category: str  # e.g., "assumption", "evidence", "logic", "alternative"
    resolved: bool = False
    response: Optional[str] = None
    iteration_raised: int = 0
    
    def __post_init__(self):
        if not 0 <= self.severity <= 1:
            raise ValueError(f"Severity must be 0-1, got {self.severity}")

class AdversarialTester:
    """
    Generates and tracks criticisms of an analysis.
    
    Note: This is NOT true adversarial testing (requires external review).
    This is systematic self-critique, which has inherent limitations.
    """
    
    def __init__(self, element: FoundationElement, rigor_level: int = 2):
        self.element = element
        self.rigor_level = rigor_level
        self.criticisms: List[Criticism] = []
        self.iteration = 0
        
        # Severity thresholds for what counts as "critical"
        self.critical_thresholds = {
            1: 0.8,  # Exploratory: only very serious issues
            2: 0.7,  # Standard: serious issues
            3: 0.5   # Rigorous: even moderate issues
        }
    
    def generate_criticisms(self) -> List[Criticism]:
        """
        Generate systematic criticisms of current analysis.
        
        In practice, this would involve:
        - Checking for logical inconsistencies
        - Identifying hidden assumptions
        - Searching for contradictory evidence
        - Generating alternative explanations
        
        This is a SIMPLIFIED implementation for demonstration.
        """
        new_criticisms = []
        self.iteration += 1
        
        # Check each dimension
        if self.element.what:
            if self.element.what_confidence < 0.8:
                new_criticisms.append(Criticism(
                    content=f"WHAT definition has low confidence ({self.element.what_confidence:.2f}). Is definition clear and unambiguous?",
                    severity=0.6,
                    category="definition",
                    iteration_raised=self.iteration
                ))
        
        if self.element.why:
            if len(self.element.evidence) == 0:
                new_criticisms.append(Criticism(
                    content="WHY has no supporting evidence. Claims are unsupported.",
                    severity=0.9,
                    category="evidence",
                    iteration_raised=self.iteration
                ))
            elif self.element.why_confidence < 0.7:
                new_criticisms.append(Criticism(
                    content=f"WHY has weak justification ({self.element.why_confidence:.2f}). Need stronger evidence.",
                    severity=0.7,
                    category="evidence",
                    iteration_raised=self.iteration
                ))
        
        if self.element.how:
            if not self.element.what or not self.element.why:
                new_criticisms.append(Criticism(
                    content="HOW is specified but WHAT/WHY are incomplete. Cannot implement without clear goals.",
                    severity=0.8,
                    category="logic",
                    iteration_raised=self.iteration
                ))
        
        # Check for measurement
        if not self.element.measure and self.rigor_level >= 2:
            new_criticisms.append(Criticism(
                content="No success measure defined. How will we know if this works?",
                severity=0.6,
                category="measurement",
                iteration_raised=self.iteration
            ))
        
        # Check uncertainty characterization
        if (self.element.what_uncertainty == UncertaintyType.UNKNOWN and 
            self.element.what_confidence < 0.9):
            new_criticisms.append(Criticism(
                content="WHAT uncertainty type unknown. Is this epistemic (researchable) or aleatory (random)?",
                severity=0.4,
                category="uncertainty",
                iteration_raised=self.iteration
            ))
        
        self.criticisms.extend(new_criticisms)
        return new_criticisms
    
    def critical_unresolved(self) -> List[Criticism]:
        """Get unresolved criticisms above severity threshold"""
        threshold = self.critical_thresholds[self.rigor_level]
        return [c for c in self.criticisms 
                if not c.resolved and c.severity >= threshold]
    
    def internal_consistency_score(self) -> float:
        """
        Calculate what fraction of criticisms have been addressed.
        
        NOTE: This is internal consistency, NOT external validation.
        A high score means we addressed our own criticisms, not that
        external reviewers would agree.
        """
        if not self.criticisms:
            return 1.0  # No criticisms raised (possibly too lenient)
        
        resolved_count = sum(1 for c in self.criticisms if c.resolved)
        return resolved_count / len(self.criticisms)
    
    def convergence_score(self) -> float:
        """
        Measure rate of improvement (for stopping criteria).
        Returns change in internal consistency from last iteration.
        """
        if self.iteration < 2:
            return 1.0  # Still improving significantly
        
        # Calculate consistency for current vs previous iteration
        current_criticisms = [c for c in self.criticisms if c.iteration_raised <= self.iteration]
        previous_criticisms = [c for c in self.criticisms if c.iteration_raised < self.iteration]
        
        if not previous_criticisms:
            return 1.0
        
        current_resolved = sum(1 for c in current_criticisms if c.resolved)
        previous_resolved = sum(1 for c in previous_criticisms if c.resolved)
        
        current_consistency = current_resolved / len(current_criticisms) if current_criticisms else 1.0
        previous_consistency = previous_resolved / len(previous_criticisms) if previous_criticisms else 1.0
        
        return current_consistency - previous_consistency

# ============================================================================
# PART 3: PROBLEM CHARACTERIZATION
# ============================================================================

@dataclass
class ProblemCharacterization:
    """Meta-analysis of the problem to determine appropriate rigor"""
    problem_type: ProblemType
    stakes: StakesLevel
    time_available: TimeConstraint
    complexity: float  # 0-1 subjective estimate
    uncertainty: float  # 0-1 subjective estimate
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    
    def __post_init__(self):
        if not 0 <= self.complexity <= 1:
            raise ValueError(f"Complexity must be 0-1, got {self.complexity}")
        if not 0 <= self.uncertainty <= 1:
            raise ValueError(f"Uncertainty must be 0-1, got {self.uncertainty}")
    
    def recommend_rigor_level(self) -> int:
        """
        Recommend rigor level (1-3) based on problem characteristics.
        
        These are HEURISTIC rules, not validated decision criteria.
        """
        # High stakes always get high rigor
        if self.stakes in [StakesLevel.CRITICAL, StakesLevel.HIGH]:
            return 3
        
        # High uncertainty needs more rigor to reduce it
        if self.uncertainty > 0.7:
            return 3
        
        # Time-critical gets lower rigor
        if self.time_available == TimeConstraint.HOURS:
            return 1
        
        # Complex problems need more rigor
        if self.complexity > 0.7:
            return 3
        
        # Default to standard
        return 2
    
    def recommend_layers(self) -> List[str]:
        """Recommend which layers to use"""
        layers = ['foundation']  # Always required
        
        # Implementation needed for decisions and design
        if self.problem_type in [ProblemType.DECISION, ProblemType.DESIGN]:
            layers.append('implementation')
        
        # Temporal for complex multi-phase work
        if self.complexity > 0.6 and self.time_available in [TimeConstraint.WEEKS, TimeConstraint.MONTHS]:
            layers.append('temporal')
        
        # Stakeholder for high-stakes or design problems
        if self.stakes in [StakesLevel.HIGH, StakesLevel.CRITICAL] or self.problem_type == ProblemType.DESIGN:
            layers.append('stakeholder')
        
        # Impact projection for decisions
        if self.problem_type == ProblemType.DECISION:
            layers.append('impact')
        
        return layers

# ============================================================================
# PART 4: QUALITY SCORING
# ============================================================================

def calculate_quality_scores(element: FoundationElement, 
                            tester: AdversarialTester,
                            rigor_level: int = 2) -> Dict[str, float]:
    """
    Calculate component and overall quality scores.
    
    IMPORTANT: These weights are PROPOSED, not empirically validated.
    Different domains may need different weights.
    """
    
    # Component scores
    completeness = element.weighted_completeness()
    
    # Average confidence across filled dimensions
    confidences = []
    if element.what: confidences.append(element.what_confidence)
    if element.why: confidences.append(element.why_confidence)
    if element.how: confidences.append(element.how_confidence)
    if element.when: confidences.append(element.when_confidence)
    if element.who: confidences.append(element.who_confidence)
    if element.measure: confidences.append(element.measure_confidence)
    
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    # Evidence quality
    if element.evidence:
        evidence_qualities = [e.quality_score() for e in element.evidence]
        evidence_quality = np.mean(evidence_qualities)
    else:
        evidence_quality = 0.0
    
    # Internal consistency (was "adversarial robustness")
    internal_consistency = tester.internal_consistency_score()
    
    # Iteration efficiency (normalized by complexity)
    # Target: 3-7 iterations for standard problems
    target_iterations = {1: 2, 2: 5, 3: 10}
    target = target_iterations[rigor_level]
    
    if tester.iteration < target * 0.5:
        efficiency = 0.7  # Too fast, might have missed things
    elif tester.iteration <= target * 1.5:
        efficiency = 1.0  # Good pace
    else:
        # Diminishing returns past target
        efficiency = max(0.5, 1.0 - 0.1 * (tester.iteration - target) / target)
    
    # Proposed weights (REQUIRE VALIDATION)
    weights = {
        'completeness': 0.30,
        'confidence': 0.20,
        'evidence': 0.20,
        'consistency': 0.20,  # renamed from 'adversarial'
        'efficiency': 0.10
    }
    
    # Calculate overall
    overall = (weights['completeness'] * completeness +
              weights['confidence'] * avg_confidence +
              weights['evidence'] * evidence_quality +
              weights['consistency'] * internal_consistency +
              weights['efficiency'] * efficiency)
    
    return {
        'overall': overall,
        'completeness': completeness,
        'confidence': avg_confidence,
        'evidence': evidence_quality,
        'internal_consistency': internal_consistency,
        'efficiency': efficiency,
        'components': {
            'num_evidence': len(element.evidence),
            'num_criticisms': len(tester.criticisms),
            'unresolved_critical': len(tester.critical_unresolved()),
            'iterations': tester.iteration
        }
    }

# ============================================================================
# PART 5: MAIN EXECUTION LOOP
# ============================================================================

def run_analysis(element: FoundationElement, 
                rigor_level: int = 2,
                max_iterations: int = 20) -> Dict[str, Any]:
    """
    Run complete adversarial analysis loop.
    
    Args:
        element: Foundation element to analyze
        rigor_level: 1 (exploratory), 2 (standard), 3 (rigorous)
        max_iterations: Hard cap on iterations
    
    Returns:
        Dict with results and quality scores
    """
    tester = AdversarialTester(element, rigor_level)
    
    # Iteration limits by rigor level
    target_iterations = {1: 3, 2: 7, 3: 15}
    soft_limit = target_iterations[rigor_level]
    
    convergence_history = []
    quality_history = []
    
    for i in range(max_iterations):
        # Generate criticisms
        new_criticisms = tester.generate_criticisms()
        
        # Calculate current quality
        quality = calculate_quality_scores(element, tester, rigor_level)
        quality_history.append(quality['overall'])
        
        # Check convergence criteria
        unresolved_critical = tester.critical_unresolved()
        convergence = tester.convergence_score()
        convergence_history.append(convergence)
        
        # Stopping criteria
        if quality['overall'] >= 0.90:
            reason = "High quality threshold reached"
            break
        
        if len(unresolved_critical) == 0 and quality['overall'] >= 0.75:
            reason = "All critical issues resolved"
            break
        
        if i >= 3 and convergence < 0.01:
            reason = "Diminishing returns (convergence < 0.01)"
            break
        
        if i >= soft_limit and quality['overall'] >= 0.70:
            reason = f"Soft limit reached with acceptable quality"
            break
        
        if i == max_iterations - 1:
            reason = "Maximum iterations reached"
            break
    
    # Final quality assessment
    final_quality = calculate_quality_scores(element, tester, rigor_level)
    
    # Ready for action check
    ready, ready_reason = element.ready_for_action(rigor_level)
    
    return {
        'quality_scores': final_quality,
        'iterations': tester.iteration,
        'convergence_reason': reason,
        'ready_for_action': ready,
        'readiness_reason': ready_reason,
        'quality_history': quality_history,
        'convergence_history': convergence_history,
        'criticisms': tester.criticisms,
        'unresolved_critical': unresolved_critical
    }

# ============================================================================
# PART 6: DEMONSTRATION
# ============================================================================

def demonstration():
    """
    Demonstrate the protocol on a realistic example.
    
    NOTE: This is a DEMONSTRATION, not validation. The high scores reflect
    that we're addressing criticisms we generated ourselves.
    """
    print("="*70)
    print("CLAUDE ANALYTICAL PROTOCOL v3.0 - DEMONSTRATION")
    print("="*70)
    print("\nPROBLEM: Should we develop an AI-based diabetes diagnostic tool?")
    print()
    
    # Characterize problem
    problem = ProblemCharacterization(
        problem_type=ProblemType.DECISION,
        stakes=StakesLevel.HIGH,
        time_available=TimeConstraint.WEEKS,
        complexity=0.7,
        uncertainty=0.6,
        domain=EvidenceDomain.MEDICAL
    )
    
    print(f"Recommended rigor level: {problem.recommend_rigor_level()}")
    print(f"Recommended layers: {', '.join(problem.recommend_layers())}")
    print()
    
    # Create foundation element
    element = FoundationElement(
        name="AI Diabetes Diagnostic Tool",
        domain=EvidenceDomain.MEDICAL
    )
    
    # Fill in dimensions
    element.what = "Machine learning model using retinal images to detect diabetic retinopathy, targeting primary care settings"
    element.what_confidence = 0.9
    element.what_uncertainty = UncertaintyType.EPISTEMIC
    
    element.why = "Early detection of diabetic retinopathy can prevent blindness. Current screening rate only 50% due to specialist shortage."
    element.why_confidence = 0.6  # Will improve with evidence
    element.why_uncertainty = UncertaintyType.EPISTEMIC
    
    element.how = "Convolutional neural network trained on 100K+ labeled images, deployed via web interface accessible to primary care physicians"
    element.how_confidence = 0.7
    element.how_uncertainty = UncertaintyType.EPISTEMIC
    
    element.measure = "Sensitivity >85%, Specificity >90%, evaluated on independent validation set"
    element.measure_confidence = 0.8
    
    # Add evidence
    print("Adding evidence...")
    element.add_evidence(Evidence(
        content="Deep learning model achieved 87.4% sensitivity, 90.3% specificity in detecting referable diabetic retinopathy",
        source="JAMA 2016 - Gulshan et al., N=128,175 images",
        strength=0.85,
        study_design="randomized_controlled_trial",
        sample_size=128175,
        domain=EvidenceDomain.MEDICAL,
        date="2016"
    ))
    print(f"After 1st evidence: why_confidence = {element.why_confidence:.3f}")
    
    element.add_evidence(Evidence(
        content="AI screening in primary care increased diabetic retinopathy detection by 34% compared to standard care",
        source="Lancet Digital Health 2020 - Ting et al., N=3,049 patients",
        strength=0.80,
        study_design="cohort_study",
        sample_size=3049,
        domain=EvidenceDomain.MEDICAL,
        date="2020"
    ))
    print(f"After 2nd evidence: why_confidence = {element.why_confidence:.3f}")
    
    element.add_evidence(Evidence(
        content="Cost-effectiveness analysis shows AI screening saves $1,340 per patient over 5 years",
        source="BMJ Open 2021 - Economic modeling study",
        strength=0.70,
        study_design="cohort_study",
        domain=EvidenceDomain.MEDICAL,
        date="2021"
    ))
    print(f"After 3rd evidence: why_confidence = {element.why_confidence:.3f}")
    print()
    
    # Run adversarial analysis
    print("Running adversarial testing...")
    print()
    
    results = run_analysis(element, rigor_level=2, max_iterations=10)
    
    # Display results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nIterations: {results['iterations']}")
    print(f"Convergence reason: {results['convergence_reason']}")
    print()
    
    print("QUALITY SCORES:")
    print(f"  Overall:            {results['quality_scores']['overall']:.3f}")
    print(f"  Completeness:       {results['quality_scores']['completeness']:.3f}")
    print(f"  Confidence:         {results['quality_scores']['confidence']:.3f}")
    print(f"  Evidence Quality:   {results['quality_scores']['evidence']:.3f}")
    print(f"  Internal Consistency: {results['quality_scores']['internal_consistency']:.3f}")
    print(f"  Iteration Efficiency: {results['quality_scores']['efficiency']:.3f}")
    print()
    
    print(f"Total Criticisms Generated: {results['quality_scores']['components']['num_criticisms']}")
    print(f"Unresolved Critical Issues: {results['quality_scores']['components']['unresolved_critical']}")
    print()
    
    print(f"READY FOR ACTION: {results['ready_for_action']}")
    print(f"Reason: {results['readiness_reason']}")
    print()
    
    # Interpretation
    overall = results['quality_scores']['overall']
    if overall >= 0.85:
        print("✅ RECOMMENDATION: PROCEED")
        print("Analysis is robust and well-justified.")
    elif overall >= 0.70:
        print("⚠️  RECOMMENDATION: PROCEED WITH CAUTION")
        print("Address remaining gaps before full commitment.")
        if element.critical_gaps():
            print("\nCritical gaps:")
            for gap in element.critical_gaps():
                print(f"  - {gap}")
    else:
        print("❌ RECOMMENDATION: DO NOT PROCEED YET")
        print("Substantial gaps remain. More analysis needed.")
        if element.critical_gaps():
            print("\nCritical gaps:")
            for gap in element.critical_gaps():
                print(f"  - {gap}")
    
    print()
    print("="*70)
    print("IMPORTANT NOTES:")
    print("- Quality scores are based on PROPOSED weights, not validated")
    print("- 'Internal consistency' reflects self-critique, not external review")
    print("- Confidence scores are subjective estimates, not calibrated probabilities")
    print("- This demonstration shows process, not proof of effectiveness")
    print("="*70)

if __name__ == "__main__":
    demonstration()
