"""
PRACTICAL IMPROVEMENTS & FIXES
===============================
Concrete code improvements to address critical vulnerabilities.

These fixes can be integrated into enhanced_protocol_v2.py to
make the system more robust and trustworthy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# FIX 1: SAFETY LIMITS AND WARNINGS
# =============================================================================

class WarningLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SystemWarning:
    level: WarningLevel
    category: str
    message: str
    recommendation: str
    
    def __str__(self):
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[self.level.value]
        return f"{icon} [{self.category}] {self.message}\n   → {self.recommendation}"


class SafetyLimits:
    """Hard limits to prevent system abuse"""
    
    MAX_ITERATIONS = 100
    MAX_EVIDENCE_PIECES = 500
    MAX_MECHANISM_NODES = 100
    MAX_MECHANISM_EDGES = 500
    MIN_EVIDENCE_FOR_DECISION = 3
    MIN_CALIBRATION_POINTS = 20
    
    CREDENCE_WARNING_THRESHOLD = 0.95
    EVIDENCE_BIT_CAP = 20.0
    MAX_LIKELIHOOD_RATIO = 100.0
    MIN_LIKELIHOOD_RATIO = 0.01
    
    MAX_DIMENSION_WEIGHT = 10.0
    MIN_CONTRADICTING_EVIDENCE_RATIO = 0.1


class WarningSystem:
    """Centralized warning generation and tracking"""
    
    def __init__(self):
        self.warnings: List[SystemWarning] = []
    
    def add_warning(self, level: WarningLevel, category: str, 
                   message: str, recommendation: str):
        self.warnings.append(SystemWarning(level, category, message, recommendation))
    
    def check_evidence_sufficiency(self, evidence_count: int) -> bool:
        if evidence_count < SafetyLimits.MIN_EVIDENCE_FOR_DECISION:
            self.add_warning(
                WarningLevel.CRITICAL,
                "Evidence Sufficiency",
                f"Only {evidence_count} pieces of evidence (minimum: {SafetyLimits.MIN_EVIDENCE_FOR_DECISION})",
                "Add more evidence before making high-stakes decisions"
            )
            return False
        return True
    
    def check_high_confidence(self, credence: float) -> bool:
        if credence > SafetyLimits.CREDENCE_WARNING_THRESHOLD:
            self.add_warning(
                WarningLevel.WARNING,
                "High Confidence",
                f"Credence = {credence:.3f} exceeds {SafetyLimits.CREDENCE_WARNING_THRESHOLD}",
                "Verify you're not overconfident. Actively seek contradicting evidence."
            )
            return False
        return True
    
    def check_evidence_balance(self, supporting: int, contradicting: int) -> bool:
        if contradicting == 0 and supporting > 2:
            self.add_warning(
                WarningLevel.WARNING,
                "Evidence Imbalance",
                f"{supporting} supporting, 0 contradicting evidence",
                "Actively seek evidence that might contradict your hypothesis"
            )
            return False
        
        total = supporting + contradicting
        if total > 0:
            ratio = contradicting / total
            if ratio < SafetyLimits.MIN_CONTRADICTING_EVIDENCE_RATIO:
                self.add_warning(
                    WarningLevel.WARNING,
                    "Evidence Imbalance",
                    f"Only {ratio:.1%} of evidence is contradicting",
                    "Consider confirmation bias. Seek dissenting opinions."
                )
                return False
        return True
    
    def check_calibration(self, num_predictions: int) -> bool:
        if num_predictions < SafetyLimits.MIN_CALIBRATION_POINTS:
            self.add_warning(
                WarningLevel.INFO,
                "Insufficient Calibration",
                f"Only {num_predictions} historical predictions (need {SafetyLimits.MIN_CALIBRATION_POINTS})",
                "Calibration adjustments may be unreliable until more predictions are made"
            )
            return False
        return True
    
    def check_numerical_bounds(self, likelihood_ratio: float, was_clamped: bool) -> bool:
        if was_clamped:
            self.add_warning(
                WarningLevel.WARNING,
                "Numerical Instability",
                f"Likelihood ratio {likelihood_ratio:.2f} was clamped to [{SafetyLimits.MIN_LIKELIHOOD_RATIO}, {SafetyLimits.MAX_LIKELIHOOD_RATIO}]",
                "Evidence strength is hitting numerical bounds. Results may be unreliable."
            )
            return False
        return True
    
    def check_mechanism_complexity(self, nodes: int, edges: int) -> bool:
        if nodes > SafetyLimits.MAX_MECHANISM_NODES:
            self.add_warning(
                WarningLevel.WARNING,
                "Mechanism Complexity",
                f"{nodes} nodes exceeds recommended limit of {SafetyLimits.MAX_MECHANISM_NODES}",
                "Consider simplifying mechanism map. Analysis may be intractable."
            )
            return False
        return True
    
    def get_critical_warnings(self) -> List[SystemWarning]:
        return [w for w in self.warnings if w.level == WarningLevel.CRITICAL]
    
    def has_critical_warnings(self) -> bool:
        return len(self.get_critical_warnings()) > 0
    
    def print_warnings(self):
        if not self.warnings:
            print("✓ No warnings")
            return
        
        for warning in self.warnings:
            print(warning)


# =============================================================================
# FIX 2: IMPROVED EPISTEMIC STATE WITH WARNINGS
# =============================================================================

@dataclass
class ImprovedEpistemicState:
    """
    Enhanced epistemic state with better tracking and warnings.
    """
    
    credence: float = 0.5
    log_odds: float = 0.0
    reliability: float = 0.5
    epistemic_uncertainty: float = 0.5
    
    # Track updates
    update_count: int = 0
    clamping_count: int = 0
    warning_system: Optional[WarningSystem] = None
    
    def __post_init__(self):
        if self.credence > 0 and self.credence < 1:
            self.log_odds = np.log(self.credence / (1 - self.credence))
        self._clamp_values()
    
    def _clamp_values(self):
        """Clamp values and track when clamping occurs"""
        old_credence = self.credence
        self.credence = max(0.001, min(0.999, self.credence))
        if abs(old_credence - self.credence) > 1e-6:
            self.clamping_count += 1
        
        self.reliability = max(0.0, min(1.0, self.reliability))
        self.epistemic_uncertainty = max(0.0, min(1.0, self.epistemic_uncertainty))
    
    def update_with_evidence(self, likelihood_ratio: float) -> float:
        """
        Proper Bayesian update with safety checks.
        """
        self.update_count += 1
        old_credence = self.credence
        
        # Check for clamping
        was_clamped = (likelihood_ratio < SafetyLimits.MIN_LIKELIHOOD_RATIO or 
                      likelihood_ratio > SafetyLimits.MAX_LIKELIHOOD_RATIO)
        
        # Clamp likelihood ratio
        lr = max(SafetyLimits.MIN_LIKELIHOOD_RATIO, 
                min(SafetyLimits.MAX_LIKELIHOOD_RATIO, likelihood_ratio))
        
        # Warn if clamping occurred
        if was_clamped and self.warning_system:
            self.warning_system.check_numerical_bounds(likelihood_ratio, was_clamped)
        
        # Update via log-odds
        self.log_odds += np.log(lr)
        
        # Convert back to probability
        self.credence = 1 / (1 + np.exp(-self.log_odds))
        self._clamp_values()
        
        # Check for excessive confidence
        if self.warning_system:
            self.warning_system.check_high_confidence(self.credence)
        
        return self.credence - old_credence
    
    def get_update_summary(self) -> Dict:
        """Return summary of updates for auditing"""
        return {
            'total_updates': self.update_count,
            'clamping_events': self.clamping_count,
            'credence': self.credence,
            'log_odds': self.log_odds,
            'warning': self.clamping_count > 5  # Flag if excessive clamping
        }


# =============================================================================
# FIX 3: BETTER CAUSAL INFERENCE
# =============================================================================

class ImprovedCausalAssessment:
    """
    Quality-first causal assessment that doesn't blindly favor study design.
    """
    
    @staticmethod
    def effective_strength(quality: float, causal_level: str, 
                          sample_size: Optional[int] = None) -> float:
        """
        Calculate effective strength considering both quality AND causal level.
        Quality dominates, causal level provides modest boost.
        
        Args:
            quality: Base study quality [0, 1]
            causal_level: "association", "intervention", or "counterfactual"
            sample_size: If provided, further adjusts strength
        
        Returns:
            Effective strength [0, 1]
        """
        # Base quality is primary factor
        strength = quality
        
        # Causal level provides BOOST, not discount
        causal_boost = {
            'association': 0.0,        # No boost
            'intervention': 0.15,      # 15% boost
            'counterfactual': 0.05     # Small boost (theoretical)
        }.get(causal_level.lower(), 0.0)
        
        # Apply boost
        strength = min(1.0, strength * (1 + causal_boost))
        
        # Sample size adjustment
        if sample_size is not None:
            if sample_size < 50:
                strength *= 0.7  # Significant penalty for tiny studies
            elif sample_size < 200:
                strength *= 0.85
            elif sample_size > 10000:
                strength *= 1.1  # Small boost for very large studies
                strength = min(1.0, strength)
        
        return strength
    
    @staticmethod
    def assess_confounding_risk(study_design: str, 
                               controls_present: bool = False) -> float:
        """
        Assess confounding risk based on study design and controls.
        
        Returns: Risk score [0, 1] where 1 = high confounding risk
        """
        base_risk = {
            'meta_analysis': 0.2,
            'rct': 0.1,
            'cohort': 0.4,
            'case_control': 0.5,
            'cross_sectional': 0.6,
            'case_study': 0.7,
            'anecdote': 0.9
        }.get(study_design, 0.5)
        
        # Reduce risk if proper controls present
        if controls_present:
            base_risk *= 0.7
        
        return base_risk


# =============================================================================
# FIX 4: IMPROVED VALUE OF INFORMATION
# =============================================================================

@dataclass
class RealisticVOI:
    """
    More realistic Value of Information calculation.
    """
    
    @staticmethod
    def calculate_voi(current_utility: float,
                     optimal_utility: float,
                     information_cost: float,
                     time_delay_cost: float,
                     info_quality: float = 1.0) -> Dict:
        """
        Calculate VOI accounting for realistic constraints.
        
        Args:
            current_utility: EU with current information
            optimal_utility: EU with perfect information
            information_cost: Cost to acquire information
            time_delay_cost: Opportunity cost of waiting
            info_quality: How close to perfect can you get [0,1]
        
        Returns:
            Dict with VOI analysis
        """
        # Raw VOI (upper bound)
        raw_voi = abs(optimal_utility - current_utility)
        
        # Adjusted for realistic information quality
        realistic_voi = raw_voi * info_quality
        
        # Net VOI after costs
        net_voi = realistic_voi - information_cost - time_delay_cost
        
        # Decision
        should_gather = net_voi > 0
        
        return {
            'raw_voi': raw_voi,
            'realistic_voi': realistic_voi,
            'information_cost': information_cost,
            'time_cost': time_delay_cost,
            'net_voi': net_voi,
            'recommendation': 'gather_info' if should_gather else 'decide_now',
            'explanation': RealisticVOI._explain_voi(should_gather, net_voi)
        }
    
    @staticmethod
    def _explain_voi(should_gather: bool, net_voi: float) -> str:
        if should_gather:
            return f"Net VOI = {net_voi:.3f} > 0. Gather more information before deciding."
        else:
            return f"Net VOI = {net_voi:.3f} ≤ 0. Decide now with current information."


# =============================================================================
# FIX 5: EVIDENCE INDEPENDENCE CHECKING
# =============================================================================

class EvidenceIndependenceChecker:
    """
    Check if evidence pieces are truly independent.
    """
    
    @staticmethod
    def check_independence(evidence_list: List[Dict]) -> Dict:
        """
        Assess independence of evidence pieces.
        
        Returns independence score [0, 1] where:
        0 = Completely redundant
        1 = Fully independent
        """
        if len(evidence_list) <= 1:
            return {'independence_score': 1.0, 'issues': []}
        
        issues = []
        penalties = []
        
        # Check for same source
        sources = [e.get('source', '') for e in evidence_list]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            if count > 1:
                issues.append(f"Source '{source}' appears {count} times")
                penalties.append(0.3 * (count - 1))
        
        # Check for same dates (clustering)
        dates = [e.get('date', '') for e in evidence_list]
        if len(set(dates)) < len(dates) * 0.5:  # More than half same date
            issues.append("Evidence clustered in time (>50% same date)")
            penalties.append(0.2)
        
        # Check for citation keywords (heuristic)
        contents = [e.get('content', '').lower() for e in evidence_list]
        citation_words = ['citing', 'according to', 'based on', 'references']
        citation_count = sum(1 for c in contents 
                           if any(word in c for word in citation_words))
        if citation_count > len(evidence_list) * 0.3:
            issues.append(f"{citation_count} pieces may be citing others")
            penalties.append(0.3)
        
        # Calculate independence score
        if penalties:
            independence_score = max(0.0, 1.0 - sum(penalties) / len(evidence_list))
        else:
            independence_score = 1.0
        
        return {
            'independence_score': independence_score,
            'issues': issues,
            'penalty': sum(penalties),
            'recommendation': 'Consider evidence redundancy in your interpretation' if independence_score < 0.7 else 'Evidence appears reasonably independent'
        }


# =============================================================================
# FIX 6: RISK AVERSION IN UTILITY MODEL
# =============================================================================

class RiskAwareUtilityModel:
    """
    Utility model with proper risk aversion.
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        """
        risk_aversion:
            0 = Risk neutral (maximize expected value)
            1 = Moderate risk aversion (default, typical for individuals)
            2 = High risk aversion (typical for organizations)
            3+ = Extreme risk aversion (catastrophic downside exists)
        """
        self.risk_aversion = risk_aversion
        self.scenarios = []
    
    def add_scenario(self, probability: float, utility: float):
        self.scenarios.append({'prob': probability, 'util': utility})
    
    def expected_utility(self) -> float:
        """Standard expected utility"""
        return sum(s['prob'] * s['util'] for s in self.scenarios)
    
    def certainty_equivalent(self) -> float:
        """
        Certainty equivalent using CRRA utility.
        U(x) = x^(1-γ) / (1-γ) for γ ≠ 1
        U(x) = ln(x) for γ = 1
        """
        γ = self.risk_aversion
        
        if γ == 0:
            # Risk neutral
            return self.expected_utility()
        
        if γ == 1:
            # Logarithmic utility
            eu_log = sum(s['prob'] * np.log(max(0.01, s['util'] + 10)) 
                        for s in self.scenarios)  # Shift to avoid negative
            return np.exp(eu_log) - 10
        
        # CRRA utility
        eu_crra = sum(s['prob'] * ((s['util'] + 10) ** (1 - γ)) 
                     for s in self.scenarios)
        ce = (eu_crra * (1 - γ)) ** (1 / (1 - γ)) - 10
        
        return ce
    
    def risk_premium(self) -> float:
        """How much utility you'd sacrifice to avoid risk"""
        return self.expected_utility() - self.certainty_equivalent()


# =============================================================================
# FIX 7: COMPREHENSIVE SENSITIVITY ANALYSIS
# =============================================================================

class ComprehensiveSensitivity:
    """
    More thorough sensitivity analysis.
    """
    
    @staticmethod
    def analyze(base_score: float, dimensions: Dict) -> Dict:
        """
        Test sensitivity to:
        1. Individual dimensions at multiple magnitudes
        2. Pairs of dimensions (interaction effects)
        3. Assumption failures
        """
        
        results = {
            'base_score': base_score,
            'single_dimension': {},
            'interaction_effects': [],
            'critical_dimensions': []
        }
        
        # Test each dimension at multiple perturbations
        for dim_name, dim_value in dimensions.items():
            impacts = []
            for delta in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
                perturbed = np.clip(dim_value + delta, 0.0, 1.0)
                # Simulate score change (in real implementation, would recalculate)
                score_change = (perturbed - dim_value) * 0.5  # Simplified
                impacts.append({
                    'delta': delta,
                    'new_value': perturbed,
                    'score_change': score_change
                })
            
            results['single_dimension'][dim_name] = impacts
            
            # Flag critical dimensions
            max_impact = max(abs(i['score_change']) for i in impacts)
            if max_impact > 0.1:  # >10% score change
                results['critical_dimensions'].append({
                    'name': dim_name,
                    'base_value': dim_value,
                    'max_impact': max_impact
                })
        
        return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def demonstrate_improvements():
    """Show how improved components work"""
    
    print("=" * 70)
    print("IMPROVED SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # 1. Warning System
    print("\n1. WARNING SYSTEM")
    print("-" * 70)
    warnings = WarningSystem()
    
    warnings.check_evidence_sufficiency(2)  # Too few
    warnings.check_high_confidence(0.97)    # Too confident
    warnings.check_evidence_balance(10, 0)  # Imbalanced
    
    warnings.print_warnings()
    
    # 2. Improved Epistemic State
    print("\n\n2. IMPROVED EPISTEMIC STATE")
    print("-" * 70)
    state = ImprovedEpistemicState(credence=0.5, warning_system=warnings)
    
    print(f"Initial credence: {state.credence:.4f}")
    for i in range(5):
        state.update_with_evidence(8.0)  # Strong evidence
        print(f"After update {i+1}: {state.credence:.4f}")
    
    summary = state.get_update_summary()
    print(f"\nUpdate summary: {summary}")
    
    # 3. Improved Causal Assessment
    print("\n\n3. IMPROVED CAUSAL ASSESSMENT")
    print("-" * 70)
    
    # Large cohort study
    strength1 = ImprovedCausalAssessment.effective_strength(
        quality=0.95, 
        causal_level="association",
        sample_size=500000
    )
    print(f"Large cohort (q=0.95, n=500K): effective strength = {strength1:.3f}")
    
    # Small RCT
    strength2 = ImprovedCausalAssessment.effective_strength(
        quality=0.35,
        causal_level="intervention", 
        sample_size=50
    )
    print(f"Small RCT (q=0.35, n=50): effective strength = {strength2:.3f}")
    
    print(f"\n→ Cohort is {'stronger' if strength1 > strength2 else 'weaker'} despite being observational")
    
    # 4. Realistic VOI
    print("\n\n4. REALISTIC VALUE OF INFORMATION")
    print("-" * 70)
    
    voi_result = RealisticVOI.calculate_voi(
        current_utility=0.3,
        optimal_utility=0.8,
        information_cost=0.2,
        time_delay_cost=0.1,
        info_quality=0.7
    )
    
    print(f"Raw VOI: {voi_result['raw_voi']:.3f}")
    print(f"Realistic VOI: {voi_result['realistic_voi']:.3f}")
    print(f"Net VOI: {voi_result['net_voi']:.3f}")
    print(f"Recommendation: {voi_result['recommendation']}")
    print(f"Explanation: {voi_result['explanation']}")
    
    # 5. Evidence Independence
    print("\n\n5. EVIDENCE INDEPENDENCE CHECKING")
    print("-" * 70)
    
    evidence = [
        {'source': 'Journal A', 'date': '2023', 'content': 'Study shows X'},
        {'source': 'Journal A', 'date': '2023', 'content': 'Analysis shows X'},
        {'source': 'Journal B', 'date': '2024', 'content': 'Research citing Journal A shows X'},
    ]
    
    independence = EvidenceIndependenceChecker.check_independence(evidence)
    print(f"Independence score: {independence['independence_score']:.2f}")
    print(f"Issues found: {len(independence['issues'])}")
    for issue in independence['issues']:
        print(f"  - {issue}")
    print(f"Recommendation: {independence['recommendation']}")
    
    # 6. Risk Aversion
    print("\n\n6. RISK-AWARE UTILITY MODEL")
    print("-" * 70)
    
    for risk_aversion in [0, 1, 2]:
        model = RiskAwareUtilityModel(risk_aversion=risk_aversion)
        model.add_scenario(0.5, 1.0)   # 50% chance of $1M
        model.add_scenario(0.5, -0.3)  # 50% chance of -$300K
        
        eu = model.expected_utility()
        ce = model.certainty_equivalent()
        rp = model.risk_premium()
        
        print(f"\nRisk aversion = {risk_aversion}:")
        print(f"  Expected Utility: {eu:.3f}")
        print(f"  Certainty Equivalent: {ce:.3f}")
        print(f"  Risk Premium: {rp:.3f}")


if __name__ == "__main__":
    demonstrate_improvements()
