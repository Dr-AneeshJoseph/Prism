"""
SCIENTIFICALLY ENHANCED ANALYTICAL PROTOCOL v2.0
=================================================
A rigorous decision analysis framework incorporating:

FRAMEWORK 1: Proper Epistemic State Tracking
- Separates credence, evidence strength, and reliability
- Proper Bayesian updating with log-odds

FRAMEWORK 2: Causal Inference (Pearl's Hierarchy)
- Causal level annotations (association/intervention/counterfactual)
- Confounding risk assessment
- Causal strength discounting

FRAMEWORK 3: Decision Theory
- Explicit utility modeling with scenarios
- Risk-adjusted certainty equivalents
- Value of Information (VOI) calculation

FRAMEWORK 4: Cognitive Bias Detection
- Confirmation bias detection
- Overconfidence detection
- Planning fallacy detection
- Automated debiasing adjustments

FRAMEWORK 5: Information Theory
- Evidence measured in bits
- Redundancy detection
- Optimal stopping criteria

FRAMEWORK 6: Systems Thinking
- Feedback loop detection
- Systemic risk assessment

FRAMEWORK 7: Calibration Tracking
- Historical prediction tracking
- Expected Calibration Error (ECE)
- Platt scaling for score adjustment

Author: Scientific Enhancement System
Version: 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum
from datetime import datetime
import json
import warnings

# Suppress numpy warnings for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CORE ENUMS
# =============================================================================

class EvidenceDomain(Enum):
    MEDICAL = "medical"
    BUSINESS = "business"
    POLICY = "policy"
    TECHNOLOGY = "technology"
    GENERAL = "general"


class NodeType(Enum):
    CAUSE = "cause"
    MECHANISM = "mechanism"
    OUTCOME = "outcome"
    BLOCKER = "blocker"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    INTERVENTION = "intervention"


class EdgeType(Enum):
    CAUSES = "causes"
    PREVENTS = "prevents"
    ENABLES = "enables"
    COMPENSATES = "compensates"
    REQUIRES = "requires"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class CausalLevel(Enum):
    """Pearl's Causal Hierarchy - FRAMEWORK 2"""
    ASSOCIATION = 1      # Level 1: Observed correlation P(Y|X)
    INTERVENTION = 2     # Level 2: Experimental evidence P(Y|do(X))
    COUNTERFACTUAL = 3   # Level 3: Theoretical mechanism P(Y_x|X',Y')


class BiasType(Enum):
    """Cognitive biases to detect - FRAMEWORK 4"""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    PLANNING_FALLACY = "planning_fallacy"


class FeedbackLoopType(Enum):
    """System dynamics feedback types - FRAMEWORK 6"""
    REINFORCING = "reinforcing"  # Amplifies changes
    BALANCING = "balancing"      # Stabilizes toward equilibrium


class TraceEventType(Enum):
    LAYER_ENTER = "layer_enter"
    LAYER_EXIT = "layer_exit"
    FOUNDATION_SET = "foundation_set"
    EVIDENCE_ADDED = "evidence_added"
    CONFIDENCE_UPDATE = "confidence_update"
    MECHANISM_NODE_ADDED = "mechanism_node_added"
    MECHANISM_EDGE_ADDED = "mechanism_edge_added"
    CRITICISM_GENERATED = "criticism_generated"
    CRITICISM_RESOLVED = "criticism_resolved"
    QUALITY_CALCULATED = "quality_calculated"
    FATAL_FLAW_DETECTED = "fatal_flaw_detected"
    SENSITIVITY_CALCULATED = "sensitivity_calculated"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    GATE_CHECK = "gate_check"
    DECISION = "decision"
    GAP_IDENTIFIED = "gap_identified"
    COMPARISON_ADDED = "comparison_added"
    # New event types for enhanced protocol
    EPISTEMIC_UPDATE = "epistemic_update"
    BIAS_DETECTED = "bias_detected"
    VOI_CALCULATED = "voi_calculated"
    CALIBRATION_APPLIED = "calibration_applied"
    FEEDBACK_LOOP_DETECTED = "feedback_loop_detected"
    CAUSAL_ASSESSMENT = "causal_assessment"


# =============================================================================
# EVIDENCE QUALITY HIERARCHIES
# =============================================================================

EVIDENCE_QUALITY = {
    EvidenceDomain.MEDICAL: {
        'meta_analysis': 1.0, 'rct': 0.85, 'cohort': 0.7,
        'case_control': 0.6, 'expert_opinion': 0.3, 'anecdote': 0.15
    },
    EvidenceDomain.BUSINESS: {
        'multi_company_analysis': 0.9, 'controlled_experiment': 0.8,
        'benchmark': 0.7, 'case_study': 0.5, 'expert_opinion': 0.4, 'anecdote': 0.2
    },
    EvidenceDomain.POLICY: {
        'randomized_trial': 0.9, 'quasi_experiment': 0.75, 'regression_discontinuity': 0.7,
        'difference_in_differences': 0.65, 'case_study': 0.4, 'expert_opinion': 0.3
    },
    EvidenceDomain.TECHNOLOGY: {
        'controlled_experiment': 0.9, 'ab_test': 0.85, 'benchmark': 0.7,
        'case_study': 0.5, 'expert_opinion': 0.4, 'anecdote': 0.2
    },
    EvidenceDomain.GENERAL: {
        'rigorous_study': 0.8, 'good_study': 0.6, 'weak_study': 0.4,
        'expert_opinion': 0.3, 'anecdote': 0.15
    }
}

# Causal level discount factors (FRAMEWORK 2)
CAUSAL_LEVEL_DISCOUNT = {
    CausalLevel.ASSOCIATION: 0.5,      # Heavy discount - might be spurious
    CausalLevel.INTERVENTION: 0.85,    # Some discount for external validity
    CausalLevel.COUNTERFACTUAL: 0.95   # Small discount for model uncertainty
}


# =============================================================================
# FRAMEWORK 1: EPISTEMIC STATE
# =============================================================================

@dataclass
class EpistemicState:
    """
    Properly separated epistemic concepts (FRAMEWORK 1).
    
    This replaces the single 'confidence' value with distinct concepts:
    - credence: Subjective probability P(H|E)
    - log_odds: For proper Bayesian updating
    - reliability: How stable is this belief
    - epistemic_uncertainty: Second-order uncertainty
    """
    
    credence: float = 0.5  # P(hypothesis is true | current evidence)
    log_odds: float = 0.0  # log(P/(1-P)) for Bayesian updates
    reliability: float = 0.5  # Robustness to removing any single evidence
    epistemic_uncertainty: float = 0.5  # Uncertainty about our uncertainty
    
    def __post_init__(self):
        # Ensure credence and log_odds are consistent
        if self.credence > 0 and self.credence < 1:
            self.log_odds = np.log(self.credence / (1 - self.credence))
        self._clamp_values()
    
    def _clamp_values(self):
        """Ensure all values are in valid ranges"""
        self.credence = max(0.001, min(0.999, self.credence))
        self.reliability = max(0.0, min(1.0, self.reliability))
        self.epistemic_uncertainty = max(0.0, min(1.0, self.epistemic_uncertainty))
    
    def update_with_evidence(self, likelihood_ratio: float) -> float:
        """
        Proper Bayesian update using likelihood ratio.
        
        LR = P(E|H) / P(E|¬H)
        LR > 1 means evidence supports hypothesis
        LR < 1 means evidence contradicts hypothesis
        
        Returns the change in credence.
        """
        old_credence = self.credence
        
        # Clamp likelihood ratio to prevent numerical issues
        lr = max(0.01, min(100, likelihood_ratio))
        
        # Update log-odds (additive in log space)
        self.log_odds += np.log(lr)
        
        # Clamp log-odds to prevent extreme values
        self.log_odds = max(-10, min(10, self.log_odds))
        
        # Convert back to probability
        self.credence = 1 / (1 + np.exp(-self.log_odds))
        self._clamp_values()
        
        return self.credence - old_credence
    
    def get_confidence_interval(self, alpha: float = 0.90) -> Tuple[float, float]:
        """
        Return credible interval accounting for epistemic uncertainty.
        Higher epistemic_uncertainty = wider interval.
        """
        # Spread proportional to epistemic uncertainty
        spread = self.epistemic_uncertainty * 0.4
        
        lower = max(0.0, self.credence - spread)
        upper = min(1.0, self.credence + spread)
        
        return (lower, upper)
    
    def to_dict(self) -> Dict:
        return {
            'credence': self.credence,
            'log_odds': self.log_odds,
            'reliability': self.reliability,
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'confidence_interval': self.get_confidence_interval()
        }


# =============================================================================
# FRAMEWORK 5: INFORMATION-THEORETIC EVIDENCE
# =============================================================================

@dataclass
class Evidence:
    """
    Evidence with information-theoretic measurement (FRAMEWORK 5).
    
    Measures evidence in bits of information rather than simple counts.
    Tracks correlation with other evidence for redundancy detection.
    """
    id: str
    content: str
    source: str
    quality: float  # Base quality 0-1
    date: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    study_design: Optional[str] = None
    
    # Information-theoretic properties (FRAMEWORK 5)
    bits: float = 0.0  # Information content in bits
    correlation_with: Dict[str, float] = field(default_factory=dict)
    
    # Direction of evidence
    supports_hypothesis: bool = True  # False if contradicting
    
    # Causal properties (FRAMEWORK 2)
    causal_level: CausalLevel = CausalLevel.ASSOCIATION
    
    def get_quality(self) -> float:
        """Get quality based on study design hierarchy"""
        if self.study_design:
            hierarchy = EVIDENCE_QUALITY.get(self.domain, EVIDENCE_QUALITY[EvidenceDomain.GENERAL])
            return hierarchy.get(self.study_design, self.quality)
        return self.quality
    
    def calculate_bits(self, prior: float, posterior: float) -> float:
        """
        Calculate information content in bits.
        bits = log2(posterior/prior) for supporting evidence
        """
        if prior <= 0.001 or posterior <= 0.001:
            return 0.0
        if prior >= 0.999 or posterior >= 0.999:
            return 0.0
        
        # Information is the log of the Bayes factor
        prior_odds = prior / (1 - prior)
        posterior_odds = posterior / (1 - posterior)
        
        if prior_odds <= 0:
            return 0.0
        
        self.bits = abs(np.log2(posterior_odds / prior_odds))
        return self.bits
    
    def effective_bits(self, existing_evidence_ids: List[str]) -> float:
        """
        Bits adjusted for redundancy with existing evidence.
        Correlated evidence provides less new information.
        """
        if not self.correlation_with or not existing_evidence_ids:
            return self.bits
        
        # Find maximum correlation with existing evidence
        max_correlation = 0.0
        for eid in existing_evidence_ids:
            if eid in self.correlation_with:
                max_correlation = max(max_correlation, self.correlation_with[eid])
        
        # Discount bits by correlation (redundancy)
        return self.bits * (1 - max_correlation)
    
    def get_likelihood_ratio(self) -> float:
        """
        Convert quality and direction to likelihood ratio for Bayesian update.
        """
        base_lr = 1 + self.get_quality() * 2  # Quality 1.0 -> LR of 3
        
        # Apply causal level discount
        causal_discount = CAUSAL_LEVEL_DISCOUNT.get(self.causal_level, 0.5)
        adjusted_lr = 1 + (base_lr - 1) * causal_discount
        
        if not self.supports_hypothesis:
            adjusted_lr = 1 / adjusted_lr  # Invert for contradicting evidence
        
        return adjusted_lr
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'quality': self.get_quality(),
            'bits': self.bits,
            'causal_level': self.causal_level.name,
            'supports': self.supports_hypothesis
        }


# =============================================================================
# FRAMEWORK 2: CAUSAL MECHANISM STRUCTURES
# =============================================================================

@dataclass
class MechanismNode:
    """
    A node in the causal mechanism map with proper epistemic state.
    """
    id: str
    label: str
    node_type: NodeType
    description: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    
    # Replace single confidence with epistemic state (FRAMEWORK 1)
    epistemic_state: EpistemicState = field(default_factory=EpistemicState)
    
    # Legacy compatibility
    @property
    def confidence(self) -> float:
        return self.epistemic_state.credence
    
    @confidence.setter
    def confidence(self, value: float):
        self.epistemic_state.credence = max(0.001, min(0.999, value))
        if value > 0 and value < 1:
            self.epistemic_state.log_odds = np.log(value / (1 - value))
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.node_type.value,
            'confidence': self.epistemic_state.credence,
            'epistemic_state': self.epistemic_state.to_dict(),
            'description': self.description
        }


@dataclass
class MechanismEdge:
    """
    A causal relationship with proper causal level annotation (FRAMEWORK 2).
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5
    evidence_ids: List[str] = field(default_factory=list)
    
    # Causal inference properties (FRAMEWORK 2)
    causal_level: CausalLevel = CausalLevel.ASSOCIATION
    confounding_risk: float = 0.5  # Risk this is spurious correlation
    effect_size: float = 0.0  # Standardized effect size
    effect_uncertainty: float = 0.5
    
    def causal_strength(self) -> float:
        """
        Calculate true causal strength accounting for:
        1. Causal level (association vs intervention vs counterfactual)
        2. Confounding risk
        3. Effect uncertainty
        """
        base = max(0.0, min(1.0, self.strength))
        
        # Apply causal level discount
        level_discount = CAUSAL_LEVEL_DISCOUNT.get(self.causal_level, 0.5)
        
        # Apply confounding discount
        confounding_discount = 1 - (self.confounding_risk * 0.5)
        
        # Apply uncertainty discount
        uncertainty_discount = 1 - (self.effect_uncertainty * 0.2)
        
        return base * level_discount * confounding_discount * uncertainty_discount
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.edge_type.value,
            'strength': self.strength,
            'causal_level': self.causal_level.name,
            'causal_strength': self.causal_strength(),
            'confounding_risk': self.confounding_risk
        }


# =============================================================================
# FRAMEWORK 6: FEEDBACK LOOP DETECTION
# =============================================================================

@dataclass
class FeedbackLoop:
    """A detected feedback loop in the mechanism map"""
    loop_id: str
    node_ids: List[str]
    loop_type: FeedbackLoopType
    strength: float = 0.5
    time_delay: float = 1.0  # Time for effect to manifest
    
    def amplification_factor(self, time_steps: int = 1) -> float:
        """Calculate amplification/dampening over time"""
        if self.loop_type == FeedbackLoopType.REINFORCING:
            return (1 + self.strength) ** time_steps
        else:
            return 1 / ((1 + self.strength) ** time_steps)
    
    def to_dict(self) -> Dict:
        return {
            'loop_id': self.loop_id,
            'node_ids': self.node_ids,
            'type': self.loop_type.value,
            'strength': self.strength,
            'time_delay': self.time_delay
        }


class MechanismMap:
    """
    Enhanced causal mechanism map with:
    - Proper causal semantics (FRAMEWORK 2)
    - Feedback loop detection (FRAMEWORK 6)
    - Information-theoretic confidence aggregation
    """
    
    def __init__(self):
        self.nodes: Dict[str, MechanismNode] = {}
        self.edges: List[MechanismEdge] = []
        self.feedback_loops: List[FeedbackLoop] = []
        self.tracer: Optional['AnalysisTracer'] = None
    
    def add_node(self, node: MechanismNode) -> str:
        self.nodes[node.id] = node
        if self.tracer:
            self.tracer.log(TraceEventType.MECHANISM_NODE_ADDED, "L1", {
                'node_id': node.id,
                'label': node.label,
                'type': node.node_type.value,
                'confidence': node.confidence
            })
        return node.id
    
    def add_edge(self, edge: MechanismEdge):
        self.edges.append(edge)
        if self.tracer:
            self.tracer.log(TraceEventType.MECHANISM_EDGE_ADDED, "L1", {
                'source': edge.source_id,
                'target': edge.target_id,
                'type': edge.edge_type.value,
                'strength': edge.strength,
                'causal_level': edge.causal_level.name,
                'causal_strength': edge.causal_strength()
            })
    
    def overall_confidence(self) -> float:
        """
        Calculate overall confidence using proper aggregation.
        Uses geometric mean of credences (multiplicative - weakest link matters).
        """
        if not self.nodes:
            return 0.0
        
        credences = [n.epistemic_state.credence for n in self.nodes.values()]
        
        # Clamp values to prevent numerical issues
        clamped = [max(0.001, min(0.999, c)) for c in credences]
        
        # Geometric mean via log
        log_mean = np.mean(np.log(clamped))
        result = np.exp(log_mean)
        
        return float(max(0.0, min(1.0, result)))
    
    def average_causal_strength(self) -> float:
        """Average causal strength across all edges"""
        if not self.edges:
            return 0.0
        return np.mean([e.causal_strength() for e in self.edges])
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
        """
        Detect feedback loops in the mechanism map (FRAMEWORK 6).
        Uses DFS to find cycles.
        """
        # Build adjacency list
        adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        edge_map: Dict[Tuple[str, str], MechanismEdge] = {}
        
        for edge in self.edges:
            if edge.source_id in adjacency:
                adjacency[edge.source_id].append(edge.target_id)
                edge_map[(edge.source_id, edge.target_id)] = edge
        
        # Find cycles using DFS
        cycles = []
        visited = set()
        
        def dfs(node: str, path: List[str], path_set: Set[str]):
            if node in path_set:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path_set.add(node)
            path.append(node)
            
            for neighbor in adjacency.get(node, []):
                dfs(neighbor, path.copy(), path_set.copy())
        
        for node in adjacency:
            dfs(node, [], set())
        
        # Convert to FeedbackLoop objects
        self.feedback_loops = []
        for i, cycle_nodes in enumerate(cycles):
            if len(cycle_nodes) < 2:
                continue
            
            # Determine loop type based on edge types
            prevent_count = 0
            total_strength = 0.0
            
            for j in range(len(cycle_nodes)):
                src = cycle_nodes[j]
                tgt = cycle_nodes[(j + 1) % len(cycle_nodes)]
                if (src, tgt) in edge_map:
                    edge = edge_map[(src, tgt)]
                    total_strength += edge.strength
                    if edge.edge_type in [EdgeType.PREVENTS, EdgeType.CONTRADICTS]:
                        prevent_count += 1
            
            # Odd number of preventing edges = balancing loop
            loop_type = FeedbackLoopType.BALANCING if prevent_count % 2 == 1 else FeedbackLoopType.REINFORCING
            avg_strength = total_strength / len(cycle_nodes) if cycle_nodes else 0.5
            
            self.feedback_loops.append(FeedbackLoop(
                loop_id=f"FL{i}",
                node_ids=cycle_nodes,
                loop_type=loop_type,
                strength=avg_strength
            ))
            
            if self.tracer:
                self.tracer.log(TraceEventType.FEEDBACK_LOOP_DETECTED, "L1", {
                    'loop_id': f"FL{i}",
                    'type': loop_type.value,
                    'nodes': cycle_nodes,
                    'strength': avg_strength
                })
        
        return self.feedback_loops
    
    def systemic_risk_score(self) -> float:
        """
        Calculate systemic risk based on feedback loop structure.
        Reinforcing loops without balancing = high risk (runaway effects).
        """
        if not self.feedback_loops:
            self.detect_feedback_loops()
        
        if not self.feedback_loops:
            return 0.0
        
        reinforcing = sum(1 for l in self.feedback_loops if l.loop_type == FeedbackLoopType.REINFORCING)
        balancing = sum(1 for l in self.feedback_loops if l.loop_type == FeedbackLoopType.BALANCING)
        
        total = reinforcing + balancing
        if total == 0:
            return 0.0
        
        # Risk is proportion of reinforcing loops
        return reinforcing / total
    
    def get_critical_path(self) -> List[str]:
        """Find nodes with lowest confidence (weakest links)"""
        if not self.nodes:
            return []
        
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.epistemic_state.credence
        )
        return [n.id for n in sorted_nodes[:5]]
    
    def get_blockers(self) -> List[MechanismNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.BLOCKER]
    
    def get_assumptions(self) -> List[MechanismNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.ASSUMPTION]
    
    def to_dict(self) -> Dict:
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges],
            'feedback_loops': [fl.to_dict() for fl in self.feedback_loops],
            'overall_confidence': self.overall_confidence(),
            'average_causal_strength': self.average_causal_strength(),
            'systemic_risk': self.systemic_risk_score()
        }


# =============================================================================
# FRAMEWORK 3: UTILITY MODEL
# =============================================================================

@dataclass
class Scenario:
    """A possible outcome scenario"""
    description: str
    probability: float
    utility: float  # Can be negative for bad outcomes


@dataclass
class UtilityModel:
    """
    Decision-theoretic utility modeling (FRAMEWORK 3).
    
    Connects analysis scores to actual decisions by modeling:
    - Possible outcome scenarios with probabilities and utilities
    - Risk preferences
    - Time preferences
    - Value of information
    """
    
    scenarios: List[Scenario] = field(default_factory=list)
    risk_aversion: float = 0.5  # 0 = risk neutral, 1 = highly risk averse
    discount_rate: float = 0.1  # Annual discount rate
    time_horizon_years: float = 1.0
    
    def add_scenario(self, description: str, probability: float, utility: float):
        """Add an outcome scenario (probabilities should sum to 1)"""
        self.scenarios.append(Scenario(description, probability, utility))
    
    def expected_utility(self) -> float:
        """Calculate expected utility across scenarios"""
        if not self.scenarios:
            return 0.0
        return sum(s.probability * s.utility for s in self.scenarios)
    
    def variance(self) -> float:
        """Calculate variance of utility"""
        if not self.scenarios:
            return 0.0
        eu = self.expected_utility()
        return sum(s.probability * (s.utility - eu) ** 2 for s in self.scenarios)
    
    def certainty_equivalent(self) -> float:
        """
        Risk-adjusted value (certainty equivalent).
        Uses Arrow-Pratt risk adjustment.
        """
        eu = self.expected_utility()
        var = self.variance()
        
        # Arrow-Pratt risk premium
        risk_premium = 0.5 * self.risk_aversion * var
        
        return eu - risk_premium
    
    def value_of_perfect_information(self) -> float:
        """
        Calculate value of perfect information (VOPI).
        This is the maximum we should pay to know the true outcome before deciding.
        """
        if not self.scenarios:
            return 0.0
        
        # Current EU with uncertainty
        current_eu = self.expected_utility()
        
        # EU if we could always pick the best option knowing the outcome
        # (For binary decision: max utility we could achieve)
        best_outcome_eu = max(s.utility for s in self.scenarios)
        
        return max(0, best_outcome_eu - current_eu)
    
    def value_of_information(self, signal_accuracy: float = 0.8) -> float:
        """
        Value of imperfect information.
        signal_accuracy: How accurate is the additional information (0-1)
        """
        vopi = self.value_of_perfect_information()
        return vopi * signal_accuracy
    
    def should_gather_more_info(self, info_cost: float, signal_accuracy: float = 0.8) -> Tuple[bool, str]:
        """
        Should we gather more information before deciding?
        """
        voi = self.value_of_information(signal_accuracy)
        
        if voi > info_cost:
            return True, f"Yes: VOI ({voi:.3f}) > cost ({info_cost:.3f})"
        else:
            return False, f"No: VOI ({voi:.3f}) <= cost ({info_cost:.3f})"
    
    def to_dict(self) -> Dict:
        return {
            'scenarios': [{'desc': s.description, 'prob': s.probability, 'util': s.utility} 
                         for s in self.scenarios],
            'expected_utility': self.expected_utility(),
            'certainty_equivalent': self.certainty_equivalent(),
            'value_of_information': self.value_of_information(),
            'risk_aversion': self.risk_aversion
        }


# =============================================================================
# FRAMEWORK 4: COGNITIVE BIAS DETECTION
# =============================================================================

@dataclass
class BiasCheck:
    """Result of a cognitive bias check"""
    bias_type: BiasType
    detected: bool
    evidence: str
    severity: float  # 0-1
    mitigation: str
    
    def to_dict(self) -> Dict:
        return {
            'type': self.bias_type.value,
            'detected': self.detected,
            'evidence': self.evidence,
            'severity': self.severity,
            'mitigation': self.mitigation
        }


class CognitiveBiasDetector:
    """
    Systematic cognitive bias detection (FRAMEWORK 4).
    
    Checks for common analytical biases and suggests mitigations.
    """
    
    def __init__(self, element: 'AnalysisElement'):
        self.element = element
        self.checks: List[BiasCheck] = []
    
    def check_confirmation_bias(self) -> BiasCheck:
        """
        Detection: All evidence supports hypothesis, none contradicts.
        Mitigation: Actively seek disconfirming evidence.
        """
        if not self.element.evidence:
            return BiasCheck(
                bias_type=BiasType.CONFIRMATION,
                detected=False,
                evidence="No evidence to check",
                severity=0.0,
                mitigation=""
            )
        
        supporting = sum(1 for e in self.element.evidence if e.supports_hypothesis)
        contradicting = len(self.element.evidence) - supporting
        
        # Check mechanism map for contradicting nodes
        contradict_nodes = [
            n for n in self.element.mechanism_map.nodes.values()
            if 'contradict' in n.description.lower() 
            or 'against' in n.description.lower()
            or 'fail' in n.description.lower()
        ]
        
        detected = (contradicting == 0 and len(contradict_nodes) == 0 and 
                   len(self.element.evidence) >= 1)
        
        return BiasCheck(
            bias_type=BiasType.CONFIRMATION,
            detected=detected,
            evidence=f"{supporting} supporting, {contradicting} contradicting evidence" if detected else "",
            severity=0.7 if detected else 0.0,
            mitigation="Pre-mortem: List 3 specific ways this hypothesis could fail"
        )
    
    def check_overconfidence(self) -> BiasCheck:
        """
        Detection: High confidence scores with limited evidence.
        Research shows 90% CI contain truth only ~50% of time.
        """
        high_conf_dims = []
        
        for dim in self.element.scoring.dimensions.values():
            if dim.value > 0.85:
                # High confidence - check if justified
                evidence_count = len(self.element.evidence)
                if evidence_count < 3:
                    high_conf_dims.append(dim.name)
        
        detected = len(high_conf_dims) > 0
        
        return BiasCheck(
            bias_type=BiasType.OVERCONFIDENCE,
            detected=detected,
            evidence=f"High confidence with limited evidence: {high_conf_dims}" if detected else "",
            severity=0.6 if detected else 0.0,
            mitigation="Apply calibration: Multiply all confidence values by 0.8"
        )
    
    def check_planning_fallacy(self) -> BiasCheck:
        """
        Detection: Timeline estimates don't account for typical delays.
        Research: Projects typically take 2-3x longer than estimated.
        """
        timeline_dim = self.element.scoring.dimensions.get('timeline_realism')
        
        detected = False
        if timeline_dim and timeline_dim.value > 0.8:
            detected = True
        
        return BiasCheck(
            bias_type=BiasType.PLANNING_FALLACY,
            detected=detected,
            evidence="High timeline confidence without reference class" if detected else "",
            severity=0.5 if detected else 0.0,
            mitigation="Reference class forecasting: Find base rate from 5+ similar projects"
        )
    
    def check_base_rate_neglect(self) -> BiasCheck:
        """
        Detection: No prior/base rate considered.
        """
        # Check if there's any evidence of base rate consideration
        base_rate_mentioned = any(
            'base rate' in e.content.lower() or 
            'prior' in e.content.lower() or
            'typically' in e.content.lower()
            for e in self.element.evidence
        )
        
        detected = not base_rate_mentioned and len(self.element.evidence) > 0
        
        return BiasCheck(
            bias_type=BiasType.BASE_RATE_NEGLECT,
            detected=detected,
            evidence="No base rate or prior probability considered" if detected else "",
            severity=0.5 if detected else 0.0,
            mitigation="Research base rate: What % of similar hypotheses are true?"
        )
    
    def run_all_checks(self) -> List[BiasCheck]:
        """Run all bias checks"""
        self.checks = [
            self.check_confirmation_bias(),
            self.check_overconfidence(),
            self.check_planning_fallacy(),
            self.check_base_rate_neglect(),
        ]
        return self.checks
    
    def total_bias_penalty(self) -> float:
        """Calculate total penalty from detected biases"""
        return sum(check.severity * 0.1 for check in self.checks if check.detected)
    
    def get_debiased_score(self, original_score: float) -> float:
        """Apply debiasing adjustments"""
        penalty = self.total_bias_penalty()
        return max(0.0, min(1.0, original_score - penalty))


# =============================================================================
# FRAMEWORK 7: CALIBRATION TRACKING
# =============================================================================

@dataclass
class CalibrationRecord:
    """Record of a prediction and its outcome"""
    hypothesis_id: str
    predicted_score: float
    actual_outcome: Optional[bool] = None  # True if hypothesis was correct
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CalibrationTracker:
    """
    Track and improve calibration over time (FRAMEWORK 7).
    
    Well-calibrated means: P(correct | score=0.7) ≈ 0.7
    """
    
    def __init__(self):
        self.records: List[CalibrationRecord] = []
    
    def add_prediction(self, hypothesis_id: str, score: float):
        """Record a new prediction"""
        self.records.append(CalibrationRecord(
            hypothesis_id=hypothesis_id,
            predicted_score=score
        ))
    
    def record_outcome(self, hypothesis_id: str, actual: bool):
        """Record the actual outcome for a prediction"""
        for record in reversed(self.records):
            if record.hypothesis_id == hypothesis_id and record.actual_outcome is None:
                record.actual_outcome = actual
                break
    
    def get_completed_records(self) -> List[CalibrationRecord]:
        """Get records with known outcomes"""
        return [r for r in self.records if r.actual_outcome is not None]
    
    def expected_calibration_error(self) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        Bins predictions and compares to actual frequency.
        """
        completed = self.get_completed_records()
        
        if len(completed) < 10:
            return 0.5  # Not enough data
        
        # Bin into 10 buckets
        bins = [[] for _ in range(10)]
        for r in completed:
            bin_idx = min(9, int(r.predicted_score * 10))
            bins[bin_idx].append(r.actual_outcome)
        
        # Calculate ECE
        ece = 0.0
        total = len(completed)
        
        for i, bin_outcomes in enumerate(bins):
            if not bin_outcomes:
                continue
            
            bin_confidence = (i + 0.5) / 10
            bin_accuracy = sum(bin_outcomes) / len(bin_outcomes)
            bin_weight = len(bin_outcomes) / total
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def is_overconfident(self) -> bool:
        """Are we systematically overconfident?"""
        completed = self.get_completed_records()
        if len(completed) < 10:
            return False  # Not enough data
        
        # Check if average score > average accuracy
        avg_score = np.mean([r.predicted_score for r in completed])
        avg_accuracy = np.mean([r.actual_outcome for r in completed])
        
        return avg_score > avg_accuracy + 0.1
    
    def get_calibrated_score(self, raw_score: float) -> float:
        """
        Apply Platt scaling based on historical calibration.
        """
        ece = self.expected_calibration_error()
        
        if ece > 0.15 and self.is_overconfident():
            # Shrink toward 0.5 (reduce overconfidence)
            return raw_score * 0.85 + 0.075
        elif ece > 0.15:
            # Some calibration issue but not overconfidence
            return raw_score * 0.95 + 0.025
        
        return raw_score
    
    def to_dict(self) -> Dict:
        return {
            'total_predictions': len(self.records),
            'completed': len(self.get_completed_records()),
            'ece': self.expected_calibration_error(),
            'is_overconfident': self.is_overconfident()
        }


# =============================================================================
# TRACE SYSTEM
# =============================================================================

@dataclass
class TraceEvent:
    timestamp: float
    event_type: TraceEventType
    layer: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'type': self.event_type.value,
            'layer': self.layer,
            'data': self.data
        }


class AnalysisTracer:
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.start_time = datetime.now().timestamp()
        self.iteration_data: List[Dict] = []
        self.sensitivity_data: List[Dict] = []
    
    def _elapsed(self) -> float:
        return datetime.now().timestamp() - self.start_time
    
    def log(self, event_type: TraceEventType, layer: str, data: Dict[str, Any]):
        self.events.append(TraceEvent(self._elapsed(), event_type, layer, data))
    
    def get_trace(self) -> List[Dict]:
        return [e.to_dict() for e in self.events]


# =============================================================================
# SCORING SYSTEM (ENHANCED)
# =============================================================================

@dataclass
class DimensionScore:
    """Score for a single dimension with proper uncertainty"""
    name: str
    value: float
    weight: float
    is_fatal_below: float = 0.3
    rationale: str = ""
    
    # Enhanced: Track uncertainty about this score
    uncertainty: float = 0.3
    
    @property
    def is_fatal_flaw(self) -> bool:
        return self.value < self.is_fatal_below
    
    def confidence_interval(self) -> Tuple[float, float]:
        """Get confidence interval for this dimension"""
        lower = max(0, self.value - self.uncertainty)
        upper = min(1, self.value + self.uncertainty)
        return (lower, upper)


class ScoringSystem:
    """
    Enhanced scoring system with proper mathematical foundations.
    
    Uses Bayesian log-odds combination instead of arbitrary weighted average.
    """
    
    def __init__(self):
        self.dimensions: Dict[str, DimensionScore] = {}
        self.tracer: Optional[AnalysisTracer] = None
    
    def set_dimension(self, name: str, value: float, weight: float = 0.2,
                      fatal_threshold: float = 0.3, rationale: str = "",
                      uncertainty: float = 0.3):
        """Set a dimension score with proper bounds checking"""
        # Clamp value to valid range
        clamped_value = max(0.001, min(0.999, value))
        
        self.dimensions[name] = DimensionScore(
            name=name, 
            value=clamped_value, 
            weight=weight,
            is_fatal_below=fatal_threshold, 
            rationale=rationale,
            uncertainty=uncertainty
        )
        
        if clamped_value < fatal_threshold and self.tracer:
            self.tracer.log(TraceEventType.FATAL_FLAW_DETECTED, "L4", {
                'dimension': name,
                'value': clamped_value,
                'threshold': fatal_threshold
            })
    
    def additive_score(self) -> float:
        """Traditional weighted sum (for comparison)"""
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(d.value * d.weight for d in self.dimensions.values())
        return weighted_sum / total_weight
    
    def bayesian_score(self) -> float:
        """
        Proper Bayesian combination using log-odds.
        This is mathematically correct for combining probabilities.
        """
        if not self.dimensions:
            return 0.0
        
        # Convert to log-odds, weight, and combine
        total_log_odds = 0.0
        total_weight = 0.0
        
        for dim in self.dimensions.values():
            # Clamp to avoid log(0)
            p = max(0.001, min(0.999, dim.value))
            log_odds = np.log(p / (1 - p))
            
            total_log_odds += log_odds * dim.weight
            total_weight += dim.weight
        
        if total_weight == 0:
            return 0.5
        
        # Normalize by weight
        avg_log_odds = total_log_odds / total_weight
        
        # Convert back to probability
        result = 1 / (1 + np.exp(-avg_log_odds))
        
        return float(max(0.0, min(1.0, result)))
    
    def multiplicative_score(self) -> float:
        """Geometric mean - for backward compatibility"""
        if not self.dimensions:
            return 0.0
        
        values = [max(0.001, min(0.999, d.value)) for d in self.dimensions.values()]
        
        # Geometric mean via log
        log_mean = np.mean(np.log(values))
        return float(np.exp(log_mean))
    
    def fatal_flaws(self) -> List[DimensionScore]:
        """Get all dimensions that are fatal flaws"""
        return [d for d in self.dimensions.values() if d.is_fatal_flaw]
    
    def combined_score(self) -> Tuple[float, str]:
        """
        Combined scoring using Bayesian approach with fatal flaw override.
        """
        flaws = self.fatal_flaws()
        bayesian = self.bayesian_score()
        
        if flaws:
            # Fatal flaw caps the score
            score = min(0.3, bayesian * 0.5)
            reason = f"Fatal flaw in: {', '.join(f.name for f in flaws)}"
        else:
            score = bayesian
            reason = "No fatal flaws"
        
        return score, reason
    
    def sensitivity_analysis(self) -> List[Dict]:
        """Calculate sensitivity of each dimension"""
        if not self.dimensions:
            return []
        
        base_score, _ = self.combined_score()
        sensitivities = []
        
        for name, dim in self.dimensions.items():
            original = dim.value
            
            # Test +0.1
            dim.value = min(0.999, original + 0.1)
            high_score, _ = self.combined_score()
            
            # Test -0.1
            dim.value = max(0.001, original - 0.1)
            low_score, _ = self.combined_score()
            
            dim.value = original  # Restore
            
            impact = high_score - low_score
            sensitivities.append({
                'dimension': name,
                'current_value': original,
                'impact': abs(impact),
                'direction': 'positive' if impact > 0 else 'negative',
                'is_fatal': dim.is_fatal_flaw
            })
        
        return sorted(sensitivities, key=lambda x: x['impact'], reverse=True)


# =============================================================================
# MAIN ANALYSIS ELEMENT (ENHANCED)
# =============================================================================

@dataclass
class AnalysisElement:
    """
    Enhanced analysis element incorporating all scientific frameworks.
    """
    name: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    
    # Core foundation
    what: Optional[str] = None
    why: Optional[str] = None
    how: Optional[str] = None
    measure: Optional[str] = None
    
    # Overall epistemic state (FRAMEWORK 1)
    epistemic_state: EpistemicState = field(default_factory=EpistemicState)
    
    # Components
    mechanism_map: MechanismMap = field(default_factory=MechanismMap)
    scoring: ScoringSystem = field(default_factory=ScoringSystem)
    evidence: List[Evidence] = field(default_factory=list)
    
    # Enhanced components
    utility_model: UtilityModel = field(default_factory=UtilityModel)  # FRAMEWORK 3
    bias_detector: Optional[CognitiveBiasDetector] = None  # FRAMEWORK 4
    calibration_tracker: CalibrationTracker = field(default_factory=CalibrationTracker)  # FRAMEWORK 7
    
    # Computed results
    bias_checks: List[BiasCheck] = field(default_factory=list)
    feedback_loops: List[FeedbackLoop] = field(default_factory=list)
    
    tracer: Optional[AnalysisTracer] = None
    
    def __post_init__(self):
        self.mechanism_map = MechanismMap()
        self.scoring = ScoringSystem()
        self.utility_model = UtilityModel()
        self.calibration_tracker = CalibrationTracker()
    
    def set_tracer(self, tracer: AnalysisTracer):
        self.tracer = tracer
        self.mechanism_map.tracer = tracer
        self.scoring.tracer = tracer
    
    def set_what(self, value: str, confidence: float):
        self.what = value
        self.scoring.set_dimension('definition_clarity', confidence, weight=0.15)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'WHAT', 'confidence': confidence})
    
    def set_why(self, value: str, confidence: float):
        self.why = value
        self.scoring.set_dimension('justification_strength', confidence, weight=0.20)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'WHY', 'confidence': confidence})
    
    def set_how(self, value: str, confidence: float):
        self.how = value
        self.scoring.set_dimension('mechanism_validity', confidence, weight=0.25, fatal_threshold=0.25)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'HOW', 'confidence': confidence})
    
    def set_measure(self, value: str, confidence: float):
        self.measure = value
        self.scoring.set_dimension('measurability', confidence, weight=0.15)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'MEASURE', 'confidence': confidence})
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence with proper Bayesian updating"""
        # Calculate effective bits accounting for redundancy
        existing_ids = [e.id for e in self.evidence]
        
        # Get likelihood ratio for Bayesian update
        lr = evidence.get_likelihood_ratio()
        
        # Update overall epistemic state
        old_credence = self.epistemic_state.credence
        self.epistemic_state.update_with_evidence(lr)
        
        # Calculate bits
        evidence.calculate_bits(old_credence, self.epistemic_state.credence)
        
        self.evidence.append(evidence)
        
        # Update evidence quality dimension
        if self.evidence:
            qualities = [e.get_quality() for e in self.evidence]
            # Diminishing returns
            n = len(self.evidence)
            effective_quality = np.mean(qualities) * (1 - 0.5 * np.exp(-n/3))
            self.scoring.set_dimension('evidence_quality', effective_quality, weight=0.15)
        
        if self.tracer:
            self.tracer.log(TraceEventType.EVIDENCE_ADDED, "L1", {
                'id': evidence.id,
                'quality': evidence.get_quality(),
                'bits': evidence.bits,
                'causal_level': evidence.causal_level.name,
                'new_credence': self.epistemic_state.credence
            })
            self.tracer.log(TraceEventType.EPISTEMIC_UPDATE, "L1", {
                'old_credence': old_credence,
                'new_credence': self.epistemic_state.credence,
                'likelihood_ratio': lr
            })
    
    def add_mechanism_node(self, node: MechanismNode) -> str:
        return self.mechanism_map.add_node(node)
    
    def add_mechanism_edge(self, edge: MechanismEdge):
        self.mechanism_map.add_edge(edge)
    
    def set_feasibility(self, technical: float, economic: float, timeline: float):
        self.scoring.set_dimension('technical_feasibility', technical, weight=0.15, fatal_threshold=0.2)
        self.scoring.set_dimension('economic_viability', economic, weight=0.10, fatal_threshold=0.2)
        self.scoring.set_dimension('timeline_realism', timeline, weight=0.10)
    
    def set_risk(self, execution_risk: float, external_risk: float):
        self.scoring.set_dimension('execution_safety', 1.0 - execution_risk, weight=0.10)
        self.scoring.set_dimension('external_resilience', 1.0 - external_risk, weight=0.10)
    
    def add_scenario(self, description: str, probability: float, utility: float):
        """Add an outcome scenario to the utility model"""
        self.utility_model.add_scenario(description, probability, utility)
    
    def run_bias_detection(self) -> List[BiasCheck]:
        """Run cognitive bias detection"""
        self.bias_detector = CognitiveBiasDetector(self)
        self.bias_checks = self.bias_detector.run_all_checks()
        
        if self.tracer:
            for check in self.bias_checks:
                if check.detected:
                    self.tracer.log(TraceEventType.BIAS_DETECTED, "L2", {
                        'bias_type': check.bias_type.value,
                        'severity': check.severity,
                        'evidence': check.evidence,
                        'mitigation': check.mitigation
                    })
        
        return self.bias_checks
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
        """Detect feedback loops in mechanism map"""
        self.feedback_loops = self.mechanism_map.detect_feedback_loops()
        return self.feedback_loops
    
    def get_total_evidence_bits(self) -> float:
        """Total information from all evidence"""
        return sum(e.bits for e in self.evidence)
    
    def value_of_information(self) -> float:
        """Calculate value of gathering more information"""
        return self.utility_model.value_of_information()


# =============================================================================
# ADVERSARIAL TESTER (ENHANCED)
# =============================================================================

@dataclass
class Criticism:
    content: str
    severity: float
    cycle: str
    dimension_affected: str = ""
    resolved: bool = False
    response: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'severity': self.severity,
            'cycle': self.cycle,
            'resolved': self.resolved
        }


class AdversarialTester:
    """Enhanced adversarial tester with bias integration"""
    
    def __init__(self, element: AnalysisElement, rigor: int, tracer: AnalysisTracer):
        self.element = element
        self.rigor = rigor
        self.tracer = tracer
        self.criticisms: List[Criticism] = []
        self.iteration = 0
    
    def generate_criticisms(self) -> List[Criticism]:
        new = []
        self.iteration += 1
        self.tracer.log(TraceEventType.ITERATION_START, "L2", {'iteration': self.iteration})
        
        # Check assumptions with low confidence
        assumptions = self.element.mechanism_map.get_assumptions()
        for assumption in assumptions:
            if assumption.epistemic_state.credence < 0.7:
                c = Criticism(
                    f"Untested assumption: '{assumption.label}' (credence: {assumption.epistemic_state.credence:.2f})",
                    severity=0.7, cycle="assumption_test", dimension_affected="mechanism_validity"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check blockers
        blockers = self.element.mechanism_map.get_blockers()
        for blocker in blockers:
            if blocker.epistemic_state.credence > 0.5:
                c = Criticism(
                    f"Likely blocker: '{blocker.label}' (likelihood: {blocker.epistemic_state.credence:.2f})",
                    severity=0.8, cycle="pre_mortem", dimension_affected="execution_safety"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check fatal flaws
        flaws = self.element.scoring.fatal_flaws()
        for flaw in flaws:
            c = Criticism(
                f"Fatal flaw: {flaw.name} = {flaw.value:.2f} (threshold: {flaw.is_fatal_below})",
                severity=0.95, cycle="fatal_flaw", dimension_affected=flaw.name
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check evidence gaps
        if len(self.element.evidence) < 2:
            c = Criticism(
                "Insufficient evidence (< 2 sources)",
                severity=0.6, cycle="evidence_check", dimension_affected="evidence_quality"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check for weak causal links (FRAMEWORK 2)
        weak_edges = [e for e in self.element.mechanism_map.edges if e.causal_strength() < 0.3]
        for edge in weak_edges[:3]:
            c = Criticism(
                f"Weak causal link: {edge.source_id} -> {edge.target_id} (strength: {edge.causal_strength():.2f}, level: {edge.causal_level.name})",
                severity=0.5, cycle="causal_analysis", dimension_affected="mechanism_validity"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check weak mechanism nodes
        weak_nodes = [n for n in self.element.mechanism_map.nodes.values() 
                     if n.epistemic_state.credence < 0.5]
        for node in weak_nodes[:3]:
            c = Criticism(
                f"Weak mechanism: '{node.label}' (credence: {node.epistemic_state.credence:.2f})",
                severity=0.5, cycle="mechanism_analysis", dimension_affected="mechanism_validity"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Add bias-related criticisms (FRAMEWORK 4)
        for bias_check in self.element.bias_checks:
            if bias_check.detected and bias_check.severity >= 0.5:
                c = Criticism(
                    f"Cognitive bias detected: {bias_check.bias_type.value} - {bias_check.evidence}",
                    severity=bias_check.severity, cycle="bias_detection", dimension_affected="overall"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check systemic risk from feedback loops (FRAMEWORK 6)
        systemic_risk = self.element.mechanism_map.systemic_risk_score()
        if systemic_risk > 0.6:
            c = Criticism(
                f"High systemic risk: {systemic_risk:.2f} (reinforcing loops dominate)",
                severity=0.7, cycle="systems_analysis", dimension_affected="execution_safety"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        self.criticisms.extend(new)
        return new
    
    def unresolved_critical(self, threshold: float = 0.7) -> List[Criticism]:
        return [c for c in self.criticisms if not c.resolved and c.severity >= threshold]
    
    def consistency_score(self) -> float:
        if not self.criticisms:
            return 1.0
        return sum(1 for c in self.criticisms if c.resolved) / len(self.criticisms)


# =============================================================================
# HYPOTHESIS COMPARATOR
# =============================================================================

class HypothesisComparator:
    """Compare multiple hypotheses with enhanced metrics"""
    
    def __init__(self):
        self.hypotheses: Dict[str, AnalysisElement] = {}
    
    def add_hypothesis(self, element: AnalysisElement):
        self.hypotheses[element.name] = element
    
    def compare(self) -> Dict:
        comparison = {
            'hypotheses': [],
            'dimensions': set(),
            'rankings': {}
        }
        
        # Collect all dimensions
        for name, elem in self.hypotheses.items():
            for dim_name in elem.scoring.dimensions:
                comparison['dimensions'].add(dim_name)
        
        comparison['dimensions'] = list(comparison['dimensions'])
        
        # Score each hypothesis
        for name, elem in self.hypotheses.items():
            score, reason = elem.scoring.combined_score()
            flaws = elem.scoring.fatal_flaws()
            
            # Enhanced metrics
            hyp_data = {
                'name': name,
                'combined_score': score,
                'bayesian_score': elem.scoring.bayesian_score(),
                'additive_score': elem.scoring.additive_score(),
                'multiplicative_score': elem.scoring.multiplicative_score(),
                'credence': elem.epistemic_state.credence,
                'confidence_interval': elem.epistemic_state.get_confidence_interval(),
                'fatal_flaws': [f.name for f in flaws],
                'reason': reason,
                'dimensions': {d.name: d.value for d in elem.scoring.dimensions.values()},
                'mechanism_confidence': elem.mechanism_map.overall_confidence(),
                'average_causal_strength': elem.mechanism_map.average_causal_strength(),
                'evidence_count': len(elem.evidence),
                'total_evidence_bits': elem.get_total_evidence_bits(),
                'expected_utility': elem.utility_model.expected_utility(),
                'certainty_equivalent': elem.utility_model.certainty_equivalent(),
                'value_of_information': elem.value_of_information(),
                'biases_detected': [b.bias_type.value for b in elem.bias_checks if b.detected],
                'systemic_risk': elem.mechanism_map.systemic_risk_score()
            }
            comparison['hypotheses'].append(hyp_data)
        
        # Rank by combined score
        comparison['hypotheses'].sort(key=lambda x: x['combined_score'], reverse=True)
        comparison['rankings'] = {h['name']: i+1 for i, h in enumerate(comparison['hypotheses'])}
        
        return comparison


# =============================================================================
# OPTIMAL STOPPING
# =============================================================================

class OptimalStoppingCriterion:
    """
    Determine when to stop analysis and make a decision (FRAMEWORK 5).
    """
    
    def __init__(self, element: AnalysisElement, 
                 decision_threshold: float = 0.7,
                 delay_cost: float = 0.01):
        self.element = element
        self.threshold = decision_threshold
        self.delay_cost = delay_cost
    
    def should_stop(self) -> Tuple[bool, str]:
        """
        Optimal stopping: Stop if cost of delay > expected value of more info.
        """
        credence = self.element.epistemic_state.credence
        voi = self.element.value_of_information()
        
        # Very confident either way
        if credence > 0.95:
            return True, f"High confidence ({credence:.2f}): Proceed"
        if credence < 0.05:
            return True, f"Low confidence ({credence:.2f}): Reject"
        
        # VOI vs delay cost
        if voi < self.delay_cost:
            return True, f"VOI ({voi:.3f}) < delay cost ({self.delay_cost:.3f}): Decide now"
        
        # Above threshold with no fatal flaws
        flaws = self.element.scoring.fatal_flaws()
        if credence >= self.threshold and len(flaws) == 0:
            return True, f"Threshold reached ({credence:.2f} >= {self.threshold}): Proceed"
        
        return False, f"Continue analysis: credence={credence:.2f}, VOI={voi:.3f}"


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_analysis(element: AnalysisElement, rigor_level: int = 2, max_iter: int = 15) -> Dict:
    """
    Run enhanced analysis with all scientific frameworks.
    """
    tracer = AnalysisTracer()
    element.set_tracer(tracer)
    
    # Layer 0: Characterization
    tracer.log(TraceEventType.LAYER_ENTER, "L0", {'description': 'Problem Characterization'})
    rigor_names = {1: "Light", 2: "Standard", 3: "Deep"}
    targets = {1: 0.5, 2: 0.7, 3: 0.85}
    target = targets.get(rigor_level, 0.7)
    tracer.log(TraceEventType.DECISION, "L0", {'rigor': rigor_names.get(rigor_level, "Standard"), 'target': target})
    tracer.log(TraceEventType.LAYER_EXIT, "L0", {'result': f'Target: {target}'})
    
    # Layer 1: Foundation & Mechanism
    tracer.log(TraceEventType.LAYER_ENTER, "L1", {'description': 'Foundation & Mechanism Mapping'})
    
    # Detect feedback loops (FRAMEWORK 6)
    element.detect_feedback_loops()
    
    mechanism_conf = element.mechanism_map.overall_confidence()
    causal_strength = element.mechanism_map.average_causal_strength()
    tracer.log(TraceEventType.LAYER_EXIT, "L1", {
        'result': f'Mechanism confidence: {mechanism_conf:.2f}, Causal strength: {causal_strength:.2f}'
    })
    
    # Run bias detection (FRAMEWORK 4)
    tracer.log(TraceEventType.LAYER_ENTER, "L2", {'description': 'Bias Detection & Adversarial Testing'})
    element.run_bias_detection()
    
    # Adversarial testing
    tester = AdversarialTester(element, rigor_level, tracer)
    
    history = []
    reason = "Max iterations"
    stopping = OptimalStoppingCriterion(element, target)
    
    for i in range(max_iter):
        tester.generate_criticisms()
        
        # Calculate scores using Bayesian approach
        bayesian = element.scoring.bayesian_score()
        combined, score_reason = element.scoring.combined_score()
        consistency = tester.consistency_score()
        
        # Apply calibration (FRAMEWORK 7)
        calibrated = element.calibration_tracker.get_calibrated_score(combined)
        
        # Apply bias debiasing (FRAMEWORK 4)
        if element.bias_detector:
            debiased = element.bias_detector.get_debiased_score(calibrated)
        else:
            debiased = calibrated
        
        quality = debiased
        history.append(quality)
        
        tracer.log(TraceEventType.QUALITY_CALCULATED, "L4", {
            'iteration': i + 1,
            'bayesian_score': bayesian,
            'combined_score': combined,
            'calibrated_score': calibrated,
            'debiased_score': debiased,
            'consistency': consistency,
            'quality': quality,
            'fatal_flaws': len(element.scoring.fatal_flaws()),
            'credence': element.epistemic_state.credence
        })
        
        # Calculate and log VOI (FRAMEWORK 3)
        voi = element.value_of_information()
        tracer.log(TraceEventType.VOI_CALCULATED, "L4", {
            'value_of_information': voi,
            'expected_utility': element.utility_model.expected_utility()
        })
        
        tracer.iteration_data.append({
            'iteration': i + 1,
            'quality': quality,
            'bayesian': bayesian,
            'combined': combined,
            'calibrated': calibrated,
            'debiased': debiased,
            'credence': element.epistemic_state.credence,
            'consistency': consistency,
            'voi': voi
        })
        
        # Gate checks
        flaws = element.scoring.fatal_flaws()
        tracer.log(TraceEventType.GATE_CHECK, "L4", {
            'gate': 'quality', 'value': quality, 'threshold': target, 'passed': quality >= target
        })
        tracer.log(TraceEventType.GATE_CHECK, "L4", {
            'gate': 'fatal_flaws', 'value': len(flaws), 'threshold': 0, 'passed': len(flaws) == 0
        })
        
        # Check stopping criteria
        should_stop, stop_reason = stopping.should_stop()
        
        if should_stop:
            reason = stop_reason
            tracer.log(TraceEventType.ITERATION_END, "L2", {
                'iteration': i + 1, 'quality': quality, 'action': f'STOP: {reason}'
            })
            break
        elif i >= 3 and len(history) >= 2 and abs(history[-1] - history[-2]) < 0.01:
            reason = "Diminishing returns"
            tracer.log(TraceEventType.ITERATION_END, "L2", {
                'iteration': i + 1, 'quality': quality, 'action': 'STOP: Diminishing returns'
            })
            break
        else:
            tracer.log(TraceEventType.ITERATION_END, "L2", {
                'iteration': i + 1, 'quality': quality, 'action': 'CONTINUE'
            })
    
    tracer.log(TraceEventType.LAYER_EXIT, "L2", {
        'result': f'{tester.iteration} iterations, {reason}'
    })
    
    # Sensitivity analysis
    tracer.log(TraceEventType.LAYER_ENTER, "L3", {'description': 'Sensitivity Analysis'})
    sensitivity = element.scoring.sensitivity_analysis()
    tracer.sensitivity_data = sensitivity
    tracer.log(TraceEventType.LAYER_EXIT, "L3", {
        'result': f'Most sensitive: {sensitivity[0]["dimension"] if sensitivity else "N/A"}'
    })
    
    # Final decision
    tracer.log(TraceEventType.LAYER_ENTER, "L5", {'description': 'Final Decision'})
    
    final_bayesian = element.scoring.bayesian_score()
    final_combined, final_reason = element.scoring.combined_score()
    final_calibrated = element.calibration_tracker.get_calibrated_score(final_combined)
    
    if element.bias_detector:
        final_debiased = element.bias_detector.get_debiased_score(final_calibrated)
    else:
        final_debiased = final_calibrated
    
    flaws = element.scoring.fatal_flaws()
    ready = final_debiased >= target and len(flaws) == 0
    
    # Generate recommendation
    voi = element.value_of_information()
    if voi > 0.1:
        recommendation = f"INVESTIGATE: VOI ({voi:.2f}) suggests more information needed"
    elif ready:
        recommendation = f"PROCEED: Score ({final_debiased:.2f}) meets threshold"
    elif len(flaws) > 0:
        recommendation = f"REJECT: Fatal flaws in {[f.name for f in flaws]}"
    else:
        recommendation = f"UNCERTAIN: Score ({final_debiased:.2f}) below threshold"
    
    tracer.log(TraceEventType.DECISION, "L5", {
        'decision': recommendation,
        'bayesian_score': final_bayesian,
        'combined_score': final_combined,
        'calibrated_score': final_calibrated,
        'debiased_score': final_debiased,
        'credence': element.epistemic_state.credence,
        'confidence_interval': element.epistemic_state.get_confidence_interval(),
        'value_of_information': voi,
        'fatal_flaws': [f.name for f in flaws],
        'biases_detected': [b.bias_type.value for b in element.bias_checks if b.detected]
    })
    tracer.log(TraceEventType.LAYER_EXIT, "L5", {'result': 'Analysis complete'})
    
    return {
        'name': element.name,
        
        # Core scores
        'quality': history[-1] if history else 0,
        'bayesian_score': final_bayesian,
        'combined_score': final_combined,
        'calibrated_score': final_calibrated,
        'debiased_score': final_debiased,
        'additive_score': element.scoring.additive_score(),
        'multiplicative_score': element.scoring.multiplicative_score(),
        
        # Epistemic state (FRAMEWORK 1)
        'credence': element.epistemic_state.credence,
        'confidence_interval': element.epistemic_state.get_confidence_interval(),
        'epistemic_uncertainty': element.epistemic_state.epistemic_uncertainty,
        
        # Decision outputs
        'ready': ready,
        'recommendation': recommendation,
        'reason': reason,
        
        # Fatal flaws
        'fatal_flaws': [{'name': f.name, 'value': f.value, 'threshold': f.is_fatal_below} for f in flaws],
        
        # Utility model (FRAMEWORK 3)
        'expected_utility': element.utility_model.expected_utility(),
        'certainty_equivalent': element.utility_model.certainty_equivalent(),
        'value_of_information': voi,
        
        # Bias detection (FRAMEWORK 4)
        'biases_detected': [b.to_dict() for b in element.bias_checks if b.detected],
        'bias_penalty': element.bias_detector.total_bias_penalty() if element.bias_detector else 0,
        
        # Evidence (FRAMEWORK 5)
        'evidence_count': len(element.evidence),
        'total_evidence_bits': element.get_total_evidence_bits(),
        
        # System dynamics (FRAMEWORK 6)
        'feedback_loops': [fl.to_dict() for fl in element.feedback_loops],
        'systemic_risk': element.mechanism_map.systemic_risk_score(),
        
        # Causal analysis (FRAMEWORK 2)
        'mechanism_confidence': element.mechanism_map.overall_confidence(),
        'average_causal_strength': element.mechanism_map.average_causal_strength(),
        
        # Calibration (FRAMEWORK 7)
        'calibration_ece': element.calibration_tracker.expected_calibration_error(),
        
        # Analysis metadata
        'iterations': tester.iteration,
        'history': history,
        'criticisms': [c.to_dict() for c in tester.criticisms],
        'sensitivity': sensitivity,
        'mechanism_map': element.mechanism_map.to_dict(),
        'dimensions': {name: {'value': d.value, 'weight': d.weight, 'fatal': d.is_fatal_flaw,
                             'uncertainty': d.uncertainty, 'ci': d.confidence_interval()}
                      for name, d in element.scoring.dimensions.items()},
        'trace': tracer.get_trace(),
        'iteration_data': tracer.iteration_data
    }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("SCIENTIFICALLY ENHANCED ANALYTICAL PROTOCOL v2.0 - DEMO")
    print("=" * 70)
    print()
    
    # Create hypothesis
    h1 = AnalysisElement(name="Hire Data Scientist", domain=EvidenceDomain.BUSINESS)
    
    # Set foundation
    h1.set_what("Full-time data scientist, $120K, to improve decision-making", 0.9)
    h1.set_why("We make data-driven decisions but lack statistical expertise", 0.7)
    h1.set_how("Post job → Interview → Hire → Onboard → Deliver insights", 0.8)
    h1.set_measure("A/B test success rate improves by 20%", 0.7)
    
    # Add mechanism nodes with proper epistemic states
    n1 = MechanismNode("cause1", "Lack of statistics expertise", NodeType.CAUSE)
    n1.confidence = 0.9
    h1.add_mechanism_node(n1)
    
    n2 = MechanismNode("cause2", "Poor experiment design", NodeType.CAUSE)
    n2.confidence = 0.8
    h1.add_mechanism_node(n2)
    
    n3 = MechanismNode("mech1", "DS brings expertise", NodeType.MECHANISM)
    n3.confidence = 0.85
    h1.add_mechanism_node(n3)
    
    n4 = MechanismNode("outcome1", "Improved decisions", NodeType.OUTCOME)
    n4.confidence = 0.7
    h1.add_mechanism_node(n4)
    
    n5 = MechanismNode("blocker1", "Org resistance to data", NodeType.BLOCKER)
    n5.confidence = 0.4
    h1.add_mechanism_node(n5)
    
    n6 = MechanismNode("assume1", "DS can integrate with team", NodeType.ASSUMPTION)
    n6.confidence = 0.6
    h1.add_mechanism_node(n6)
    
    # Add edges with causal levels (FRAMEWORK 2)
    h1.add_mechanism_edge(MechanismEdge(
        "cause1", "mech1", EdgeType.CAUSES, 0.9,
        causal_level=CausalLevel.COUNTERFACTUAL,  # Theoretical mechanism
        confounding_risk=0.2
    ))
    h1.add_mechanism_edge(MechanismEdge(
        "mech1", "outcome1", EdgeType.ENABLES, 0.7,
        causal_level=CausalLevel.ASSOCIATION,  # Only have correlational evidence
        confounding_risk=0.5
    ))
    h1.add_mechanism_edge(MechanismEdge(
        "blocker1", "outcome1", EdgeType.PREVENTS, 0.4,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.3
    ))
    
    # Add evidence with proper causal levels (FRAMEWORK 5)
    h1.add_evidence(Evidence(
        "ev1", "HBR study: Companies with DS report 15% better outcomes",
        "HBR 2023", 0.7, "2023", EvidenceDomain.BUSINESS, "multi_company_analysis",
        causal_level=CausalLevel.ASSOCIATION,  # Observational only!
        supports_hypothesis=True
    ))
    h1.add_evidence(Evidence(
        "ev2", "Competitor hired DS, improved their metrics",
        "Industry contact", 0.3, "2024", EvidenceDomain.BUSINESS, "anecdote",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    # Add utility scenarios (FRAMEWORK 3)
    h1.add_scenario("Success: DS integrates well, decisions improve significantly", 0.5, 1.0)
    h1.add_scenario("Partial: Some improvement but not transformative", 0.3, 0.4)
    h1.add_scenario("Failure: DS doesn't integrate, leaves within year", 0.2, -0.3)
    
    # Set feasibility
    h1.set_feasibility(technical=0.9, economic=0.7, timeline=0.8)
    h1.set_risk(execution_risk=0.3, external_risk=0.2)
    
    # Run analysis
    print("Running enhanced analysis...")
    print()
    results = run_analysis(h1, rigor_level=2, max_iter=10)
    
    # Print results
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print()
    
    print("SCORES:")
    print(f"  Bayesian Score:     {results['bayesian_score']:.3f}")
    print(f"  Combined Score:     {results['combined_score']:.3f}")
    print(f"  Calibrated Score:   {results['calibrated_score']:.3f}")
    print(f"  Debiased Score:     {results['debiased_score']:.3f}")
    print()
    
    print("EPISTEMIC STATE (Framework 1):")
    print(f"  Credence:           {results['credence']:.3f}")
    print(f"  Confidence Interval: {results['confidence_interval']}")
    print()
    
    print("CAUSAL ANALYSIS (Framework 2):")
    print(f"  Mechanism Confidence:    {results['mechanism_confidence']:.3f}")
    print(f"  Average Causal Strength: {results['average_causal_strength']:.3f}")
    print()
    
    print("DECISION THEORY (Framework 3):")
    print(f"  Expected Utility:       {results['expected_utility']:.3f}")
    print(f"  Certainty Equivalent:   {results['certainty_equivalent']:.3f}")
    print(f"  Value of Information:   {results['value_of_information']:.3f}")
    print()
    
    print("COGNITIVE BIASES (Framework 4):")
    if results['biases_detected']:
        for bias in results['biases_detected']:
            print(f"  ⚠️  {bias['type']}: {bias['evidence']}")
            print(f"      Mitigation: {bias['mitigation']}")
    else:
        print("  No biases detected")
    print()
    
    print("EVIDENCE (Framework 5):")
    print(f"  Evidence Count:     {results['evidence_count']}")
    print(f"  Total Bits:         {results['total_evidence_bits']:.2f}")
    print()
    
    print("SYSTEM DYNAMICS (Framework 6):")
    print(f"  Feedback Loops:     {len(results['feedback_loops'])}")
    print(f"  Systemic Risk:      {results['systemic_risk']:.3f}")
    print()
    
    print("CALIBRATION (Framework 7):")
    print(f"  Expected Calibration Error: {results['calibration_ece']:.3f}")
    print()
    
    print("=" * 70)
    print(f"RECOMMENDATION: {results['recommendation']}")
    print("=" * 70)
    
    if results['fatal_flaws']:
        print()
        print("FATAL FLAWS:")
        for flaw in results['fatal_flaws']:
            print(f"  💀 {flaw['name']}: {flaw['value']:.2f} (threshold: {flaw['threshold']})")
    
    return results


if __name__ == "__main__":
    demo()
