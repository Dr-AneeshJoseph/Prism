"""
PRISM v1.0 - Protocol for Rigorous Investigation of Scientific Mechanisms
==========================================================================

A hardened decision analysis framework addressing all vulnerabilities identified
in the red team analysis of the Enhanced Analytical Protocol v2.0.

MAJOR IMPROVEMENTS OVER v2.0:
1. Safety Limits - Hard caps on iterations, evidence, complexity
2. Warning System - Proactive alerts for dangerous configurations
3. Numerical Stability - Clamping with tracking and warnings
4. Quality-First Causal Inference - Strong cohort > weak RCT
5. Evidence Independence Checking - Detects redundancy
6. Realistic VOI - Accounts for costs, time, quality limits
7. Risk-Aware Utility - CRRA utility functions
8. Content-Based Fatal Flaw Detection - Scans for legal/safety issues
9. Anti-Gaming Measures - Weight reasonableness checks
10. Improved Bias Detection - Doesn't penalize established truths
11. Calibration Cold-Start Handling - Explicit uncertainty when uncalibrated

Author: Dr. Aneesh Joseph (Architecture) + Claude (Implementation)
Version: 1.0
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set, Callable
from enum import Enum
from datetime import datetime
import json
import warnings
import re
from collections import defaultdict

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# SAFETY LIMITS - P0 FIX
# =============================================================================

class SafetyLimits:
    """
    Hard limits to prevent system abuse and numerical instability.
    These are NON-NEGOTIABLE safety rails.
    """
    # Iteration limits
    MAX_ITERATIONS: int = 100
    MIN_ITERATIONS: int = 3
    
    # Evidence limits
    MAX_EVIDENCE_PIECES: int = 500
    MIN_EVIDENCE_FOR_HIGH_STAKES: int = 3
    MAX_EVIDENCE_BITS: float = 20.0  # Cap total information
    
    # Mechanism map limits
    MAX_MECHANISM_NODES: int = 100
    MAX_MECHANISM_EDGES: int = 500
    
    # Numerical stability
    MAX_LIKELIHOOD_RATIO: float = 100.0
    MIN_LIKELIHOOD_RATIO: float = 0.01
    MAX_LOG_ODDS: float = 10.0
    MIN_LOG_ODDS: float = -10.0
    
    # Confidence limits
    CREDENCE_HARD_CAP: float = 0.99
    CREDENCE_HARD_FLOOR: float = 0.01
    CREDENCE_WARNING_THRESHOLD: float = 0.90
    CREDENCE_EXTREME_THRESHOLD: float = 0.95
    
    # Evidence balance
    MIN_CONTRADICTING_RATIO: float = 0.05  # At least 5% should contradict
    
    # Dimension weights
    MAX_WEIGHT_RATIO: float = 10.0  # No dimension >10x another
    MAX_SINGLE_WEIGHT: float = 5.0
    
    # Calibration
    MIN_CALIBRATION_POINTS: int = 20
    
    # VOI limits
    MAX_REASONABLE_VOI: float = 0.5  # VOI > 50% of utility is suspicious


# =============================================================================
# ENUMS
# =============================================================================

class WarningLevel(Enum):
    """Severity levels for system warnings"""
    INFO = "info"
    WARNING = "warning"  
    CRITICAL = "critical"
    FATAL = "fatal"


class EvidenceDomain(Enum):
    """Domain-specific evidence hierarchies"""
    MEDICAL = "medical"
    BUSINESS = "business"
    POLICY = "policy"
    TECHNOLOGY = "technology"
    SCIENTIFIC = "scientific"
    GENERAL = "general"


class NodeType(Enum):
    """Types of nodes in mechanism maps"""
    CAUSE = "cause"
    MECHANISM = "mechanism"
    OUTCOME = "outcome"
    BLOCKER = "blocker"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    INTERVENTION = "intervention"


class EdgeType(Enum):
    """Types of causal relationships"""
    CAUSES = "causes"
    PREVENTS = "prevents"
    ENABLES = "enables"
    COMPENSATES = "compensates"
    REQUIRES = "requires"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class CausalLevel(Enum):
    """
    Pearl's Causal Hierarchy - but used as BOOST, not discount.
    Higher causal level = bonus on top of quality, not replacement for quality.
    """
    ASSOCIATION = 1      # P(Y|X) - Observational
    INTERVENTION = 2     # P(Y|do(X)) - Experimental
    COUNTERFACTUAL = 3   # P(Y_x|X',Y') - Theoretical mechanism


class BiasType(Enum):
    """Cognitive biases to detect"""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    PLANNING_FALLACY = "planning_fallacy"
    SUNK_COST = "sunk_cost"


class FeedbackLoopType(Enum):
    """System dynamics feedback types"""
    REINFORCING = "reinforcing"
    BALANCING = "balancing"


class DecisionReadiness(Enum):
    """Decision readiness states"""
    READY = "ready"
    NEEDS_MORE_INFO = "needs_more_info"
    REJECT = "reject"
    UNCERTAIN = "uncertain"
    FATAL_FLAW = "fatal_flaw"


class TraceEventType(Enum):
    """Event types for audit trail"""
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
    WARNING_ISSUED = "warning_issued"
    SAFETY_LIMIT_HIT = "safety_limit_hit"
    INDEPENDENCE_CHECK = "independence_check"
    CONTENT_SCAN = "content_scan"
    CALIBRATION_APPLIED = "calibration_applied"
    BIAS_DETECTED = "bias_detected"
    VOI_CALCULATED = "voi_calculated"


# =============================================================================
# EVIDENCE QUALITY HIERARCHIES (Quality-First, Design as Boost)
# =============================================================================

EVIDENCE_QUALITY_BASE = {
    EvidenceDomain.MEDICAL: {
        'meta_analysis': 0.95, 'systematic_review': 0.90, 'rct': 0.85,
        'cohort': 0.75, 'case_control': 0.65, 'cross_sectional': 0.55,
        'case_series': 0.40, 'expert_opinion': 0.30, 'anecdote': 0.15
    },
    EvidenceDomain.BUSINESS: {
        'meta_analysis': 0.90, 'multi_company_analysis': 0.85,
        'controlled_experiment': 0.80, 'ab_test': 0.75, 'benchmark': 0.65,
        'case_study': 0.50, 'expert_opinion': 0.35, 'anecdote': 0.20
    },
    EvidenceDomain.POLICY: {
        'meta_analysis': 0.90, 'randomized_trial': 0.85,
        'quasi_experiment': 0.75, 'regression_discontinuity': 0.70,
        'difference_in_differences': 0.65, 'instrumental_variables': 0.60,
        'case_study': 0.40, 'expert_opinion': 0.30
    },
    EvidenceDomain.TECHNOLOGY: {
        'meta_analysis': 0.90, 'controlled_experiment': 0.85, 'ab_test': 0.80,
        'benchmark': 0.70, 'performance_study': 0.60,
        'case_study': 0.45, 'expert_opinion': 0.35, 'anecdote': 0.20
    },
    EvidenceDomain.SCIENTIFIC: {
        'meta_analysis': 0.95, 'systematic_review': 0.90,
        'replicated_experiment': 0.85, 'single_experiment': 0.70,
        'observational': 0.55, 'theoretical': 0.50, 'expert_opinion': 0.35
    },
    EvidenceDomain.GENERAL: {
        'rigorous_study': 0.80, 'good_study': 0.60, 'weak_study': 0.40,
        'expert_opinion': 0.30, 'anecdote': 0.15
    }
}

# CAUSAL LEVEL BOOST (not discount!) - Quality dominates, causal level adds bonus
CAUSAL_LEVEL_BOOST = {
    CausalLevel.ASSOCIATION: 0.0,       # No boost for observational
    CausalLevel.INTERVENTION: 0.15,     # 15% boost for experimental
    CausalLevel.COUNTERFACTUAL: 0.05    # Small boost for theoretical mechanism
}

# Sample size adjustments
SAMPLE_SIZE_MODIFIERS = {
    'tiny': (0, 50, 0.6),           # n < 50: 40% penalty
    'small': (50, 200, 0.8),        # 50 <= n < 200: 20% penalty
    'medium': (200, 1000, 1.0),     # 200 <= n < 1000: no adjustment
    'large': (1000, 10000, 1.05),   # 1000 <= n < 10000: 5% bonus
    'very_large': (10000, float('inf'), 1.10)  # n >= 10000: 10% bonus
}

# Fatal content patterns - things that should trigger immediate review
FATAL_CONTENT_PATTERNS = [
    (r'\b(illegal|unlawful|violates?\s+(law|regulation|statute))\b', 'legal'),
    (r'\b(prohibited|banned|forbidden)\b', 'legal'),
    (r'\b(fatal|lethal|death|mortality)\s+(risk|rate|outcome)', 'safety'),
    (r'\b(unsafe|dangerous|hazardous)\b', 'safety'),
    (r'\b(fraud|fraudulent|deceptive)\b', 'ethical'),
    (r'\b(unethical|immoral)\b', 'ethical'),
    (r'\b(bankruptcy|insolvent|default)\s+risk', 'financial'),
    (r'\b(cannot|impossible|infeasible)\b', 'feasibility'),
]


# =============================================================================
# WARNING SYSTEM - P0 FIX
# =============================================================================

@dataclass
class SystemWarning:
    """A warning issued by the system"""
    level: WarningLevel
    category: str
    message: str
    recommendation: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __str__(self) -> str:
        icons = {
            WarningLevel.INFO: "ℹ️",
            WarningLevel.WARNING: "⚠️",
            WarningLevel.CRITICAL: "🚨",
            WarningLevel.FATAL: "💀"
        }
        icon = icons.get(self.level, "❓")
        return f"{icon} [{self.category}] {self.message}\n   → {self.recommendation}"
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'category': self.category,
            'message': self.message,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp
        }


class WarningSystem:
    """
    Centralized warning generation and tracking.
    Proactively alerts users to dangerous configurations.
    """
    
    def __init__(self):
        self.warnings: List[SystemWarning] = []
        self._suppressed_categories: Set[str] = set()
    
    def add_warning(self, level: WarningLevel, category: str,
                   message: str, recommendation: str) -> SystemWarning:
        """Add a warning to the system"""
        if category in self._suppressed_categories:
            return None
        
        warning = SystemWarning(level, category, message, recommendation)
        self.warnings.append(warning)
        return warning
    
    def suppress_category(self, category: str):
        """Suppress warnings of a specific category"""
        self._suppressed_categories.add(category)
    
    def check_evidence_sufficiency(self, count: int, is_high_stakes: bool = True) -> bool:
        """Check if we have enough evidence"""
        min_required = SafetyLimits.MIN_EVIDENCE_FOR_HIGH_STAKES if is_high_stakes else 1
        
        if count < min_required:
            self.add_warning(
                WarningLevel.CRITICAL,
                "Evidence Sufficiency",
                f"Only {count} evidence pieces (minimum for high-stakes: {min_required})",
                "Add more independent evidence before making decisions"
            )
            return False
        
        if count < 5:
            self.add_warning(
                WarningLevel.WARNING,
                "Evidence Sufficiency",
                f"Only {count} evidence pieces - limited confidence warranted",
                "Consider gathering additional evidence for robustness"
            )
        return True
    
    def check_high_credence(self, credence: float) -> bool:
        """Check for suspiciously high confidence"""
        if credence > SafetyLimits.CREDENCE_EXTREME_THRESHOLD:
            self.add_warning(
                WarningLevel.CRITICAL,
                "Extreme Confidence",
                f"Credence {credence:.1%} exceeds {SafetyLimits.CREDENCE_EXTREME_THRESHOLD:.0%}",
                "Verify evidence independence. Actively seek contradicting evidence. "
                "Consider: Is this confidence justified or a numerical artifact?"
            )
            return False
        
        if credence > SafetyLimits.CREDENCE_WARNING_THRESHOLD:
            self.add_warning(
                WarningLevel.WARNING,
                "High Confidence",
                f"Credence {credence:.1%} exceeds {SafetyLimits.CREDENCE_WARNING_THRESHOLD:.0%}",
                "Double-check: Have you sought disconfirming evidence?"
            )
            return False
        return True
    
    def check_evidence_balance(self, supporting: int, contradicting: int) -> bool:
        """Check for evidence imbalance (possible confirmation bias)"""
        total = supporting + contradicting
        
        if total == 0:
            return True  # No evidence to check
        
        if contradicting == 0 and supporting >= 3:
            self.add_warning(
                WarningLevel.WARNING,
                "Evidence Imbalance",
                f"All {supporting} evidence pieces support hypothesis, none contradict",
                "This pattern suggests confirmation bias OR the hypothesis is well-established. "
                "If novel hypothesis: actively seek contradicting evidence. "
                "If established fact: this warning can be acknowledged."
            )
            return False
        
        ratio = contradicting / total if total > 0 else 0
        if ratio < SafetyLimits.MIN_CONTRADICTING_RATIO and supporting >= 5:
            self.add_warning(
                WarningLevel.INFO,
                "Evidence Imbalance",
                f"Only {ratio:.1%} of evidence contradicts hypothesis",
                "Consider whether you've adequately sought opposing views"
            )
        return True
    
    def check_numerical_stability(self, original_lr: float, clamped_lr: float,
                                  was_clamped: bool) -> bool:
        """Check for numerical clamping issues"""
        if was_clamped:
            self.add_warning(
                WarningLevel.WARNING,
                "Numerical Stability",
                f"Likelihood ratio {original_lr:.2f} clamped to {clamped_lr:.2f}",
                "Evidence strength hit numerical bounds. "
                "Results may be less precise than displayed."
            )
            return False
        return True
    
    def check_evidence_bits(self, total_bits: float) -> bool:
        """Check if evidence bits are approaching cap"""
        if total_bits > SafetyLimits.MAX_EVIDENCE_BITS * 0.9:
            self.add_warning(
                WarningLevel.WARNING,
                "Evidence Saturation",
                f"Total evidence bits ({total_bits:.1f}) approaching cap ({SafetyLimits.MAX_EVIDENCE_BITS})",
                "Additional evidence may have diminishing returns. "
                "Verify evidence independence to ensure bits aren't inflated."
            )
            return False
        return True
    
    def check_calibration_sufficiency(self, num_predictions: int) -> bool:
        """Check if calibration data is sufficient"""
        if num_predictions < SafetyLimits.MIN_CALIBRATION_POINTS:
            self.add_warning(
                WarningLevel.INFO,
                "Uncalibrated",
                f"Only {num_predictions} historical predictions "
                f"(need {SafetyLimits.MIN_CALIBRATION_POINTS} for reliable calibration)",
                "Calibration adjustments are unreliable. "
                "Treat confidence intervals with extra skepticism."
            )
            return False
        return True
    
    def check_weight_gaming(self, weights: Dict[str, float]) -> bool:
        """Check for potential dimension weight gaming"""
        if not weights:
            return True
        
        values = list(weights.values())
        max_weight = max(values)
        min_weight = min(values) if min(values) > 0 else 0.01
        
        ratio = max_weight / min_weight
        if ratio > SafetyLimits.MAX_WEIGHT_RATIO:
            self.add_warning(
                WarningLevel.WARNING,
                "Weight Imbalance",
                f"Weight ratio {ratio:.1f}:1 exceeds limit of {SafetyLimits.MAX_WEIGHT_RATIO:.0f}:1",
                "Large weight disparities can be gamed. Verify weights reflect true importance."
            )
            return False
        
        if max_weight > SafetyLimits.MAX_SINGLE_WEIGHT:
            self.add_warning(
                WarningLevel.INFO,
                "High Weight",
                f"Maximum weight {max_weight:.1f} exceeds recommended {SafetyLimits.MAX_SINGLE_WEIGHT}",
                "Consider whether this dimension truly dominates all others."
            )
        return True
    
    def check_mechanism_complexity(self, nodes: int, edges: int) -> bool:
        """Check for mechanism map complexity"""
        if nodes > SafetyLimits.MAX_MECHANISM_NODES:
            self.add_warning(
                WarningLevel.CRITICAL,
                "Mechanism Complexity",
                f"{nodes} nodes exceeds limit of {SafetyLimits.MAX_MECHANISM_NODES}",
                "Analysis may be intractable. Simplify mechanism map."
            )
            return False
        
        if nodes > SafetyLimits.MAX_MECHANISM_NODES * 0.7:
            self.add_warning(
                WarningLevel.WARNING,
                "Mechanism Complexity",
                f"{nodes} nodes approaching complexity limit",
                "Consider simplifying for more reliable analysis."
            )
        return True
    
    def check_voi_reasonableness(self, voi: float, expected_utility: float) -> bool:
        """Check if VOI calculation is realistic"""
        if expected_utility != 0:
            voi_ratio = abs(voi / expected_utility) if expected_utility != 0 else float('inf')
            if voi_ratio > SafetyLimits.MAX_REASONABLE_VOI:
                self.add_warning(
                    WarningLevel.WARNING,
                    "Unrealistic VOI",
                    f"VOI ({voi:.3f}) is {voi_ratio:.0%} of expected utility",
                    "This VOI assumes perfect, free, instant information. "
                    "Real information has costs, delays, and quality limits."
                )
                return False
        return True
    
    def get_critical_warnings(self) -> List[SystemWarning]:
        """Get all critical and fatal warnings"""
        return [w for w in self.warnings 
                if w.level in [WarningLevel.CRITICAL, WarningLevel.FATAL]]
    
    def get_warnings_by_level(self, level: WarningLevel) -> List[SystemWarning]:
        """Get warnings of a specific level"""
        return [w for w in self.warnings if w.level == level]
    
    def has_fatal_warnings(self) -> bool:
        """Check if any fatal warnings exist"""
        return any(w.level == WarningLevel.FATAL for w in self.warnings)
    
    def has_critical_warnings(self) -> bool:
        """Check if any critical warnings exist"""
        return any(w.level == WarningLevel.CRITICAL for w in self.warnings)
    
    def summary(self) -> Dict[str, int]:
        """Get warning counts by level"""
        counts = {level.value: 0 for level in WarningLevel}
        for w in self.warnings:
            counts[w.level.value] += 1
        return counts
    
    def print_warnings(self, min_level: WarningLevel = WarningLevel.INFO):
        """Print all warnings at or above minimum level"""
        level_order = [WarningLevel.INFO, WarningLevel.WARNING, 
                      WarningLevel.CRITICAL, WarningLevel.FATAL]
        min_idx = level_order.index(min_level)
        
        filtered = [w for w in self.warnings 
                   if level_order.index(w.level) >= min_idx]
        
        if not filtered:
            print("✓ No warnings")
            return
        
        for warning in filtered:
            print(warning)
            print()
    
    def to_dict(self) -> Dict:
        return {
            'warnings': [w.to_dict() for w in self.warnings],
            'summary': self.summary(),
            'has_critical': self.has_critical_warnings(),
            'has_fatal': self.has_fatal_warnings()
        }
    
    def clear(self):
        """Clear all warnings"""
        self.warnings = []


# =============================================================================
# TRACE SYSTEM
# =============================================================================

@dataclass
class TraceEvent:
    """A single event in the analysis trace"""
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
    """Comprehensive audit trail for all analysis operations"""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.start_time = datetime.now().timestamp()
        self.iteration_data: List[Dict] = []
        self.sensitivity_data: List[Dict] = []
    
    def _elapsed(self) -> float:
        return datetime.now().timestamp() - self.start_time
    
    def log(self, event_type: TraceEventType, layer: str, data: Dict[str, Any]):
        """Log an event to the trace"""
        self.events.append(TraceEvent(self._elapsed(), event_type, layer, data))
    
    def get_trace(self) -> List[Dict]:
        """Get complete trace as list of dicts"""
        return [e.to_dict() for e in self.events]
    
    def get_events_by_type(self, event_type: TraceEventType) -> List[TraceEvent]:
        """Get all events of a specific type"""
        return [e for e in self.events if e.event_type == event_type]


# =============================================================================
# EPISTEMIC STATE - P0 FIX: Numerical Stability
# =============================================================================

@dataclass
class EpistemicState:
    """
    Properly tracked epistemic state with numerical stability safeguards.
    
    FIXES:
    - Tracks clamping events and warns when they occur
    - Hard caps on credence to prevent false precision
    - Proper confidence intervals accounting for uncertainty
    - Audit trail of all updates
    """
    
    credence: float = 0.5
    log_odds: float = 0.0
    reliability: float = 0.5
    epistemic_uncertainty: float = 0.5
    
    # Tracking for safety
    update_count: int = 0
    clamping_count: int = 0
    total_bits_accumulated: float = 0.0
    update_history: List[Dict] = field(default_factory=list)
    
    # Warning system reference
    warning_system: Optional[WarningSystem] = None
    
    def __post_init__(self):
        self._sync_credence_and_log_odds()
        self._clamp_values()
    
    def _sync_credence_and_log_odds(self):
        """Ensure credence and log_odds are consistent"""
        if 0 < self.credence < 1:
            self.log_odds = np.log(self.credence / (1 - self.credence))
    
    def _clamp_values(self) -> bool:
        """Clamp values to valid ranges, return True if clamping occurred"""
        clamped = False
        
        # Clamp credence
        old_credence = self.credence
        self.credence = max(SafetyLimits.CREDENCE_HARD_FLOOR, 
                           min(SafetyLimits.CREDENCE_HARD_CAP, self.credence))
        if abs(old_credence - self.credence) > 1e-9:
            clamped = True
            self.clamping_count += 1
        
        # Clamp log_odds
        self.log_odds = max(SafetyLimits.MIN_LOG_ODDS,
                          min(SafetyLimits.MAX_LOG_ODDS, self.log_odds))
        
        # Clamp other values
        self.reliability = max(0.0, min(1.0, self.reliability))
        self.epistemic_uncertainty = max(0.0, min(1.0, self.epistemic_uncertainty))
        
        return clamped
    
    def update_with_evidence(self, likelihood_ratio: float, 
                            evidence_id: str = None) -> Tuple[float, bool]:
        """
        Bayesian update with full tracking and safety checks.
        
        Returns: (change_in_credence, was_clamped)
        """
        self.update_count += 1
        old_credence = self.credence
        old_log_odds = self.log_odds
        
        # Check and clamp likelihood ratio
        original_lr = likelihood_ratio
        was_clamped = (likelihood_ratio < SafetyLimits.MIN_LIKELIHOOD_RATIO or
                      likelihood_ratio > SafetyLimits.MAX_LIKELIHOOD_RATIO)
        
        lr = max(SafetyLimits.MIN_LIKELIHOOD_RATIO,
                min(SafetyLimits.MAX_LIKELIHOOD_RATIO, likelihood_ratio))
        
        # Warn if clamped
        if was_clamped and self.warning_system:
            self.warning_system.check_numerical_stability(original_lr, lr, was_clamped)
        
        # Update log-odds (additive in log space)
        self.log_odds += np.log(lr)
        
        # Clamp log-odds
        self.log_odds = max(SafetyLimits.MIN_LOG_ODDS,
                          min(SafetyLimits.MAX_LOG_ODDS, self.log_odds))
        
        # Convert back to probability
        self.credence = 1 / (1 + np.exp(-self.log_odds))
        
        # Final clamping
        credence_clamped = self._clamp_values()
        
        # Calculate bits of information
        bits = abs(np.log2(lr)) if lr > 0 else 0
        self.total_bits_accumulated += bits
        
        # Check evidence bit cap
        if self.warning_system:
            self.warning_system.check_evidence_bits(self.total_bits_accumulated)
            self.warning_system.check_high_credence(self.credence)
        
        # Record update
        self.update_history.append({
            'update_num': self.update_count,
            'evidence_id': evidence_id,
            'likelihood_ratio': lr,
            'original_lr': original_lr,
            'was_clamped': was_clamped,
            'old_credence': old_credence,
            'new_credence': self.credence,
            'bits': bits,
            'total_bits': self.total_bits_accumulated
        })
        
        return self.credence - old_credence, was_clamped or credence_clamped
    
    def get_confidence_interval(self, alpha: float = 0.90) -> Tuple[float, float]:
        """
        Get credible interval accounting for epistemic uncertainty.
        
        Higher epistemic_uncertainty = wider interval.
        Also widens interval if we've hit numerical bounds.
        """
        # Base spread from epistemic uncertainty
        base_spread = self.epistemic_uncertainty * 0.4
        
        # Additional spread if we've had clamping issues
        clamping_penalty = min(0.2, self.clamping_count * 0.02)
        
        # Additional spread if calibration is unknown
        total_spread = base_spread + clamping_penalty
        
        lower = max(0.0, self.credence - total_spread)
        upper = min(1.0, self.credence + total_spread)
        
        return (round(lower, 3), round(upper, 3))
    
    def get_point_estimate_warning(self) -> Optional[str]:
        """
        Return warning if point estimate should not be trusted.
        """
        warnings = []
        
        if self.clamping_count > 3:
            warnings.append(f"Multiple numerical bounds hit ({self.clamping_count}x)")
        
        if self.credence > SafetyLimits.CREDENCE_EXTREME_THRESHOLD:
            warnings.append(f"Extreme confidence ({self.credence:.1%})")
        
        if self.total_bits_accumulated > SafetyLimits.MAX_EVIDENCE_BITS * 0.8:
            warnings.append(f"High evidence accumulation ({self.total_bits_accumulated:.1f} bits)")
        
        return "; ".join(warnings) if warnings else None
    
    def to_dict(self) -> Dict:
        return {
            'credence': round(self.credence, 4),
            'log_odds': round(self.log_odds, 4),
            'reliability': round(self.reliability, 4),
            'epistemic_uncertainty': round(self.epistemic_uncertainty, 4),
            'confidence_interval': self.get_confidence_interval(),
            'update_count': self.update_count,
            'clamping_count': self.clamping_count,
            'total_bits': round(self.total_bits_accumulated, 2),
            'point_estimate_warning': self.get_point_estimate_warning()
        }


# =============================================================================
# EVIDENCE - P0 FIX: Quality-First, Independence Checking
# =============================================================================

@dataclass
class Evidence:
    """
    Evidence with quality-first assessment and independence tracking.
    
    FIXES:
    - Quality dominates over study design (strong cohort > weak RCT)
    - Sample size adjustments
    - Content scanning for fatal issues
    - Independence tracking for redundancy detection
    """
    
    id: str
    content: str
    source: str
    quality: float  # Base quality 0-1 (self-assessed or from hierarchy)
    date: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    study_design: Optional[str] = None
    
    # Causal level (for boost, not discount)
    causal_level: CausalLevel = CausalLevel.ASSOCIATION
    
    # Direction
    supports_hypothesis: bool = True
    
    # Sample size (critical for proper weighting)
    sample_size: Optional[int] = None
    
    # Independence tracking
    authors: List[str] = field(default_factory=list)
    cites: List[str] = field(default_factory=list)  # IDs of other evidence this cites
    funding_source: Optional[str] = None
    underlying_data: Optional[str] = None  # ID of underlying dataset if known
    
    # Computed values
    effective_quality: float = 0.0
    bits: float = 0.0
    independence_score: float = 1.0  # 1.0 = fully independent
    
    # Content scan results
    fatal_content_flags: List[Tuple[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        self._calculate_effective_quality()
        self._scan_content_for_fatal_flags()
    
    def _calculate_effective_quality(self):
        """
        Calculate effective quality: Quality-first, then causal boost, then sample size.
        
        This is the KEY FIX for the causal inference illusion vulnerability.
        """
        # Start with base quality
        if self.study_design and self.domain in EVIDENCE_QUALITY_BASE:
            hierarchy = EVIDENCE_QUALITY_BASE[self.domain]
            base = hierarchy.get(self.study_design, self.quality)
        else:
            base = self.quality
        
        # Apply causal level BOOST (not discount!)
        causal_boost = CAUSAL_LEVEL_BOOST.get(self.causal_level, 0.0)
        boosted = base * (1 + causal_boost)
        
        # Apply sample size modifier
        if self.sample_size is not None:
            for name, (low, high, modifier) in SAMPLE_SIZE_MODIFIERS.items():
                if low <= self.sample_size < high:
                    boosted *= modifier
                    break
        
        self.effective_quality = min(1.0, max(0.0, boosted))
    
    def _scan_content_for_fatal_flags(self):
        """Scan content for patterns that should trigger fatal flaw review"""
        self.fatal_content_flags = []
        content_lower = self.content.lower()
        
        for pattern, category in FATAL_CONTENT_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                self.fatal_content_flags.append((pattern, category))
    
    def has_fatal_content(self) -> bool:
        """Check if evidence contains fatal content flags"""
        return len(self.fatal_content_flags) > 0
    
    def get_quality(self) -> float:
        """Get effective quality after all adjustments"""
        return self.effective_quality
    
    def calculate_bits(self, prior: float, posterior: float) -> float:
        """Calculate information content in bits"""
        if prior <= 0.001 or posterior <= 0.001:
            return 0.0
        if prior >= 0.999 or posterior >= 0.999:
            return 0.0
        
        prior_odds = prior / (1 - prior)
        posterior_odds = posterior / (1 - posterior)
        
        if prior_odds <= 0:
            return 0.0
        
        self.bits = abs(np.log2(posterior_odds / prior_odds))
        return self.bits
    
    def get_likelihood_ratio(self) -> float:
        """
        Convert quality and direction to likelihood ratio for Bayesian update.
        
        Quality 1.0 → LR of 3 (strong evidence)
        Quality 0.5 → LR of 2 (moderate evidence)
        Quality 0.0 → LR of 1 (no information)
        """
        # Base LR scales with quality
        base_lr = 1 + self.effective_quality * 2
        
        # Invert for contradicting evidence
        if not self.supports_hypothesis:
            base_lr = 1 / base_lr
        
        return base_lr
    
    def get_effective_bits(self, independence_discount: float = 1.0) -> float:
        """Get bits adjusted for independence"""
        return self.bits * independence_discount * self.independence_score
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content[:200] + "..." if len(self.content) > 200 else self.content,
            'source': self.source,
            'base_quality': self.quality,
            'effective_quality': round(self.effective_quality, 3),
            'study_design': self.study_design,
            'causal_level': self.causal_level.name,
            'sample_size': self.sample_size,
            'supports': self.supports_hypothesis,
            'bits': round(self.bits, 3),
            'independence_score': round(self.independence_score, 3),
            'has_fatal_content': self.has_fatal_content(),
            'fatal_flags': [(cat, pat) for pat, cat in self.fatal_content_flags]
        }


class EvidenceIndependenceChecker:
    """
    Check evidence independence to prevent redundancy inflation.
    
    FIXES the evidence redundancy blind spot vulnerability.
    """
    
    @staticmethod
    def check_pairwise_independence(e1: Evidence, e2: Evidence) -> Tuple[float, List[str]]:
        """
        Check independence between two pieces of evidence.
        Returns (independence_score, list_of_issues)
        """
        issues = []
        penalties = []
        
        # Same source
        if e1.source and e2.source and e1.source.lower() == e2.source.lower():
            issues.append(f"Same source: {e1.source}")
            penalties.append(0.5)
        
        # Citation relationship
        if e1.id in e2.cites or e2.id in e1.cites:
            issues.append("Citation relationship")
            penalties.append(0.4)
        
        # Shared authors
        if e1.authors and e2.authors:
            shared = set(a.lower() for a in e1.authors) & set(a.lower() for a in e2.authors)
            if shared:
                issues.append(f"Shared authors: {shared}")
                penalties.append(0.3)
        
        # Same underlying data
        if e1.underlying_data and e2.underlying_data:
            if e1.underlying_data == e2.underlying_data:
                issues.append("Same underlying dataset")
                penalties.append(0.6)
        
        # Same funding source (potential bias correlation)
        if e1.funding_source and e2.funding_source:
            if e1.funding_source.lower() == e2.funding_source.lower():
                issues.append(f"Same funding: {e1.funding_source}")
                penalties.append(0.2)
        
        # Content similarity (basic check)
        content1_words = set(e1.content.lower().split())
        content2_words = set(e2.content.lower().split())
        if len(content1_words) > 5 and len(content2_words) > 5:
            overlap = len(content1_words & content2_words)
            union = len(content1_words | content2_words)
            jaccard = overlap / union if union > 0 else 0
            if jaccard > 0.5:
                issues.append(f"High content similarity ({jaccard:.0%})")
                penalties.append(0.3)
        
        independence = max(0.0, 1.0 - sum(penalties))
        return independence, issues
    
    @staticmethod
    def check_all_independence(evidence_list: List[Evidence], 
                               warning_system: Optional[WarningSystem] = None) -> Dict:
        """
        Check independence across all evidence pairs.
        Updates independence_score on each evidence object.
        """
        n = len(evidence_list)
        if n <= 1:
            return {
                'overall_independence': 1.0,
                'issues': [],
                'pairwise_scores': {}
            }
        
        all_issues = []
        pairwise_scores = {}
        evidence_penalties = defaultdict(list)
        
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = evidence_list[i], evidence_list[j]
                score, issues = EvidenceIndependenceChecker.check_pairwise_independence(e1, e2)
                
                pairwise_scores[(e1.id, e2.id)] = score
                
                if issues:
                    all_issues.append({
                        'evidence_1': e1.id,
                        'evidence_2': e2.id,
                        'independence': score,
                        'issues': issues
                    })
                    
                    # Track penalties for each evidence
                    penalty = 1.0 - score
                    evidence_penalties[e1.id].append(penalty)
                    evidence_penalties[e2.id].append(penalty)
        
        # Update independence scores on evidence objects
        for evidence in evidence_list:
            if evidence.id in evidence_penalties:
                penalties = evidence_penalties[evidence.id]
                # Average penalty across all comparisons
                avg_penalty = sum(penalties) / len(penalties)
                evidence.independence_score = max(0.1, 1.0 - avg_penalty)
            else:
                evidence.independence_score = 1.0
        
        # Calculate overall independence
        if pairwise_scores:
            overall = sum(pairwise_scores.values()) / len(pairwise_scores)
        else:
            overall = 1.0
        
        # Issue warnings
        if warning_system and all_issues:
            severe_issues = [i for i in all_issues if i['independence'] < 0.5]
            if severe_issues:
                warning_system.add_warning(
                    WarningLevel.WARNING,
                    "Evidence Independence",
                    f"{len(severe_issues)} evidence pairs have low independence (<50%)",
                    "Evidence may be redundant. Information content could be inflated. "
                    "Review the independence report and consider discounting."
                )
        
        return {
            'overall_independence': round(overall, 3),
            'issues': all_issues,
            'pairwise_scores': {f"{k[0]}-{k[1]}": round(v, 3) for k, v in pairwise_scores.items()}
        }


# =============================================================================
# MECHANISM MAP STRUCTURES
# =============================================================================

@dataclass
class MechanismNode:
    """A node in the causal mechanism map"""
    id: str
    label: str
    node_type: NodeType
    description: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    epistemic_state: EpistemicState = field(default_factory=EpistemicState)
    
    @property
    def confidence(self) -> float:
        return self.epistemic_state.credence
    
    @confidence.setter
    def confidence(self, value: float):
        self.epistemic_state.credence = max(0.001, min(0.999, value))
        if 0 < value < 1:
            self.epistemic_state.log_odds = np.log(value / (1 - value))
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.node_type.value,
            'confidence': round(self.epistemic_state.credence, 3),
            'description': self.description
        }


@dataclass
class MechanismEdge:
    """A causal relationship with proper causal annotation"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5
    evidence_ids: List[str] = field(default_factory=list)
    
    causal_level: CausalLevel = CausalLevel.ASSOCIATION
    confounding_risk: float = 0.5
    effect_size: float = 0.0
    effect_uncertainty: float = 0.5
    
    def causal_strength(self) -> float:
        """
        Calculate causal strength with quality-first approach.
        """
        base = max(0.0, min(1.0, self.strength))
        
        # Causal level provides BOOST (not discount)
        causal_boost = CAUSAL_LEVEL_BOOST.get(self.causal_level, 0.0)
        boosted = base * (1 + causal_boost)
        
        # Confounding risk provides discount
        confounding_discount = 1 - (self.confounding_risk * 0.3)
        
        # Uncertainty provides discount
        uncertainty_discount = 1 - (self.effect_uncertainty * 0.2)
        
        return min(1.0, boosted * confounding_discount * uncertainty_discount)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.edge_type.value,
            'strength': round(self.strength, 3),
            'causal_level': self.causal_level.name,
            'causal_strength': round(self.causal_strength(), 3),
            'confounding_risk': self.confounding_risk
        }


@dataclass
class FeedbackLoop:
    """A detected feedback loop"""
    loop_id: str
    node_ids: List[str]
    loop_type: FeedbackLoopType
    strength: float = 0.5
    time_delay: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'loop_id': self.loop_id,
            'nodes': self.node_ids,
            'type': self.loop_type.value,
            'strength': self.strength
        }


class MechanismMap:
    """Causal mechanism map with complexity controls"""
    
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.nodes: Dict[str, MechanismNode] = {}
        self.edges: List[MechanismEdge] = []
        self.feedback_loops: List[FeedbackLoop] = []
        self.warning_system = warning_system
        self.tracer: Optional[AnalysisTracer] = None
    
    def add_node(self, node: MechanismNode) -> str:
        # Check complexity limit
        if len(self.nodes) >= SafetyLimits.MAX_MECHANISM_NODES:
            if self.warning_system:
                self.warning_system.add_warning(
                    WarningLevel.CRITICAL,
                    "Mechanism Complexity",
                    f"Cannot add node: limit of {SafetyLimits.MAX_MECHANISM_NODES} reached",
                    "Simplify mechanism map or increase limit deliberately"
                )
            return None
        
        self.nodes[node.id] = node
        
        if self.warning_system:
            self.warning_system.check_mechanism_complexity(len(self.nodes), len(self.edges))
        
        if self.tracer:
            self.tracer.log(TraceEventType.MECHANISM_NODE_ADDED, "L1", node.to_dict())
        
        return node.id
    
    def add_edge(self, edge: MechanismEdge) -> bool:
        if len(self.edges) >= SafetyLimits.MAX_MECHANISM_EDGES:
            if self.warning_system:
                self.warning_system.add_warning(
                    WarningLevel.CRITICAL,
                    "Mechanism Complexity",
                    f"Cannot add edge: limit of {SafetyLimits.MAX_MECHANISM_EDGES} reached",
                    "Simplify mechanism map"
                )
            return False
        
        self.edges.append(edge)
        
        if self.tracer:
            self.tracer.log(TraceEventType.MECHANISM_EDGE_ADDED, "L1", edge.to_dict())
        
        return True
    
    def overall_confidence(self) -> float:
        """Geometric mean of node confidences"""
        if not self.nodes:
            return 0.0
        
        credences = [max(0.001, n.epistemic_state.credence) for n in self.nodes.values()]
        log_mean = np.mean(np.log(credences))
        return float(min(1.0, max(0.0, np.exp(log_mean))))
    
    def average_causal_strength(self) -> float:
        """Average causal strength across edges"""
        if not self.edges:
            return 0.0
        return np.mean([e.causal_strength() for e in self.edges])
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
        """Detect cycles in the mechanism map"""
        adjacency = {nid: [] for nid in self.nodes}
        edge_map = {}
        
        for edge in self.edges:
            if edge.source_id in adjacency:
                adjacency[edge.source_id].append(edge.target_id)
                edge_map[(edge.source_id, edge.target_id)] = edge
        
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
        
        self.feedback_loops = []
        for i, cycle_nodes in enumerate(cycles):
            if len(cycle_nodes) < 2:
                continue
            
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
            
            loop_type = FeedbackLoopType.BALANCING if prevent_count % 2 == 1 else FeedbackLoopType.REINFORCING
            avg_strength = total_strength / len(cycle_nodes) if cycle_nodes else 0.5
            
            self.feedback_loops.append(FeedbackLoop(
                loop_id=f"FL{i}",
                node_ids=cycle_nodes,
                loop_type=loop_type,
                strength=avg_strength
            ))
        
        return self.feedback_loops
    
    def systemic_risk_score(self) -> float:
        """Risk from unbalanced feedback loops"""
        if not self.feedback_loops:
            self.detect_feedback_loops()
        
        if not self.feedback_loops:
            return 0.0
        
        reinforcing = sum(1 for l in self.feedback_loops if l.loop_type == FeedbackLoopType.REINFORCING)
        total = len(self.feedback_loops)
        
        return reinforcing / total if total > 0 else 0.0
    
    def get_blockers(self) -> List[MechanismNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.BLOCKER]
    
    def get_assumptions(self) -> List[MechanismNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.ASSUMPTION]
    
    def get_critical_path(self) -> List[str]:
        """Get nodes with lowest confidence"""
        if not self.nodes:
            return []
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.epistemic_state.credence)
        return [n.id for n in sorted_nodes[:5]]
    
    def to_dict(self) -> Dict:
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges],
            'feedback_loops': [fl.to_dict() for fl in self.feedback_loops],
            'overall_confidence': round(self.overall_confidence(), 3),
            'average_causal_strength': round(self.average_causal_strength(), 3),
            'systemic_risk': round(self.systemic_risk_score(), 3),
            'node_count': len(self.nodes),
            'edge_count': len(self.edges)
        }


# =============================================================================
# UTILITY MODEL - P0 FIX: Risk Aversion
# =============================================================================

@dataclass
class Scenario:
    """An outcome scenario"""
    description: str
    probability: float
    utility: float  # Can be negative


class RiskAwareUtilityModel:
    """
    Utility model with proper risk aversion (CRRA).
    
    FIXES the risk-neutral assumption vulnerability.
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        """
        risk_aversion (γ):
            0 = Risk neutral
            1 = Moderate (log utility, default)
            2 = High risk aversion
            3+ = Extreme risk aversion
        """
        self.risk_aversion = risk_aversion
        self.scenarios: List[Scenario] = []
        self.discount_rate: float = 0.1
        self.time_horizon_years: float = 1.0
    
    def add_scenario(self, description: str, probability: float, utility: float):
        """Add outcome scenario (probabilities should sum to ~1)"""
        self.scenarios.append(Scenario(description, probability, utility))
    
    def expected_utility(self) -> float:
        """Simple expected utility (risk-neutral)"""
        if not self.scenarios:
            return 0.0
        return sum(s.probability * s.utility for s in self.scenarios)
    
    def variance(self) -> float:
        """Variance of utility outcomes"""
        if not self.scenarios:
            return 0.0
        eu = self.expected_utility()
        return sum(s.probability * (s.utility - eu) ** 2 for s in self.scenarios)
    
    def certainty_equivalent(self) -> float:
        """
        Risk-adjusted value using CRRA utility.
        
        CRRA: U(x) = x^(1-γ) / (1-γ) for γ ≠ 1
              U(x) = ln(x) for γ = 1
        """
        γ = self.risk_aversion
        
        if γ == 0:
            return self.expected_utility()
        
        # Shift utilities to be positive for CRRA
        min_util = min(s.utility for s in self.scenarios) if self.scenarios else 0
        shift = abs(min_util) + 1 if min_util <= 0 else 0
        
        if γ == 1:
            # Log utility
            eu_log = sum(s.probability * np.log(s.utility + shift) 
                        for s in self.scenarios)
            return np.exp(eu_log) - shift
        
        # CRRA
        try:
            eu_crra = sum(s.probability * ((s.utility + shift) ** (1 - γ)) / (1 - γ)
                         for s in self.scenarios)
            ce = (eu_crra * (1 - γ)) ** (1 / (1 - γ)) - shift
            return ce
        except:
            return self.expected_utility()  # Fallback
    
    def risk_premium(self) -> float:
        """How much utility sacrificed due to risk aversion"""
        return self.expected_utility() - self.certainty_equivalent()
    
    def value_of_perfect_information(self) -> float:
        """Maximum value of knowing true outcome before deciding"""
        if not self.scenarios:
            return 0.0
        
        # With perfect info, always choose best outcome
        best_outcome = max(s.utility for s in self.scenarios)
        current_eu = self.expected_utility()
        
        return max(0, best_outcome - current_eu)
    
    def value_of_information(self, signal_accuracy: float = 0.8,
                            information_cost: float = 0.0,
                            time_delay_cost: float = 0.0) -> Dict:
        """
        REALISTIC VOI accounting for imperfect, costly, delayed information.
        
        FIXES the VOI manipulation vulnerability.
        """
        vopi = self.value_of_perfect_information()
        
        # Adjust for imperfect information
        realistic_voi = vopi * signal_accuracy
        
        # Subtract costs
        net_voi = realistic_voi - information_cost - time_delay_cost
        
        should_gather = net_voi > 0
        
        return {
            'raw_voi': round(vopi, 4),
            'signal_accuracy': signal_accuracy,
            'realistic_voi': round(realistic_voi, 4),
            'information_cost': information_cost,
            'time_delay_cost': time_delay_cost,
            'net_voi': round(net_voi, 4),
            'recommendation': 'gather_info' if should_gather else 'decide_now',
            'explanation': f"Net VOI = {net_voi:.3f}. " + 
                          ("Gathering more info appears worthwhile." if should_gather 
                           else "Decide with current information.")
        }
    
    def should_gather_more_info(self, info_cost: float, 
                               signal_accuracy: float = 0.8,
                               time_cost: float = 0.0) -> Tuple[bool, str]:
        """Should we gather more information?"""
        voi_result = self.value_of_information(signal_accuracy, info_cost, time_cost)
        return voi_result['net_voi'] > 0, voi_result['explanation']
    
    def to_dict(self) -> Dict:
        return {
            'scenarios': [{'desc': s.description, 'prob': s.probability, 'util': s.utility}
                         for s in self.scenarios],
            'expected_utility': round(self.expected_utility(), 4),
            'certainty_equivalent': round(self.certainty_equivalent(), 4),
            'risk_premium': round(self.risk_premium(), 4),
            'risk_aversion': self.risk_aversion,
            'voi': self.value_of_information()
        }


# =============================================================================
# BIAS DETECTION - P0 FIX: Don't Penalize Established Facts
# =============================================================================

@dataclass
class BiasCheck:
    """Result of a cognitive bias check"""
    bias_type: BiasType
    detected: bool
    evidence: str
    severity: float
    mitigation: str
    is_acknowledged: bool = False  # User can acknowledge and proceed
    
    def to_dict(self) -> Dict:
        return {
            'type': self.bias_type.value,
            'detected': self.detected,
            'evidence': self.evidence,
            'severity': self.severity,
            'mitigation': self.mitigation,
            'acknowledged': self.is_acknowledged
        }


class ImprovedBiasDetector:
    """
    Cognitive bias detection with nuance.
    
    FIXES the bias detector paradox - doesn't penalize established truths.
    """
    
    def __init__(self, element: 'AnalysisElement'):
        self.element = element
        self.checks: List[BiasCheck] = []
        self.hypothesis_is_established: bool = False  # User can flag
    
    def set_established_hypothesis(self, is_established: bool):
        """Flag if hypothesis is already well-established (like smoking-cancer)"""
        self.hypothesis_is_established = is_established
    
    def check_confirmation_bias(self) -> BiasCheck:
        """
        Check for confirmation bias - BUT handle established facts properly.
        """
        if not self.element.evidence:
            return BiasCheck(
                BiasType.CONFIRMATION, False, "No evidence to check", 0.0, ""
            )
        
        supporting = sum(1 for e in self.element.evidence if e.supports_hypothesis)
        contradicting = len(self.element.evidence) - supporting
        
        # If hypothesis is flagged as established, don't penalize
        if self.hypothesis_is_established:
            return BiasCheck(
                BiasType.CONFIRMATION,
                detected=False,
                evidence=f"Hypothesis marked as established. {supporting} supporting, {contradicting} contradicting.",
                severity=0.0,
                mitigation="Established hypothesis - evidence imbalance is expected."
            )
        
        # For novel hypotheses, check balance
        detected = contradicting == 0 and supporting >= 3
        
        return BiasCheck(
            BiasType.CONFIRMATION,
            detected=detected,
            evidence=f"{supporting} supporting, {contradicting} contradicting" if detected else "",
            severity=0.5 if detected else 0.0,  # Reduced from 0.7 - it's a flag, not condemnation
            mitigation="Pre-mortem: List 3 specific ways this hypothesis could fail. "
                      "If this is an established fact, use set_established_hypothesis(True)."
        )
    
    def check_overconfidence(self) -> BiasCheck:
        """Check for high confidence with limited evidence"""
        high_conf_dims = []
        
        for name, dim in self.element.scoring.dimensions.items():
            if dim.value > 0.85 and len(self.element.evidence) < 5:
                high_conf_dims.append(name)
        
        # Also check overall credence
        credence_issue = self.element.epistemic_state.credence > 0.9 and len(self.element.evidence) < 5
        
        detected = len(high_conf_dims) > 0 or credence_issue
        
        return BiasCheck(
            BiasType.OVERCONFIDENCE,
            detected=detected,
            evidence=f"High confidence ({high_conf_dims}) with <5 evidence pieces" if detected else "",
            severity=0.4 if detected else 0.0,
            mitigation="Consider: Would you bet significant resources at these odds? "
                      "If uncomfortable, lower confidence estimates."
        )
    
    def check_planning_fallacy(self) -> BiasCheck:
        """Check for optimistic timeline without reference class"""
        timeline_dim = self.element.scoring.dimensions.get('timeline_realism')
        
        detected = timeline_dim and timeline_dim.value > 0.8
        
        return BiasCheck(
            BiasType.PLANNING_FALLACY,
            detected=detected,
            evidence="High timeline confidence without reference class" if detected else "",
            severity=0.4 if detected else 0.0,
            mitigation="Reference class forecasting: Find 5+ similar projects and their actual timelines. "
                      "Typical projects take 2-3x estimated time."
        )
    
    def check_base_rate_neglect(self) -> BiasCheck:
        """Check if base rates are considered"""
        base_rate_keywords = ['base rate', 'prior', 'typically', 'usually', 
                            'on average', 'historically', 'base case']
        
        has_base_rate = any(
            any(kw in e.content.lower() for kw in base_rate_keywords)
            for e in self.element.evidence
        )
        
        detected = not has_base_rate and len(self.element.evidence) > 2
        
        return BiasCheck(
            BiasType.BASE_RATE_NEGLECT,
            detected=detected,
            evidence="No base rate or prior probability referenced" if detected else "",
            severity=0.3 if detected else 0.0,
            mitigation="What % of similar hypotheses turn out to be true? "
                      "Anchor your prior on this base rate."
        )
    
    def check_sunk_cost(self) -> BiasCheck:
        """Check for sunk cost indicators in evidence"""
        sunk_cost_keywords = ['already invested', 'spent', "can't go back", 
                            'too far', 'committed']
        
        sunk_cost_evidence = [
            e for e in self.element.evidence
            if any(kw in e.content.lower() for kw in sunk_cost_keywords)
        ]
        
        detected = len(sunk_cost_evidence) > 0
        
        return BiasCheck(
            BiasType.SUNK_COST,
            detected=detected,
            evidence=f"Sunk cost language in evidence: {[e.id for e in sunk_cost_evidence]}" if detected else "",
            severity=0.5 if detected else 0.0,
            mitigation="Evaluate: Would you make this decision fresh, ignoring past investments?"
        )
    
    def run_all_checks(self) -> List[BiasCheck]:
        """Run all bias checks"""
        self.checks = [
            self.check_confirmation_bias(),
            self.check_overconfidence(),
            self.check_planning_fallacy(),
            self.check_base_rate_neglect(),
            self.check_sunk_cost(),
        ]
        return self.checks
    
    def total_bias_penalty(self) -> float:
        """Calculate penalty from detected, unacknowledged biases"""
        return sum(
            check.severity * 0.1 
            for check in self.checks 
            if check.detected and not check.is_acknowledged
        )
    
    def get_debiased_score(self, original_score: float) -> float:
        """Apply debiasing (but don't over-penalize)"""
        penalty = self.total_bias_penalty()
        # Cap penalty at 20% reduction
        penalty = min(0.2, penalty)
        return max(0.0, min(1.0, original_score - penalty))
    
    def acknowledge_bias(self, bias_type: BiasType):
        """User acknowledges a bias (has considered it)"""
        for check in self.checks:
            if check.bias_type == bias_type:
                check.is_acknowledged = True


# =============================================================================
# CALIBRATION TRACKER - P0 FIX: Cold Start Handling
# =============================================================================

@dataclass
class CalibrationRecord:
    """A prediction and its outcome"""
    hypothesis_id: str
    predicted_score: float
    actual_outcome: Optional[bool] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CalibrationTracker:
    """
    Track calibration over time.
    
    FIXES: Explicit uncertainty when uncalibrated (cold start).
    """
    
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.records: List[CalibrationRecord] = []
        self.warning_system = warning_system
    
    def add_prediction(self, hypothesis_id: str, score: float):
        """Record a new prediction"""
        self.records.append(CalibrationRecord(hypothesis_id, score))
    
    def record_outcome(self, hypothesis_id: str, actual: bool):
        """Record actual outcome"""
        for record in reversed(self.records):
            if record.hypothesis_id == hypothesis_id and record.actual_outcome is None:
                record.actual_outcome = actual
                break
    
    def get_completed_records(self) -> List[CalibrationRecord]:
        """Get records with known outcomes"""
        return [r for r in self.records if r.actual_outcome is not None]
    
    def is_sufficiently_calibrated(self) -> bool:
        """Check if we have enough data for reliable calibration"""
        return len(self.get_completed_records()) >= SafetyLimits.MIN_CALIBRATION_POINTS
    
    def expected_calibration_error(self) -> Tuple[float, bool]:
        """
        Calculate ECE.
        Returns (ece, is_reliable).
        """
        completed = self.get_completed_records()
        
        if len(completed) < SafetyLimits.MIN_CALIBRATION_POINTS:
            if self.warning_system:
                self.warning_system.check_calibration_sufficiency(len(completed))
            return 0.5, False  # Return 0.5 (uncertain) with unreliable flag
        
        # Bin into 10 buckets
        bins = [[] for _ in range(10)]
        for r in completed:
            bin_idx = min(9, int(r.predicted_score * 10))
            bins[bin_idx].append(r.actual_outcome)
        
        ece = 0.0
        total = len(completed)
        
        for i, bin_outcomes in enumerate(bins):
            if not bin_outcomes:
                continue
            
            bin_confidence = (i + 0.5) / 10
            bin_accuracy = sum(bin_outcomes) / len(bin_outcomes)
            bin_weight = len(bin_outcomes) / total
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece, True
    
    def is_overconfident(self) -> bool:
        """Check systematic overconfidence"""
        completed = self.get_completed_records()
        if len(completed) < SafetyLimits.MIN_CALIBRATION_POINTS:
            return False  # Can't tell
        
        avg_score = np.mean([r.predicted_score for r in completed])
        avg_accuracy = np.mean([r.actual_outcome for r in completed])
        
        return avg_score > avg_accuracy + 0.1
    
    def get_calibrated_score(self, raw_score: float) -> Tuple[float, str]:
        """
        Apply calibration adjustment.
        Returns (calibrated_score, explanation).
        """
        ece, is_reliable = self.expected_calibration_error()
        
        if not is_reliable:
            return raw_score, "Uncalibrated (insufficient historical data)"
        
        if ece > 0.15 and self.is_overconfident():
            # Shrink toward 0.5
            calibrated = raw_score * 0.85 + 0.075
            return calibrated, f"Shrunk toward 0.5 (historical overconfidence, ECE={ece:.2f})"
        elif ece > 0.15:
            calibrated = raw_score * 0.95 + 0.025
            return calibrated, f"Minor adjustment (ECE={ece:.2f})"
        
        return raw_score, f"Well calibrated (ECE={ece:.2f})"
    
    def to_dict(self) -> Dict:
        ece, reliable = self.expected_calibration_error()
        return {
            'total_predictions': len(self.records),
            'completed': len(self.get_completed_records()),
            'is_sufficiently_calibrated': self.is_sufficiently_calibrated(),
            'ece': round(ece, 3),
            'ece_reliable': reliable,
            'is_overconfident': self.is_overconfident() if reliable else None
        }


# =============================================================================
# SCORING SYSTEM - P0 FIX: Anti-Gaming
# =============================================================================

@dataclass
class DimensionScore:
    """A single dimension score"""
    name: str
    value: float
    weight: float
    is_fatal_below: float = 0.3
    rationale: str = ""
    uncertainty: float = 0.3
    
    @property
    def is_fatal_flaw(self) -> bool:
        return self.value < self.is_fatal_below
    
    def confidence_interval(self) -> Tuple[float, float]:
        lower = max(0, self.value - self.uncertainty)
        upper = min(1, self.value + self.uncertainty)
        return (round(lower, 3), round(upper, 3))


class ScoringSystem:
    """
    Scoring with anti-gaming measures.
    
    FIXES: Weight reasonableness checks.
    """
    
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.dimensions: Dict[str, DimensionScore] = {}
        self.warning_system = warning_system
        self.tracer: Optional[AnalysisTracer] = None
    
    def set_dimension(self, name: str, value: float, weight: float = 1.0,
                     fatal_threshold: float = 0.3, rationale: str = "",
                     uncertainty: float = 0.3):
        """Set a dimension with anti-gaming checks"""
        # Clamp value
        clamped_value = max(0.001, min(0.999, value))
        
        # Cap weight
        capped_weight = min(SafetyLimits.MAX_SINGLE_WEIGHT, max(0.1, weight))
        
        if weight != capped_weight and self.warning_system:
            self.warning_system.add_warning(
                WarningLevel.INFO,
                "Weight Adjustment",
                f"Weight for '{name}' capped from {weight} to {capped_weight}",
                "Extreme weights can be gamed."
            )
        
        self.dimensions[name] = DimensionScore(
            name=name,
            value=clamped_value,
            weight=capped_weight,
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
        
        # Check weight gaming
        if self.warning_system:
            weights = {n: d.weight for n, d in self.dimensions.items()}
            self.warning_system.check_weight_gaming(weights)
    
    def additive_score(self) -> float:
        """Weighted average (for comparison)"""
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions.values())
        if total_weight == 0:
            return 0.0
        return sum(d.value * d.weight for d in self.dimensions.values()) / total_weight
    
    def bayesian_score(self) -> float:
        """Bayesian combination via log-odds"""
        if not self.dimensions:
            return 0.0
        
        total_log_odds = 0.0
        total_weight = 0.0
        
        for dim in self.dimensions.values():
            p = max(0.001, min(0.999, dim.value))
            log_odds = np.log(p / (1 - p))
            total_log_odds += log_odds * dim.weight
            total_weight += dim.weight
        
        if total_weight == 0:
            return 0.5
        
        avg_log_odds = total_log_odds / total_weight
        return float(1 / (1 + np.exp(-avg_log_odds)))
    
    def multiplicative_score(self) -> float:
        """Geometric mean (weakest link emphasis)"""
        if not self.dimensions:
            return 0.0
        
        values = [max(0.001, d.value) for d in self.dimensions.values()]
        return float(np.exp(np.mean(np.log(values))))
    
    def fatal_flaws(self) -> List[DimensionScore]:
        """Get all fatal flaws"""
        return [d for d in self.dimensions.values() if d.is_fatal_flaw]
    
    def combined_score(self) -> Tuple[float, str]:
        """Combined score with fatal flaw override"""
        flaws = self.fatal_flaws()
        bayesian = self.bayesian_score()
        
        if flaws:
            score = min(0.3, bayesian * 0.5)
            reason = f"Fatal flaw in: {', '.join(f.name for f in flaws)}"
        else:
            score = bayesian
            reason = "No fatal flaws"
        
        return round(score, 4), reason
    
    def sensitivity_analysis(self, perturbations: List[float] = None) -> List[Dict]:
        """
        Comprehensive sensitivity analysis.
        
        FIXES: Tests multiple perturbation magnitudes.
        """
        if perturbations is None:
            perturbations = [-0.2, -0.1, 0.1, 0.2]
        
        if not self.dimensions:
            return []
        
        base_score, _ = self.combined_score()
        sensitivities = []
        
        for name, dim in self.dimensions.items():
            original = dim.value
            impacts = []
            
            for delta in perturbations:
                dim.value = max(0.001, min(0.999, original + delta))
                new_score, _ = self.combined_score()
                impacts.append({
                    'delta': delta,
                    'new_score': round(new_score, 4),
                    'change': round(new_score - base_score, 4)
                })
            
            dim.value = original  # Restore
            
            max_impact = max(abs(i['change']) for i in impacts)
            
            sensitivities.append({
                'dimension': name,
                'current_value': original,
                'max_impact': round(max_impact, 4),
                'is_critical': max_impact > 0.1,
                'is_fatal': dim.is_fatal_flaw,
                'impacts': impacts
            })
        
        return sorted(sensitivities, key=lambda x: x['max_impact'], reverse=True)
    
    def to_dict(self) -> Dict:
        return {
            'dimensions': {n: {'value': d.value, 'weight': d.weight, 
                              'is_fatal': d.is_fatal_flaw, 'ci': d.confidence_interval()}
                          for n, d in self.dimensions.items()},
            'bayesian_score': round(self.bayesian_score(), 4),
            'additive_score': round(self.additive_score(), 4),
            'multiplicative_score': round(self.multiplicative_score(), 4),
            'fatal_flaws': [f.name for f in self.fatal_flaws()]
        }


# =============================================================================
# CONTENT-BASED FATAL FLAW DETECTION - P0 FIX
# =============================================================================

class ContentFatalFlawDetector:
    """
    Scan evidence content for fatal issues that numeric scores might miss.
    
    FIXES: Fatal flaw bypass vulnerability.
    """
    
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.warning_system = warning_system
        self.detected_issues: List[Dict] = []
    
    def scan_evidence(self, evidence_list: List[Evidence]) -> List[Dict]:
        """Scan all evidence for fatal content"""
        self.detected_issues = []
        
        for evidence in evidence_list:
            if evidence.has_fatal_content():
                for pattern, category in evidence.fatal_content_flags:
                    issue = {
                        'evidence_id': evidence.id,
                        'category': category,
                        'pattern': pattern,
                        'content_snippet': evidence.content[:100],
                        'supports_hypothesis': evidence.supports_hypothesis
                    }
                    self.detected_issues.append(issue)
        
        # Issue warnings
        if self.detected_issues and self.warning_system:
            categories = set(i['category'] for i in self.detected_issues)
            self.warning_system.add_warning(
                WarningLevel.CRITICAL,
                "Content Fatal Flags",
                f"Found {len(self.detected_issues)} potential fatal issues in categories: {categories}",
                "Review flagged evidence manually. These issues may not be captured in numeric scores."
            )
        
        return self.detected_issues
    
    def has_fatal_content(self) -> bool:
        return len(self.detected_issues) > 0
    
    def get_issues_by_category(self, category: str) -> List[Dict]:
        return [i for i in self.detected_issues if i['category'] == category]


# =============================================================================
# MAIN ANALYSIS ELEMENT
# =============================================================================

@dataclass
class AnalysisElement:
    """
    The core analysis container with all safety improvements.
    """
    name: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    
    # Foundation
    what: Optional[str] = None
    why: Optional[str] = None
    how: Optional[str] = None
    measure: Optional[str] = None
    
    # Core components
    epistemic_state: EpistemicState = field(default_factory=EpistemicState)
    mechanism_map: MechanismMap = field(default_factory=MechanismMap)
    scoring: ScoringSystem = field(default_factory=ScoringSystem)
    evidence: List[Evidence] = field(default_factory=list)
    
    # Enhanced components
    utility_model: RiskAwareUtilityModel = field(default_factory=RiskAwareUtilityModel)
    bias_detector: Optional[ImprovedBiasDetector] = None
    calibration_tracker: CalibrationTracker = field(default_factory=CalibrationTracker)
    
    # Safety systems
    warning_system: WarningSystem = field(default_factory=WarningSystem)
    content_scanner: Optional[ContentFatalFlawDetector] = None
    
    # Results
    bias_checks: List[BiasCheck] = field(default_factory=list)
    feedback_loops: List[FeedbackLoop] = field(default_factory=list)
    independence_report: Optional[Dict] = None
    
    # Flags
    is_established_hypothesis: bool = False
    is_high_stakes: bool = True
    
    tracer: Optional[AnalysisTracer] = None
    
    def __post_init__(self):
        # Initialize with warning system
        self.mechanism_map = MechanismMap(self.warning_system)
        self.scoring = ScoringSystem(self.warning_system)
        self.calibration_tracker = CalibrationTracker(self.warning_system)
        self.content_scanner = ContentFatalFlawDetector(self.warning_system)
        self.epistemic_state.warning_system = self.warning_system
    
    def set_tracer(self, tracer: AnalysisTracer):
        self.tracer = tracer
        self.mechanism_map.tracer = tracer
        self.scoring.tracer = tracer
    
    def set_established_hypothesis(self, is_established: bool):
        """Mark this as an established fact (like smoking-cancer)"""
        self.is_established_hypothesis = is_established
        if self.bias_detector:
            self.bias_detector.set_established_hypothesis(is_established)
    
    def set_high_stakes(self, is_high_stakes: bool):
        """Set whether this is a high-stakes decision"""
        self.is_high_stakes = is_high_stakes
    
    def set_what(self, value: str, confidence: float):
        self.what = value
        self.scoring.set_dimension('definition_clarity', confidence, weight=1.0)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'WHAT', 'confidence': confidence})
    
    def set_why(self, value: str, confidence: float):
        self.why = value
        self.scoring.set_dimension('justification_strength', confidence, weight=1.2)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'WHY', 'confidence': confidence})
    
    def set_how(self, value: str, confidence: float):
        self.how = value
        self.scoring.set_dimension('mechanism_validity', confidence, weight=1.5, 
                                   fatal_threshold=0.25)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'HOW', 'confidence': confidence})
    
    def set_measure(self, value: str, confidence: float):
        self.measure = value
        self.scoring.set_dimension('measurability', confidence, weight=1.0)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                           {'dimension': 'MEASURE', 'confidence': confidence})
    
    def set_dimension(self, name: str, value: float, weight: float = 1.0,
                     is_fatal_below: float = 0.3):
        """Set a custom dimension"""
        self.scoring.set_dimension(name, value, weight, is_fatal_below)
    
    def add_evidence(self, evidence: Evidence) -> bool:
        """Add evidence with full safety checks"""
        # Check evidence limit
        if len(self.evidence) >= SafetyLimits.MAX_EVIDENCE_PIECES:
            self.warning_system.add_warning(
                WarningLevel.CRITICAL,
                "Evidence Limit",
                f"Cannot add evidence: limit of {SafetyLimits.MAX_EVIDENCE_PIECES} reached",
                "Review existing evidence for redundancy"
            )
            return False
        
        # Bayesian update
        lr = evidence.get_likelihood_ratio()
        old_credence = self.epistemic_state.credence
        change, was_clamped = self.epistemic_state.update_with_evidence(lr, evidence.id)
        
        # Calculate bits
        evidence.calculate_bits(old_credence, self.epistemic_state.credence)
        
        self.evidence.append(evidence)
        
        # Update evidence quality dimension
        qualities = [e.get_quality() for e in self.evidence]
        n = len(self.evidence)
        effective_quality = np.mean(qualities) * (1 - 0.5 * np.exp(-n/3))
        self.scoring.set_dimension('evidence_quality', effective_quality, weight=1.0)
        
        # Check evidence balance
        supporting = sum(1 for e in self.evidence if e.supports_hypothesis)
        contradicting = len(self.evidence) - supporting
        self.warning_system.check_evidence_balance(supporting, contradicting)
        
        # Scan for fatal content
        if evidence.has_fatal_content():
            self.content_scanner.scan_evidence([evidence])
        
        if self.tracer:
            self.tracer.log(TraceEventType.EVIDENCE_ADDED, "L1", {
                'id': evidence.id,
                'quality': evidence.get_quality(),
                'bits': evidence.bits,
                'causal_level': evidence.causal_level.name,
                'new_credence': self.epistemic_state.credence
            })
        
        return True
    
    def add_mechanism_node(self, node: MechanismNode) -> str:
        return self.mechanism_map.add_node(node)
    
    def add_mechanism_edge(self, edge: MechanismEdge) -> bool:
        return self.mechanism_map.add_edge(edge)
    
    def set_feasibility(self, technical: float, economic: float, timeline: float):
        self.scoring.set_dimension('technical_feasibility', technical, weight=1.2, 
                                   fatal_threshold=0.2)
        self.scoring.set_dimension('economic_viability', economic, weight=1.0,
                                   fatal_threshold=0.2)
        self.scoring.set_dimension('timeline_realism', timeline, weight=0.8)
    
    def set_risk(self, execution_risk: float, external_risk: float):
        self.scoring.set_dimension('execution_safety', 1.0 - execution_risk, weight=1.0)
        self.scoring.set_dimension('external_resilience', 1.0 - external_risk, weight=0.8)
    
    def add_scenario(self, description: str, probability: float, utility: float):
        """Add outcome scenario"""
        self.utility_model.add_scenario(description, probability, utility)
    
    def set_risk_aversion(self, gamma: float):
        """Set risk aversion parameter"""
        self.utility_model.risk_aversion = gamma
    
    def check_evidence_independence(self) -> Dict:
        """Run independence check on all evidence"""
        self.independence_report = EvidenceIndependenceChecker.check_all_independence(
            self.evidence, self.warning_system
        )
        return self.independence_report
    
    def run_bias_detection(self) -> List[BiasCheck]:
        """Run cognitive bias detection"""
        self.bias_detector = ImprovedBiasDetector(self)
        self.bias_detector.set_established_hypothesis(self.is_established_hypothesis)
        self.bias_checks = self.bias_detector.run_all_checks()
        
        if self.tracer:
            for check in self.bias_checks:
                if check.detected:
                    self.tracer.log(TraceEventType.BIAS_DETECTED, "L2", check.to_dict())
        
        return self.bias_checks
    
    def scan_for_fatal_content(self) -> List[Dict]:
        """Scan evidence for fatal content flags"""
        return self.content_scanner.scan_evidence(self.evidence)
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
        """Detect feedback loops in mechanism map"""
        self.feedback_loops = self.mechanism_map.detect_feedback_loops()
        return self.feedback_loops
    
    def get_total_evidence_bits(self) -> float:
        """Total information bits from evidence"""
        return sum(e.bits for e in self.evidence)
    
    def get_effective_evidence_bits(self) -> float:
        """Evidence bits adjusted for independence"""
        if not self.independence_report:
            self.check_evidence_independence()
        
        overall_independence = self.independence_report.get('overall_independence', 1.0)
        return self.get_total_evidence_bits() * overall_independence
    
    def value_of_information(self, info_cost: float = 0.0, 
                            signal_accuracy: float = 0.8,
                            time_cost: float = 0.0) -> Dict:
        """Calculate realistic VOI"""
        voi_result = self.utility_model.value_of_information(
            signal_accuracy, info_cost, time_cost
        )
        
        # Check reasonableness
        eu = self.utility_model.expected_utility()
        self.warning_system.check_voi_reasonableness(voi_result['net_voi'], eu)
        
        if self.tracer:
            self.tracer.log(TraceEventType.VOI_CALCULATED, "L4", voi_result)
        
        return voi_result
    
    def pre_flight_check(self) -> Tuple[bool, List[str]]:
        """Run all safety checks before analysis"""
        issues = []
        
        # Evidence sufficiency
        if not self.warning_system.check_evidence_sufficiency(
            len(self.evidence), self.is_high_stakes
        ):
            issues.append("Insufficient evidence")
        
        # Run independence check
        independence = self.check_evidence_independence()
        if independence['overall_independence'] < 0.5:
            issues.append(f"Low evidence independence ({independence['overall_independence']:.1%})")
        
        # Scan for fatal content
        fatal_content = self.scan_for_fatal_content()
        if fatal_content:
            issues.append(f"Fatal content flags in {len(fatal_content)} evidence pieces")
        
        # Check mechanism complexity
        self.warning_system.check_mechanism_complexity(
            len(self.mechanism_map.nodes),
            len(self.mechanism_map.edges)
        )
        
        return len(issues) == 0, issues


# =============================================================================
# ADVERSARIAL TESTER
# =============================================================================

@dataclass
class Criticism:
    """A criticism of the analysis"""
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
    """Generate criticisms of the analysis"""
    
    def __init__(self, element: AnalysisElement, rigor: int, tracer: AnalysisTracer):
        self.element = element
        self.rigor = rigor
        self.tracer = tracer
        self.criticisms: List[Criticism] = []
        self.iteration = 0
    
    def generate_criticisms(self) -> List[Criticism]:
        """Generate criticisms based on current state"""
        new = []
        self.iteration += 1
        self.tracer.log(TraceEventType.ITERATION_START, "L2", {'iteration': self.iteration})
        
        # Check assumptions
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
        if len(self.element.evidence) < 3:
            c = Criticism(
                f"Insufficient evidence ({len(self.element.evidence)} pieces)",
                severity=0.6, cycle="evidence_check", dimension_affected="evidence_quality"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check weak causal links
        weak_edges = [e for e in self.element.mechanism_map.edges if e.causal_strength() < 0.3]
        for edge in weak_edges[:3]:
            c = Criticism(
                f"Weak causal link: {edge.source_id} -> {edge.target_id} "
                f"(strength: {edge.causal_strength():.2f})",
                severity=0.5, cycle="causal_analysis", dimension_affected="mechanism_validity"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check bias detections
        for bias_check in self.element.bias_checks:
            if bias_check.detected and not bias_check.is_acknowledged and bias_check.severity >= 0.4:
                c = Criticism(
                    f"Cognitive bias: {bias_check.bias_type.value} - {bias_check.evidence}",
                    severity=bias_check.severity, cycle="bias_detection", 
                    dimension_affected="overall"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check content-based fatal flags
        if self.element.content_scanner.has_fatal_content():
            for issue in self.element.content_scanner.detected_issues[:3]:
                c = Criticism(
                    f"CONTENT ALERT ({issue['category']}): {issue['content_snippet'][:50]}...",
                    severity=0.9, cycle="content_scan", dimension_affected="overall"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", c.to_dict())
        
        # Check evidence independence
        if self.element.independence_report:
            if self.element.independence_report['overall_independence'] < 0.6:
                c = Criticism(
                    f"Low evidence independence ({self.element.independence_report['overall_independence']:.0%})",
                    severity=0.6, cycle="independence_check", dimension_affected="evidence_quality"
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
# OPTIMAL STOPPING
# =============================================================================

class OptimalStoppingCriterion:
    """Determine when to stop analysis"""
    
    def __init__(self, element: AnalysisElement, target: float, 
                 delay_cost: float = 0.05):
        self.element = element
        self.target = target
        self.delay_cost = delay_cost
    
    def should_stop(self) -> Tuple[bool, str]:
        """Check if analysis should stop"""
        credence = self.element.epistemic_state.credence
        
        # Check for extreme confidence (likely numerical artifact)
        if credence > SafetyLimits.CREDENCE_EXTREME_THRESHOLD:
            return True, f"Extreme confidence ({credence:.2f}) - likely numerical artifact"
        
        # Very low confidence - reject
        if credence < 0.2:
            return True, f"Low confidence ({credence:.2f}): Reject"
        
        # Fatal flaws present
        flaws = self.element.scoring.fatal_flaws()
        if flaws:
            return True, f"Fatal flaws detected: {[f.name for f in flaws]}"
        
        # Content-based fatal flags
        if self.element.content_scanner.has_fatal_content():
            return True, "Content-based fatal flags require human review"
        
        # Calculate VOI
        voi_result = self.element.utility_model.value_of_information()
        net_voi = voi_result['net_voi']
        
        if net_voi < self.delay_cost:
            return True, f"VOI ({net_voi:.3f}) < delay cost ({self.delay_cost:.3f}): Decide now"
        
        # Threshold reached
        if credence >= self.target:
            return True, f"Target reached ({credence:.2f} >= {self.target})"
        
        return False, f"Continue: credence={credence:.2f}, net_voi={net_voi:.3f}"


# =============================================================================
# HYPOTHESIS COMPARATOR
# =============================================================================

class HypothesisComparator:
    """Compare multiple hypotheses"""
    
    def __init__(self):
        self.hypotheses: Dict[str, AnalysisElement] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_hypothesis(self, element: AnalysisElement):
        self.hypotheses[element.name] = element
    
    def compare(self) -> Dict:
        """Compare all hypotheses"""
        comparison = {
            'hypotheses': [],
            'rankings': {},
            'best_choice': None,
            'summary': ""
        }
        
        for name, elem in self.hypotheses.items():
            score, reason = elem.scoring.combined_score()
            flaws = elem.scoring.fatal_flaws()
            
            hyp_data = {
                'name': name,
                'combined_score': round(score, 4),
                'bayesian_score': round(elem.scoring.bayesian_score(), 4),
                'credence': round(elem.epistemic_state.credence, 4),
                'confidence_interval': elem.epistemic_state.get_confidence_interval(),
                'fatal_flaws': [f.name for f in flaws],
                'expected_utility': round(elem.utility_model.expected_utility(), 4),
                'certainty_equivalent': round(elem.utility_model.certainty_equivalent(), 4),
                'evidence_count': len(elem.evidence),
                'has_critical_warnings': elem.warning_system.has_critical_warnings()
            }
            comparison['hypotheses'].append(hyp_data)
        
        # Rank by combined score (excluding those with fatal flaws)
        valid_hyps = [h for h in comparison['hypotheses'] if not h['fatal_flaws']]
        
        if valid_hyps:
            sorted_valid = sorted(valid_hyps, key=lambda x: x['combined_score'], reverse=True)
            for i, h in enumerate(sorted_valid):
                comparison['rankings'][h['name']] = i + 1
            comparison['best_choice'] = sorted_valid[0]['name']
        
        # Add rankings for flawed hypotheses at bottom
        flawed = [h for h in comparison['hypotheses'] if h['fatal_flaws']]
        start_rank = len(valid_hyps) + 1
        for i, h in enumerate(flawed):
            comparison['rankings'][h['name']] = start_rank + i
        
        return comparison


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_analysis(element: AnalysisElement, rigor_level: int = 2, 
                max_iter: int = 15) -> Dict:
    """
    Run comprehensive analysis with all safety features.
    
    Args:
        element: The analysis element to evaluate
        rigor_level: 1=Light, 2=Standard, 3=Deep
        max_iter: Maximum iterations (capped by SafetyLimits)
    
    Returns:
        Complete analysis results with warnings and recommendations
    """
    # Cap iterations
    max_iter = min(max_iter, SafetyLimits.MAX_ITERATIONS)
    
    tracer = AnalysisTracer()
    element.set_tracer(tracer)
    
    # Layer 0: Characterization
    tracer.log(TraceEventType.LAYER_ENTER, "L0", {'description': 'Problem Characterization'})
    rigor_names = {1: "Light", 2: "Standard", 3: "Deep"}
    targets = {1: 0.5, 2: 0.7, 3: 0.85}
    target = targets.get(rigor_level, 0.7)
    tracer.log(TraceEventType.DECISION, "L0", {
        'rigor': rigor_names.get(rigor_level, "Standard"), 
        'target': target
    })
    tracer.log(TraceEventType.LAYER_EXIT, "L0", {'result': f'Target: {target}'})
    
    # Pre-flight checks
    tracer.log(TraceEventType.LAYER_ENTER, "L0.5", {'description': 'Pre-flight Safety Checks'})
    passed, issues = element.pre_flight_check()
    tracer.log(TraceEventType.LAYER_EXIT, "L0.5", {
        'passed': passed,
        'issues': issues
    })
    
    # Layer 1: Foundation & Mechanism
    tracer.log(TraceEventType.LAYER_ENTER, "L1", {'description': 'Foundation & Mechanism Mapping'})
    element.detect_feedback_loops()
    mechanism_conf = element.mechanism_map.overall_confidence()
    causal_strength = element.mechanism_map.average_causal_strength()
    tracer.log(TraceEventType.LAYER_EXIT, "L1", {
        'mechanism_confidence': mechanism_conf,
        'causal_strength': causal_strength
    })
    
    # Layer 2: Bias Detection & Testing
    tracer.log(TraceEventType.LAYER_ENTER, "L2", {'description': 'Bias Detection & Adversarial Testing'})
    element.run_bias_detection()
    
    tester = AdversarialTester(element, rigor_level, tracer)
    
    history = []
    reason = "Max iterations"
    stopping = OptimalStoppingCriterion(element, target)
    
    for i in range(max_iter):
        tester.generate_criticisms()
        
        # Calculate scores
        bayesian = element.scoring.bayesian_score()
        combined, score_reason = element.scoring.combined_score()
        consistency = tester.consistency_score()
        
        # Apply calibration
        calibrated, cal_explanation = element.calibration_tracker.get_calibrated_score(combined)
        
        # Apply debiasing
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
            'calibration_note': cal_explanation,
            'debiased_score': debiased,
            'consistency': consistency,
            'quality': quality,
            'credence': element.epistemic_state.credence
        })
        
        tracer.iteration_data.append({
            'iteration': i + 1,
            'quality': quality,
            'bayesian': bayesian,
            'combined': combined,
            'calibrated': calibrated,
            'debiased': debiased,
            'credence': element.epistemic_state.credence
        })
        
        # Gate checks
        flaws = element.scoring.fatal_flaws()
        tracer.log(TraceEventType.GATE_CHECK, "L4", {
            'gate': 'quality', 'value': quality, 'threshold': target, 
            'passed': quality >= target
        })
        tracer.log(TraceEventType.GATE_CHECK, "L4", {
            'gate': 'fatal_flaws', 'value': len(flaws), 'threshold': 0, 
            'passed': len(flaws) == 0
        })
        
        # Check stopping
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
    
    # Layer 3: Sensitivity Analysis
    tracer.log(TraceEventType.LAYER_ENTER, "L3", {'description': 'Sensitivity Analysis'})
    sensitivity = element.scoring.sensitivity_analysis()
    tracer.sensitivity_data = sensitivity
    tracer.log(TraceEventType.LAYER_EXIT, "L3", {
        'result': f'Most sensitive: {sensitivity[0]["dimension"] if sensitivity else "N/A"}'
    })
    
    # Layer 5: Final Decision
    tracer.log(TraceEventType.LAYER_ENTER, "L5", {'description': 'Final Decision'})
    
    final_bayesian = element.scoring.bayesian_score()
    final_combined, final_reason = element.scoring.combined_score()
    final_calibrated, cal_note = element.calibration_tracker.get_calibrated_score(final_combined)
    
    if element.bias_detector:
        final_debiased = element.bias_detector.get_debiased_score(final_calibrated)
    else:
        final_debiased = final_calibrated
    
    flaws = element.scoring.fatal_flaws()
    
    # Determine readiness
    ready = final_debiased >= target and len(flaws) == 0
    has_critical_warnings = element.warning_system.has_critical_warnings()
    has_fatal_content = element.content_scanner.has_fatal_content()
    
    # Generate recommendation
    voi_result = element.value_of_information()
    net_voi = voi_result['net_voi']
    
    if has_fatal_content:
        recommendation = f"HUMAN REVIEW REQUIRED: Content flags detected"
        decision_state = DecisionReadiness.FATAL_FLAW
    elif len(flaws) > 0:
        recommendation = f"REJECT: Fatal flaws in {[f.name for f in flaws]}"
        decision_state = DecisionReadiness.FATAL_FLAW
    elif net_voi > 0.1:
        recommendation = f"INVESTIGATE: Net VOI ({net_voi:.2f}) suggests more info needed"
        decision_state = DecisionReadiness.NEEDS_MORE_INFO
    elif ready and not has_critical_warnings:
        recommendation = f"PROCEED: Score ({final_debiased:.2f}) meets threshold"
        decision_state = DecisionReadiness.READY
    elif ready and has_critical_warnings:
        recommendation = f"PROCEED WITH CAUTION: Score meets threshold but critical warnings exist"
        decision_state = DecisionReadiness.READY
    else:
        recommendation = f"UNCERTAIN: Score ({final_debiased:.2f}) below threshold"
        decision_state = DecisionReadiness.UNCERTAIN
    
    tracer.log(TraceEventType.DECISION, "L5", {
        'decision': recommendation,
        'decision_state': decision_state.value,
        'final_score': final_debiased,
        'credence': element.epistemic_state.credence
    })
    tracer.log(TraceEventType.LAYER_EXIT, "L5", {'result': 'Analysis complete'})
    
    # Compile results
    return {
        'name': element.name,
        
        # Core scores (presented as ranges, not false precision)
        'bayesian_score': round(final_bayesian, 3),
        'combined_score': round(final_combined, 3),
        'calibrated_score': round(final_calibrated, 3),
        'calibration_note': cal_note,
        'debiased_score': round(final_debiased, 3),
        
        # Epistemic state
        'credence': round(element.epistemic_state.credence, 3),
        'confidence_interval': element.epistemic_state.get_confidence_interval(),
        'epistemic_uncertainty': round(element.epistemic_state.epistemic_uncertainty, 3),
        'credence_warning': element.epistemic_state.get_point_estimate_warning(),
        
        # Decision outputs
        'ready': ready,
        'decision_state': decision_state.value,
        'recommendation': recommendation,
        'reason': reason,
        
        # Fatal flaws (numeric and content-based)
        'fatal_flaws': [{'name': f.name, 'value': f.value, 'threshold': f.is_fatal_below} 
                       for f in flaws],
        'content_fatal_flags': element.content_scanner.detected_issues,
        
        # Utility model
        'expected_utility': round(element.utility_model.expected_utility(), 3),
        'certainty_equivalent': round(element.utility_model.certainty_equivalent(), 3),
        'risk_premium': round(element.utility_model.risk_premium(), 3),
        'voi': voi_result,
        
        # Bias detection
        'biases_detected': [b.to_dict() for b in element.bias_checks if b.detected],
        'bias_penalty': round(element.bias_detector.total_bias_penalty(), 3) if element.bias_detector else 0,
        
        # Evidence
        'evidence_count': len(element.evidence),
        'total_evidence_bits': round(element.get_total_evidence_bits(), 2),
        'effective_evidence_bits': round(element.get_effective_evidence_bits(), 2),
        'independence_report': element.independence_report,
        
        # System dynamics
        'feedback_loops': [fl.to_dict() for fl in element.feedback_loops],
        'systemic_risk': round(element.mechanism_map.systemic_risk_score(), 3),
        
        # Causal analysis
        'mechanism_confidence': round(element.mechanism_map.overall_confidence(), 3),
        'average_causal_strength': round(element.mechanism_map.average_causal_strength(), 3),
        
        # Calibration
        'calibration': element.calibration_tracker.to_dict(),
        
        # Warnings (critical for user awareness)
        'warnings': element.warning_system.to_dict(),
        'has_critical_warnings': has_critical_warnings,
        
        # Analysis metadata
        'iterations': tester.iteration,
        'history': history,
        'criticisms': [c.to_dict() for c in tester.criticisms],
        'sensitivity': sensitivity,
        'mechanism_map': element.mechanism_map.to_dict(),
        'dimensions': element.scoring.to_dict(),
        'trace': tracer.get_trace(),
        'iteration_data': tracer.iteration_data
    }


# =============================================================================
# RESULT EXPLAINER
# =============================================================================

def explain_result(results: Dict) -> str:
    """Generate human-readable explanation of analysis results"""
    lines = []
    
    lines.append(f"ANALYSIS: {results['name']}")
    lines.append("=" * 60)
    
    # Decision
    lines.append(f"\n📋 RECOMMENDATION: {results['recommendation']}")
    lines.append(f"   Decision State: {results['decision_state']}")
    
    # Scores (as ranges)
    ci = results['confidence_interval']
    lines.append(f"\n📊 CONFIDENCE:")
    lines.append(f"   Credence: {results['credence']:.1%} (range: {ci[0]:.0%} - {ci[1]:.0%})")
    
    if results['credence_warning']:
        lines.append(f"   ⚠️  {results['credence_warning']}")
    
    # Fatal issues
    if results['fatal_flaws']:
        lines.append(f"\n💀 FATAL FLAWS:")
        for flaw in results['fatal_flaws']:
            lines.append(f"   • {flaw['name']}: {flaw['value']:.2f} (threshold: {flaw['threshold']})")
    
    if results['content_fatal_flags']:
        lines.append(f"\n🚨 CONTENT ALERTS:")
        for flag in results['content_fatal_flags'][:3]:
            lines.append(f"   • [{flag['category']}] {flag['content_snippet'][:50]}...")
    
    # Warnings
    warning_summary = results['warnings']['summary']
    if warning_summary.get('critical', 0) > 0 or warning_summary.get('warning', 0) > 0:
        lines.append(f"\n⚠️  WARNINGS:")
        for w in results['warnings']['warnings'][:5]:
            if w['level'] in ['critical', 'warning']:
                lines.append(f"   [{w['level'].upper()}] {w['message']}")
    
    # Biases
    if results['biases_detected']:
        lines.append(f"\n🧠 COGNITIVE BIASES DETECTED:")
        for bias in results['biases_detected']:
            lines.append(f"   • {bias['type']}: {bias['evidence']}")
    
    # Evidence
    lines.append(f"\n📚 EVIDENCE:")
    lines.append(f"   Count: {results['evidence_count']}")
    lines.append(f"   Total bits: {results['total_evidence_bits']:.1f}")
    lines.append(f"   Effective bits: {results['effective_evidence_bits']:.1f} (after independence adjustment)")
    
    if results['independence_report']:
        lines.append(f"   Independence: {results['independence_report']['overall_independence']:.0%}")
    
    # Utility
    lines.append(f"\n💰 DECISION THEORY:")
    lines.append(f"   Expected Utility: {results['expected_utility']:.3f}")
    lines.append(f"   Certainty Equivalent: {results['certainty_equivalent']:.3f}")
    lines.append(f"   Risk Premium: {results['risk_premium']:.3f}")
    lines.append(f"   Net VOI: {results['voi']['net_voi']:.3f} ({results['voi']['recommendation']})")
    
    # Sensitivity
    if results['sensitivity']:
        lines.append(f"\n🎚️  MOST SENSITIVE DIMENSIONS:")
        for dim in results['sensitivity'][:3]:
            status = "⚠️ CRITICAL" if dim['is_critical'] else ""
            lines.append(f"   • {dim['dimension']}: max impact {dim['max_impact']:.2f} {status}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate PRISM 1.0 capabilities"""
    print("=" * 70)
    print("PRISM v1.0 - Protocol for Rigorous Investigation of Scientific Mechanisms")
    print("=" * 70)
    print()
    
    # Create hypothesis
    h = AnalysisElement(
        name="Hire Data Scientist",
        domain=EvidenceDomain.BUSINESS
    )
    
    # Set foundation
    h.set_what("Full-time data scientist, $120K, to improve decision-making", 0.9)
    h.set_why("We make data-driven decisions but lack statistical expertise", 0.7)
    h.set_how("Post job → Interview → Hire → Onboard → Deliver insights", 0.8)
    h.set_measure("A/B test success rate improves by 20%", 0.7)
    
    # Add mechanism nodes
    n1 = MechanismNode("cause1", "Lack of statistics expertise", NodeType.CAUSE)
    n1.confidence = 0.9
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("mech1", "DS brings expertise", NodeType.MECHANISM)
    n2.confidence = 0.85
    h.add_mechanism_node(n2)
    
    n3 = MechanismNode("outcome1", "Improved decisions", NodeType.OUTCOME)
    n3.confidence = 0.7
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("blocker1", "Org resistance to data", NodeType.BLOCKER)
    n4.confidence = 0.4
    h.add_mechanism_node(n4)
    
    n5 = MechanismNode("assume1", "DS can integrate with team", NodeType.ASSUMPTION)
    n5.confidence = 0.6
    h.add_mechanism_node(n5)
    
    # Add edges with quality-first causal assessment
    h.add_mechanism_edge(MechanismEdge(
        "cause1", "mech1", EdgeType.CAUSES, 0.9,
        causal_level=CausalLevel.COUNTERFACTUAL,
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "mech1", "outcome1", EdgeType.ENABLES, 0.7,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.5
    ))
    h.add_mechanism_edge(MechanismEdge(
        "blocker1", "outcome1", EdgeType.PREVENTS, 0.4,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.3
    ))
    
    # Add evidence with full metadata for independence checking
    h.add_evidence(Evidence(
        id="ev1",
        content="HBR study: Companies with DS report 15% better outcomes",
        source="Harvard Business Review",
        quality=0.7,
        date="2023-03",
        domain=EvidenceDomain.BUSINESS,
        study_design="multi_company_analysis",
        sample_size=500,
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True,
        authors=["Smith, J.", "Johnson, K."]
    ))
    
    h.add_evidence(Evidence(
        id="ev2",
        content="Competitor hired DS, saw metric improvements after 6 months",
        source="Industry contact",
        quality=0.3,
        date="2024-01",
        domain=EvidenceDomain.BUSINESS,
        study_design="anecdote",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    h.add_evidence(Evidence(
        id="ev3",
        content="Internal survey: Team skeptical about DS integration",
        source="HR Department",
        quality=0.6,
        date="2024-02",
        domain=EvidenceDomain.BUSINESS,
        study_design="case_study",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=False  # Contradicting evidence!
    ))
    
    # Add utility scenarios with risk aversion
    h.set_risk_aversion(1.0)  # Moderate risk aversion
    h.add_scenario("Success: DS integrates well, major improvements", 0.5, 1.0)
    h.add_scenario("Partial: Some improvement, not transformative", 0.3, 0.4)
    h.add_scenario("Failure: DS doesn't fit, leaves within year", 0.2, -0.3)
    
    # Set feasibility
    h.set_feasibility(technical=0.9, economic=0.7, timeline=0.8)
    h.set_risk(execution_risk=0.3, external_risk=0.2)
    
    # Run analysis
    print("Running PRISM analysis...")
    print()
    results = run_analysis(h, rigor_level=2, max_iter=10)
    
    # Print human-readable explanation
    print(explain_result(results))
    
    # Print warnings
    print("\n📢 SYSTEM WARNINGS:")
    h.warning_system.print_warnings(WarningLevel.WARNING)
    
    return results


if __name__ == "__main__":
    results = demo()
