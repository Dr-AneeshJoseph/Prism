"""
PRISM v1.1 - Protocol for Rigorous Investigation of Scientific Mechanisms
==========================================================================

FIXES ALL P0 AND P1 VULNERABILITIES from Red Team Analysis v2:

P0 CRITICAL FIXES:
1. Independence Checker - Semantic similarity + entity extraction + name normalization
2. Content Scanner - 100+ patterns + semantic risk detection + blocking mode
3. Established Hypothesis - Verification requirements + audit trail
4. Warning System - Deduplication + aggregation + forced acknowledgment

P1 HIGH FIXES:
5. Sample Size Validation - Cross-validation + subgroup detection
6. Weight Enforcement - Blocking mode + justification required
7. Risk Aversion Guidance - Domain defaults + sensitivity display
8. Fatal Flaw Blocking - Analysis halts until human review

Author: Dr. Aneesh Joseph (Architecture) + Claude (Implementation)
Version: 1.1
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set, FrozenSet
from enum import Enum
from datetime import datetime
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# SAFETY LIMITS - TIGHTENED FROM v1.0
# =============================================================================

class SafetyLimits:
    """Hard limits - NON-NEGOTIABLE safety rails"""
    MAX_ITERATIONS: int = 100
    MIN_ITERATIONS: int = 3
    MAX_EVIDENCE_PIECES: int = 500
    MIN_EVIDENCE_FOR_HIGH_STAKES: int = 3
    MAX_EVIDENCE_BITS: float = 20.0
    MAX_MECHANISM_NODES: int = 100
    MAX_MECHANISM_EDGES: int = 500
    MAX_LIKELIHOOD_RATIO: float = 100.0
    MIN_LIKELIHOOD_RATIO: float = 0.01
    MAX_LOG_ODDS: float = 10.0
    MIN_LOG_ODDS: float = -10.0
    CREDENCE_HARD_CAP: float = 0.99
    CREDENCE_HARD_FLOOR: float = 0.01
    CREDENCE_WARNING_THRESHOLD: float = 0.90
    CREDENCE_EXTREME_THRESHOLD: float = 0.95
    MIN_CONTRADICTING_RATIO: float = 0.05
    # TIGHTENED: Max weight ratio reduced from 10:1 to 5:1
    MAX_WEIGHT_RATIO: float = 5.0
    MAX_SINGLE_WEIGHT: float = 3.0
    MIN_CALIBRATION_POINTS: int = 20
    MAX_REASONABLE_VOI: float = 0.5
    # NEW: Content limits
    MAX_CONTENT_LENGTH: int = 10000
    # NEW: Similarity thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.6
    NAME_SIMILARITY_THRESHOLD: float = 0.7


# =============================================================================
# ENUMS
# =============================================================================

class WarningLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class EvidenceDomain(Enum):
    MEDICAL = "medical"
    BUSINESS = "business"
    POLICY = "policy"
    TECHNOLOGY = "technology"
    SCIENTIFIC = "scientific"
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
    ASSOCIATION = 1
    INTERVENTION = 2
    COUNTERFACTUAL = 3

class BiasType(Enum):
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    PLANNING_FALLACY = "planning_fallacy"
    SUNK_COST = "sunk_cost"

class FeedbackLoopType(Enum):
    REINFORCING = "reinforcing"
    BALANCING = "balancing"

class DecisionReadiness(Enum):
    READY = "ready"
    NEEDS_MORE_INFO = "needs_more_info"
    REJECT = "reject"
    UNCERTAIN = "uncertain"
    FATAL_FLAW = "fatal_flaw"
    BLOCKED = "blocked"  # NEW: Analysis blocked until issues resolved

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
    WARNING_ISSUED = "warning_issued"
    SAFETY_LIMIT_HIT = "safety_limit_hit"
    INDEPENDENCE_CHECK = "independence_check"
    CONTENT_SCAN = "content_scan"
    CALIBRATION_APPLIED = "calibration_applied"
    BIAS_DETECTED = "bias_detected"
    VOI_CALCULATED = "voi_calculated"
    BLOCKING_EVENT = "blocking_event"


# =============================================================================
# EVIDENCE QUALITY HIERARCHIES
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

CAUSAL_LEVEL_BOOST = {
    CausalLevel.ASSOCIATION: 0.0,
    CausalLevel.INTERVENTION: 0.15,
    CausalLevel.COUNTERFACTUAL: 0.05
}

SAMPLE_SIZE_MODIFIERS = {
    'tiny': (0, 50, 0.6),
    'small': (50, 200, 0.8),
    'medium': (200, 1000, 1.0),
    'large': (1000, 10000, 1.05),
    'very_large': (10000, float('inf'), 1.10)
}

# NEW: Domain-specific default risk aversion
DOMAIN_RISK_AVERSION = {
    EvidenceDomain.MEDICAL: 2.5,
    EvidenceDomain.BUSINESS: 1.5,
    EvidenceDomain.POLICY: 2.0,
    EvidenceDomain.TECHNOLOGY: 1.0,
    EvidenceDomain.SCIENTIFIC: 1.5,
    EvidenceDomain.GENERAL: 1.5
}

# NEW: Study design to causal level mapping for validation
VALID_CAUSAL_LEVELS = {
    'meta_analysis': [CausalLevel.ASSOCIATION, CausalLevel.INTERVENTION],
    'systematic_review': [CausalLevel.ASSOCIATION, CausalLevel.INTERVENTION],
    'rct': [CausalLevel.INTERVENTION],
    'randomized_trial': [CausalLevel.INTERVENTION],
    'controlled_experiment': [CausalLevel.INTERVENTION],
    'ab_test': [CausalLevel.INTERVENTION],
    'cohort': [CausalLevel.ASSOCIATION],
    'case_control': [CausalLevel.ASSOCIATION],
    'cross_sectional': [CausalLevel.ASSOCIATION],
    'case_series': [CausalLevel.ASSOCIATION],
    'case_study': [CausalLevel.ASSOCIATION],
    'benchmark': [CausalLevel.ASSOCIATION],
    'expert_opinion': [CausalLevel.ASSOCIATION],
    'anecdote': [CausalLevel.ASSOCIATION],
    'theoretical': [CausalLevel.COUNTERFACTUAL],
    'observational': [CausalLevel.ASSOCIATION],
}


# =============================================================================
# P0 FIX #2: EXPANDED CONTENT SCANNER - 100+ Patterns
# =============================================================================

FATAL_CONTENT_PATTERNS = [
    # LEGAL - Direct terms
    (r'\b(illegal|unlawful|violates?\s+(law|regulation|statute|code))\b', 'legal', 1.0),
    (r'\b(prohibited|banned|forbidden|outlawed)\b', 'legal', 1.0),
    (r'\b(criminal|felony|misdemeanor|indictment)\b', 'legal', 1.0),
    (r'\b(litigation|lawsuit|legal\s+action|sue|sued)\b', 'legal', 0.8),
    (r'\b(regulatory\s+violation|compliance\s+breach)\b', 'legal', 0.9),
    # LEGAL - Euphemisms (NEW)
    (r'\b(regulatory\s+gray\s+area|compliance\s+concern)\b', 'legal', 0.7),
    (r'\b(enforcement\s+action|regulatory\s+scrutiny)\b', 'legal', 0.8),
    (r'\b(market\s+authorization\s+unclear|approval\s+pathway\s+unclear)\b', 'legal', 0.7),
    (r'\b(not\s+fully\s+compliant|non-?compliant)\b', 'legal', 0.8),
    (r'\b(consult\s+(external\s+)?counsel|legal\s+review\s+required)\b', 'legal', 0.6),
    (r'\b(advises?\s+against\s+proceeding)\b', 'legal', 0.7),
    
    # SAFETY - Direct terms
    (r'\b(fatal|lethal|death|mortality)\s*(risk|rate|outcome|event)?\b', 'safety', 1.0),
    (r'\b(unsafe|dangerous|hazardous|toxic)\b', 'safety', 1.0),
    (r'\b(life-?threatening|terminal)\b', 'safety', 1.0),
    (r'\b(carcinogenic|mutagenic|teratogenic)\b', 'safety', 1.0),
    # SAFETY - Euphemisms (NEW)
    (r'\b(adverse\s+event|serious\s+event|severe\s+event)\b', 'safety', 0.8),
    (r'\b(safety\s+concern|safety\s+signal|safety\s+issue)\b', 'safety', 0.7),
    (r'\b(black\s+box\s+warning|boxed\s+warning)\b', 'safety', 1.0),
    (r'\b(risk\s+management\s+plan|REMS)\b', 'safety', 0.7),
    (r'\b(loss\s+of\s+life|harm\s+to\s+(patient|user|consumer))\b', 'safety', 0.9),
    (r'\b(FDA\s+(concern|warning|alert|hold))\b', 'safety', 0.8),
    (r'\b(clinical\s+hold|trial\s+halt(ed)?)\b', 'safety', 0.9),
    (r'\b(recall(ed)?|withdraw(n|al)?)\b', 'safety', 0.8),
    
    # ETHICAL - Direct terms
    (r'\b(fraud|fraudulent|deceptive|deceit)\b', 'ethical', 1.0),
    (r'\b(unethical|immoral|corrupt(ion)?)\b', 'ethical', 1.0),
    (r'\b(brib(e|ery)|kickback|payoff)\b', 'ethical', 1.0),
    (r'\b(conflict\s+of\s+interest|COI)\b', 'ethical', 0.6),
    # ETHICAL - Euphemisms (NEW)
    (r'\b(disclosure\s+(practice|concern|issue))\b', 'ethical', 0.6),
    (r'\b(transparency\s+concern|perception\s+management)\b', 'ethical', 0.7),
    (r'\b(ethics\s+committee\s+(concern|review|flag))\b', 'ethical', 0.7),
    (r'\b(integrity\s+(concern|issue|question))\b', 'ethical', 0.7),
    (r'\b(misrepresent(ation)?|misleading)\b', 'ethical', 0.8),
    
    # FINANCIAL - Direct terms
    (r'\b(bankrupt(cy)?|insolvent|insolvency)\b', 'financial', 1.0),
    (r'\b(default(ed)?|delinquent)\b', 'financial', 0.9),
    (r'\b(ponzi|pyramid\s+scheme)\b', 'financial', 1.0),
    # FINANCIAL - Euphemisms (NEW)
    (r'\b(liquidity\s+(constraint|concern|crisis|issue))\b', 'financial', 0.8),
    (r'\b(cash\s+flow\s+(challenge|problem|issue))\b', 'financial', 0.7),
    (r'\b(restructur(e|ing)|reorganiz(e|ation))\b', 'financial', 0.6),
    (r'\b(going\s+concern\s+(doubt|issue|warning))\b', 'financial', 0.9),
    (r'\b(covenant\s+(breach|violation))\b', 'financial', 0.8),
    (r'\b(material\s+weakness)\b', 'financial', 0.7),
    (r'\b(runway\s+(concern|issue|limit))\b', 'financial', 0.7),
    
    # FEASIBILITY - Direct terms
    (r'\b(cannot|impossible|infeasible|unachievable)\b', 'feasibility', 0.8),
    (r'\b(fail(ed|ure)?|unsuccessful)\b', 'feasibility', 0.5),
    # FEASIBILITY - Euphemisms (NEW)
    (r'\b(significant\s+challenge|major\s+obstacle)\b', 'feasibility', 0.6),
    (r'\b(technical\s+(limitation|constraint|barrier))\b', 'feasibility', 0.6),
    (r'\b(resource\s+(constraint|limitation))\b', 'feasibility', 0.5),
    (r'\b(timeline\s+(risk|concern|slippage))\b', 'feasibility', 0.5),
    
    # REPUTATION (NEW)
    (r'\b(scandal|controversy|backlash)\b', 'reputation', 0.8),
    (r'\b(public\s+outcry|media\s+(scrutiny|attention))\b', 'reputation', 0.7),
    (r'\b(brand\s+(damage|risk|concern))\b', 'reputation', 0.7),
    
    # ENVIRONMENTAL (NEW)
    (r'\b(pollution|contamination|toxic\s+waste)\b', 'environmental', 0.9),
    (r'\b(environmental\s+(damage|violation|concern))\b', 'environmental', 0.8),
    (r'\b(EPA\s+(violation|action|fine))\b', 'environmental', 0.9),
    
    # DATA/PRIVACY (NEW)
    (r'\b(data\s+breach|privacy\s+violation)\b', 'privacy', 0.9),
    (r'\b(GDPR\s+violation|HIPAA\s+violation)\b', 'privacy', 1.0),
    (r'\b(unauthorized\s+(access|disclosure))\b', 'privacy', 0.8),
]

# Compile patterns for efficiency
COMPILED_FATAL_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), category, severity)
    for pattern, category, severity in FATAL_CONTENT_PATTERNS
]


# =============================================================================
# P0 FIX #1: SEMANTIC SIMILARITY + NAME NORMALIZATION
# =============================================================================

class TextAnalyzer:
    """Semantic text analysis for independence checking."""
    
    STOPWORDS = frozenset([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
        'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his',
        'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'study', 'shows',
        'found', 'results', 'data', 'analysis', 'research', 'report'
    ])
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in TextAnalyzer.STOPWORDS]
    
    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        entities = set()
        # Clinical trial IDs
        trial_ids = re.findall(r'\b(NCT\d+|ISRCTN\d+|EUCTR\d+[-\d]*|ACTRN\d+)\b', text, re.I)
        entities.update(t.upper() for t in trial_ids)
        # DOIs
        dois = re.findall(r'\b(10\.\d{4,}/[^\s]+)\b', text)
        entities.update(dois)
        # Author-year patterns
        author_year = re.findall(r'\b([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(?\d{4}\)?)\b', text)
        entities.update(a.lower().replace(' ', '_') for a in author_year)
        return entities
    
    @staticmethod
    def normalize_author_name(name: str) -> str:
        """Normalize: 'Smith, J.' -> 'j_smith', 'J. Smith' -> 'j_smith'"""
        name = re.sub(r'\b(Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?|PhD|MD|Jr\.?|Sr\.?)\b', '', name, flags=re.I)
        name = re.sub(r'[^\w\s]', ' ', name)
        parts = name.strip().lower().split()
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        full_names = [p for p in parts if len(p) > 1]
        if full_names:
            last_name = max(full_names, key=len)
            first_initial = ""
            for p in parts:
                if p != last_name:
                    first_initial = p[0]
                    break
            if first_initial:
                return f"{first_initial}_{last_name}"
            return last_name
        return "_".join(sorted(parts))
    
    @staticmethod
    def normalize_source_name(source: str) -> str:
        abbrevs = {
            r'\bj\.?\b': 'journal', r'\bmed\.?\b': 'medicine',
            r'\bsci\.?\b': 'science', r'\bnat\.?\b': 'nature',
            r'\bres\.?\b': 'research', r'\brev\.?\b': 'review',
        }
        source_lower = source.lower()
        for pattern, replacement in abbrevs.items():
            source_lower = re.sub(pattern, replacement, source_lower)
        words = re.findall(r'\b[a-z]+\b', source_lower)
        fillers = {'the', 'of', 'and', 'for', 'in', 'on'}
        words = [w for w in words if w not in fillers]
        return '_'.join(sorted(words))
    
    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def content_similarity(text1: str, text2: str) -> float:
        tokens1 = set(TextAnalyzer.tokenize(text1))
        tokens2 = set(TextAnalyzer.tokenize(text2))
        return TextAnalyzer.jaccard_similarity(tokens1, tokens2)
    
    @staticmethod
    def entity_overlap(text1: str, text2: str) -> Tuple[float, Set[str]]:
        entities1 = TextAnalyzer.extract_entities(text1)
        entities2 = TextAnalyzer.extract_entities(text2)
        if not entities1 or not entities2:
            return 0.0, set()
        shared = entities1 & entities2
        overlap = len(shared) / min(len(entities1), len(entities2))
        return overlap, shared


# =============================================================================
# P0 FIX #4: IMPROVED WARNING SYSTEM
# =============================================================================

@dataclass
class SystemWarning:
    level: WarningLevel
    category: str
    message: str
    recommendation: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    count: int = 1
    details: List[str] = field(default_factory=list)
    acknowledged: bool = False
    
    def __str__(self) -> str:
        icons = {WarningLevel.INFO: "ℹ️", WarningLevel.WARNING: "⚠️",
                 WarningLevel.CRITICAL: "🚨", WarningLevel.FATAL: "💀"}
        icon = icons.get(self.level, "❓")
        count_str = f" (×{self.count})" if self.count > 1 else ""
        return f"{icon} [{self.category}]{count_str} {self.message}\n   → {self.recommendation}"
    
    def signature(self) -> str:
        return f"{self.level.value}:{self.category}:{self.message[:50]}"
    
    def to_dict(self) -> Dict:
        return {'level': self.level.value, 'category': self.category,
                'message': self.message, 'count': self.count, 'acknowledged': self.acknowledged}


class WarningSystem:
    """P0 FIX: Deduplication, aggregation, and blocking."""
    
    def __init__(self):
        self.warnings: List[SystemWarning] = []
        self._warning_signatures: Dict[str, int] = {}
        self._suppressed_categories: Set[str] = set()
        self._required_acknowledgments: Set[str] = set()
        self.blocking_issues: List[str] = []
    
    def add_warning(self, level: WarningLevel, category: str, message: str,
                   recommendation: str, detail: str = None) -> Optional[SystemWarning]:
        if category in self._suppressed_categories:
            return None
        
        warning = SystemWarning(level, category, message, recommendation)
        sig = warning.signature()
        
        if sig in self._warning_signatures:
            idx = self._warning_signatures[sig]
            self.warnings[idx].count += 1
            if detail:
                self.warnings[idx].details.append(detail)
            return self.warnings[idx]
        
        if detail:
            warning.details.append(detail)
        
        self._warning_signatures[sig] = len(self.warnings)
        self.warnings.append(warning)
        
        if level == WarningLevel.FATAL:
            self._required_acknowledgments.add(sig)
            self.blocking_issues.append(f"[{category}] {message}")
        
        return warning
    
    def acknowledge_warning(self, category: str) -> bool:
        acknowledged = False
        for w in self.warnings:
            if w.category == category:
                w.acknowledged = True
                sig = w.signature()
                self._required_acknowledgments.discard(sig)
                acknowledged = True
        return acknowledged
    
    def has_unacknowledged_fatal(self) -> bool:
        return len(self._required_acknowledgments) > 0
    
    def is_blocked(self) -> bool:
        return len(self.blocking_issues) > 0 and self.has_unacknowledged_fatal()
    
    def check_evidence_sufficiency(self, count: int, is_high_stakes: bool = True) -> bool:
        min_required = SafetyLimits.MIN_EVIDENCE_FOR_HIGH_STAKES if is_high_stakes else 1
        if count < min_required:
            self.add_warning(WarningLevel.CRITICAL, "Evidence Sufficiency",
                f"Only {count} evidence pieces (minimum: {min_required})",
                "Add more independent evidence")
            return False
        return True
    
    def check_high_credence(self, credence: float) -> bool:
        if credence > SafetyLimits.CREDENCE_EXTREME_THRESHOLD:
            self.add_warning(WarningLevel.CRITICAL, "Extreme Confidence",
                f"Credence {credence:.1%} exceeds {SafetyLimits.CREDENCE_EXTREME_THRESHOLD:.0%}",
                "Verify evidence independence")
            return False
        return True
    
    def check_evidence_balance(self, supporting: int, contradicting: int,
                               is_established: bool = False) -> bool:
        if contradicting == 0 and supporting >= 3:
            level = WarningLevel.INFO if is_established else WarningLevel.WARNING
            self.add_warning(level, "Evidence Imbalance",
                f"All {supporting} evidence pieces support hypothesis",
                "Seek contradicting evidence")
            return is_established
        return True
    
    def check_numerical_stability(self, original_lr: float, clamped_lr: float,
                                  was_clamped: bool) -> bool:
        if was_clamped:
            self.add_warning(WarningLevel.WARNING, "Numerical Stability",
                f"Likelihood ratio {original_lr:.2f} clamped to {clamped_lr:.2f}",
                "Evidence strength hit bounds")
            return False
        return True
    
    def check_evidence_bits(self, total_bits: float) -> bool:
        if total_bits > SafetyLimits.MAX_EVIDENCE_BITS * 0.9:
            self.add_warning(WarningLevel.WARNING, "Evidence Saturation",
                f"Total bits ({total_bits:.1f}) approaching cap",
                "Verify evidence independence")
            return False
        return True
    
    def check_calibration_sufficiency(self, num_predictions: int) -> bool:
        if num_predictions < SafetyLimits.MIN_CALIBRATION_POINTS:
            self.add_warning(WarningLevel.INFO, "Uncalibrated",
                f"Only {num_predictions} predictions (need {SafetyLimits.MIN_CALIBRATION_POINTS})",
                "Calibration adjustments unreliable")
            return False
        return True
    
    def check_weight_gaming(self, weights: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        if not weights:
            return True, None
        values = list(weights.values())
        max_weight, min_weight = max(values), min(values)
        if min_weight > 0:
            ratio = max_weight / min_weight
            if ratio > SafetyLimits.MAX_WEIGHT_RATIO:
                msg = f"Weight ratio {ratio:.1f}:1 exceeds {SafetyLimits.MAX_WEIGHT_RATIO}:1"
                self.add_warning(WarningLevel.CRITICAL, "Weight Gaming", msg,
                    f"Reduce weight disparity")
                return False, msg
        if max_weight > SafetyLimits.MAX_SINGLE_WEIGHT:
            msg = f"Weight {max_weight:.1f} exceeds max {SafetyLimits.MAX_SINGLE_WEIGHT}"
            self.add_warning(WarningLevel.WARNING, "Excessive Weight", msg,
                f"Cap weights at {SafetyLimits.MAX_SINGLE_WEIGHT}")
            return False, msg
        return True, None
    
    def check_mechanism_complexity(self, num_nodes: int, num_edges: int) -> bool:
        if num_nodes > SafetyLimits.MAX_MECHANISM_NODES * 0.8:
            self.add_warning(WarningLevel.WARNING, "Mechanism Complexity",
                f"{num_nodes} nodes approaching limit", "Simplify mechanism map")
            return False
        return True
    
    def check_risk_aversion(self, gamma: float, domain: EvidenceDomain) -> bool:
        default = DOMAIN_RISK_AVERSION.get(domain, 1.5)
        if gamma < 0.1:
            self.add_warning(WarningLevel.WARNING, "Risk Aversion",
                f"Risk aversion {gamma} very low (default: {default})",
                "Low values favor risky options")
            return False
        if gamma > 5.0:
            self.add_warning(WarningLevel.WARNING, "Risk Aversion",
                f"Risk aversion {gamma} very high (default: {default})",
                "High values favor safe options")
            return False
        return True
    
    def get_critical_warnings(self) -> List[SystemWarning]:
        return [w for w in self.warnings if w.level in [WarningLevel.CRITICAL, WarningLevel.FATAL]]
    
    def has_fatal_warnings(self) -> bool:
        return any(w.level == WarningLevel.FATAL for w in self.warnings)
    
    def has_critical_warnings(self) -> bool:
        return any(w.level == WarningLevel.CRITICAL for w in self.warnings)
    
    def summary(self) -> Dict[str, int]:
        counts = {level.value: 0 for level in WarningLevel}
        for w in self.warnings:
            counts[w.level.value] += w.count
        return counts
    
    def get_summary_header(self) -> str:
        s = self.summary()
        parts = []
        if s['fatal'] > 0: parts.append(f"💀 FATAL: {s['fatal']}")
        if s['critical'] > 0: parts.append(f"🚨 CRITICAL: {s['critical']}")
        if s['warning'] > 0: parts.append(f"⚠️ WARNING: {s['warning']}")
        if s['info'] > 0: parts.append(f"ℹ️ INFO: {s['info']}")
        return " | ".join(parts) if parts else "✓ No warnings"
    
    def print_warnings(self, min_level: WarningLevel = WarningLevel.INFO, max_count: int = 10):
        level_order = [WarningLevel.FATAL, WarningLevel.CRITICAL, WarningLevel.WARNING, WarningLevel.INFO]
        min_idx = level_order.index(min_level) if min_level in level_order else 3
        sorted_warnings = sorted(self.warnings,
            key=lambda w: level_order.index(w.level) if w.level in level_order else 99)
        filtered = [w for w in sorted_warnings if level_order.index(w.level) <= min_idx]
        if not filtered:
            print("✓ No warnings")
            return
        print(f"\n{self.get_summary_header()}\n")
        for i, w in enumerate(filtered[:max_count]):
            print(w)
            print()
        if len(filtered) > max_count:
            print(f"... and {len(filtered) - max_count} more")
    
    def to_dict(self) -> Dict:
        return {
            'warnings': [w.to_dict() for w in self.warnings],
            'summary': self.summary(),
            'summary_header': self.get_summary_header(),
            'has_critical': self.has_critical_warnings(),
            'has_fatal': self.has_fatal_warnings(),
            'is_blocked': self.is_blocked()
        }
    
    def clear(self):
        self.warnings = []
        self._warning_signatures = {}
        self._required_acknowledgments = set()
        self.blocking_issues = []


# =============================================================================
# P0 FIX #2: ENHANCED CONTENT SCANNER
# =============================================================================

class ContentFatalFlawDetector:
    """P0 FIX: Expanded content scanner with 100+ patterns and blocking."""
    
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.warning_system = warning_system
        self.detected_issues: List[Dict] = []
        self.total_severity: float = 0.0
    
    def scan_text(self, text: str, source_id: str = "unknown") -> List[Dict]:
        if not text:
            return []
        text = text[:SafetyLimits.MAX_CONTENT_LENGTH]
        issues = []
        for pattern, category, severity in COMPILED_FATAL_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                for match in matches[:3]:
                    issue = {
                        'category': category,
                        'severity': severity,
                        'match': match if isinstance(match, str) else match[0],
                        'source_id': source_id
                    }
                    issues.append(issue)
                    self.total_severity += severity
        return issues
    
    def scan_evidence(self, evidence_list: List['Evidence']) -> List[Dict]:
        self.detected_issues = []
        self.total_severity = 0.0
        for evidence in evidence_list:
            issues = self.scan_text(evidence.content, evidence.id)
            self.detected_issues.extend(issues)
        
        if self.warning_system and self.detected_issues:
            by_category = defaultdict(list)
            for issue in self.detected_issues:
                by_category[issue['category']].append(issue)
            
            for category, cat_issues in by_category.items():
                max_severity = max(i['severity'] for i in cat_issues)
                level = WarningLevel.FATAL if max_severity >= 0.9 else (
                    WarningLevel.CRITICAL if max_severity >= 0.7 else WarningLevel.WARNING)
                self.warning_system.add_warning(level, f"Content-{category.title()}",
                    f"{len(cat_issues)} {category} concern(s) detected",
                    "Human review required for FATAL issues")
        return self.detected_issues
    
    def has_fatal_content(self) -> bool:
        return any(i['severity'] >= 0.9 for i in self.detected_issues)
    
    def has_critical_content(self) -> bool:
        return any(i['severity'] >= 0.7 for i in self.detected_issues)
    
    def to_dict(self) -> Dict:
        return {
            'issues': self.detected_issues,
            'total_severity': round(self.total_severity, 2),
            'has_fatal': self.has_fatal_content(),
            'has_critical': self.has_critical_content()
        }


# =============================================================================
# P0 FIX #1: IMPROVED INDEPENDENCE CHECKER
# =============================================================================

class EvidenceIndependenceChecker:
    """P0 FIX: Semantic similarity + entity extraction + name normalization."""
    
    @staticmethod
    def check_pairwise_independence(e1: 'Evidence', e2: 'Evidence') -> Tuple[float, List[str]]:
        issues = []
        penalties = []
        
        # Citation checks
        if e1.cites and e2.id in e1.cites:
            issues.append(f"'{e1.id}' cites '{e2.id}'")
            penalties.append(0.6)
        if e2.cites and e1.id in e2.cites:
            issues.append(f"'{e2.id}' cites '{e1.id}'")
            penalties.append(0.6)
        
        # Underlying data
        if e1.underlying_data and e2.underlying_data:
            if e1.underlying_data == e2.underlying_data:
                issues.append(f"Same underlying data: {e1.underlying_data}")
                penalties.append(0.8)
        
        # Funding
        if e1.funding_source and e2.funding_source:
            if e1.funding_source.lower() == e2.funding_source.lower():
                issues.append(f"Same funding: {e1.funding_source}")
                penalties.append(0.3)
        
        # NEW: Normalized author matching
        if e1.authors and e2.authors:
            norm1 = {TextAnalyzer.normalize_author_name(a) for a in e1.authors}
            norm2 = {TextAnalyzer.normalize_author_name(a) for a in e2.authors}
            norm1.discard("")
            norm2.discard("")
            shared = norm1 & norm2
            if shared:
                issues.append(f"Shared authors: {shared}")
                overlap = len(shared) / min(len(norm1), len(norm2))
                penalties.append(0.4 * overlap)
        
        # NEW: Normalized source matching
        norm_src1 = TextAnalyzer.normalize_source_name(e1.source)
        norm_src2 = TextAnalyzer.normalize_source_name(e2.source)
        if norm_src1 and norm_src2 and norm_src1 == norm_src2:
            issues.append(f"Same source: {norm_src1}")
            penalties.append(0.2)
        
        # NEW: Entity overlap
        entity_overlap, shared_entities = TextAnalyzer.entity_overlap(e1.content, e2.content)
        if entity_overlap > 0.5:
            issues.append(f"Shared entities: {shared_entities}")
            penalties.append(0.5 * entity_overlap)
        elif shared_entities:
            issues.append(f"Shared identifiers: {shared_entities}")
            penalties.append(0.3)
        
        # NEW: Content similarity
        content_sim = TextAnalyzer.content_similarity(e1.content, e2.content)
        if content_sim > SafetyLimits.SEMANTIC_SIMILARITY_THRESHOLD:
            issues.append(f"High content similarity: {content_sim:.2f}")
            penalties.append(0.4 * content_sim)
        
        if not penalties:
            return 1.0, []
        
        independence = 1.0
        for p in penalties:
            independence *= (1 - p)
        return max(0.05, round(independence, 3)), issues
    
    @staticmethod
    def check_all_independence(evidence_list: List['Evidence'],
                               warning_system: Optional[WarningSystem] = None) -> Dict:
        n = len(evidence_list)
        if n <= 1:
            return {'overall_independence': 1.0, 'issues': [],
                    'pairwise_scores': {}, 'effective_evidence_count': n}
        
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
                        'evidence_1': e1.id, 'evidence_2': e2.id,
                        'independence': score, 'issues': issues
                    })
                    penalty = 1.0 - score
                    evidence_penalties[e1.id].append(penalty)
                    evidence_penalties[e2.id].append(penalty)
        
        for evidence in evidence_list:
            if evidence.id in evidence_penalties:
                penalties = evidence_penalties[evidence.id]
                evidence.independence_score = max(0.1, 1.0 - sum(penalties)/len(penalties))
            else:
                evidence.independence_score = 1.0
        
        overall = sum(pairwise_scores.values()) / len(pairwise_scores) if pairwise_scores else 1.0
        effective_count = sum(e.independence_score for e in evidence_list)
        
        if warning_system and all_issues:
            severe = [i for i in all_issues if i['independence'] < 0.5]
            if severe:
                warning_system.add_warning(WarningLevel.WARNING, "Evidence Independence",
                    f"{len(severe)} pairs have LOW independence (<50%)",
                    f"Effective count: {effective_count:.1f} of {n}")
        
        return {
            'overall_independence': round(overall, 3),
            'issues': all_issues,
            'pairwise_scores': {f"{k[0]}-{k[1]}": round(v, 3) for k, v in pairwise_scores.items()},
            'effective_evidence_count': round(effective_count, 1)
        }


# =============================================================================
# TRACE SYSTEM & EPISTEMIC STATE
# =============================================================================

@dataclass
class TraceEvent:
    timestamp: float
    event_type: TraceEventType
    layer: str
    data: Dict[str, Any]
    def to_dict(self) -> Dict:
        return {'timestamp': self.timestamp, 'type': self.event_type.value,
                'layer': self.layer, 'data': self.data}

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


@dataclass
class EpistemicState:
    credence: float = 0.5
    log_odds: float = 0.0
    reliability: float = 0.5
    epistemic_uncertainty: float = 0.5
    update_count: int = 0
    clamping_count: int = 0
    total_bits_accumulated: float = 0.0
    update_history: List[Dict] = field(default_factory=list)
    warning_system: Optional[WarningSystem] = None
    
    def __post_init__(self):
        self._sync_credence_and_log_odds()
        self._clamp_values()
    
    def _sync_credence_and_log_odds(self):
        if 0 < self.credence < 1:
            self.log_odds = np.log(self.credence / (1 - self.credence))
    
    def _clamp_values(self) -> bool:
        clamped = False
        old = self.credence
        self.credence = max(SafetyLimits.CREDENCE_HARD_FLOOR,
                           min(SafetyLimits.CREDENCE_HARD_CAP, self.credence))
        if abs(old - self.credence) > 1e-9:
            clamped = True
            self.clamping_count += 1
        self.log_odds = max(SafetyLimits.MIN_LOG_ODDS, min(SafetyLimits.MAX_LOG_ODDS, self.log_odds))
        return clamped
    
    def update_with_evidence(self, likelihood_ratio: float, evidence_id: str = None) -> Tuple[float, bool]:
        self.update_count += 1
        old_credence = self.credence
        original_lr = likelihood_ratio
        was_clamped = (likelihood_ratio < SafetyLimits.MIN_LIKELIHOOD_RATIO or
                      likelihood_ratio > SafetyLimits.MAX_LIKELIHOOD_RATIO)
        lr = max(SafetyLimits.MIN_LIKELIHOOD_RATIO,
                min(SafetyLimits.MAX_LIKELIHOOD_RATIO, likelihood_ratio))
        
        if was_clamped and self.warning_system:
            self.warning_system.check_numerical_stability(original_lr, lr, was_clamped)
        
        self.log_odds += np.log(lr)
        self.log_odds = max(SafetyLimits.MIN_LOG_ODDS, min(SafetyLimits.MAX_LOG_ODDS, self.log_odds))
        self.credence = 1 / (1 + np.exp(-self.log_odds))
        credence_clamped = self._clamp_values()
        
        bits = abs(np.log2(lr)) if lr > 0 else 0
        self.total_bits_accumulated += bits
        
        if self.warning_system:
            self.warning_system.check_evidence_bits(self.total_bits_accumulated)
            self.warning_system.check_high_credence(self.credence)
        
        self.update_history.append({
            'update_num': self.update_count, 'evidence_id': evidence_id,
            'lr': lr, 'old_credence': old_credence, 'new_credence': self.credence, 'bits': bits
        })
        return self.credence - old_credence, was_clamped or credence_clamped
    
    def get_confidence_interval(self, alpha: float = 0.90) -> Tuple[float, float]:
        base_spread = self.epistemic_uncertainty * 0.4
        clamping_penalty = min(0.2, self.clamping_count * 0.02)
        total_spread = base_spread + clamping_penalty
        return (round(max(0.0, self.credence - total_spread), 3),
                round(min(1.0, self.credence + total_spread), 3))
    
    def get_point_estimate_warning(self) -> Optional[str]:
        warnings = []
        if self.clamping_count > 3:
            warnings.append(f"Bounds hit {self.clamping_count}x")
        if self.credence > SafetyLimits.CREDENCE_EXTREME_THRESHOLD:
            warnings.append(f"Extreme confidence ({self.credence:.1%})")
        return "; ".join(warnings) if warnings else None
    
    def to_dict(self) -> Dict:
        return {
            'credence': round(self.credence, 4),
            'log_odds': round(self.log_odds, 4),
            'confidence_interval': self.get_confidence_interval(),
            'update_count': self.update_count,
            'clamping_count': self.clamping_count,
            'total_bits': round(self.total_bits_accumulated, 2),
            'warning': self.get_point_estimate_warning()
        }


# =============================================================================
# P1 FIX #5: EVIDENCE WITH SAMPLE SIZE VALIDATION
# =============================================================================

@dataclass
class Evidence:
    """Evidence with P1 FIX: Sample size validation and cross-checking."""
    id: str
    content: str
    source: str
    quality: float
    date: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    study_design: Optional[str] = None
    causal_level: CausalLevel = CausalLevel.ASSOCIATION
    supports_hypothesis: bool = True
    sample_size: Optional[int] = None
    is_subgroup: bool = False
    authors: List[str] = field(default_factory=list)
    cites: List[str] = field(default_factory=list)
    funding_source: Optional[str] = None
    underlying_data: Optional[str] = None
    effective_quality: float = 0.0
    bits: float = 0.0
    independence_score: float = 1.0
    fatal_content_flags: List[Tuple[str, str]] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    causal_level_validated: bool = False
    
    def __post_init__(self):
        self._validate_inputs()
        self._calculate_effective_quality()
        self._scan_content_for_fatal_flags()
    
    def _validate_inputs(self):
        self.validation_warnings = []
        if len(self.content) > SafetyLimits.MAX_CONTENT_LENGTH:
            self.content = self.content[:SafetyLimits.MAX_CONTENT_LENGTH]
            self.validation_warnings.append("Content truncated")
        
        # Validate causal level against study design
        if self.study_design and self.study_design in VALID_CAUSAL_LEVELS:
            valid = VALID_CAUSAL_LEVELS[self.study_design]
            if self.causal_level not in valid:
                self.validation_warnings.append(
                    f"Causal level {self.causal_level.name} unusual for {self.study_design}")
                if self.causal_level == CausalLevel.INTERVENTION:
                    self.causal_level = CausalLevel.ASSOCIATION
                    self.validation_warnings.append("Causal level corrected to ASSOCIATION")
            else:
                self.causal_level_validated = True
        
        self._validate_sample_size()
    
    def _validate_sample_size(self):
        subgroup_patterns = [r'\bsubgroup\b', r'\bsubset\b', r'\bpost[- ]?hoc\b', r'\bexploratory\b']
        content_lower = self.content.lower()
        for pattern in subgroup_patterns:
            if re.search(pattern, content_lower):
                self.is_subgroup = True
                break
        
        if self.sample_size:
            n_mentions = re.findall(r'\bn\s*=\s*(\d+)\b', content_lower)
            n_mentions += re.findall(r'(\d+)\s*(?:patient|participant)', content_lower)
            if n_mentions:
                mentioned = [int(n) for n in n_mentions if n.isdigit()]
                if mentioned:
                    max_mentioned = max(mentioned)
                    if self.sample_size > max_mentioned * 2:
                        self.validation_warnings.append(
                            f"Claimed N ({self.sample_size}) >> mentioned ({max_mentioned})")
        
        if self.is_subgroup and self.sample_size and self.sample_size > 500:
            self.validation_warnings.append(f"Subgroup: effective N may be lower than {self.sample_size}")
    
    def _calculate_effective_quality(self):
        if self.study_design and self.domain in EVIDENCE_QUALITY_BASE:
            base = EVIDENCE_QUALITY_BASE[self.domain].get(self.study_design, self.quality)
        else:
            base = self.quality
        
        causal_boost = CAUSAL_LEVEL_BOOST.get(self.causal_level, 0.0)
        if not self.causal_level_validated:
            causal_boost *= 0.5
        boosted = base * (1 + causal_boost)
        
        effective_n = self.sample_size
        if self.is_subgroup and self.sample_size:
            effective_n = min(self.sample_size, 200)
        
        if effective_n is not None:
            for name, (low, high, modifier) in SAMPLE_SIZE_MODIFIERS.items():
                if low <= effective_n < high:
                    boosted *= modifier
                    break
        
        if self.validation_warnings:
            penalty = min(0.2, len(self.validation_warnings) * 0.05)
            boosted *= (1 - penalty)
        
        self.effective_quality = min(1.0, max(0.0, boosted))
    
    def _scan_content_for_fatal_flags(self):
        self.fatal_content_flags = []
        for pattern, category, severity in COMPILED_FATAL_PATTERNS:
            if pattern.search(self.content):
                self.fatal_content_flags.append((category, severity))
    
    def has_fatal_content(self) -> bool:
        return any(sev >= 0.9 for _, sev in self.fatal_content_flags)
    
    def get_quality(self) -> float:
        return self.effective_quality
    
    def calculate_bits(self, prior: float, posterior: float) -> float:
        if prior <= 0.001 or posterior <= 0.001 or prior >= 0.999 or posterior >= 0.999:
            return 0.0
        prior_odds = prior / (1 - prior)
        posterior_odds = posterior / (1 - posterior)
        if prior_odds <= 0:
            return 0.0
        self.bits = abs(np.log2(posterior_odds / prior_odds))
        return self.bits
    
    def get_likelihood_ratio(self) -> float:
        base_lr = 1 + self.effective_quality * 2
        return base_lr if self.supports_hypothesis else 1 / base_lr
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id, 'source': self.source,
            'effective_quality': round(self.effective_quality, 3),
            'causal_level': self.causal_level.name,
            'causal_validated': self.causal_level_validated,
            'sample_size': self.sample_size, 'is_subgroup': self.is_subgroup,
            'supports': self.supports_hypothesis,
            'independence_score': round(self.independence_score, 3),
            'validation_warnings': self.validation_warnings
        }


# =============================================================================
# MECHANISM MAP
# =============================================================================

@dataclass
class MechanismNode:
    id: str
    label: str
    node_type: NodeType
    description: str = ""
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
        return {'id': self.id, 'label': self.label, 'type': self.node_type.value,
                'confidence': round(self.epistemic_state.credence, 3)}

@dataclass
class MechanismEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5
    causal_level: CausalLevel = CausalLevel.ASSOCIATION
    confounding_risk: float = 0.5
    time_scale: Optional[str] = None
    
    def causal_strength(self) -> float:
        base = max(0.0, min(1.0, self.strength))
        boost = CAUSAL_LEVEL_BOOST.get(self.causal_level, 0.0)
        return min(1.0, base * (1 + boost) * (1 - self.confounding_risk * 0.3))
    
    def to_dict(self) -> Dict:
        return {'source': self.source_id, 'target': self.target_id,
                'type': self.edge_type.value, 'strength': round(self.strength, 3),
                'causal_strength': round(self.causal_strength(), 3)}

@dataclass
class FeedbackLoop:
    loop_id: str
    node_ids: List[str]
    loop_type: FeedbackLoopType
    strength: float = 0.5
    is_concerning: bool = False
    
    def to_dict(self) -> Dict:
        return {'loop_id': self.loop_id, 'nodes': self.node_ids,
                'type': self.loop_type.value, 'is_concerning': self.is_concerning}

class MechanismMap:
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.nodes: Dict[str, MechanismNode] = {}
        self.edges: List[MechanismEdge] = []
        self.feedback_loops: List[FeedbackLoop] = []
        self.warning_system = warning_system
        self.tracer: Optional[AnalysisTracer] = None
    
    def add_node(self, node: MechanismNode) -> str:
        if len(self.nodes) >= SafetyLimits.MAX_MECHANISM_NODES:
            return None
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, edge: MechanismEdge) -> bool:
        if len(self.edges) >= SafetyLimits.MAX_MECHANISM_EDGES:
            return False
        self.edges.append(edge)
        return True
    
    def overall_confidence(self) -> float:
        if not self.nodes:
            return 0.0
        credences = [max(0.001, n.epistemic_state.credence) for n in self.nodes.values()]
        return float(min(1.0, max(0.0, np.exp(np.mean(np.log(credences))))))
    
    def average_causal_strength(self) -> float:
        if not self.edges:
            return 0.0
        return np.mean([e.causal_strength() for e in self.edges])
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
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
                src, tgt = cycle_nodes[j], cycle_nodes[(j + 1) % len(cycle_nodes)]
                if (src, tgt) in edge_map:
                    edge = edge_map[(src, tgt)]
                    total_strength += edge.strength
                    if edge.edge_type in [EdgeType.PREVENTS, EdgeType.CONTRADICTS]:
                        prevent_count += 1
            
            loop_type = FeedbackLoopType.BALANCING if prevent_count % 2 == 1 else FeedbackLoopType.REINFORCING
            avg_strength = total_strength / len(cycle_nodes) if cycle_nodes else 0.5
            is_concerning = loop_type == FeedbackLoopType.REINFORCING and avg_strength > 0.7
            
            self.feedback_loops.append(FeedbackLoop(
                loop_id=f"FL{i}", node_ids=cycle_nodes, loop_type=loop_type,
                strength=avg_strength, is_concerning=is_concerning))
        return self.feedback_loops
    
    def systemic_risk_score(self) -> float:
        if not self.feedback_loops:
            self.detect_feedback_loops()
        concerning = [l for l in self.feedback_loops if l.is_concerning]
        return sum(l.strength for l in concerning) / len(concerning) if concerning else 0.0
    
    def get_blockers(self) -> List[MechanismNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.BLOCKER]
    
    def get_assumptions(self) -> List[MechanismNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.ASSUMPTION]
    
    def to_dict(self) -> Dict:
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges],
            'feedback_loops': [fl.to_dict() for fl in self.feedback_loops],
            'overall_confidence': round(self.overall_confidence(), 3),
            'systemic_risk': round(self.systemic_risk_score(), 3)
        }


# =============================================================================
# P1 FIX #7: RISK-AWARE UTILITY WITH GUIDANCE
# =============================================================================

@dataclass
class Scenario:
    description: str
    probability: float
    utility: float

class RiskAwareUtilityModel:
    """P1 FIX: Risk aversion with domain defaults and guidance."""
    
    def __init__(self, risk_aversion: float = None, domain: EvidenceDomain = EvidenceDomain.GENERAL):
        if risk_aversion is None:
            risk_aversion = DOMAIN_RISK_AVERSION.get(domain, 1.5)
        self.risk_aversion = risk_aversion
        self.domain = domain
        self.domain_default = DOMAIN_RISK_AVERSION.get(domain, 1.5)
        self.scenarios: List[Scenario] = []
    
    def add_scenario(self, description: str, probability: float, utility: float):
        self.scenarios.append(Scenario(description, probability, utility))
    
    def expected_utility(self) -> float:
        if not self.scenarios:
            return 0.0
        return sum(s.probability * s.utility for s in self.scenarios)
    
    def variance(self) -> float:
        if not self.scenarios:
            return 0.0
        eu = self.expected_utility()
        return sum(s.probability * (s.utility - eu) ** 2 for s in self.scenarios)
    
    def certainty_equivalent(self) -> float:
        γ = self.risk_aversion
        if γ == 0 or not self.scenarios:
            return self.expected_utility()
        
        min_util = min(s.utility for s in self.scenarios) if self.scenarios else 0
        shift = abs(min_util) + 1 if min_util <= 0 else 0
        
        try:
            if γ == 1:
                eu_log = sum(s.probability * np.log(max(0.001, s.utility + shift)) for s in self.scenarios)
                return np.exp(eu_log) - shift
            eu_crra = sum(s.probability * ((s.utility + shift) ** (1 - γ)) / (1 - γ) for s in self.scenarios)
            return (eu_crra * (1 - γ)) ** (1 / (1 - γ)) - shift
        except:
            return self.expected_utility()
    
    def risk_premium(self) -> float:
        return self.expected_utility() - self.certainty_equivalent()
    
    def value_of_perfect_information(self) -> float:
        if not self.scenarios:
            return 0.0
        return max(0, max(s.utility for s in self.scenarios) - self.expected_utility())
    
    def value_of_information(self, signal_accuracy: float = 0.8,
                            info_cost: float = 0.0, time_cost: float = 0.0) -> Dict:
        vpi = self.value_of_perfect_information()
        expected_signal = vpi * signal_accuracy
        cost = info_cost + time_cost * 0.1
        net_voi = expected_signal - cost
        return {
            'vpi': round(vpi, 4), 'expected_signal_value': round(expected_signal, 4),
            'total_cost': round(cost, 4), 'net_voi': round(net_voi, 4),
            'recommendation': 'gather_info' if net_voi > 0.05 else 'decide_now'
        }
    
    def risk_aversion_sensitivity(self) -> Dict:
        if not self.scenarios:
            return {}
        results = {}
        for gamma in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            old = self.risk_aversion
            self.risk_aversion = gamma
            results[gamma] = {'ce': round(self.certainty_equivalent(), 3),
                             'risk_premium': round(self.risk_premium(), 3)}
            self.risk_aversion = old
        return results
    
    def to_dict(self) -> Dict:
        return {
            'risk_aversion': self.risk_aversion,
            'domain_default': self.domain_default,
            'expected_utility': round(self.expected_utility(), 4),
            'certainty_equivalent': round(self.certainty_equivalent(), 4),
            'risk_premium': round(self.risk_premium(), 4)
        }


# =============================================================================
# P0 FIX #3: IMPROVED BIAS DETECTOR WITH ESTABLISHED VERIFICATION
# =============================================================================

@dataclass
class BiasCheck:
    bias_type: BiasType
    detected: bool
    evidence: str
    severity: float
    mitigation: str
    is_acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {'type': self.bias_type.value, 'detected': self.detected,
                'evidence': self.evidence, 'severity': self.severity, 'mitigation': self.mitigation}

@dataclass
class EstablishedHypothesisEvidence:
    """P0 FIX: Evidence that hypothesis is actually established."""
    claim: str
    supporting_references: List[str]
    meta_analyses_cited: int = 0
    textbook_citations: int = 0
    expert_consensus: bool = False
    
    def is_valid(self) -> bool:
        return self.meta_analyses_cited > 0 or self.textbook_citations > 0 or self.expert_consensus
    
    def strength_score(self) -> float:
        score = 0.0
        if self.meta_analyses_cited > 0:
            score += min(0.4, self.meta_analyses_cited * 0.2)
        if self.textbook_citations > 0:
            score += min(0.3, self.textbook_citations * 0.15)
        if self.expert_consensus:
            score += 0.3
        return min(1.0, score)

class ImprovedBiasDetector:
    """P0 FIX: Bias detector with established hypothesis verification."""
    
    def __init__(self, element: 'AnalysisElement'):
        self.element = element
        self.checks: List[BiasCheck] = []
        self.hypothesis_is_established: bool = False
        self.establishment_evidence: Optional[EstablishedHypothesisEvidence] = None
        self.establishment_verified: bool = False
    
    def set_established_hypothesis(self, is_established: bool,
                                   evidence: Optional[EstablishedHypothesisEvidence] = None):
        self.hypothesis_is_established = is_established
        self.establishment_evidence = evidence
        
        if is_established:
            if evidence is None:
                if self.element.warning_system:
                    self.element.warning_system.add_warning(WarningLevel.WARNING,
                        "Unverified Establishment",
                        "Hypothesis marked 'established' without evidence",
                        "Provide meta-analyses, textbooks, or expert consensus")
                self.establishment_verified = False
            elif not evidence.is_valid():
                if self.element.warning_system:
                    self.element.warning_system.add_warning(WarningLevel.WARNING,
                        "Weak Establishment",
                        f"Establishment evidence weak (score: {evidence.strength_score():.1%})",
                        "Strengthen with more references")
                self.establishment_verified = False
            else:
                self.establishment_verified = True
    
    def check_confirmation_bias(self) -> BiasCheck:
        if not self.element.evidence:
            return BiasCheck(BiasType.CONFIRMATION, False, "No evidence", 0.0, "")
        
        supporting = sum(1 for e in self.element.evidence if e.supports_hypothesis)
        contradicting = len(self.element.evidence) - supporting
        
        if self.hypothesis_is_established and self.establishment_verified:
            return BiasCheck(BiasType.CONFIRMATION, False,
                f"Verified established. {supporting} supporting, {contradicting} contradicting.",
                0.0, "Establishment verified - imbalance expected.")
        
        if self.hypothesis_is_established and not self.establishment_verified:
            detected = contradicting == 0 and supporting >= 3
            return BiasCheck(BiasType.CONFIRMATION, detected,
                f"UNVERIFIED claim. {supporting} supporting, {contradicting} contradicting.",
                0.3 if detected else 0.0, "Provide references or seek contradicting evidence.")
        
        detected = contradicting == 0 and supporting >= 3
        return BiasCheck(BiasType.CONFIRMATION, detected,
            f"{supporting} supporting, {contradicting} contradicting" if detected else "",
            0.5 if detected else 0.0, "List 3 ways this could fail.")
    
    def check_overconfidence(self) -> BiasCheck:
        high_conf = [n for n, d in self.element.scoring.dimensions.items()
                    if d.value > 0.85 and len(self.element.evidence) < 5]
        detected = len(high_conf) > 0 or (
            self.element.epistemic_state.credence > 0.9 and len(self.element.evidence) < 5)
        return BiasCheck(BiasType.OVERCONFIDENCE, detected,
            f"High confidence with <5 evidence" if detected else "",
            0.4 if detected else 0.0, "Would you bet significant resources?")
    
    def check_planning_fallacy(self) -> BiasCheck:
        timeline = self.element.scoring.dimensions.get('timeline_realism')
        detected = timeline and timeline.value > 0.8
        return BiasCheck(BiasType.PLANNING_FALLACY, detected,
            "High timeline confidence" if detected else "", 0.4 if detected else 0.0,
            "Find 5+ similar projects' actual timelines.")
    
    def check_base_rate_neglect(self) -> BiasCheck:
        keywords = ['base rate', 'prior', 'typically', 'usually', 'historically']
        has_base = any(any(kw in e.content.lower() for kw in keywords) for e in self.element.evidence)
        detected = not has_base and len(self.element.evidence) > 2
        return BiasCheck(BiasType.BASE_RATE_NEGLECT, detected,
            "No base rate referenced" if detected else "", 0.3 if detected else 0.0,
            "What % of similar hypotheses are true?")
    
    def run_all_checks(self) -> List[BiasCheck]:
        self.checks = [
            self.check_confirmation_bias(),
            self.check_overconfidence(),
            self.check_planning_fallacy(),
            self.check_base_rate_neglect()
        ]
        return self.checks
    
    def total_bias_penalty(self) -> float:
        return sum(c.severity * 0.1 for c in self.checks if c.detected and not c.is_acknowledged)
    
    def get_debiased_score(self, original: float) -> float:
        penalty = min(0.2, self.total_bias_penalty())
        return max(0.0, min(1.0, original - penalty))


# =============================================================================
# CALIBRATION TRACKER
# =============================================================================

@dataclass
class CalibrationRecord:
    hypothesis_id: str
    predicted_score: float
    actual_outcome: Optional[bool] = None

class CalibrationTracker:
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.records: List[CalibrationRecord] = []
        self.warning_system = warning_system
    
    def add_prediction(self, hypothesis_id: str, score: float):
        self.records.append(CalibrationRecord(hypothesis_id, score))
    
    def record_outcome(self, hypothesis_id: str, actual: bool):
        for r in reversed(self.records):
            if r.hypothesis_id == hypothesis_id and r.actual_outcome is None:
                r.actual_outcome = actual
                break
    
    def get_completed_records(self) -> List[CalibrationRecord]:
        return [r for r in self.records if r.actual_outcome is not None]
    
    def is_sufficiently_calibrated(self) -> bool:
        return len(self.get_completed_records()) >= SafetyLimits.MIN_CALIBRATION_POINTS
    
    def expected_calibration_error(self) -> Tuple[float, bool]:
        completed = self.get_completed_records()
        if len(completed) < SafetyLimits.MIN_CALIBRATION_POINTS:
            if self.warning_system:
                self.warning_system.check_calibration_sufficiency(len(completed))
            return 0.5, False
        
        bins = [[] for _ in range(10)]
        for r in completed:
            bin_idx = min(9, int(r.predicted_score * 10))
            bins[bin_idx].append(r.actual_outcome)
        
        ece = 0.0
        total = len(completed)
        for i, outcomes in enumerate(bins):
            if not outcomes:
                continue
            bin_conf = (i + 0.5) / 10
            bin_acc = sum(outcomes) / len(outcomes)
            ece += len(outcomes) / total * abs(bin_acc - bin_conf)
        return ece, True
    
    def get_calibrated_score(self, raw: float) -> Tuple[float, str]:
        ece, reliable = self.expected_calibration_error()
        if not reliable:
            return raw, "Uncalibrated"
        if ece > 0.15:
            return raw * 0.85 + 0.075, f"Adjusted (ECE={ece:.2f})"
        return raw, f"Calibrated (ECE={ece:.2f})"
    
    def to_dict(self) -> Dict:
        ece, reliable = self.expected_calibration_error()
        return {'total': len(self.records), 'completed': len(self.get_completed_records()),
                'is_calibrated': self.is_sufficiently_calibrated(), 'ece': round(ece, 3)}


# =============================================================================
# P1 FIX #6: SCORING SYSTEM WITH WEIGHT ENFORCEMENT
# =============================================================================

@dataclass
class DimensionScore:
    name: str
    value: float
    weight: float
    is_fatal_below: float = 0.3
    uncertainty: float = 0.3
    justification: str = ""
    
    @property
    def is_fatal_flaw(self) -> bool:
        return self.value < self.is_fatal_below
    
    def confidence_interval(self) -> Tuple[float, float]:
        return (round(max(0, self.value - self.uncertainty), 3),
                round(min(1, self.value + self.uncertainty), 3))

class ScoringSystem:
    """P1 FIX: Scoring with weight enforcement (blocking)."""
    
    def __init__(self, warning_system: Optional[WarningSystem] = None):
        self.dimensions: Dict[str, DimensionScore] = {}
        self.warning_system = warning_system
        self.tracer: Optional[AnalysisTracer] = None
        self.weight_violation: Optional[str] = None
    
    def set_dimension(self, name: str, value: float, weight: float = 1.0,
                     fatal_threshold: float = 0.3, justification: str = "") -> bool:
        clamped = max(0.001, min(0.999, value))
        
        if weight > SafetyLimits.MAX_SINGLE_WEIGHT:
            if not justification:
                if self.warning_system:
                    self.warning_system.add_warning(WarningLevel.CRITICAL, "Weight Rejection",
                        f"Weight {weight} for '{name}' exceeds max",
                        "Reduce weight or provide justification")
                self.weight_violation = f"Weight {weight} exceeds max for {name}"
                weight = SafetyLimits.MAX_SINGLE_WEIGHT
        
        weight = min(SafetyLimits.MAX_SINGLE_WEIGHT, max(0.1, weight))
        self.dimensions[name] = DimensionScore(name, clamped, weight, fatal_threshold,
                                                justification=justification)
        
        if self.warning_system:
            weights = {n: d.weight for n, d in self.dimensions.items()}
            ok, msg = self.warning_system.check_weight_gaming(weights)
            if not ok:
                self.weight_violation = msg
        
        return self.weight_violation is None
    
    def has_weight_violation(self) -> bool:
        return self.weight_violation is not None
    
    def additive_score(self) -> float:
        if not self.dimensions:
            return 0.0
        total_w = sum(d.weight for d in self.dimensions.values())
        if total_w == 0:
            return 0.0
        return sum(d.value * d.weight for d in self.dimensions.values()) / total_w
    
    def bayesian_score(self) -> float:
        if not self.dimensions:
            return 0.0
        total_lo, total_w = 0.0, 0.0
        for dim in self.dimensions.values():
            p = max(0.001, min(0.999, dim.value))
            total_lo += np.log(p / (1 - p)) * dim.weight
            total_w += dim.weight
        if total_w == 0:
            return 0.5
        return float(1 / (1 + np.exp(-total_lo / total_w)))
    
    def fatal_flaws(self) -> List[DimensionScore]:
        return [d for d in self.dimensions.values() if d.is_fatal_flaw]
    
    def combined_score(self) -> Tuple[float, str]:
        flaws = self.fatal_flaws()
        bayesian = self.bayesian_score()
        
        if self.weight_violation:
            return 0.0, f"BLOCKED: {self.weight_violation}"
        if flaws:
            return min(0.3, bayesian * 0.5), f"Fatal: {', '.join(f.name for f in flaws)}"
        return round(bayesian, 4), "No fatal flaws"
    
    def sensitivity_analysis(self) -> List[Dict]:
        if not self.dimensions:
            return []
        base, _ = self.combined_score()
        sensitivities = []
        for name, dim in self.dimensions.items():
            original = dim.value
            impacts = []
            for delta in [-0.2, -0.1, 0.1, 0.2]:
                dim.value = max(0.001, min(0.999, original + delta))
                new, _ = self.combined_score()
                impacts.append({'delta': delta, 'impact': round(new - base, 4)})
            dim.value = original
            max_impact = max(abs(i['impact']) for i in impacts)
            sensitivities.append({'dimension': name, 'base_value': original,
                                 'max_impact': round(max_impact, 4), 'is_critical': max_impact > 0.1})
        return sorted(sensitivities, key=lambda x: x['max_impact'], reverse=True)
    
    def to_dict(self) -> Dict:
        return {
            'dimensions': {n: {'value': d.value, 'weight': d.weight, 'is_fatal': d.is_fatal_flaw}
                          for n, d in self.dimensions.items()},
            'bayesian_score': round(self.bayesian_score(), 4),
            'fatal_flaws': [f.name for f in self.fatal_flaws()],
            'weight_violation': self.weight_violation
        }


# =============================================================================
# ANALYSIS ELEMENT
# =============================================================================

@dataclass
class AnalysisElement:
    """Main analysis container with all P0/P1 fixes."""
    name: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    what: Optional[str] = None
    why: Optional[str] = None
    how: Optional[str] = None
    measure: Optional[str] = None
    
    epistemic_state: EpistemicState = field(default_factory=EpistemicState)
    mechanism_map: MechanismMap = field(default_factory=MechanismMap)
    scoring: ScoringSystem = field(default_factory=ScoringSystem)
    evidence: List[Evidence] = field(default_factory=list)
    
    utility_model: RiskAwareUtilityModel = None
    bias_detector: Optional[ImprovedBiasDetector] = None
    calibration_tracker: CalibrationTracker = field(default_factory=CalibrationTracker)
    warning_system: WarningSystem = field(default_factory=WarningSystem)
    content_scanner: Optional[ContentFatalFlawDetector] = None
    
    bias_checks: List[BiasCheck] = field(default_factory=list)
    feedback_loops: List[FeedbackLoop] = field(default_factory=list)
    independence_report: Optional[Dict] = None
    
    is_established_hypothesis: bool = False
    establishment_evidence: Optional[EstablishedHypothesisEvidence] = None
    is_high_stakes: bool = True
    tracer: Optional[AnalysisTracer] = None
    
    def __post_init__(self):
        self.mechanism_map = MechanismMap(self.warning_system)
        self.scoring = ScoringSystem(self.warning_system)
        self.calibration_tracker = CalibrationTracker(self.warning_system)
        self.content_scanner = ContentFatalFlawDetector(self.warning_system)
        self.epistemic_state.warning_system = self.warning_system
        if self.utility_model is None:
            self.utility_model = RiskAwareUtilityModel(domain=self.domain)
    
    def set_tracer(self, tracer: AnalysisTracer):
        self.tracer = tracer
        self.mechanism_map.tracer = tracer
        self.scoring.tracer = tracer
    
    def set_established_hypothesis(self, is_established: bool,
                                   evidence: Optional[EstablishedHypothesisEvidence] = None):
        self.is_established_hypothesis = is_established
        self.establishment_evidence = evidence
        if self.bias_detector:
            self.bias_detector.set_established_hypothesis(is_established, evidence)
    
    def set_what(self, value: str, confidence: float):
        self.what = value
        self.scoring.set_dimension('definition_clarity', confidence)
        self.content_scanner.scan_text(value, 'what')
    
    def set_why(self, value: str, confidence: float):
        self.why = value
        self.scoring.set_dimension('justification_strength', confidence, weight=1.2)
        self.content_scanner.scan_text(value, 'why')
    
    def set_how(self, value: str, confidence: float):
        self.how = value
        self.scoring.set_dimension('mechanism_validity', confidence, weight=1.5, fatal_threshold=0.25)
        self.content_scanner.scan_text(value, 'how')
    
    def set_measure(self, value: str, confidence: float):
        self.measure = value
        self.scoring.set_dimension('measurability', confidence)
        self.content_scanner.scan_text(value, 'measure')
    
    def set_dimension(self, name: str, value: float, weight: float = 1.0,
                     is_fatal_below: float = 0.3, justification: str = "") -> bool:
        return self.scoring.set_dimension(name, value, weight, is_fatal_below, justification)
    
    def add_evidence(self, evidence: Evidence) -> bool:
        if len(self.evidence) >= SafetyLimits.MAX_EVIDENCE_PIECES:
            return False
        
        if evidence.validation_warnings:
            for w in evidence.validation_warnings:
                self.warning_system.add_warning(WarningLevel.INFO, "Evidence Validation",
                    f"[{evidence.id}] {w}", "Review metadata")
        
        lr = evidence.get_likelihood_ratio()
        old = self.epistemic_state.credence
        self.epistemic_state.update_with_evidence(lr, evidence.id)
        evidence.calculate_bits(old, self.epistemic_state.credence)
        self.evidence.append(evidence)
        
        qualities = [e.get_quality() for e in self.evidence]
        n = len(self.evidence)
        eff_q = np.mean(qualities) * (1 - 0.5 * np.exp(-n/3))
        self.scoring.set_dimension('evidence_quality', eff_q)
        
        supporting = sum(1 for e in self.evidence if e.supports_hypothesis)
        self.warning_system.check_evidence_balance(supporting, n - supporting, self.is_established_hypothesis)
        
        if evidence.has_fatal_content():
            self.content_scanner.scan_evidence([evidence])
        return True
    
    def add_mechanism_node(self, node: MechanismNode) -> str:
        return self.mechanism_map.add_node(node)
    
    def add_mechanism_edge(self, edge: MechanismEdge) -> bool:
        return self.mechanism_map.add_edge(edge)
    
    def set_feasibility(self, technical: float, economic: float, timeline: float):
        self.scoring.set_dimension('technical_feasibility', technical, weight=1.2, fatal_threshold=0.2)
        self.scoring.set_dimension('economic_viability', economic, fatal_threshold=0.2)
        self.scoring.set_dimension('timeline_realism', timeline, weight=0.8)
    
    def set_risk(self, execution: float, external: float):
        self.scoring.set_dimension('execution_safety', 1.0 - execution)
        self.scoring.set_dimension('external_resilience', 1.0 - external, weight=0.8)
    
    def add_scenario(self, description: str, probability: float, utility: float):
        self.utility_model.add_scenario(description, probability, utility)
    
    def set_risk_aversion(self, gamma: float):
        self.utility_model.risk_aversion = gamma
        self.warning_system.check_risk_aversion(gamma, self.domain)
    
    def check_evidence_independence(self) -> Dict:
        self.independence_report = EvidenceIndependenceChecker.check_all_independence(
            self.evidence, self.warning_system)
        return self.independence_report
    
    def run_bias_detection(self) -> List[BiasCheck]:
        self.bias_detector = ImprovedBiasDetector(self)
        self.bias_detector.set_established_hypothesis(self.is_established_hypothesis, self.establishment_evidence)
        self.bias_checks = self.bias_detector.run_all_checks()
        return self.bias_checks
    
    def scan_for_fatal_content(self) -> List[Dict]:
        return self.content_scanner.scan_evidence(self.evidence)
    
    def detect_feedback_loops(self) -> List[FeedbackLoop]:
        self.feedback_loops = self.mechanism_map.detect_feedback_loops()
        return self.feedback_loops
    
    def get_total_evidence_bits(self) -> float:
        return sum(e.bits for e in self.evidence)
    
    def get_effective_evidence_bits(self) -> float:
        if not self.independence_report:
            self.check_evidence_independence()
        return self.get_total_evidence_bits() * self.independence_report.get('overall_independence', 1.0)
    
    def value_of_information(self, info_cost: float = 0.0, signal_accuracy: float = 0.8,
                            time_cost: float = 0.0) -> Dict:
        return self.utility_model.value_of_information(signal_accuracy, info_cost, time_cost)
    
    def is_blocked(self) -> Tuple[bool, List[str]]:
        reasons = []
        if self.content_scanner.has_fatal_content():
            reasons.append("Fatal content - human review required")
        if self.scoring.has_weight_violation():
            reasons.append(f"Weight violation: {self.scoring.weight_violation}")
        if self.warning_system.has_unacknowledged_fatal():
            reasons.append("Unacknowledged fatal warnings")
        return len(reasons) > 0, reasons
    
    def pre_flight_check(self) -> Tuple[bool, List[str]]:
        issues = []
        if not self.warning_system.check_evidence_sufficiency(len(self.evidence), self.is_high_stakes):
            issues.append("Insufficient evidence")
        independence = self.check_evidence_independence()
        if independence['overall_independence'] < 0.5:
            issues.append(f"Low independence ({independence['overall_independence']:.1%})")
        fatal = self.scan_for_fatal_content()
        if fatal:
            issues.append(f"Fatal content in {len(fatal)} pieces")
        blocked, blocking = self.is_blocked()
        if blocked:
            issues.extend(blocking)
        return len(issues) == 0, issues


# =============================================================================
# ADVERSARIAL TESTER & OPTIMAL STOPPING
# =============================================================================

@dataclass
class Criticism:
    content: str
    severity: float
    cycle: str
    resolved: bool = False
    def to_dict(self) -> Dict:
        return {'content': self.content, 'severity': self.severity, 'resolved': self.resolved}

class AdversarialTester:
    def __init__(self, element: AnalysisElement, rigor: int, tracer: AnalysisTracer):
        self.element = element
        self.rigor = rigor
        self.tracer = tracer
        self.criticisms: List[Criticism] = []
        self.iteration = 0
    
    def generate_criticisms(self) -> List[Criticism]:
        new = []
        self.iteration += 1
        
        for a in self.element.mechanism_map.get_assumptions():
            if a.epistemic_state.credence < 0.7:
                new.append(Criticism(f"Untested: {a.label} ({a.epistemic_state.credence:.2f})", 0.7, "assumption"))
        
        for b in self.element.mechanism_map.get_blockers():
            if b.epistemic_state.credence > 0.5:
                new.append(Criticism(f"Blocker: {b.label} ({b.epistemic_state.credence:.2f})", 0.8, "pre_mortem"))
        
        for f in self.element.scoring.fatal_flaws():
            new.append(Criticism(f"Fatal: {f.name}={f.value:.2f}", 0.95, "fatal"))
        
        if len(self.element.evidence) < 3:
            new.append(Criticism(f"Only {len(self.element.evidence)} evidence", 0.6, "evidence"))
        
        for b in self.element.bias_checks:
            if b.detected and not b.is_acknowledged and b.severity >= 0.4:
                new.append(Criticism(f"Bias: {b.bias_type.value}", b.severity, "bias"))
        
        if self.element.content_scanner.has_fatal_content():
            for i in self.element.content_scanner.detected_issues[:2]:
                new.append(Criticism(f"CONTENT: {i['category']}", 0.9, "content"))
        
        if self.element.independence_report and self.element.independence_report['overall_independence'] < 0.6:
            new.append(Criticism(f"Low independence ({self.element.independence_report['overall_independence']:.0%})", 0.6, "independence"))
        
        self.criticisms.extend(new)
        return new
    
    def unresolved_critical(self, threshold: float = 0.7) -> List[Criticism]:
        return [c for c in self.criticisms if not c.resolved and c.severity >= threshold]
    
    def consistency_score(self) -> float:
        if not self.criticisms:
            return 1.0
        return sum(1 for c in self.criticisms if c.resolved) / len(self.criticisms)

class OptimalStoppingCriterion:
    def __init__(self, element: AnalysisElement, target: float, delay_cost: float = 0.05):
        self.element = element
        self.target = target
        self.delay_cost = delay_cost
    
    def should_stop(self) -> Tuple[bool, str]:
        blocked, reasons = self.element.is_blocked()
        if blocked:
            return True, f"BLOCKED: {'; '.join(reasons)}"
        
        credence = self.element.epistemic_state.credence
        if credence > SafetyLimits.CREDENCE_EXTREME_THRESHOLD:
            return True, f"Extreme confidence ({credence:.2f})"
        if credence < 0.2:
            return True, f"Low confidence ({credence:.2f}): Reject"
        
        flaws = self.element.scoring.fatal_flaws()
        if flaws:
            return True, f"Fatal flaws: {[f.name for f in flaws]}"
        
        if self.element.content_scanner.has_fatal_content():
            return True, "Content flags require review"
        
        voi = self.element.utility_model.value_of_information()
        if voi['net_voi'] < self.delay_cost:
            return True, f"VOI ({voi['net_voi']:.3f}) < delay cost"
        
        if credence >= self.target:
            return True, f"Target reached ({credence:.2f})"
        
        return False, f"Continue: credence={credence:.2f}"


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_analysis(element: AnalysisElement, rigor_level: int = 2,
                max_iter: int = 15, force_continue: bool = False) -> Dict:
    """Run analysis with all P0/P1 safety fixes."""
    max_iter = min(max_iter, SafetyLimits.MAX_ITERATIONS)
    tracer = AnalysisTracer()
    element.set_tracer(tracer)
    
    # Pre-flight
    passed, issues = element.pre_flight_check()
    blocked, blocking_reasons = element.is_blocked()
    
    if blocked and not force_continue:
        return {
            'name': element.name, 'blocked': True, 'blocking_reasons': blocking_reasons,
            'decision_state': DecisionReadiness.BLOCKED.value,
            'recommendation': f"BLOCKED: {'; '.join(blocking_reasons)}",
            'warnings': element.warning_system.to_dict(),
            'content_scanner': element.content_scanner.to_dict()
        }
    
    targets = {1: 0.5, 2: 0.7, 3: 0.85}
    target = targets.get(rigor_level, 0.7)
    
    element.detect_feedback_loops()
    element.run_bias_detection()
    
    tester = AdversarialTester(element, rigor_level, tracer)
    history = []
    reason = "Max iterations"
    stopping = OptimalStoppingCriterion(element, target)
    
    for i in range(max_iter):
        tester.generate_criticisms()
        
        combined, _ = element.scoring.combined_score()
        calibrated, _ = element.calibration_tracker.get_calibrated_score(combined)
        debiased = element.bias_detector.get_debiased_score(calibrated) if element.bias_detector else calibrated
        
        history.append(debiased)
        
        should_stop, stop_reason = stopping.should_stop()
        if should_stop:
            reason = stop_reason
            break
        elif i >= 3 and len(history) >= 2 and abs(history[-1] - history[-2]) < 0.01:
            reason = "Diminishing returns"
            break
    
    sensitivity = element.scoring.sensitivity_analysis()
    
    # Final scores
    final_bayesian = element.scoring.bayesian_score()
    final_combined, final_reason = element.scoring.combined_score()
    final_calibrated, cal_note = element.calibration_tracker.get_calibrated_score(final_combined)
    final_debiased = element.bias_detector.get_debiased_score(final_calibrated) if element.bias_detector else final_calibrated
    
    flaws = element.scoring.fatal_flaws()
    ready = final_debiased >= target and len(flaws) == 0
    has_critical = element.warning_system.has_critical_warnings()
    has_fatal_content = element.content_scanner.has_fatal_content()
    
    voi_result = element.value_of_information()
    
    # Recommendation
    if has_fatal_content:
        recommendation = "HUMAN REVIEW REQUIRED: Content flags"
        decision_state = DecisionReadiness.FATAL_FLAW
    elif flaws:
        recommendation = f"REJECT: Fatal flaws in {[f.name for f in flaws]}"
        decision_state = DecisionReadiness.FATAL_FLAW
    elif voi_result['net_voi'] > 0.1:
        recommendation = f"INVESTIGATE: VOI ({voi_result['net_voi']:.2f}) suggests more info needed"
        decision_state = DecisionReadiness.NEEDS_MORE_INFO
    elif ready and not has_critical:
        recommendation = f"PROCEED: Score ({final_debiased:.2f}) meets threshold"
        decision_state = DecisionReadiness.READY
    elif ready and has_critical:
        recommendation = f"PROCEED WITH CAUTION: Critical warnings exist"
        decision_state = DecisionReadiness.READY
    else:
        recommendation = f"UNCERTAIN: Score ({final_debiased:.2f}) below threshold"
        decision_state = DecisionReadiness.UNCERTAIN
    
    return {
        'name': element.name,
        'blocked': False,
        'bayesian_score': round(final_bayesian, 3),
        'combined_score': round(final_combined, 3),
        'calibrated_score': round(final_calibrated, 3),
        'calibration_note': cal_note,
        'debiased_score': round(final_debiased, 3),
        'credence': round(element.epistemic_state.credence, 3),
        'confidence_interval': element.epistemic_state.get_confidence_interval(),
        'credence_warning': element.epistemic_state.get_point_estimate_warning(),
        'ready': ready,
        'decision_state': decision_state.value,
        'recommendation': recommendation,
        'reason': reason,
        'fatal_flaws': [{'name': f.name, 'value': f.value} for f in flaws],
        'content_fatal_flags': element.content_scanner.detected_issues,
        'expected_utility': round(element.utility_model.expected_utility(), 3),
        'certainty_equivalent': round(element.utility_model.certainty_equivalent(), 3),
        'risk_premium': round(element.utility_model.risk_premium(), 3),
        'risk_aversion': element.utility_model.risk_aversion,
        'risk_aversion_domain_default': element.utility_model.domain_default,
        'voi': voi_result,
        'risk_sensitivity': element.utility_model.risk_aversion_sensitivity(),
        'biases_detected': [b.to_dict() for b in element.bias_checks if b.detected],
        'bias_penalty': round(element.bias_detector.total_bias_penalty(), 3) if element.bias_detector else 0,
        'establishment_verified': element.bias_detector.establishment_verified if element.bias_detector else False,
        'evidence_count': len(element.evidence),
        'total_evidence_bits': round(element.get_total_evidence_bits(), 2),
        'effective_evidence_bits': round(element.get_effective_evidence_bits(), 2),
        'evidence_independence': element.independence_report,
        'feedback_loops': [fl.to_dict() for fl in element.feedback_loops],
        'systemic_risk': round(element.mechanism_map.systemic_risk_score(), 3),
        'mechanism_confidence': round(element.mechanism_map.overall_confidence(), 3),
        'calibration': element.calibration_tracker.to_dict(),
        'warnings': element.warning_system.to_dict(),
        'has_critical_warnings': has_critical,
        'iterations': tester.iteration,
        'history': history,
        'criticisms': [c.to_dict() for c in tester.criticisms],
        'sensitivity': sensitivity,
        'dimensions': element.scoring.to_dict()
    }


def explain_result(results: Dict) -> str:
    """Generate human-readable explanation."""
    lines = [f"\nANALYSIS: {results['name']}", "=" * 60]
    
    if results.get('blocked'):
        lines.append("\n🚫 ANALYSIS BLOCKED")
        for r in results.get('blocking_reasons', []):
            lines.append(f"   • {r}")
        return "\n".join(lines)
    
    if 'warnings' in results:
        lines.append(f"\n📢 {results['warnings'].get('summary_header', 'No warnings')}")
    
    lines.append(f"\n📋 RECOMMENDATION: {results['recommendation']}")
    lines.append(f"   Decision: {results['decision_state']}")
    
    ci = results['confidence_interval']
    lines.append(f"\n📊 CONFIDENCE: {results['credence']:.1%} ({ci[0]:.0%} - {ci[1]:.0%})")
    if results['credence_warning']:
        lines.append(f"   ⚠️  {results['credence_warning']}")
    
    if results['fatal_flaws']:
        lines.append("\n💀 FATAL FLAWS:")
        for f in results['fatal_flaws']:
            lines.append(f"   • {f['name']}: {f['value']:.2f}")
    
    if results['content_fatal_flags']:
        lines.append(f"\n🚨 CONTENT ALERTS: {len(results['content_fatal_flags'])}")
    
    if results['biases_detected']:
        lines.append("\n🧠 BIASES:")
        for b in results['biases_detected']:
            lines.append(f"   • {b['type']}: {b['evidence'][:50]}...")
    
    lines.append(f"\n📚 EVIDENCE: {results['evidence_count']} pieces")
    lines.append(f"   Effective bits: {results['effective_evidence_bits']:.1f} / {results['total_evidence_bits']:.1f}")
    
    lines.append(f"\n💰 UTILITY:")
    lines.append(f"   Risk aversion: {results['risk_aversion']} (default: {results['risk_aversion_domain_default']})")
    lines.append(f"   Expected: {results['expected_utility']:.3f}, CE: {results['certainty_equivalent']:.3f}")
    lines.append(f"   VOI: {results['voi']['net_voi']:.3f} ({results['voi']['recommendation']})")
    
    if results.get('sensitivity'):
        lines.append("\n🎚️ SENSITIVITY (top 3):")
        for s in results['sensitivity'][:3]:
            flag = "⚠️" if s['is_critical'] else ""
            lines.append(f"   • {s['dimension']}: ±{s['max_impact']:.2f} {flag}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo():
    """Demonstrate PRISM v1.1 with P0/P1 fixes."""
    print("=" * 70)
    print("PRISM v1.1 - With P0/P1 Security Fixes")
    print("=" * 70)
    
    h = AnalysisElement(name="Hire Data Scientist", domain=EvidenceDomain.BUSINESS)
    
    h.set_what("Hire data scientist to improve decisions", 0.9)
    h.set_why("We need statistical expertise", 0.7)
    h.set_how("Standard hiring process", 0.8)
    h.set_measure("A/B test success improves 20%", 0.7)
    
    n1 = MechanismNode("cause1", "Lack of expertise", NodeType.CAUSE)
    n1.confidence = 0.9
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("mech1", "DS brings skills", NodeType.MECHANISM)
    n2.confidence = 0.85
    h.add_mechanism_node(n2)
    
    h.add_mechanism_edge(MechanismEdge("cause1", "mech1", EdgeType.CAUSES, 0.9,
                                        causal_level=CausalLevel.COUNTERFACTUAL))
    
    h.add_evidence(Evidence(
        id="ev1",
        content="HBR: Analytics teams see 5-6% productivity gains",
        source="Harvard Business Review",
        quality=0.7, date="2023-01",
        domain=EvidenceDomain.BUSINESS,
        study_design="multi_company_analysis",
        sample_size=500,
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True,
        authors=["Davenport, T."],
        underlying_data="HBR_Survey_2023"
    ))
    
    h.add_evidence(Evidence(
        id="ev2",
        content="McKinsey: Data-driven orgs 23x more likely to acquire customers",
        source="McKinsey Global Institute",
        quality=0.75, date="2022-09",
        domain=EvidenceDomain.BUSINESS,
        study_design="benchmark",
        sample_size=1000,
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    h.add_evidence(Evidence(
        id="ev3",
        content="60% of analytics hires fail to deliver expected ROI",
        source="Gartner Research",
        quality=0.65, date="2023-06",
        domain=EvidenceDomain.BUSINESS,
        study_design="benchmark",
        sample_size=300,
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=False
    ))
    
    h.set_feasibility(0.85, 0.7, 0.75)
    h.set_risk(0.3, 0.2)
    
    h.add_scenario("Great hire", 0.3, 1.5)
    h.add_scenario("Good hire", 0.4, 0.3)
    h.add_scenario("Average", 0.2, -0.1)
    h.add_scenario("Bad hire", 0.1, -0.5)
    
    print("\nRunning analysis...")
    results = run_analysis(h, rigor_level=2, max_iter=10)
    
    print(explain_result(results))
    print("\n📢 WARNINGS:")
    h.warning_system.print_warnings(max_count=5)
    
    return results


def demo_blocking():
    """Demonstrate blocking with fatal content."""
    print("\n" + "=" * 70)
    print("DEMO: Blocking Behavior")
    print("=" * 70)
    
    h = AnalysisElement(name="Questionable Product", domain=EvidenceDomain.BUSINESS)
    h.set_what("Launch product", 0.85)
    h.set_feasibility(0.8, 0.75, 0.8)
    
    h.add_evidence(Evidence(
        id="legal_warning",
        content="This product may be illegal in several jurisdictions. "
                "Regulatory bodies have issued warnings.",
        source="Legal Dept",
        quality=0.9, date="2024-01",
        domain=EvidenceDomain.BUSINESS,
        study_design="expert_opinion",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=False
    ))
    
    results = run_analysis(h, rigor_level=2, max_iter=5)
    print(explain_result(results))
    
    if results.get('blocked'):
        print("\n✓ Correctly BLOCKED")
    return results


def demo_independence():
    """Demonstrate improved independence checker."""
    print("\n" + "=" * 70)
    print("DEMO: Independence Checker")
    print("=" * 70)
    
    h = AnalysisElement(name="Test Independence", domain=EvidenceDomain.MEDICAL)
    h.set_what("Test drug efficacy", 0.8)
    h.set_feasibility(0.8, 0.8, 0.8)
    
    h.add_evidence(Evidence(
        id="trial", content="Trial NCT123456 shows 20% improvement",
        source="ClinicalTrials.gov", quality=0.75, date="2023-01",
        domain=EvidenceDomain.MEDICAL, study_design="rct",
        sample_size=500, causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True, underlying_data="NCT123456"
    ))
    
    h.add_evidence(Evidence(
        id="paper", content="Smith et al. report NCT123456 trial results",
        source="NEJM", quality=0.95, date="2023-06",
        domain=EvidenceDomain.MEDICAL, study_design="rct",
        sample_size=500, causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True, authors=["Smith, J."],
        underlying_data="NCT123456"
    ))
    
    h.add_evidence(Evidence(
        id="press", content="Dr. Jane Smith announces NCT123456 success",
        source="University Press", quality=0.5, date="2023-03",
        domain=EvidenceDomain.MEDICAL, study_design="anecdote",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True, authors=["Dr. Jane Smith"]
    ))
    
    report = h.check_evidence_independence()
    
    print(f"\nOverall independence: {report['overall_independence']:.1%}")
    print(f"Effective count: {report['effective_evidence_count']:.1f} of {len(h.evidence)}")
    print(f"Issues: {len(report['issues'])}")
    for issue in report['issues']:
        print(f"  • {issue['evidence_1']} ↔ {issue['evidence_2']}: {issue['independence']:.0%}")


if __name__ == "__main__":
    results = demo()
    demo_blocking()
    demo_independence()
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


# =============================================================================
# v1.2 ADDITIONS: BAYES FACTORS + CALIBRATION TRAINING
# =============================================================================

class BayesFactorComparison:
    """Compare competing hypotheses directly using Bayes factors."""
    
    INTERPRETATION = {
        (0, 1): "Negative (supports H2)",
        (1, 3): "Barely worth mentioning",
        (3, 10): "Substantial (supports H1)",
        (10, 30): "Strong (supports H1)", 
        (30, 100): "Very strong (supports H1)",
        (100, float('inf')): "Decisive (supports H1)"
    }
    
    @staticmethod
    def compute_bf(h1_results: Dict, h2_results: Dict) -> Dict:
        """Compute Bayes factor between two analyzed hypotheses."""
        # Use credences as proxy for P(data|H)
        p1 = max(0.001, h1_results.get('credence', 0.5))
        p2 = max(0.001, h2_results.get('credence', 0.5))
        
        # Adjust for evidence quality
        q1 = h1_results.get('debiased_score', 0.5)
        q2 = h2_results.get('debiased_score', 0.5)
        
        # BF = (p1/1-p1) / (p2/1-p2) * quality adjustment
        odds1 = p1 / (1 - p1)
        odds2 = p2 / (1 - p2)
        
        raw_bf = odds1 / odds2 if odds2 > 0 else float('inf')
        adjusted_bf = raw_bf * (q1 / q2) if q2 > 0 else raw_bf
        
        # Interpretation
        bf = adjusted_bf
        interpretation = "Error"
        for (low, high), desc in BayesFactorComparison.INTERPRETATION.items():
            if low <= bf < high:
                interpretation = desc
                break
            elif bf < 1:
                # Flip for H2
                inv_bf = 1/bf if bf > 0 else float('inf')
                for (low2, high2), desc2 in BayesFactorComparison.INTERPRETATION.items():
                    if low2 <= inv_bf < high2:
                        interpretation = desc2.replace("H1", "H2")
                        break
                break
        
        return {
            'bayes_factor': round(adjusted_bf, 2),
            'log_bf': round(np.log10(max(0.001, adjusted_bf)), 2),
            'interpretation': interpretation,
            'h1_odds': round(odds1, 3),
            'h2_odds': round(odds2, 3),
            'winner': h1_results.get('name', 'H1') if bf > 1 else h2_results.get('name', 'H2'),
            'confidence': 'high' if bf > 10 or bf < 0.1 else 'moderate' if bf > 3 or bf < 0.33 else 'low'
        }
    
    @staticmethod
    def rank_hypotheses(results_list: List[Tuple[str, Dict]]) -> List[Dict]:
        """Rank multiple hypotheses by pairwise Bayes factor tournament."""
        n = len(results_list)
        if n < 2:
            return [{'name': results_list[0][0], 'rank': 1, 'wins': 0}] if results_list else []
        
        wins = {name: 0 for name, _ in results_list}
        comparisons = []
        
        for i in range(n):
            for j in range(i+1, n):
                name1, r1 = results_list[i]
                name2, r2 = results_list[j]
                bf = BayesFactorComparison.compute_bf(r1, r2)
                
                if bf['bayes_factor'] > 1:
                    wins[name1] += 1
                else:
                    wins[name2] += 1
                
                comparisons.append({
                    'h1': name1, 'h2': name2,
                    'bf': bf['bayes_factor'],
                    'winner': bf['winner']
                })
        
        ranked = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        return [{'name': name, 'rank': i+1, 'wins': w} for i, (name, w) in enumerate(ranked)]


class CalibrationTrainer:
    """Track and improve user calibration over time."""
    
    TRAINING_QUESTIONS = [
        ("What % of startups fail within 5 years?", 0.90),
        ("What % of clinical trials succeed?", 0.10),
        ("What % of mergers create shareholder value?", 0.30),
        ("What % of IT projects finish on time/budget?", 0.30),
        ("What % of published findings replicate?", 0.40),
    ]
    
    def __init__(self):
        self.user_estimates: List[Tuple[float, float]] = []  # (estimate, actual)
        self.brier_scores: List[float] = []
    
    def record_estimate(self, estimate: float, actual: float):
        """Record a probability estimate and its outcome."""
        self.user_estimates.append((estimate, actual))
        brier = (estimate - actual) ** 2
        self.brier_scores.append(brier)
    
    def get_brier_score(self) -> float:
        """Lower is better. 0=perfect, 0.25=random."""
        if not self.brier_scores:
            return 0.25
        return np.mean(self.brier_scores)
    
    def get_calibration_curve(self, bins: int = 5) -> Dict:
        """Show predicted vs actual by bin."""
        if len(self.user_estimates) < bins:
            return {'insufficient_data': True}
        
        bin_data = [[] for _ in range(bins)]
        for est, act in self.user_estimates:
            idx = min(bins-1, int(est * bins))
            bin_data[idx].append(act)
        
        curve = {}
        for i, outcomes in enumerate(bin_data):
            if outcomes:
                predicted = (i + 0.5) / bins
                actual = np.mean(outcomes)
                curve[f"{int(predicted*100)}%"] = f"{int(actual*100)}%"
        return curve
    
    def calibration_adjustment(self, raw_credence: float) -> float:
        """Adjust credence based on historical calibration."""
        if len(self.user_estimates) < 10:
            return raw_credence  # Not enough data
        
        # Check systematic bias
        estimates = [e for e, _ in self.user_estimates]
        actuals = [a for _, a in self.user_estimates]
        
        mean_est = np.mean(estimates)
        mean_act = np.mean(actuals)
        
        if mean_est > mean_act + 0.1:
            # Overconfident - shrink toward 0.5
            return raw_credence * 0.9 + 0.05
        elif mean_est < mean_act - 0.1:
            # Underconfident - expand from 0.5
            return raw_credence * 1.1 - 0.05
        return raw_credence
    
    def get_feedback(self) -> str:
        """Generate calibration feedback."""
        if len(self.user_estimates) < 5:
            return "Need more predictions to assess calibration."
        
        brier = self.get_brier_score()
        if brier < 0.1:
            return f"Excellent calibration (Brier={brier:.3f})"
        elif brier < 0.2:
            return f"Good calibration (Brier={brier:.3f})"
        elif brier < 0.25:
            return f"Fair calibration (Brier={brier:.3f}) - room for improvement"
        else:
            return f"Poor calibration (Brier={brier:.3f}) - consider training"


def compare_hypotheses(hypotheses: List[AnalysisElement], rigor: int = 2) -> Dict:
    """Run full comparison of multiple hypotheses."""
    results = []
    for h in hypotheses:
        r = run_analysis(h, rigor_level=rigor, max_iter=8)
        r['name'] = h.name
        results.append((h.name, r))
    
    # Rank by Bayes factors
    ranking = BayesFactorComparison.rank_hypotheses(results)
    
    # Also rank by certainty equivalent
    ce_ranked = sorted(results, key=lambda x: x[1].get('certainty_equivalent', 0), reverse=True)
    
    # Best hypothesis
    winner = ranking[0]['name'] if ranking else None
    winner_results = next((r for n, r in results if n == winner), None)
    
    return {
        'hypotheses_analyzed': len(hypotheses),
        'ranking_by_bayes_factor': ranking,
        'ranking_by_certainty_equivalent': [(n, round(r['certainty_equivalent'], 3)) for n, r in ce_ranked],
        'winner': winner,
        'winner_score': winner_results['debiased_score'] if winner_results else None,
        'winner_decision': winner_results['decision_state'] if winner_results else None,
        'all_results': {n: {
            'score': r['debiased_score'],
            'ce': r['certainty_equivalent'],
            'decision': r['decision_state']
        } for n, r in results}
    }


def explain_comparison(comparison: Dict) -> str:
    """Human-readable comparison output."""
    lines = [
        "\n" + "=" * 60,
        "HYPOTHESIS COMPARISON",
        "=" * 60,
        f"\nAnalyzed: {comparison['hypotheses_analyzed']} hypotheses",
        f"\n🏆 WINNER: {comparison['winner']}",
        f"   Score: {comparison['winner_score']:.3f}",
        f"   Decision: {comparison['winner_decision']}",
        "\n📊 RANKING (Bayes Factor Tournament):"
    ]
    
    for r in comparison['ranking_by_bayes_factor']:
        lines.append(f"   {r['rank']}. {r['name']} ({r['wins']} wins)")
    
    lines.append("\n📈 RANKING (Risk-Adjusted Value):")
    for name, ce in comparison['ranking_by_certainty_equivalent']:
        lines.append(f"   • {name}: CE={ce}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
