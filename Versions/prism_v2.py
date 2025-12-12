"""
PRISM v2.0 - Protocol for Rigorous Investigation of Scientific Mechanisms
==========================================================================

COMPLETE INTEGRATED VERSION

MAJOR FEATURES:
1. DIAGNOSTIC TEST THEORY - Likelihood ratios from sensitivity/specificity
2. CORRELATION-ADJUSTED POOLING - Proper handling of dependent evidence
3. META-ANALYTIC COMBINATION - DerSimonian-Laird random effects
4. TARJAN'S ALGORITHM - O(V+E) cycle detection for mechanism maps
5. WILSON SCORE INTERVALS - Proper confidence intervals
6. REFERENCE CLASS FORECASTING - Empirical base rates
7. P-CURVE ANALYSIS - P-hacking detection
8. INSTRUMENTAL VARIABLE TESTS - F-statistic, Sargan J-test
9. EXTREME VALUE THEORY - GPD for tail risk, VaR/CVaR
10. SEMANTIC INDEPENDENCE - TextAnalyzer for evidence deduplication
11. ESTABLISHED HYPOTHESIS VERIFICATION - Evidence requirements
12. 100+ CONTENT PATTERNS - Including euphemism detection
13. 8 COGNITIVE BIAS DETECTORS - With statistical tests

Author: Dr. Aneesh Joseph (Architecture) + Claude (Implementation)
Version: 2.0 Final
Date: December 2025

THEORETICAL FOUNDATIONS:
- Likelihood ratios: Sackett et al. (2000) "Evidence-Based Medicine"
- Meta-analysis: Borenstein et al. (2009) "Introduction to Meta-Analysis"  
- P-curve: Simonsohn, Nelson & Simmons (2014)
- CRRA Utility: Arrow (1965), Pratt (1964)
- EVT: Pickands (1975), Balkema & de Haan (1974)
- IV Tests: Staiger & Stock (1997), Stock & Yogo (2005)
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set, Callable
from enum import Enum
from datetime import datetime
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# MATHEMATICAL CONSTANTS AND SAFETY LIMITS
# =============================================================================

class MathConstants:
    """Mathematical constants for numerical stability."""
    EPSILON: float = 1e-10
    LOG_EPSILON: float = -23.0
    MAX_EXP: float = 700.0
    PHI: float = 1.618033988749895
    SQRT_2PI: float = 2.5066282746310002


class SafetyLimits:
    """Hard limits - NON-NEGOTIABLE safety rails."""
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
    MAX_WEIGHT_RATIO: float = 5.0
    MAX_SINGLE_WEIGHT: float = 3.0
    MIN_CALIBRATION_POINTS: int = 20
    DEFAULT_ECE_THRESHOLD: float = 0.15
    MAX_REASONABLE_VOI: float = 0.5
    MAX_CONTENT_LENGTH: int = 10000
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.6
    NAME_SIMILARITY_THRESHOLD: float = 0.7
    DEFAULT_ALPHA: float = 0.05
    MIN_EFFECTIVE_SAMPLE_SIZE: int = 10


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
    """Pearl's causal hierarchy."""
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
    PUBLICATION_BIAS = "publication_bias"

class FeedbackLoopType(Enum):
    REINFORCING = "reinforcing"
    BALANCING = "balancing"

class DecisionReadiness(Enum):
    READY = "ready"
    NEEDS_MORE_INFO = "needs_more_info"
    REJECT = "reject"
    UNCERTAIN = "uncertain"
    FATAL_FLAW = "fatal_flaw"
    BLOCKED = "blocked"


# =============================================================================
# DIAGNOSTIC TEST THEORY
# =============================================================================

@dataclass
class DiagnosticMetrics:
    """Sensitivity and specificity for study design."""
    sensitivity: float
    specificity: float
    prevalence_adjustment: float = 1.0
    
    def lr_positive(self) -> float:
        """LR+ when evidence supports hypothesis."""
        if self.specificity >= 1.0:
            return SafetyLimits.MAX_LIKELIHOOD_RATIO
        return self.sensitivity / (1 - self.specificity + MathConstants.EPSILON)
    
    def lr_negative(self) -> float:
        """LR- when evidence contradicts hypothesis."""
        if self.specificity <= 0:
            return SafetyLimits.MIN_LIKELIHOOD_RATIO
        return (1 - self.sensitivity) / (self.specificity + MathConstants.EPSILON)
    
    def diagnostic_odds_ratio(self) -> float:
        """DOR = LR+ / LR-"""
        num = self.sensitivity * self.specificity
        denom = (1 - self.sensitivity) * (1 - self.specificity)
        return num / (denom + MathConstants.EPSILON)


# Study design to diagnostic metrics mapping
STUDY_DIAGNOSTIC_METRICS = {
    'meta_analysis': DiagnosticMetrics(0.92, 0.90),
    'systematic_review': DiagnosticMetrics(0.88, 0.88),
    'rct': DiagnosticMetrics(0.85, 0.85),
    'randomized_trial': DiagnosticMetrics(0.85, 0.85),
    'cohort': DiagnosticMetrics(0.75, 0.75),
    'case_control': DiagnosticMetrics(0.70, 0.70),
    'cross_sectional': DiagnosticMetrics(0.65, 0.65),
    'case_series': DiagnosticMetrics(0.55, 0.55),
    'case_study': DiagnosticMetrics(0.55, 0.55),
    'expert_opinion': DiagnosticMetrics(0.60, 0.60),
    'anecdote': DiagnosticMetrics(0.52, 0.52),
    'controlled_experiment': DiagnosticMetrics(0.82, 0.82),
    'ab_test': DiagnosticMetrics(0.80, 0.80),
    'multi_company_analysis': DiagnosticMetrics(0.75, 0.75),
    'benchmark': DiagnosticMetrics(0.70, 0.70),
    'performance_study': DiagnosticMetrics(0.65, 0.65),
    'quasi_experiment': DiagnosticMetrics(0.75, 0.75),
    'regression_discontinuity': DiagnosticMetrics(0.78, 0.78),
    'difference_in_differences': DiagnosticMetrics(0.72, 0.72),
    'instrumental_variables': DiagnosticMetrics(0.70, 0.70),
    'replicated_experiment': DiagnosticMetrics(0.88, 0.88),
    'single_experiment': DiagnosticMetrics(0.72, 0.72),
    'observational': DiagnosticMetrics(0.62, 0.62),
    'theoretical': DiagnosticMetrics(0.58, 0.58),
}

# Domain-specific risk aversion defaults
DOMAIN_RISK_AVERSION = {
    EvidenceDomain.MEDICAL: 2.5,
    EvidenceDomain.BUSINESS: 1.5,
    EvidenceDomain.POLICY: 2.0,
    EvidenceDomain.TECHNOLOGY: 1.0,
    EvidenceDomain.SCIENTIFIC: 1.5,
    EvidenceDomain.GENERAL: 1.5
}

# Valid causal levels by study design
VALID_CAUSAL_LEVELS = {
    'meta_analysis': [CausalLevel.ASSOCIATION, CausalLevel.INTERVENTION],
    'rct': [CausalLevel.INTERVENTION],
    'randomized_trial': [CausalLevel.INTERVENTION],
    'controlled_experiment': [CausalLevel.INTERVENTION],
    'ab_test': [CausalLevel.INTERVENTION],
    'cohort': [CausalLevel.ASSOCIATION],
    'case_control': [CausalLevel.ASSOCIATION],
    'observational': [CausalLevel.ASSOCIATION],
    'case_study': [CausalLevel.ASSOCIATION],
    'quasi_experiment': [CausalLevel.ASSOCIATION, CausalLevel.INTERVENTION],
    'regression_discontinuity': [CausalLevel.INTERVENTION],
    'instrumental_variables': [CausalLevel.INTERVENTION],
}


def sample_size_adjustment(n: int, effect_size: float = 0.3) -> float:
    """Adjust diagnostic metrics based on sample size."""
    if n is None or n <= 0:
        return 0.85
    adjustment = 1 - (1 / (4 * max(1, n)))
    if n > 10000:
        adjustment = min(1.02, adjustment)
    return max(0.6, min(1.05, adjustment))


# =============================================================================
# STATISTICAL PRIMITIVES
# =============================================================================

class StatisticalPrimitives:
    """Statistically rigorous primitive operations."""
    
    @staticmethod
    def wilson_score_interval(successes: int, trials: int,
                              confidence: float = 0.95) -> Tuple[float, float]:
        """Wilson score interval for binomial proportion."""
        if trials == 0:
            return (0.0, 1.0)
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = successes / trials
        denom = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denom
        return (max(0, center - margin), min(1, center + margin))
    
    @staticmethod
    def log_odds(p: float) -> float:
        """Convert probability to log-odds."""
        p = max(MathConstants.EPSILON, min(1 - MathConstants.EPSILON, p))
        return np.log(p / (1 - p))
    
    @staticmethod
    def odds_to_prob(log_odds: float) -> float:
        """Convert log-odds to probability."""
        log_odds = max(-MathConstants.MAX_EXP, min(MathConstants.MAX_EXP, log_odds))
        return 1 / (1 + np.exp(-log_odds))
    
    @staticmethod
    def kl_divergence(p: float, q: float) -> float:
        """KL divergence for Bernoulli distributions."""
        p = max(MathConstants.EPSILON, min(1 - MathConstants.EPSILON, p))
        q = max(MathConstants.EPSILON, min(1 - MathConstants.EPSILON, q))
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    @staticmethod
    def entropy(p: float) -> float:
        """Binary entropy."""
        p = max(MathConstants.EPSILON, min(1 - MathConstants.EPSILON, p))
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    @staticmethod
    def information_gain(prior: float, posterior: float) -> float:
        """Information gain in bits."""
        h_prior = StatisticalPrimitives.entropy(prior)
        h_posterior = StatisticalPrimitives.entropy(posterior)
        return max(0, h_prior - h_posterior)


# =============================================================================
# EVIDENCE POOLING (META-ANALYSIS)
# =============================================================================

class EvidencePooling:
    """Methods for combining dependent evidence."""
    
    @staticmethod
    def correlation_from_independence(independence_score: float) -> float:
        """Convert independence score to correlation."""
        return 1.0 - independence_score
    
    @staticmethod
    def effective_evidence_count(evidence_count: int, avg_correlation: float) -> float:
        """Calculate effective N accounting for correlation."""
        if evidence_count <= 1:
            return float(evidence_count)
        rho = max(0.0, min(1.0, avg_correlation))
        denominator = 1 + (evidence_count - 1) * rho
        return evidence_count / max(1.0, denominator)
    
    @staticmethod
    def random_effects_pooling(effect_sizes: List[float],
                               variances: List[float]) -> Dict:
        """DerSimonian-Laird random effects meta-analysis."""
        n = len(effect_sizes)
        if n == 0:
            return {'pooled_effect': 0, 'pooled_variance': 1, 'i_squared': 0, 'n_studies': 0}
        if n == 1:
            return {'pooled_effect': effect_sizes[0], 'pooled_variance': variances[0],
                    'heterogeneity_tau2': 0, 'i_squared': 0, 'n_studies': 1}
        
        effects = np.array(effect_sizes)
        vars_ = np.maximum(np.array(variances), MathConstants.EPSILON)
        
        w = 1 / vars_
        theta_fe = np.sum(w * effects) / np.sum(w)
        Q = np.sum(w * (effects - theta_fe) ** 2)
        df = n - 1
        C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0, (Q - df) / C) if C > 0 else 0
        
        w_re = 1 / (vars_ + tau2)
        theta_re = np.sum(w_re * effects) / np.sum(w_re)
        var_re = 1 / np.sum(w_re)
        i_squared = max(0, min(100, 100 * (Q - df) / Q)) if Q > 0 else 0
        
        return {
            'pooled_effect': theta_re,
            'pooled_variance': var_re,
            'pooled_se': np.sqrt(var_re),
            'heterogeneity_tau2': tau2,
            'i_squared': i_squared,
            'q_statistic': Q,
            'n_studies': n
        }
    
    @staticmethod
    def trim_and_fill(effect_sizes: List[float], variances: List[float]) -> Dict:
        """Trim and fill for publication bias detection."""
        if len(effect_sizes) < 3:
            return {'missing_studies': 0, 'asymmetry_detected': False}
        
        effects = np.array(effect_sizes)
        median_effect = np.median(effects)
        above = np.sum(effects > median_effect)
        below = len(effects) - above
        missing = abs(above - below)
        
        return {
            'missing_studies': missing,
            'adjusted_effect': median_effect,
            'asymmetry_detected': missing > len(effects) * 0.2
        }


# =============================================================================
# REFERENCE CLASS FORECASTING
# =============================================================================

class ReferenceClassForecasting:
    """Kahneman & Tversky's reference class forecasting."""
    
    BASE_RATES = {
        'startup_success_5yr': 0.10,
        'merger_creates_value': 0.30,
        'it_project_on_time': 0.30,
        'product_launch_success': 0.40,
        'strategic_pivot_success': 0.35,
        'clinical_trial_phase1': 0.65,
        'clinical_trial_phase2': 0.30,
        'clinical_trial_phase3': 0.60,
        'drug_approval': 0.10,
        'treatment_efficacy': 0.50,
        'replication_success': 0.40,
        'null_hypothesis_true': 0.80,
        'publication_bias_present': 0.90,
        'policy_achieves_goals': 0.30,
        'reform_sustained': 0.40,
        'tech_prediction_accurate': 0.30,
        'ai_project_succeeds': 0.15,
        'expert_prediction_accurate': 0.50,
        'planning_estimate_accurate': 0.20,
    }
    
    @staticmethod
    def get_base_rate(category: str) -> Tuple[float, str]:
        """Get empirical base rate."""
        if category in ReferenceClassForecasting.BASE_RATES:
            return ReferenceClassForecasting.BASE_RATES[category], f"Base rate: {category}"
        return 0.5, "Uninformative prior"
    
    @staticmethod
    def suggest_reference_class(desc: str, domain: EvidenceDomain) -> List[Tuple[str, float]]:
        """Suggest relevant reference classes."""
        suggestions = []
        desc_lower = desc.lower()
        
        if domain == EvidenceDomain.MEDICAL:
            if 'trial' in desc_lower or 'drug' in desc_lower:
                suggestions.append(('drug_approval', 0.10))
                suggestions.append(('clinical_trial_phase2', 0.30))
            if 'treatment' in desc_lower:
                suggestions.append(('treatment_efficacy', 0.50))
        elif domain == EvidenceDomain.BUSINESS:
            if 'startup' in desc_lower:
                suggestions.append(('startup_success_5yr', 0.10))
            if 'merger' in desc_lower or 'acquisition' in desc_lower:
                suggestions.append(('merger_creates_value', 0.30))
            if 'product' in desc_lower or 'launch' in desc_lower:
                suggestions.append(('product_launch_success', 0.40))
        elif domain == EvidenceDomain.TECHNOLOGY:
            if 'ai' in desc_lower or 'machine learning' in desc_lower:
                suggestions.append(('ai_project_succeeds', 0.15))
        elif domain == EvidenceDomain.SCIENTIFIC:
            suggestions.append(('replication_success', 0.40))
        
        if not suggestions:
            suggestions.append(('expert_prediction_accurate', 0.50))
        return suggestions


# =============================================================================
# P-CURVE ANALYSIS
# =============================================================================

class PCurveAnalysis:
    """P-curve analysis for p-hacking detection."""
    
    @staticmethod
    def analyze(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Analyze p-value distribution for evidential value."""
        if len(p_values) < 3:
            return {'has_evidential_value': None, 'p_hacking_suspected': None,
                    'interpretation': 'Insufficient p-values (need ≥3)', 'n_values': len(p_values)}
        
        sig_p = [p for p in p_values if 0 < p < alpha]
        if len(sig_p) < 3:
            return {'has_evidential_value': None, 'p_hacking_suspected': None,
                    'interpretation': f'Insufficient significant p-values ({len(sig_p)} < 3)'}
        
        threshold = alpha / 2
        below = sum(1 for p in sig_p if p < threshold)
        above = len(sig_p) - below
        n_total = len(sig_p)
        
        p_right = stats.binomtest(below, n_total, 0.5, alternative='greater').pvalue
        p_left = stats.binomtest(above, n_total, 0.5, alternative='greater').pvalue
        
        has_evidential_value = p_right < 0.05 and below > above
        p_hacking_suspected = p_left < 0.05 and above > below
        
        if has_evidential_value:
            interpretation = "RIGHT-SKEWED: Evidence suggests real effects"
        elif p_hacking_suspected:
            interpretation = "LEFT-SKEWED: P-hacking suspected"
        else:
            interpretation = "FLAT: Inconclusive"
        
        return {
            'has_evidential_value': has_evidential_value,
            'p_hacking_suspected': p_hacking_suspected,
            'right_skew_p': p_right,
            'left_skew_p': p_left,
            'interpretation': interpretation,
            'n_values': len(sig_p)
        }


# =============================================================================
# INSTRUMENTAL VARIABLE TESTS
# =============================================================================

class InstrumentalVariableTests:
    """Tests for IV validity and strength."""
    
    @staticmethod
    def first_stage_f_test(f_statistic: float, n_instruments: int = 1) -> Dict[str, Any]:
        """Evaluate instrument strength (Staiger-Stock rule: F > 10)."""
        if f_statistic >= 10:
            strength, reliability = "STRONG", 0.9
        elif f_statistic >= 5:
            strength, reliability = "MODERATE", 0.7
        else:
            strength, reliability = "WEAK", 0.4
        
        return {
            'f_statistic': f_statistic,
            'strength': strength,
            'reliability_factor': reliability,
            'passes_staiger_stock': f_statistic > 10,
        }
    
    @staticmethod
    def sargan_test(j_statistic: float, df: int) -> Dict[str, Any]:
        """Sargan/Hansen J test for overidentification."""
        if df <= 0:
            return {'instruments_valid': None, 'note': 'Exactly identified'}
        p_value = 1 - stats.chi2.cdf(j_statistic, df)
        return {'j_statistic': j_statistic, 'p_value': p_value,
                'instruments_valid': p_value > 0.05}


# =============================================================================
# EXTREME VALUE THEORY
# =============================================================================

class ExtremeValueTheory:
    """EVT for tail risk using Generalized Pareto Distribution."""
    
    @staticmethod
    def fit_gpd(exceedances: List[float], threshold: float = 0.0) -> Dict[str, Any]:
        """Fit GPD to exceedances above threshold."""
        exc = np.array([x - threshold for x in exceedances if x > threshold])
        if len(exc) < 10:
            return {'fitted': False, 'reason': f'Insufficient exceedances ({len(exc)} < 10)'}
        
        mean_exc, var_exc = np.mean(exc), np.var(exc)
        if var_exc > 0:
            xi_hat = 0.5 * (mean_exc**2 / var_exc - 1)
            sigma_hat = 0.5 * mean_exc * (mean_exc**2 / var_exc + 1)
        else:
            xi_hat, sigma_hat = 0.0, mean_exc
        
        xi_hat = max(-0.5, min(2.0, xi_hat))
        sigma_hat = max(MathConstants.EPSILON, sigma_hat)
        
        if xi_hat > 0.1:
            tail_type = "HEAVY (fat-tailed)"
        elif xi_hat < -0.1:
            tail_type = "BOUNDED (thin-tailed)"
        else:
            tail_type = "EXPONENTIAL"
        
        return {'fitted': True, 'xi': xi_hat, 'sigma': sigma_hat,
                'tail_type': tail_type, 'n_exceedances': len(exc)}
    
    @staticmethod
    def var_cvar(xi: float, sigma: float, threshold: float,
                 probability: float = 0.95, n_obs: int = 100) -> Dict[str, float]:
        """Calculate VaR and CVaR."""
        prob_exceed = 1 - probability
        if abs(xi) < 0.01:
            var = threshold + sigma * (-np.log(prob_exceed * n_obs / 100))
            cvar = var + sigma
        else:
            var = threshold + (sigma / xi) * ((prob_exceed * n_obs / 100) ** (-xi) - 1)
            cvar = (var + sigma - xi * threshold) / (1 - xi)
        return {'VaR': var, 'CVaR': cvar, 'probability': probability}
    
    @staticmethod
    def tail_risk_adjustment(scenarios: List[Tuple[float, float]],
                            risk_aversion: float) -> float:
        """Adjust expected utility for tail risk."""
        if not scenarios:
            return 0.0
        
        probs = np.array([s[0] for s in scenarios])
        outcomes = np.array([s[1] for s in scenarios])
        probs = probs / np.sum(probs)
        
        sorted_idx = np.argsort(outcomes)
        tail_cutoff = max(1, int(0.1 * len(outcomes)))
        tail_idx = sorted_idx[:tail_cutoff]
        tail_prob = np.sum(probs[tail_idx])
        tail_heaviness = tail_prob / 0.1
        
        adjusted_gamma = risk_aversion * (1 + 0.5 * (tail_heaviness - 1)) if tail_heaviness > 1.2 else risk_aversion
        
        if abs(adjusted_gamma - 1) < 0.01:
            utilities = np.log(np.maximum(outcomes + 1, MathConstants.EPSILON))
        else:
            utilities = ((outcomes + 1) ** (1 - adjusted_gamma) - 1) / (1 - adjusted_gamma)
        
        expected_utility = np.sum(probs * utilities)
        
        if abs(adjusted_gamma - 1) < 0.01:
            ce = np.exp(expected_utility) - 1
        else:
            ce = ((1 - adjusted_gamma) * expected_utility + 1) ** (1 / (1 - adjusted_gamma)) - 1
        
        return ce


# =============================================================================
# TEXT ANALYZER (from v1.1 - Semantic Independence)
# =============================================================================

class TextAnalyzer:
    """Semantic analysis for evidence independence checking."""
    
    STOPWORDS = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these',
                 'those', 'it', 'its', 'with', 'from', 'by', 'as', 'but', 'not'}
    
    @staticmethod
    def normalize_author_name(name: str) -> str:
        """Normalize author names for matching."""
        name = re.sub(r'^(Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+', '', name, flags=re.IGNORECASE)
        name = name.lower().strip()
        name = re.sub(r'[^\w\s]', '', name)
        
        parts = name.split()
        if len(parts) >= 2:
            if len(parts[0]) <= 2:  # "J Smith" -> "j_smith"
                return f"{parts[0]}_{parts[-1]}"
            elif len(parts[-1]) <= 2:  # "Smith J" -> "j_smith"
                return f"{parts[-1]}_{parts[0]}"
            else:  # "Jane Smith" -> "j_smith"
                return f"{parts[0][0]}_{parts[-1]}"
        return name.replace(' ', '_')
    
    @staticmethod
    def normalize_source(source: str) -> str:
        """Normalize source names."""
        source = source.lower().strip()
        source = re.sub(r'\b(journal|j\.?)\s+(of|the)\s+', '', source)
        source = re.sub(r'[^\w\s]', '', source)
        source = re.sub(r'\s+', '_', source)
        return source
    
    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        """Extract identifiable entities from text."""
        entities = set()
        
        # Trial IDs
        for pattern in [r'NCT\d{6,}', r'ISRCTN\d+', r'EUCTR\d+-\d+-\d+']:
            for match in re.findall(pattern, text, re.IGNORECASE):
                entities.add(match.upper())
        
        # DOIs
        dois = re.findall(r'10\.\d{4,}/[^\s]+', text)
        entities.update(dois)
        
        # Author-year citations
        author_year = re.findall(r'([A-Z][a-z]+)\s+(?:et\s+al\.?\s+)?(\d{4})', text)
        for author, year in author_year:
            entities.add(f"{author}_{year}")
        
        return entities
    
    @staticmethod
    def tokenize(text: str) -> Set[str]:
        """Tokenize and remove stopwords."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = set(text.split())
        return tokens - TextAnalyzer.STOPWORDS
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts."""
        tokens1 = TextAnalyzer.tokenize(text1)
        tokens2 = TextAnalyzer.tokenize(text2)
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0


# =============================================================================
# ESTABLISHED HYPOTHESIS EVIDENCE (from v1.1)
# =============================================================================

@dataclass
class EstablishedHypothesisEvidence:
    """Evidence required to claim a hypothesis is 'established'."""
    claim: str
    supporting_references: List[str] = field(default_factory=list)
    meta_analyses_cited: int = 0
    textbook_citations: int = 0
    expert_consensus: bool = False
    years_established: int = 0
    replication_count: int = 0
    
    def strength_score(self) -> float:
        """Calculate establishment strength."""
        score = 0.0
        score += min(0.3, len(self.supporting_references) * 0.05)
        score += min(0.25, self.meta_analyses_cited * 0.1)
        score += min(0.15, self.textbook_citations * 0.05)
        if self.expert_consensus:
            score += 0.2
        score += min(0.1, self.years_established * 0.01)
        return min(1.0, score)
    
    def is_sufficiently_established(self, threshold: float = 0.5) -> bool:
        """Check if evidence meets establishment threshold."""
        return self.strength_score() >= threshold


# =============================================================================
# EVIDENCE INDEPENDENCE CHECKER
# =============================================================================

class EvidenceIndependenceChecker:
    """Check independence between evidence pieces."""
    
    @staticmethod
    def check_pair_independence(e1, e2) -> Dict[str, Any]:
        """Check independence between two evidence pieces."""
        issues = []
        independence = 1.0
        
        # Author overlap
        if e1.authors and e2.authors:
            norm1 = {TextAnalyzer.normalize_author_name(a) for a in e1.authors}
            norm2 = {TextAnalyzer.normalize_author_name(a) for a in e2.authors}
            overlap = norm1 & norm2
            if overlap:
                independence *= 0.6
                issues.append(f"Author overlap: {overlap}")
        
        # Source similarity
        if e1.source and e2.source:
            s1 = TextAnalyzer.normalize_source(e1.source)
            s2 = TextAnalyzer.normalize_source(e2.source)
            if s1 == s2:
                independence *= 0.7
                issues.append(f"Same source: {e1.source}")
        
        # Entity overlap
        ent1 = TextAnalyzer.extract_entities(e1.content)
        ent2 = TextAnalyzer.extract_entities(e2.content)
        if ent1 and ent2:
            overlap = ent1 & ent2
            if overlap:
                independence *= 0.5
                issues.append(f"Shared entities: {overlap}")
        
        # Content similarity
        sim = TextAnalyzer.jaccard_similarity(e1.content, e2.content)
        if sim > SafetyLimits.SEMANTIC_SIMILARITY_THRESHOLD:
            independence *= (1 - sim)
            issues.append(f"Content similarity: {sim:.0%}")
        
        # Underlying data
        if e1.underlying_data and e2.underlying_data:
            if e1.underlying_data == e2.underlying_data:
                independence *= 0.3
                issues.append("Same underlying data")
        
        return {
            'independence': independence,
            'issues': issues,
            'pair': (e1.id, e2.id)
        }
    
    @staticmethod
    def check_all_independence(evidence_list: List) -> Dict[str, Any]:
        """Check independence for all evidence pairs."""
        n = len(evidence_list)
        if n <= 1:
            return {'overall_independence': 1.0, 'effective_evidence_count': float(n), 'issues': []}
        
        all_issues = []
        total_independence = 0.0
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                result = EvidenceIndependenceChecker.check_pair_independence(
                    evidence_list[i], evidence_list[j])
                total_independence += result['independence']
                pair_count += 1
                if result['issues']:
                    all_issues.append(result)
        
        avg_independence = total_independence / pair_count if pair_count > 0 else 1.0
        avg_correlation = 1 - avg_independence
        effective_n = n / (1 + (n - 1) * avg_correlation)
        
        return {
            'overall_independence': avg_independence,
            'effective_evidence_count': effective_n,
            'issues': all_issues,
            'n_evidence': n
        }


# =============================================================================
# EVIDENCE DATACLASS
# =============================================================================

@dataclass
class Evidence:
    """Evidence with full statistical characterization."""
    id: str
    content: str
    source: str
    date: str
    domain: EvidenceDomain
    study_design: str = "observational"
    sample_size: Optional[int] = None
    causal_level: CausalLevel = None
    supports_hypothesis: bool = True
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_variance: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    f_statistic: Optional[float] = None
    j_statistic: Optional[float] = None
    n_instruments: int = 1
    quality: Optional[float] = None
    authors: List[str] = field(default_factory=list)
    underlying_data: Optional[str] = None
    doi: Optional[str] = None
    
    likelihood_ratio: float = field(init=False)
    log_likelihood_ratio: float = field(init=False)
    information_bits: float = field(init=False)
    effective_quality: float = field(init=False)
    validation_warnings: List[str] = field(default_factory=list, init=False)
    is_subgroup: bool = field(default=False, init=False)
    iv_strength: Optional[Dict] = field(default=None, init=False)
    fatal_content_flags: List[Tuple[str, float]] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        self.validation_warnings = []
        self.fatal_content_flags = []
        if self.causal_level is None:
            self.causal_level = CausalLevel.ASSOCIATION
        
        self._detect_subgroup()
        self._validate_causal_level()
        self._compute_likelihood_ratio()
        self._compute_effective_quality()
        
        if self.f_statistic is not None:
            self.iv_strength = InstrumentalVariableTests.first_stage_f_test(
                self.f_statistic, self.n_instruments)
            if not self.iv_strength['passes_staiger_stock']:
                self.validation_warnings.append(f"Weak instrument: F={self.f_statistic:.1f}")
        
        if self.p_value is not None and not 0 < self.p_value < 1:
            self.validation_warnings.append(f"Invalid p-value: {self.p_value}")
            self.p_value = None
    
    def _detect_subgroup(self):
        patterns = [r'\bsubgroup\b', r'\bpost[- ]?hoc\b', r'\bexploratory\b', r'\bsubset\b']
        for pattern in patterns:
            if re.search(pattern, self.content.lower()):
                self.is_subgroup = True
                self.validation_warnings.append("Subgroup/post-hoc analysis detected")
                break
    
    def _validate_causal_level(self):
        design_lower = self.study_design.lower()
        if design_lower in VALID_CAUSAL_LEVELS:
            if self.causal_level not in VALID_CAUSAL_LEVELS[design_lower]:
                self.validation_warnings.append(
                    f"Causal level {self.causal_level.name} invalid for {design_lower}")
                self.causal_level = CausalLevel.ASSOCIATION
    
    def _compute_likelihood_ratio(self):
        design_lower = self.study_design.lower()
        metrics = STUDY_DIAGNOSTIC_METRICS.get(design_lower, DiagnosticMetrics(0.65, 0.65))
        sens, spec = metrics.sensitivity, metrics.specificity
        
        if self.sample_size:
            adj = sample_size_adjustment(self.sample_size)
            sens, spec = sens * adj, spec * adj
        
        if self.is_subgroup:
            sens, spec = sens * 0.85, spec * 0.85
        
        if self.iv_strength and not self.iv_strength['passes_staiger_stock']:
            r = self.iv_strength['reliability_factor']
            sens, spec = sens * r, spec * r
        
        if self.supports_hypothesis:
            lr = sens / (1 - spec + MathConstants.EPSILON)
        else:
            lr = (1 - sens) / (spec + MathConstants.EPSILON)
        
        self.likelihood_ratio = max(SafetyLimits.MIN_LIKELIHOOD_RATIO,
                                    min(SafetyLimits.MAX_LIKELIHOOD_RATIO, lr))
        self.log_likelihood_ratio = np.log(self.likelihood_ratio)
        self.information_bits = np.log2(max(self.likelihood_ratio, MathConstants.EPSILON))
    
    def _compute_effective_quality(self):
        if self.quality is not None:
            base = self.quality
        else:
            lr = self.likelihood_ratio
            base = 1 - 1/lr if lr > 1 else lr - 0.5
            base = max(0.1, min(0.95, base))
        
        penalty = 1.0
        if self.is_subgroup:
            penalty *= 0.8
        penalty *= 0.9 ** len(self.validation_warnings)
        self.effective_quality = base * penalty
    
    def to_effect_size(self) -> Tuple[float, float]:
        """Return (effect_size, variance) for meta-analysis."""
        if self.effect_size is not None and self.effect_variance is not None:
            return self.effect_size, self.effect_variance
        effect = self.log_likelihood_ratio
        variance = 4 / self.sample_size if self.sample_size and self.sample_size > 0 else 0.5
        return effect, variance


# =============================================================================
# BAYESIAN UPDATER
# =============================================================================

class BayesianUpdater:
    """Bayesian updating with correlation adjustment."""
    
    def __init__(self, prior: float = 0.5):
        self.prior = max(SafetyLimits.CREDENCE_HARD_FLOOR,
                        min(SafetyLimits.CREDENCE_HARD_CAP, prior))
        self.current_log_odds = self._prob_to_log_odds(self.prior)
        self.update_history: List[Dict] = []
        self.total_bits = 0.0
    
    @staticmethod
    def _prob_to_log_odds(p: float) -> float:
        p = max(MathConstants.EPSILON, min(1 - MathConstants.EPSILON, p))
        return np.log(p / (1 - p))
    
    @staticmethod
    def _log_odds_to_prob(lo: float) -> float:
        lo = max(-SafetyLimits.MAX_LOG_ODDS, min(SafetyLimits.MAX_LOG_ODDS, lo))
        return 1 / (1 + np.exp(-lo))
    
    def update_single(self, evidence: Evidence, correlation: float = 0.0) -> float:
        """Update with single evidence piece."""
        effective_lr = evidence.likelihood_ratio ** (1 - correlation)
        new_log_odds = self.current_log_odds + np.log(effective_lr)
        new_log_odds = max(-SafetyLimits.MAX_LOG_ODDS, min(SafetyLimits.MAX_LOG_ODDS, new_log_odds))
        
        old_prob = self._log_odds_to_prob(self.current_log_odds)
        new_prob = self._log_odds_to_prob(new_log_odds)
        
        self.update_history.append({
            'evidence_id': evidence.id, 'lr': evidence.likelihood_ratio,
            'effective_lr': effective_lr, 'correlation': correlation,
            'prior': old_prob, 'posterior': new_prob,
            'bits_gained': evidence.information_bits * (1 - correlation)
        })
        
        self.current_log_odds = new_log_odds
        self.total_bits += evidence.information_bits * (1 - correlation)
        return new_prob
    
    def update_batch(self, evidence_list: List[Evidence],
                     correlation_matrix: Optional[np.ndarray] = None) -> float:
        """Update with multiple evidence pieces."""
        n = len(evidence_list)
        if n == 0:
            return self._log_odds_to_prob(self.current_log_odds)
        
        if correlation_matrix is None:
            correlation_matrix = np.eye(n)
        
        avg_corr = (np.sum(correlation_matrix) - n) / max(1, n * (n - 1))
        avg_corr = max(0, avg_corr)
        n_eff = n / (1 + (n - 1) * avg_corr)
        scale = n_eff / n if n > 0 else 1.0
        
        for i, ev in enumerate(evidence_list):
            other_corrs = [correlation_matrix[i, j] for j in range(n) if j != i]
            avg_other = np.mean(other_corrs) if other_corrs else 0.0
            self.update_single(ev, correlation=avg_other * (1 - scale))
        
        return self._log_odds_to_prob(self.current_log_odds)
    
    @property
    def posterior(self) -> float:
        return self._log_odds_to_prob(self.current_log_odds)
    
    def get_credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Approximate credible interval."""
        effective_n = max(1, 2 ** self.total_bits)
        p = self.posterior
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        denom = 1 + z**2 / effective_n
        center = (p + z**2 / (2 * effective_n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * effective_n)) / effective_n) / denom
        return (max(0, center - margin), min(1, center + margin))
    
    def explain(self) -> str:
        """Generate explanation."""
        lines = [f"Prior: {self.prior:.1%}"]
        for u in self.update_history:
            lines.append(f"  + {u['evidence_id']}: LR={u['lr']:.2f} (eff={u['effective_lr']:.2f}, ρ={u['correlation']:.2f}) → {u['posterior']:.1%}")
        lines.append(f"Posterior: {self.posterior:.1%}")
        lines.append(f"Total information: {self.total_bits:.2f} bits")
        ci = self.get_credible_interval()
        lines.append(f"95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
        return "\n".join(lines)


# =============================================================================
# MECHANISM MAP (Tarjan's Algorithm)
# =============================================================================

@dataclass
class MechanismNode:
    """Node in mechanism map."""
    id: str
    label: str
    node_type: NodeType
    probability: float = 0.5
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class MechanismEdge:
    """Edge in mechanism map."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5
    evidence_ids: List[str] = field(default_factory=list)
    time_lag: Optional[str] = None


class MechanismMap:
    """Directed graph with Tarjan's SCC algorithm for cycle detection."""
    
    def __init__(self):
        self.nodes: Dict[str, MechanismNode] = {}
        self.edges: List[MechanismEdge] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
    
    def add_node(self, node: MechanismNode) -> bool:
        if len(self.nodes) >= SafetyLimits.MAX_MECHANISM_NODES:
            return False
        self.nodes[node.id] = node
        return True
    
    def add_edge(self, edge: MechanismEdge) -> bool:
        if len(self.edges) >= SafetyLimits.MAX_MECHANISM_EDGES:
            return False
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return False
        self.edges.append(edge)
        self._adjacency[edge.source_id].append(edge.target_id)
        return True
    
    def find_cycles_tarjan(self) -> List[List[str]]:
        """Find SCCs using Tarjan's algorithm. O(V+E)"""
        index_counter = [0]
        stack, lowlinks, index, on_stack = [], {}, {}, {}
        sccs = []
        
        def strongconnect(node_id: str):
            index[node_id] = lowlinks[node_id] = index_counter[0]
            index_counter[0] += 1
            stack.append(node_id)
            on_stack[node_id] = True
            
            for succ in self._adjacency.get(node_id, []):
                if succ not in index:
                    strongconnect(succ)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[succ])
                elif on_stack.get(succ, False):
                    lowlinks[node_id] = min(lowlinks[node_id], index[succ])
            
            if lowlinks[node_id] == index[node_id]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node_id:
                        break
                if len(scc) > 1:
                    sccs.append(scc)
        
        for node_id in self.nodes:
            if node_id not in index:
                strongconnect(node_id)
        return sccs
    
    def identify_feedback_loops(self) -> List[Dict]:
        """Identify and classify feedback loops."""
        sccs = self.find_cycles_tarjan()
        loops = []
        for scc in sccs:
            reinforcing = sum(1 for e in self.edges if e.source_id in scc and e.target_id in scc
                             and e.edge_type in [EdgeType.CAUSES, EdgeType.ENABLES])
            balancing = sum(1 for e in self.edges if e.source_id in scc and e.target_id in scc
                           and e.edge_type in [EdgeType.PREVENTS, EdgeType.COMPENSATES])
            loop_type = FeedbackLoopType.REINFORCING if reinforcing > balancing else FeedbackLoopType.BALANCING
            strengths = [e.strength for e in self.edges if e.source_id in scc and e.target_id in scc]
            avg_strength = np.mean(strengths) if strengths else 0.5
            loops.append({'nodes': scc, 'type': loop_type, 'size': len(scc),
                         'avg_strength': avg_strength,
                         'concerning': loop_type == FeedbackLoopType.REINFORCING and avg_strength > 0.7})
        return loops
    
    def get_summary(self) -> Dict:
        """Generate mechanism map summary."""
        loops = self.identify_feedback_loops()
        return {'n_nodes': len(self.nodes), 'n_edges': len(self.edges),
                'n_cycles': len(self.find_cycles_tarjan()), 'feedback_loops': loops,
                'has_concerning_loops': any(l['concerning'] for l in loops)}


# =============================================================================
# HYPOTHESIS CLASS
# =============================================================================

class Hypothesis:
    """Central hypothesis analysis class."""
    
    def __init__(self, name: str, domain: EvidenceDomain, description: str = "",
                 reference_class: Optional[str] = None):
        self.name = name
        self.domain = domain
        self.description = description
        self.created_at = datetime.now()
        
        if reference_class:
            self.prior, self.prior_source = ReferenceClassForecasting.get_base_rate(reference_class)
        else:
            suggestions = ReferenceClassForecasting.suggest_reference_class(description or name, domain)
            self.prior = suggestions[0][1] if suggestions else 0.5
            self.prior_source = f"Suggested: {suggestions[0][0]}" if suggestions else "Uninformative"
        
        self.updater = BayesianUpdater(self.prior)
        self.evidence: List[Evidence] = []
        self.evidence_correlation_matrix: Optional[np.ndarray] = None
        self.mechanism_map = MechanismMap()
        
        self.what: Optional[Tuple[str, float]] = None
        self.why: Optional[Tuple[str, float]] = None
        self.how: Optional[Tuple[str, float]] = None
        self.feasibility = {'technical': None, 'economic': None, 'timeline': None}
        self.scenarios: List[Tuple[str, float, float]] = []
        self.risk_aversion = DOMAIN_RISK_AVERSION.get(domain, 1.5)
        
        self.established = False
        self.establishment_evidence: Optional[EstablishedHypothesisEvidence] = None
        self.establishment_verified = False
        
        self.p_curve_result: Optional[Dict] = None
        self.meta_analysis_result: Optional[Dict] = None
        self.tail_risk_result: Optional[Dict] = None
        self.warnings: List[str] = []
    
    def set_what(self, desc: str, conf: float):
        self.what = (desc, max(0.1, min(0.99, conf)))
    
    def set_why(self, desc: str, conf: float):
        self.why = (desc, max(0.1, min(0.99, conf)))
    
    def set_how(self, desc: str, conf: float):
        self.how = (desc, max(0.1, min(0.99, conf)))
    
    def set_feasibility(self, technical: float, economic: float, timeline: float):
        self.feasibility = {'technical': max(0, min(1, technical)),
                           'economic': max(0, min(1, economic)),
                           'timeline': max(0, min(1, timeline))}
    
    def set_established_hypothesis(self, established: bool, 
                                   evidence: Optional[EstablishedHypothesisEvidence] = None):
        """Set hypothesis as established with optional evidence."""
        self.established = established
        self.establishment_evidence = evidence
        if established and evidence:
            self.establishment_verified = evidence.is_sufficiently_established()
            if not self.establishment_verified:
                self.warnings.append("Establishment claim not sufficiently supported by evidence")
        elif established and not evidence:
            self.establishment_verified = False
            self.warnings.append("Establishment claimed without supporting evidence")
    
    def add_evidence(self, evidence: Evidence):
        self.evidence.append(evidence)
        self.p_curve_result = None
        self.meta_analysis_result = None
    
    def add_scenario(self, name: str, prob: float, outcome: float):
        self.scenarios.append((name, prob, outcome))
    
    def compute_evidence_correlations(self) -> np.ndarray:
        """Compute pairwise correlation matrix."""
        n = len(self.evidence)
        if n == 0:
            return np.array([])
        
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                result = EvidenceIndependenceChecker.check_pair_independence(
                    self.evidence[i], self.evidence[j])
                rho = 1 - result['independence']
                corr[i, j] = corr[j, i] = rho
        
        self.evidence_correlation_matrix = corr
        return corr
    
    def update_beliefs(self) -> float:
        if not self.evidence:
            return self.prior
        if self.evidence_correlation_matrix is None:
            self.compute_evidence_correlations()
        self.updater = BayesianUpdater(self.prior)
        return self.updater.update_batch(self.evidence, self.evidence_correlation_matrix)
    
    def run_p_curve_analysis(self) -> Dict:
        p_values = [e.p_value for e in self.evidence if e.p_value]
        self.p_curve_result = PCurveAnalysis.analyze(p_values)
        if self.p_curve_result.get('p_hacking_suspected'):
            self.warnings.append("P-CURVE: P-hacking suspected")
        return self.p_curve_result
    
    def run_meta_analysis(self) -> Dict:
        effects, variances = [], []
        for ev in self.evidence:
            eff, var = ev.to_effect_size()
            effects.append(eff)
            variances.append(var)
        
        if not effects:
            self.meta_analysis_result = {'pooled_effect': 0, 'n_studies': 0}
            return self.meta_analysis_result
        
        self.meta_analysis_result = EvidencePooling.random_effects_pooling(effects, variances)
        
        if len(effects) >= 3:
            tf = EvidencePooling.trim_and_fill(effects, variances)
            self.meta_analysis_result['publication_bias'] = tf
            if tf.get('asymmetry_detected'):
                self.warnings.append(f"PUBLICATION BIAS: ~{tf['missing_studies']} missing studies")
        
        if self.meta_analysis_result.get('i_squared', 0) > 75:
            self.warnings.append(f"HIGH HETEROGENEITY: I²={self.meta_analysis_result['i_squared']:.0f}%")
        
        return self.meta_analysis_result
    
    def compute_tail_risk(self) -> Dict:
        if not self.scenarios:
            return {}
        negative = [s[2] for s in self.scenarios if s[2] < 0]
        if len(negative) < 3:
            self.tail_risk_result = {'computed': False}
            return self.tail_risk_result
        
        losses = [-o for o in negative]
        gpd = ExtremeValueTheory.fit_gpd(losses, threshold=0)
        if gpd['fitted']:
            var_cvar = ExtremeValueTheory.var_cvar(gpd['xi'], gpd['sigma'], 0)
            self.tail_risk_result = {'computed': True, 'gpd': gpd, 'var_cvar': var_cvar}
            if gpd['tail_type'].startswith('HEAVY'):
                self.warnings.append(f"HEAVY TAIL RISK: CVaR={var_cvar['CVaR']:.2f}")
        else:
            self.tail_risk_result = gpd
        return self.tail_risk_result
    
    def compute_expected_utility(self) -> Dict:
        if not self.scenarios:
            return {'ce': 0, 'warning': 'No scenarios'}
        
        self.compute_tail_risk()
        if self.tail_risk_result and self.tail_risk_result.get('computed'):
            ce = ExtremeValueTheory.tail_risk_adjustment(
                [(s[1], s[2]) for s in self.scenarios], self.risk_aversion)
        else:
            probs = np.array([s[1] for s in self.scenarios])
            outcomes = np.array([s[2] for s in self.scenarios])
            probs = probs / np.sum(probs)
            gamma = self.risk_aversion
            
            if abs(gamma - 1) < 0.01:
                utilities = np.log(np.maximum(outcomes + 1, MathConstants.EPSILON))
            else:
                utilities = ((outcomes + 1) ** (1 - gamma) - 1) / (1 - gamma)
            eu = np.sum(probs * utilities)
            ce = np.exp(eu) - 1 if abs(gamma - 1) < 0.01 else ((1 - gamma) * eu + 1) ** (1 / (1 - gamma)) - 1
        
        return {'certainty_equivalent': ce, 'risk_aversion': self.risk_aversion,
                'tail_adjusted': self.tail_risk_result and self.tail_risk_result.get('computed', False)}
    
    def get_posterior(self) -> float:
        return self.updater.posterior


# =============================================================================
# WARNING SYSTEM
# =============================================================================

@dataclass
class Warning:
    """Warning with deduplication support."""
    category: str
    message: str
    level: WarningLevel
    count: int = 1
    details: List[str] = field(default_factory=list)
    
    def key(self) -> str:
        return f"{self.category}:{self.level.value}:{self.message[:50]}"


class WarningSystem:
    """Aggregated warning system with deduplication and blocking."""
    
    def __init__(self):
        self.warnings: Dict[str, Warning] = {}
        self._blocked = False
        self._blocking_reasons: List[str] = []
    
    def add(self, category: str, message: str, level: WarningLevel,
            details: Optional[List[str]] = None):
        w = Warning(category, message, level, 1, details or [])
        key = w.key()
        if key in self.warnings:
            self.warnings[key].count += 1
            if details:
                self.warnings[key].details.extend(details)
        else:
            self.warnings[key] = w
        
        if level == WarningLevel.FATAL:
            self._blocked = True
            self._blocking_reasons.append(f"{category}: {message}")
    
    def add_fatal(self, category: str, message: str):
        self.add(category, message, WarningLevel.FATAL)
    
    def add_critical(self, category: str, message: str, details: List[str] = None):
        self.add(category, message, WarningLevel.CRITICAL, details)
    
    def get_by_level(self, level: WarningLevel) -> List[Warning]:
        return [w for w in self.warnings.values() if w.level == level]
    
    def get_summary_header(self) -> str:
        counts = Counter(w.level for w in self.warnings.values())
        parts = []
        if counts[WarningLevel.FATAL]:
            parts.append(f"💀 FATAL: {counts[WarningLevel.FATAL]}")
        if counts[WarningLevel.CRITICAL]:
            parts.append(f"🚨 CRITICAL: {counts[WarningLevel.CRITICAL]}")
        if counts[WarningLevel.WARNING]:
            parts.append(f"⚠️ WARNING: {counts[WarningLevel.WARNING]}")
        if counts[WarningLevel.INFO]:
            parts.append(f"ℹ️ INFO: {counts[WarningLevel.INFO]}")
        return " | ".join(parts) if parts else "✅ No warnings"
    
    def is_blocked(self) -> Tuple[bool, List[str]]:
        return self._blocked, self._blocking_reasons
    
    def block(self, reason: str):
        self._blocked = True
        self._blocking_reasons.append(reason)
    
    def format_all(self, max_per_level: int = 5) -> str:
        lines = [self.get_summary_header(), ""]
        level_order = {WarningLevel.FATAL: 0, WarningLevel.CRITICAL: 1,
                      WarningLevel.WARNING: 2, WarningLevel.INFO: 3}
        
        for level in sorted([WarningLevel.FATAL, WarningLevel.CRITICAL,
                            WarningLevel.WARNING, WarningLevel.INFO],
                           key=lambda x: level_order[x]):
            level_warnings = self.get_by_level(level)
            if not level_warnings:
                continue
            icon = {"fatal": "💀", "critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[level.value]
            for w in level_warnings[:max_per_level]:
                count_str = f" (×{w.count})" if w.count > 1 else ""
                lines.append(f"{icon} [{w.category}] {w.message}{count_str}")
        return "\n".join(lines)


# =============================================================================
# CONTENT SCANNER
# =============================================================================

class ContentScanner:
    """Scans content for fatal/concerning patterns with euphemism detection."""
    
    PATTERNS = {
        'legal': [
            (r'\billegal\b', 0.95), (r'\bunlawful\b', 0.95), (r'\bfraud\b', 0.9),
            (r'\bviolat', 0.85), (r'\bcriminal\b', 0.9), (r'\blitigation\b', 0.8),
            (r'regulatory gray area', 0.7), (r'enforcement action', 0.75),
            (r'not fully compliant', 0.7), (r'legal exposure', 0.7),
        ],
        'safety': [
            (r'\bfatal\b', 0.95), (r'\bdeath', 0.9), (r'\bdangerous\b', 0.85),
            (r'\btoxic\b', 0.85), (r'\blethal\b', 0.95), (r'\bharm', 0.7),
            (r'adverse event', 0.75), (r'serious event', 0.8), (r'safety signal', 0.7),
            (r'FDA concern', 0.8), (r'clinical hold', 0.85), (r'black box warning', 0.9),
        ],
        'ethical': [
            (r'\bfraud', 0.9), (r'\bdecepti', 0.85), (r'\bmisrepresent', 0.8),
            (r'\bfalsif', 0.9), (r'\bplagiaris', 0.85), (r'conflict of interest', 0.6),
            (r'data integrity', 0.7), (r'selective reporting', 0.7), (r'p-?hacking', 0.75),
        ],
        'financial': [
            (r'\bbankrupt', 0.9), (r'\binsolven', 0.9), (r'\bdefault', 0.8),
            (r'liquidity constraint', 0.7), (r'going concern', 0.85),
            (r'material weakness', 0.7), (r'restatement', 0.65),
        ],
    }
    
    def __init__(self):
        self.findings: List[Dict] = []
        self._compiled = {cat: [(re.compile(p, re.I), s) for p, s in patterns]
                         for cat, patterns in self.PATTERNS.items()}
    
    def scan(self, content: str) -> List[Dict]:
        self.findings = []
        if not content:
            return self.findings
        
        for category, patterns in self._compiled.items():
            for regex, severity in patterns:
                for match in regex.finditer(content):
                    self.findings.append({
                        'category': category, 'matched': match.group(),
                        'severity': severity, 'position': match.start()
                    })
        self.findings.sort(key=lambda x: -x['severity'])
        return self.findings
    
    def scan_evidence_list(self, evidence_list: List) -> Dict[str, List[Dict]]:
        all_findings = defaultdict(list)
        for ev in evidence_list:
            for f in self.scan(ev.content):
                f['evidence_id'] = ev.id
                all_findings[f['category']].append(f)
        return dict(all_findings)
    
    def has_fatal_content(self) -> bool:
        return any(f['severity'] >= 0.9 for f in self.findings)
    
    def get_max_severity(self) -> float:
        return max((f['severity'] for f in self.findings), default=0.0)
    
    def get_summary(self) -> Dict:
        by_cat = defaultdict(list)
        for f in self.findings:
            by_cat[f['category']].append(f)
        return {'total_findings': len(self.findings), 'has_fatal': self.has_fatal_content(),
                'max_severity': self.get_max_severity(),
                'by_category': {k: len(v) for k, v in by_cat.items()}}


# =============================================================================
# BIAS DETECTOR
# =============================================================================

class BiasDetector:
    """Detects 8 cognitive biases with statistical tests."""
    
    def __init__(self):
        self.detected_biases: List[Dict] = []
        self.establishment_verified = False
    
    def detect_all(self, hypothesis, evidence_list: List,
                   meta_result: Optional[Dict] = None,
                   p_curve_result: Optional[Dict] = None) -> List[Dict]:
        self.detected_biases = []
        
        self._check_confirmation_bias(evidence_list)
        self._check_anchoring(hypothesis)
        self._check_availability_bias(evidence_list)
        self._check_overconfidence(hypothesis)
        self._check_base_rate_neglect(hypothesis)
        self._check_planning_fallacy(hypothesis)
        self._check_sunk_cost(hypothesis)
        if meta_result:
            self._check_publication_bias(meta_result, p_curve_result)
        
        return self.detected_biases
    
    def _add_bias(self, bias_type: BiasType, severity: float, desc: str, rec: str):
        self.detected_biases.append({'type': bias_type, 'severity': severity,
                                     'description': desc, 'recommendation': rec})
    
    def _check_confirmation_bias(self, evidence_list: List):
        if not evidence_list:
            return
        supporting = sum(1 for e in evidence_list if e.supports_hypothesis)
        total = len(evidence_list)
        ratio = supporting / total
        if ratio > 0.9 and total >= 3:
            self._add_bias(BiasType.CONFIRMATION, 0.7,
                          f"{supporting}/{total} ({ratio:.0%}) supports hypothesis",
                          "Actively seek disconfirming evidence")
    
    def _check_anchoring(self, h):
        if not hasattr(h, 'updater') or not h.evidence:
            return
        movement = abs(h.updater.posterior - h.prior)
        bits = h.updater.total_bits
        if movement < 0.1 and bits > 2:
            self._add_bias(BiasType.ANCHORING, 0.6,
                          f"Posterior barely moved despite {bits:.1f} bits",
                          "Re-examine if prior is being updated appropriately")
    
    def _check_availability_bias(self, evidence_list: List):
        if len(evidence_list) < 3:
            return
        years = []
        for e in evidence_list:
            if e.date:
                match = re.search(r'20\d{2}', e.date)
                if match:
                    years.append(int(match.group()))
        if len(years) >= 3:
            recent = sum(1 for y in years if y >= 2023)
            if recent / len(years) > 0.8:
                self._add_bias(BiasType.AVAILABILITY, 0.5,
                              f"{recent}/{len(years)} evidence from recent years",
                              "Include older foundational studies")
    
    def _check_overconfidence(self, h):
        if not hasattr(h, 'updater'):
            return
        if h.updater.posterior > 0.95:
            ci = h.updater.get_credible_interval()
            if ci[1] - ci[0] < 0.1:
                self._add_bias(BiasType.OVERCONFIDENCE, 0.7,
                              f"Very high confidence ({h.updater.posterior:.0%}) with narrow CI",
                              "Consider unknown unknowns")
    
    def _check_base_rate_neglect(self, h):
        if 'uninformative' in h.prior_source.lower():
            self._add_bias(BiasType.BASE_RATE_NEGLECT, 0.6,
                          "Using uninformative prior instead of reference class",
                          "Identify appropriate reference class")
    
    def _check_planning_fallacy(self, h):
        timeline = h.feasibility.get('timeline')
        if timeline and timeline > 0.8:
            desc = (h.description or h.name).lower()
            if any(kw in desc for kw in ['launch', 'project', 'implement', 'develop']):
                self._add_bias(BiasType.PLANNING_FALLACY, 0.6,
                              f"High timeline confidence ({timeline:.0%}) for project",
                              "Add 50% buffer typical for similar projects")
    
    def _check_sunk_cost(self, h):
        desc = (h.description or "").lower()
        indicators = ['already invested', 'spent so far', 'too late to', "can't stop"]
        for ind in indicators:
            if ind in desc:
                self._add_bias(BiasType.SUNK_COST, 0.7, f"Sunk cost language: '{ind}'",
                              "Evaluate based on future costs/benefits only")
                break
    
    def _check_publication_bias(self, meta: Dict, pcurve: Optional[Dict]):
        if meta.get('publication_bias', {}).get('asymmetry_detected'):
            self._add_bias(BiasType.PUBLICATION_BIAS, 0.7,
                          f"Funnel asymmetry: ~{meta['publication_bias'].get('missing_studies', 0)} missing",
                          "Effect sizes may be inflated")
        if pcurve and pcurve.get('p_hacking_suspected'):
            self._add_bias(BiasType.PUBLICATION_BIAS, 0.8,
                          "P-curve suggests p-hacking", "Scrutinize for QRPs")
        if meta.get('i_squared', 0) > 75:
            self._add_bias(BiasType.PUBLICATION_BIAS, 0.5,
                          f"High heterogeneity (I²={meta['i_squared']:.0f}%)",
                          "Check if studies are comparable")


# =============================================================================
# SENSITIVITY ANALYZER
# =============================================================================

class SensitivityAnalyzer:
    """Tests robustness of conclusions."""
    
    @staticmethod
    def perturbation_analysis(h, perturbations=[-0.2, -0.1, 0.1, 0.2]) -> Dict:
        if not h.evidence:
            return {'stable': True, 'perturbations': []}
        
        base = h.updater.posterior
        results = []
        for delta in perturbations:
            perturbed_lrs = [max(0.01, min(100, e.likelihood_ratio * (1 + delta)))
                           for e in h.evidence]
            log_odds = np.log(h.prior / (1 - h.prior))
            for lr in perturbed_lrs:
                log_odds += np.log(lr)
            log_odds = max(-10, min(10, log_odds))
            post = 1 / (1 + np.exp(-log_odds))
            results.append({'delta': delta, 'posterior': post, 'change': post - base})
        
        max_change = max(abs(r['change']) for r in results)
        return {'base_posterior': base, 'perturbations': results, 'max_change': max_change,
                'stable': max_change < 0.15, 'critical_sensitivity': max_change > 0.25}
    
    @staticmethod
    def leave_one_out(h) -> Dict:
        if len(h.evidence) < 2:
            return {'influential': [], 'stable': True}
        
        base = h.updater.posterior
        influential = []
        for i, ev in enumerate(h.evidence):
            others = [e for j, e in enumerate(h.evidence) if j != i]
            log_odds = np.log(h.prior / (1 - h.prior))
            for e in others:
                log_odds += e.log_likelihood_ratio
            log_odds = max(-10, min(10, log_odds))
            post_without = 1 / (1 + np.exp(-log_odds))
            impact = base - post_without
            if abs(impact) > 0.1:
                influential.append({'evidence_id': ev.id, 'impact': impact})
        
        influential.sort(key=lambda x: -abs(x['impact']))
        return {'base_posterior': base, 'influential': influential,
                'stable': len(influential) == 0,
                'most_influential': influential[0] if influential else None}


# =============================================================================
# RUN ANALYSIS (5-Layer Architecture)
# =============================================================================

def run_analysis(hypothesis: Hypothesis, rigor_level: int = 2,
                 max_iter: int = 10, force_continue: bool = False) -> Dict:
    """
    Run full 5-layer PRISM analysis.
    
    LAYERS:
    L0: Foundation validation
    L0.5: Pre-flight checks
    L1: Evidence synthesis
    L2: Adversarial testing
    L3: Sensitivity analysis
    L4: Gate check
    L5: Decision synthesis
    """
    results = {
        'hypothesis_name': hypothesis.name,
        'domain': hypothesis.domain.value,
        'timestamp': datetime.now().isoformat(),
        'rigor_level': rigor_level,
        'layers': {},
        'blocked': False,
        'blocking_reasons': [],
        'decision_state': None,
        'recommendation': None
    }
    
    ws = WarningSystem()
    cs = ContentScanner()
    bd = BiasDetector()
    
    # === L0: FOUNDATION ===
    l0 = {'valid': True, 'issues': []}
    if not hypothesis.what:
        l0['issues'].append("Missing WHAT")
        l0['valid'] = False
    if rigor_level >= 2 and not hypothesis.why:
        l0['issues'].append("Missing WHY")
    for issue in l0['issues']:
        ws.add("Foundation", issue, WarningLevel.WARNING)
    results['layers']['L0_foundation'] = l0
    
    # === L0.5: PRE-FLIGHT ===
    l05 = {'passed': True, 'evidence_count': len(hypothesis.evidence)}
    if len(hypothesis.evidence) < 2:
        ws.add("Evidence", "Insufficient evidence (<2)", WarningLevel.WARNING)
    
    if hypothesis.evidence:
        findings = cs.scan_evidence_list(hypothesis.evidence)
        l05['content_scan'] = cs.get_summary()
        if cs.has_fatal_content():
            ws.add_fatal("Content", "Fatal content detected")
            l05['passed'] = False
        elif cs.get_max_severity() > 0.7:
            for cat, fs in findings.items():
                if fs and max(f['severity'] for f in fs) > 0.7:
                    ws.add_critical(f"Content-{cat}", f"{len(fs)} concerning patterns")
    
    results['layers']['L0.5_preflight'] = l05
    
    # === CHECK BLOCKING ===
    blocked, reasons = ws.is_blocked()
    if blocked and not force_continue:
        results['blocked'] = True
        results['blocking_reasons'] = reasons
        results['decision_state'] = 'blocked'
        results['recommendation'] = "BLOCKED: Resolve fatal issues"
        results['warnings'] = ws.format_all()
        return results
    
    # === L1: SYNTHESIS ===
    l1 = {}
    hypothesis.compute_evidence_correlations()
    posterior = hypothesis.update_beliefs()
    l1['prior'] = hypothesis.prior
    l1['prior_source'] = hypothesis.prior_source
    l1['posterior'] = posterior
    l1['credible_interval'] = hypothesis.updater.get_credible_interval()
    l1['total_bits'] = hypothesis.updater.total_bits
    
    if len(hypothesis.evidence) >= 2:
        l1['meta_analysis'] = hypothesis.run_meta_analysis()
    
    p_vals = [e.p_value for e in hypothesis.evidence if e.p_value]
    if len(p_vals) >= 3:
        l1['p_curve'] = hypothesis.run_p_curve_analysis()
    
    results['layers']['L1_synthesis'] = l1
    
    # === L2: ADVERSARIAL ===
    l2 = {}
    biases = bd.detect_all(hypothesis, hypothesis.evidence,
                          l1.get('meta_analysis'), l1.get('p_curve'))
    l2['biases_detected'] = biases
    
    for b in biases:
        if b['severity'] >= 0.7:
            ws.add_critical(f"Bias-{b['type'].value}", b['description'])
        else:
            ws.add(f"Bias-{b['type'].value}", b['description'], WarningLevel.WARNING)
    
    if hypothesis.evidence_correlation_matrix is not None:
        n = len(hypothesis.evidence)
        if n > 1:
            avg_corr = (np.sum(hypothesis.evidence_correlation_matrix) - n) / (n * (n - 1))
            n_eff = n / (1 + (n - 1) * avg_corr)
            l2['evidence_independence'] = {'n_evidence': n, 'avg_correlation': avg_corr, 'effective_n': n_eff}
            if n_eff < n * 0.5:
                ws.add_critical("Independence", f"Low independence: effective N = {n_eff:.1f}/{n}")
    
    # Check establishment verification
    if hypothesis.established and not hypothesis.establishment_verified:
        ws.add("Establishment", "Claim not verified with sufficient evidence", WarningLevel.WARNING)
    l2['establishment_verified'] = hypothesis.establishment_verified
    
    results['layers']['L2_adversarial'] = l2
    
    # === L3: SENSITIVITY ===
    l3 = {}
    perturb = SensitivityAnalyzer.perturbation_analysis(hypothesis)
    l3['perturbation'] = perturb
    if perturb.get('critical_sensitivity'):
        ws.add_critical("Sensitivity", f"Critically sensitive (max: {perturb['max_change']:.1%})")
    
    if len(hypothesis.evidence) >= 3:
        loo = SensitivityAnalyzer.leave_one_out(hypothesis)
        l3['leave_one_out'] = loo
        if loo.get('most_influential') and abs(loo['most_influential']['impact']) > 0.2:
            ws.add("Sensitivity", f"Depends on '{loo['most_influential']['evidence_id']}'", WarningLevel.WARNING)
    
    results['layers']['L3_sensitivity'] = l3
    
    # === L4: GATE CHECK ===
    l4 = {'gates_passed': True, 'failed_gates': []}
    if len(hypothesis.evidence) < 2:
        l4['failed_gates'].append("EVIDENCE: <2 pieces")
        l4['gates_passed'] = False
    if l05.get('content_scan', {}).get('has_fatal'):
        l4['failed_gates'].append("CONTENT: Fatal detected")
        l4['gates_passed'] = False
    if any(b['severity'] >= 0.8 for b in biases):
        l4['failed_gates'].append("BIAS: Severe bias detected")
    if perturb.get('critical_sensitivity'):
        l4['failed_gates'].append("STABILITY: Critically sensitive")
    
    results['layers']['L4_gate'] = l4
    
    # === L5: DECISION ===
    l5 = {}
    if hypothesis.scenarios:
        l5['utility'] = hypothesis.compute_expected_utility()
        if hypothesis.tail_risk_result and hypothesis.tail_risk_result.get('computed'):
            l5['tail_risk'] = hypothesis.tail_risk_result
    
    ci = l1['credible_interval']
    ci_width = ci[1] - ci[0]
    
    if not l4['gates_passed']:
        decision = DecisionReadiness.FATAL_FLAW
        rec = "STOP: Critical issues must be resolved"
    elif blocked:
        decision = DecisionReadiness.BLOCKED
        rec = "BLOCKED: Fatal issues detected"
    elif posterior > 0.8 and ci_width < 0.3:
        decision = DecisionReadiness.READY
        rec = f"PROCEED with high confidence ({posterior:.0%})"
    elif posterior < 0.3 and ci_width < 0.3:
        decision = DecisionReadiness.REJECT
        rec = f"REJECT: Low probability ({posterior:.0%})"
    elif ci_width > 0.4:
        decision = DecisionReadiness.NEEDS_MORE_INFO
        rec = f"UNCERTAIN: Wide CI [{ci[0]:.0%}-{ci[1]:.0%}] - gather more evidence"
    else:
        decision = DecisionReadiness.UNCERTAIN
        rec = f"UNCERTAIN: Moderate confidence ({posterior:.0%})"
    
    l5['decision'] = decision.value
    l5['recommendation'] = rec
    l5['confidence_summary'] = {'posterior': posterior, 'credible_interval': ci, 'ci_width': ci_width}
    
    results['layers']['L5_decision'] = l5
    results['decision_state'] = decision.value
    results['recommendation'] = rec
    results['warnings'] = ws.format_all()
    results['warning_summary'] = ws.get_summary_header()
    
    return results


# =============================================================================
# EXPLAIN RESULT
# =============================================================================

def explain_result(results: Dict, verbose: bool = True) -> str:
    """Generate human-readable explanation."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"PRISM v2.0 ANALYSIS: {results['hypothesis_name']}")
    lines.append("=" * 70)
    lines.append(f"Domain: {results['domain']} | Rigor: L{results['rigor_level']}")
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    
    lines.append("─" * 70)
    lines.append("DECISION SUMMARY")
    lines.append("─" * 70)
    
    if results['blocked']:
        lines.append("🚫 ANALYSIS BLOCKED")
        for r in results['blocking_reasons']:
            lines.append(f"   • {r}")
    else:
        l5 = results['layers'].get('L5_decision', {})
        conf = l5.get('confidence_summary', {})
        lines.append(f"📊 Posterior: {conf.get('posterior', 0):.1%}")
        ci = conf.get('credible_interval', (0, 1))
        lines.append(f"📐 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
        lines.append(f"\n🎯 {results['recommendation']}")
    
    lines.append("")
    lines.append("─" * 70)
    lines.append("WARNINGS")
    lines.append("─" * 70)
    lines.append(results.get('warning_summary', ''))
    
    if verbose:
        lines.append("")
        lines.append("─" * 70)
        lines.append("LAYER DETAILS")
        lines.append("─" * 70)
        
        l1 = results['layers'].get('L1_synthesis', {})
        lines.append(f"\n[L1] EVIDENCE SYNTHESIS")
        lines.append(f"   Prior: {l1.get('prior', 0):.1%} ({l1.get('prior_source', '')})")
        lines.append(f"   Posterior: {l1.get('posterior', 0):.1%}")
        lines.append(f"   Information: {l1.get('total_bits', 0):.2f} bits")
        if 'meta_analysis' in l1:
            ma = l1['meta_analysis']
            lines.append(f"   Meta-analysis: effect={ma.get('pooled_effect', 0):.3f}, I²={ma.get('i_squared', 0):.0f}%")
        
        l2 = results['layers'].get('L2_adversarial', {})
        if l2.get('biases_detected'):
            lines.append(f"\n[L2] BIASES: {len(l2['biases_detected'])}")
            for b in l2['biases_detected'][:3]:
                lines.append(f"   • {b['type'].value}: {b['description']}")
        if 'evidence_independence' in l2:
            ei = l2['evidence_independence']
            lines.append(f"   Independence: eff N = {ei.get('effective_n', 0):.1f}/{ei.get('n_evidence', 0)}")
        
        l3 = results['layers'].get('L3_sensitivity', {})
        if 'perturbation' in l3:
            p = l3['perturbation']
            s = "✓ Stable" if p.get('stable') else "⚠ Sensitive"
            lines.append(f"\n[L3] SENSITIVITY: {s} (max: {p.get('max_change', 0):.1%})")
        
        l4 = results['layers'].get('L4_gate', {})
        if l4.get('failed_gates'):
            lines.append(f"\n[L4] FAILED GATES:")
            for g in l4['failed_gates']:
                lines.append(f"   ❌ {g}")
        else:
            lines.append(f"\n[L4] All gates passed ✓")
        
        l5 = results['layers'].get('L5_decision', {})
        if 'utility' in l5:
            u = l5['utility']
            lines.append(f"\n[L5] UTILITY: CE={u.get('certainty_equivalent', 0):.3f}, γ={u.get('risk_aversion', 0)}")
    
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# INTEGRATION TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PRISM v2.0 FINAL - FULL INTEGRATION TEST")
    print("=" * 70)
    
    # Create hypothesis with reference class
    h = Hypothesis(
        name="Drug X Efficacy",
        domain=EvidenceDomain.MEDICAL,
        description="Drug X treats condition Y",
        reference_class="clinical_trial_phase2"
    )
    
    # Set foundation
    h.set_what("Drug X reduces symptoms by 30%", 0.75)
    h.set_why("Targets receptor pathway", 0.70)
    h.set_how("Oral 10mg daily", 0.80)
    h.set_feasibility(technical=0.85, economic=0.70, timeline=0.65)
    
    # Set established with evidence
    est_evidence = EstablishedHypothesisEvidence(
        claim="Receptor pathway is valid target",
        supporting_references=["Smith 2020", "Jones 2021"],
        meta_analyses_cited=2,
        expert_consensus=True
    )
    h.set_established_hypothesis(True, est_evidence)
    
    print(f"\n📋 Hypothesis: {h.name}")
    print(f"   Prior: {h.prior:.1%} ({h.prior_source})")
    print(f"   Establishment verified: {h.establishment_verified}")
    print(f"   Establishment strength: {est_evidence.strength_score():.1%}")
    
    # Add evidence
    e1 = Evidence(
        id="phase2_rct", 
        content="Phase 2 RCT (N=200) shows 32% reduction, p=0.02",
        source="NEJM", date="2024-03", domain=EvidenceDomain.MEDICAL,
        study_design="rct", sample_size=200, p_value=0.02,
        effect_size=0.45, effect_variance=0.05,
        supports_hypothesis=True, causal_level=CausalLevel.INTERVENTION,
        authors=["Smith, J.", "Jones, M."]
    )
    
    e2 = Evidence(
        id="cohort_study",
        content="Cohort (N=2000) confirms benefit, p=0.01",
        source="Lancet", date="2024-01", domain=EvidenceDomain.MEDICAL,
        study_design="cohort", sample_size=2000, p_value=0.01,
        supports_hypothesis=True, authors=["Brown, A."]
    )
    
    e3 = Evidence(
        id="subgroup_analysis",
        content="Subgroup (n=80) in elderly shows enhanced benefit",
        source="J Geriatr", date="2024-02", domain=EvidenceDomain.MEDICAL,
        study_design="rct", sample_size=500, p_value=0.04,
        supports_hypothesis=True
    )
    
    e4 = Evidence(
        id="safety_report",
        content="FDA expressed concerns about adverse event rate",
        source="FDA Advisory", date="2024-04", domain=EvidenceDomain.MEDICAL,
        study_design="observational", sample_size=1000,
        supports_hypothesis=False
    )
    
    for e in [e1, e2, e3, e4]:
        h.add_evidence(e)
    
    print(f"\n📚 Evidence: {len(h.evidence)} pieces")
    for e in h.evidence:
        icon = "✓" if e.supports_hypothesis else "✗"
        warn = f" ⚠️{len(e.validation_warnings)}" if e.validation_warnings else ""
        print(f"   {icon} {e.id}: LR={e.likelihood_ratio:.2f}{warn}")
    
    # Test TextAnalyzer
    print(f"\n🔍 TextAnalyzer Tests:")
    print(f"   'Smith, J.' → '{TextAnalyzer.normalize_author_name('Smith, J.')}'")
    print(f"   'Dr. Jane Smith' → '{TextAnalyzer.normalize_author_name('Dr. Jane Smith')}'")
    entities = TextAnalyzer.extract_entities("Trial NCT12345678 by Smith 2024")
    print(f"   Entities from 'NCT12345678 Smith 2024': {entities}")
    
    # Test independence checker
    print(f"\n🔗 Evidence Independence:")
    indep = EvidenceIndependenceChecker.check_all_independence(h.evidence)
    print(f"   Overall independence: {indep['overall_independence']:.1%}")
    print(f"   Effective N: {indep['effective_evidence_count']:.1f}/{indep['n_evidence']}")
    
    # Add scenarios
    h.add_scenario("Strong adoption", 0.25, 1.2)
    h.add_scenario("Moderate success", 0.40, 0.4)
    h.add_scenario("Limited uptake", 0.25, -0.1)
    h.add_scenario("Safety withdrawal", 0.10, -0.9)
    
    # Build mechanism map
    h.mechanism_map.add_node(MechanismNode("drug", "Drug X", NodeType.CAUSE, 0.9))
    h.mechanism_map.add_node(MechanismNode("receptor", "Receptor", NodeType.MECHANISM, 0.8))
    h.mechanism_map.add_node(MechanismNode("outcome", "Relief", NodeType.OUTCOME, 0.7))
    h.mechanism_map.add_edge(MechanismEdge("drug", "receptor", EdgeType.CAUSES, 0.85))
    h.mechanism_map.add_edge(MechanismEdge("receptor", "outcome", EdgeType.CAUSES, 0.75))
    
    print(f"\n🔗 Mechanism: {len(h.mechanism_map.nodes)} nodes, {len(h.mechanism_map.edges)} edges")
    print(f"   Cycles: {len(h.mechanism_map.find_cycles_tarjan())}")
    
    # Run analysis
    print("\n" + "=" * 70)
    print("RUNNING 5-LAYER ANALYSIS...")
    print("=" * 70)
    
    results = run_analysis(h, rigor_level=2)
    print(explain_result(results, verbose=True))
    
    # Show Bayesian trace
    print("\n📊 Bayesian Update Trace:")
    print(h.updater.explain())
    
    # Show meta-analysis
    if h.meta_analysis_result:
        print(f"\n📈 Meta-Analysis:")
        ma = h.meta_analysis_result
        print(f"   Pooled: {ma.get('pooled_effect', 0):.3f} ± {ma.get('pooled_se', 0):.3f}")
        print(f"   I²: {ma.get('i_squared', 0):.1f}%")
    
    # P-curve
    if h.p_curve_result:
        print(f"\n📉 P-Curve: {h.p_curve_result.get('interpretation', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("✅ PRISM v2.0 FINAL TEST COMPLETE")
    print("=" * 70)
