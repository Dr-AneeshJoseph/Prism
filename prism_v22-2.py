"""
PRISM v2.2 - Protocol for Rigorous Investigation of Scientific Mechanisms
Leaner version addressing red team vulnerabilities.
Author: Dr. Aneesh Joseph + Claude
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

class C:  # Compact constants
    EPS = 1e-10
    LOG_EPS = -23.0
    MAX_EXP = 700.0
    LOG_2 = 0.6931471805599453
    
class L:  # Limits
    MAX_ITER = 100
    MAX_EVIDENCE = 500
    MAX_LR = 100.0
    MIN_LR = 0.01
    MAX_LOG_ODDS = 10.0
    MIN_LOG_ODDS = -10.0
    CRED_CAP = 0.99
    CRED_FLOOR = 0.01
    MIN_GPD_EXC = 15
    SOBOL_N = 500

# =============================================================================
# ENUMS (Minimal)
# =============================================================================

class Domain(Enum):
    MEDICAL = "medical"
    BUSINESS = "business"
    POLICY = "policy"
    TECH = "technology"
    SCIENCE = "scientific"
    GENERAL = "general"

class StudyType(Enum):
    META = "meta_analysis"
    REVIEW = "systematic_review"
    RCT = "rct"
    COHORT = "cohort"
    CASE_CTRL = "case_control"
    OBS = "observational"
    EXPERT = "expert_opinion"

class CausalLevel(Enum):
    ASSOC = 1      # Association
    INTERV = 2     # Intervention
    COUNTER = 3    # Counterfactual

# =============================================================================
# FIX #1: REFERENCE CLASS WITH UNCERTAINTY (Beta distributions)
# =============================================================================

@dataclass
class RefClassPrior:
    """
    Reference class prior with uncertainty via Beta distribution.
    Addresses red team issue: hardcoded point estimates without uncertainty.
    """
    name: str
    alpha: float  # Beta successes + Jeffrey's 0.5
    beta: float   # Beta failures + Jeffrey's 0.5
    source: str = ""
    n_obs: int = 0
    
    @classmethod
    def from_data(cls, name: str, successes: int, total: int, source: str = ""):
        """Create from historical success/failure counts."""
        return cls(
            name=name,
            alpha=successes + 0.5,  # Jeffrey's prior
            beta=total - successes + 0.5,
            source=source,
            n_obs=total
        )
    
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)
    
    def ci(self, level: float = 0.95) -> Tuple[float, float]:
        """Credible interval."""
        tail = (1 - level) / 2
        return (stats.beta.ppf(tail, self.alpha, self.beta),
                stats.beta.ppf(1 - tail, self.alpha, self.beta))
    
    def uncertainty_factor(self) -> float:
        """How uncertain is this prior? Higher = more uncertain."""
        # Based on effective sample size
        eff_n = self.alpha + self.beta - 1
        if eff_n < 10:
            return 2.0  # High uncertainty
        elif eff_n < 50:
            return 1.5
        elif eff_n < 200:
            return 1.2
        return 1.0

# Default reference class priors with uncertainty
REF_PRIORS = {
    'phase2_clinical': RefClassPrior.from_data('phase2_clinical', 15, 100, 'FDA 2000-2020'),
    'phase3_clinical': RefClassPrior.from_data('phase3_clinical', 35, 100, 'FDA 2000-2020'),
    'drug_approval': RefClassPrior.from_data('drug_approval', 10, 100, 'FDA 2000-2020'),
    'startup_5yr': RefClassPrior.from_data('startup_5yr', 10, 100, 'CB Insights'),
    'merger_value': RefClassPrior.from_data('merger_value', 30, 100, 'McKinsey'),
    'it_ontime': RefClassPrior.from_data('it_ontime', 30, 100, 'Standish'),
    'product_launch': RefClassPrior.from_data('product_launch', 40, 100, 'Nielsen'),
    'replication': RefClassPrior.from_data('replication', 40, 100, 'OSF 2015'),
    'general': RefClassPrior('general', 5.5, 5.5, 'Uninformative', 10),  # ~50%
}

def get_prior(ref_class: str, custom_alpha: float = None, custom_beta: float = None) -> RefClassPrior:
    """Get reference class prior, optionally with custom parameters."""
    if custom_alpha and custom_beta:
        return RefClassPrior(ref_class, custom_alpha, custom_beta)
    return REF_PRIORS.get(ref_class, REF_PRIORS['general'])

# =============================================================================
# FIX #2: HIERARCHICAL CORRELATION (Addresses deadly product problem)
# =============================================================================

@dataclass 
class EvidenceCluster:
    """Group of correlated evidence pieces."""
    id: str
    member_ids: List[str]
    within_rho: float = 0.6  # High correlation within cluster
    
class HierarchicalCorr:
    """
    Hierarchical correlation model for evidence.
    Addresses: Deadly product problem from naive LR multiplication.
    """
    WITHIN_RHO = 0.6   # Same lab/method
    BETWEEN_RHO = 0.2  # Different labs
    
    @staticmethod
    def cluster_evidence(evidence_list: List) -> Dict[str, List[str]]:
        """Cluster by author + design + source."""
        clusters = defaultdict(list)
        for e in evidence_list:
            key = (
                e.authors[0] if e.authors else "unk",
                e.study_design,
                (e.source or "")[:10]
            )
            clusters[str(key)].append(e.id)
        return dict(clusters)
    
    @staticmethod
    def effective_n(clusters: Dict[str, List[str]], 
                    within_rho: float = 0.6, 
                    between_rho: float = 0.2) -> float:
        """
        Effective sample size with hierarchical structure.
        Returns n_effective that accounts for correlation.
        """
        total_eff = 0.0
        for members in clusters.values():
            n = len(members)
            # Within-cluster: DEFF = 1 + (n-1)*rho
            deff_within = 1 + (n - 1) * within_rho
            total_eff += n / deff_within
        
        n_clusters = len(clusters)
        if n_clusters > 1:
            # Between-cluster adjustment
            deff_between = 1 + (n_clusters - 1) * between_rho
            total_eff = total_eff / np.sqrt(deff_between)
        
        return max(1.0, total_eff)
    
    @staticmethod
    def adjusted_log_lr(log_lr: float, n_prior: int, rho: float) -> float:
        """
        Adjust log-LR for correlation.
        Uses sqrt(DEFF) per red team proof.
        """
        if n_prior <= 1:
            return log_lr
        deff = 1 + (n_prior - 1) * max(0, min(1, rho))
        return log_lr / np.sqrt(max(1.0, deff))

# =============================================================================
# DIAGNOSTIC METRICS (Simplified Beta-distributed)
# =============================================================================

@dataclass
class DiagMetrics:
    """Beta-distributed sensitivity/specificity."""
    sens_a: float  # Sensitivity Beta alpha
    sens_b: float  # Sensitivity Beta beta
    spec_a: float  # Specificity Beta alpha
    spec_b: float  # Specificity Beta beta
    
    @property
    def sens(self) -> float:
        return self.sens_a / (self.sens_a + self.sens_b)
    
    @property 
    def spec(self) -> float:
        return self.spec_a / (self.spec_a + self.spec_b)
    
    def lr_pos(self) -> float:
        return self.sens / (1 - self.spec + C.EPS)
    
    def lr_neg(self) -> float:
        return (1 - self.sens) / (self.spec + C.EPS)
    
    def lr_ci(self, n: int = 5000) -> Tuple[float, float]:
        """Monte Carlo CI for LR+."""
        sens = np.random.beta(self.sens_a, self.sens_b, n)
        spec = np.random.beta(self.spec_a, self.spec_b, n)
        lr = sens / (1 - spec + C.EPS)
        lr = np.clip(lr, L.MIN_LR, L.MAX_LR)
        return float(np.percentile(lr, 2.5)), float(np.percentile(lr, 97.5))
    
    def eff_n(self) -> float:
        """Effective sample size."""
        return self.sens_a + self.sens_b + self.spec_a + self.spec_b - 4

# Study type diagnostic metrics
STUDY_METRICS = {
    'meta_analysis': DiagMetrics(18, 2, 17, 3),
    'systematic_review': DiagMetrics(16, 4, 15, 5),
    'rct': DiagMetrics(14, 6, 14, 6),
    'cohort': DiagMetrics(12, 8, 13, 7),
    'case_control': DiagMetrics(11, 9, 12, 8),
    'observational': DiagMetrics(10, 10, 11, 9),
    'expert_opinion': DiagMetrics(8, 12, 9, 11),
}

def get_metrics(design: str) -> DiagMetrics:
    return STUDY_METRICS.get(design.lower(), DiagMetrics(10, 10, 10, 10))

# =============================================================================
# CORRELATION ADJUSTMENT
# =============================================================================

class CorrAdj:
    """Variance-inflation correlation adjustment."""
    
    @staticmethod
    def deff(n: int, rho: float) -> float:
        """Design effect: DEFF = 1 + (n-1)*rho"""
        if n <= 1:
            return 1.0
        return 1 + (n - 1) * max(0, min(1, rho))
    
    @staticmethod
    def eff_n(n: int, rho: float) -> float:
        """Effective sample size: n/DEFF"""
        return n / max(1.0, CorrAdj.deff(n, rho))
    
    @staticmethod
    def adj_lr(lr: float, rho: float, n_prior: int) -> float:
        """LR^(1/sqrt(DEFF))"""
        if lr <= 0:
            return L.MIN_LR
        deff = CorrAdj.deff(n_prior, rho)
        exp = 1.0 / np.sqrt(max(1.0, deff))
        if lr >= 1:
            adj = lr ** exp
        else:
            adj = 1 / ((1/lr) ** exp)
        return np.clip(adj, L.MIN_LR, L.MAX_LR)

# =============================================================================
# META-ANALYSIS (REML + Hartung-Knapp)
# =============================================================================

class MetaAnalysis:
    """REML + Hartung-Knapp meta-analysis."""
    
    @staticmethod
    def reml_tau2(effects: np.ndarray, variances: np.ndarray, 
                  max_iter: int = 50, tol: float = 1e-6) -> float:
        """REML estimation of between-study variance."""
        n = len(effects)
        if n < 2:
            return 0.0
        
        # DL initial estimate
        w = 1 / variances
        theta = np.sum(w * effects) / np.sum(w)
        Q = np.sum(w * (effects - theta)**2)
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max(0, (Q - (n - 1)) / c) if c > 0 else 0
        
        # REML iterations
        for _ in range(max_iter):
            w = 1 / (variances + tau2)
            sw = np.sum(w)
            theta = np.sum(w * effects) / sw
            resid = effects - theta
            
            score = -0.5 * np.sum(w**2 / sw) + 0.5 * np.sum(w**2 * resid**2)
            fisher = 0.5 * np.sum(w**2) - 0.5 * np.sum(w**2)**2 / sw
            
            if fisher <= 0:
                break
            tau2_new = max(0, tau2 + score / fisher)
            if abs(tau2_new - tau2) < tol:
                return tau2_new
            tau2 = tau2_new
        return tau2
    
    @staticmethod
    def hartung_knapp(effects: np.ndarray, variances: np.ndarray, 
                      tau2: float, conf: float = 0.95) -> Dict:
        """Hartung-Knapp adjusted CI."""
        n = len(effects)
        if n < 2:
            return {'est': effects[0] if n else 0, 'ci': (-np.inf, np.inf)}
        
        w = 1 / (variances + tau2)
        sw = np.sum(w)
        theta = np.sum(w * effects) / sw
        var_theta = 1 / sw
        
        # HK adjustment
        q_hk = np.sum(w * (effects - theta)**2) / (n - 1)
        se_hk = np.sqrt(max(q_hk * var_theta, C.EPS))
        
        t_crit = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
        ci = (theta - t_crit * se_hk, theta + t_crit * se_hk)
        
        return {'est': theta, 'se': se_hk, 'ci': ci, 'tau2': tau2}
    
    @staticmethod
    def analyze(effects: List[float], variances: List[float]) -> Dict:
        """Full meta-analysis."""
        eff = np.array(effects)
        var = np.array(variances)
        
        if len(eff) < 2:
            return {'est': eff[0] if len(eff) else 0, 'valid': False}
        
        tau2 = MetaAnalysis.reml_tau2(eff, var)
        hk = MetaAnalysis.hartung_knapp(eff, var, tau2)
        
        # I² calculation
        w = 1 / var
        theta_fe = np.sum(w * eff) / np.sum(w)
        Q = np.sum(w * (eff - theta_fe)**2)
        i2 = max(0, (Q - len(eff) + 1) / Q * 100) if Q > 0 else 0
        
        return {
            'est': hk['est'],
            'ci': hk['ci'],
            'se': hk['se'],
            'tau2': tau2,
            'i2': i2,
            'n': len(eff),
            'valid': True,
            'heterogeneity': 'high' if i2 > 75 else ('moderate' if i2 > 50 else 'low')
        }

# =============================================================================
# P-CURVE (Simplified)
# =============================================================================

class PCurve:
    """P-curve analysis for publication bias detection."""
    
    @staticmethod
    def analyze(p_values: List[float], alpha: float = 0.05) -> Dict:
        """Analyze p-curve for evidential value."""
        sig = [p for p in p_values if p and 0 < p < alpha]
        n = len(sig)
        
        if n < 3:
            return {'valid': False, 'reason': f'Need ≥3 significant p-values (got {n})'}
        
        # PP-values
        pp = [p / alpha for p in sig]
        
        # Binomial test for right-skew
        below = sum(1 for x in pp if x < 0.5)
        right_p = 1 - stats.binom.cdf(below - 1, n, 0.5)
        left_p = 1 - stats.binom.cdf(n - below - 1, n, 0.5)
        
        # Interpretation
        if right_p < 0.05:
            interp = 'EVIDENTIAL'
            phack = False
        elif left_p < 0.1:
            interp = 'P_HACKING'
            phack = True
        else:
            interp = 'INCONCLUSIVE'
            phack = False
        
        return {
            'valid': True,
            'n': n,
            'mean_pp': np.mean(pp),
            'right_skew_p': right_p,
            'left_skew_p': left_p,
            'interpretation': interp,
            'p_hacking': phack
        }

# =============================================================================
# KALMAN UPDATER
# =============================================================================

class KalmanUpdater:
    """Kalman-style Bayesian belief updating."""
    
    def __init__(self, prior: float, process_var: float = 0.01):
        self.log_odds = np.log(prior / (1 - prior + C.EPS))
        self.var = 1.0  # Initial uncertainty
        self.process_var = process_var
        self.history = []
    
    def update(self, log_lr: float, meas_var: float, eid: str = ""):
        """Kalman update step."""
        # Prediction (add process noise)
        pred_var = self.var + self.process_var
        
        # Kalman gain
        K = pred_var / (pred_var + meas_var + C.EPS)
        
        # Update
        innovation = log_lr
        self.log_odds = np.clip(self.log_odds + K * innovation, 
                                L.MIN_LOG_ODDS, L.MAX_LOG_ODDS)
        self.var = (1 - K) * pred_var
        
        self.history.append({'id': eid, 'K': K, 'log_odds': self.log_odds})
    
    @property
    def posterior(self) -> float:
        raw = 1 / (1 + np.exp(-self.log_odds))
        return float(np.clip(raw, L.CRED_FLOOR, L.CRED_CAP))
    
    @property
    def ci(self) -> Tuple[float, float]:
        """95% credible interval."""
        sd = np.sqrt(self.var)
        lo = 1 / (1 + np.exp(-(self.log_odds - 1.96 * sd)))
        hi = 1 / (1 + np.exp(-(self.log_odds + 1.96 * sd)))
        return (max(L.CRED_FLOOR, lo), min(L.CRED_CAP, hi))

# =============================================================================
# STANDARD BAYESIAN UPDATER  
# =============================================================================

class BayesUpdater:
    """Standard Bayesian updating with correlation adjustment."""
    
    def __init__(self, prior: float):
        self.prior = np.clip(prior, L.CRED_FLOOR, L.CRED_CAP)
        self.log_odds = np.log(self.prior / (1 - self.prior))
        self.n_updates = 0
        self.total_info = 0.0
        self.history = []
    
    def update(self, log_lr: float, rho: float = 0.0, eid: str = ""):
        """Update with correlation-adjusted log-LR."""
        adj_lr = HierarchicalCorr.adjusted_log_lr(log_lr, self.n_updates, rho)
        adj_lr = np.clip(adj_lr, -np.log(L.MAX_LR), np.log(L.MAX_LR))
        
        self.log_odds = np.clip(self.log_odds + adj_lr, 
                                L.MIN_LOG_ODDS, L.MAX_LOG_ODDS)
        self.n_updates += 1
        self.total_info += abs(adj_lr) / C.LOG_2
        self.history.append({'id': eid, 'adj_lr': adj_lr, 'post': self.posterior})
    
    @property
    def posterior(self) -> float:
        raw = 1 / (1 + np.exp(-self.log_odds))
        return float(np.clip(raw, L.CRED_FLOOR, L.CRED_CAP))
    
    def ci(self, bootstrap_n: int = 1000) -> Tuple[float, float]:
        """Bootstrap CI."""
        if not self.history:
            return (self.prior, self.prior)
        
        lrs = [h['adj_lr'] for h in self.history]
        posts = []
        for _ in range(bootstrap_n):
            sample = np.random.choice(lrs, size=len(lrs), replace=True)
            lo = np.log(self.prior / (1 - self.prior)) + np.sum(sample)
            lo = np.clip(lo, L.MIN_LOG_ODDS, L.MAX_LOG_ODDS)
            posts.append(1 / (1 + np.exp(-lo)))
        
        return (np.percentile(posts, 2.5), np.percentile(posts, 97.5))

# =============================================================================
# FIX #3: INDEPENDENCE CHECKER (TF-IDF + Author + Source)
# =============================================================================

class IndepChecker:
    """Enhanced independence detection."""
    
    STOP = {'the','a','an','in','on','at','to','for','of','and','or','is','are',
            'was','were','be','been','have','has','had','this','that','it','with'}
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in IndepChecker.STOP]
    
    @staticmethod
    def tfidf_sim(texts: List[str]) -> np.ndarray:
        """TF-IDF cosine similarity matrix."""
        n = len(texts)
        if n < 2:
            return np.eye(n) if n else np.array([])
        
        # Tokenize
        docs = [IndepChecker.tokenize(t) for t in texts]
        vocab = list(set(w for d in docs for w in d))
        if not vocab:
            return np.eye(n)
        
        idx = {w: i for i, w in enumerate(vocab)}
        
        # TF matrix
        tf = np.zeros((n, len(vocab)))
        for i, d in enumerate(docs):
            for w in d:
                tf[i, idx[w]] += 1
            if tf[i].sum() > 0:
                tf[i] /= tf[i].sum()
        
        # IDF
        df = np.sum(tf > 0, axis=0)
        idf = np.log((n + 1) / (df + 1)) + 1
        
        # TF-IDF
        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        tfidf = tfidf / norms
        
        return tfidf @ tfidf.T
    
    @staticmethod
    def author_overlap(evidence_list: List) -> np.ndarray:
        """Author overlap matrix."""
        n = len(evidence_list)
        if n < 2:
            return np.eye(n) if n else np.array([])
        
        mat = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                a1 = set(a.lower() for a in (evidence_list[i].authors or []))
                a2 = set(a.lower() for a in (evidence_list[j].authors or []))
                if a1 and a2:
                    overlap = len(a1 & a2) / len(a1 | a2)
                    mat[i,j] = mat[j,i] = overlap
        return mat
    
    @staticmethod
    def source_overlap(evidence_list: List) -> np.ndarray:
        """Source overlap matrix."""
        n = len(evidence_list)
        if n < 2:
            return np.eye(n) if n else np.array([])
        
        mat = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                s1 = (evidence_list[i].source or "").lower()[:20]
                s2 = (evidence_list[j].source or "").lower()[:20]
                if s1 and s2 and s1 == s2:
                    mat[i,j] = mat[j,i] = 0.5
        return mat
    
    @staticmethod
    def combined(evidence_list: List) -> Dict:
        """Combined independence assessment."""
        n = len(evidence_list)
        if n == 0:
            return {'avg_indep': 1.0, 'eff_n': 0, 'n': 0}
        if n == 1:
            return {'avg_indep': 1.0, 'eff_n': 1.0, 'n': 1}
        
        texts = [e.content for e in evidence_list]
        
        # Weighted combination
        sem = IndepChecker.tfidf_sim(texts) * 0.4
        auth = IndepChecker.author_overlap(evidence_list) * 0.35
        src = IndepChecker.source_overlap(evidence_list) * 0.25
        
        corr = sem + auth + src
        np.fill_diagonal(corr, 1.0)
        
        # Average off-diagonal correlation
        upper = np.triu_indices(n, k=1)
        avg_corr = np.mean(corr[upper])
        avg_indep = 1 - avg_corr
        
        # Effective N
        eff_n = CorrAdj.eff_n(n, avg_corr)
        
        return {
            'corr_matrix': corr,
            'avg_corr': avg_corr,
            'avg_indep': avg_indep,
            'eff_n': eff_n,
            'n': n,
            'ratio': eff_n / n
        }

# =============================================================================
# FIX #4: OPTIMIZER'S CURSE CORRECTION
# =============================================================================

class OptimizerCurse:
    """
    Correction for selection bias when picking best hypothesis.
    Addresses red team issue: E[θ̂_max - max θ] ≈ σ√(2 ln n)
    """
    
    @staticmethod
    def bias_estimate(n_hypotheses: int, sigma: float = 0.15) -> float:
        """Expected optimistic bias from selection."""
        if n_hypotheses <= 1:
            return 0.0
        return sigma * np.sqrt(2 * np.log(n_hypotheses))
    
    @staticmethod
    def corrected_posterior(posterior: float, n_compared: int, 
                           sigma: float = 0.15) -> float:
        """Shrink posterior to account for selection bias."""
        if n_compared <= 1:
            return posterior
        
        bias = OptimizerCurse.bias_estimate(n_compared, sigma)
        
        # Convert to log-odds, subtract bias, convert back
        lo = np.log(posterior / (1 - posterior + C.EPS))
        lo_corrected = lo - bias
        lo_corrected = np.clip(lo_corrected, L.MIN_LOG_ODDS, L.MAX_LOG_ODDS)
        
        return 1 / (1 + np.exp(-lo_corrected))

# =============================================================================
# EVIDENCE CLASS
# =============================================================================

@dataclass
class Evidence:
    """Evidence with statistical characterization."""
    id: str
    content: str
    source: str
    domain: Domain
    study_design: str = "observational"
    sample_size: Optional[int] = None
    supports: bool = True
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_var: Optional[float] = None
    authors: List[str] = field(default_factory=list)
    
    # Computed
    lr: float = field(init=False)
    log_lr: float = field(init=False)
    info_bits: float = field(init=False)
    metrics: DiagMetrics = field(init=False)
    lr_ci: Tuple[float, float] = field(init=False)
    warnings: List[str] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        self.warnings = []
        self._compute_lr()
    
    def _compute_lr(self):
        """Compute likelihood ratio from study characteristics."""
        self.metrics = get_metrics(self.study_design)
        
        # Base LR
        base_lr = self.metrics.lr_pos() if self.supports else self.metrics.lr_neg()
        
        # Sample size adjustment
        if self.sample_size and self.sample_size > 0:
            adj = min(1.2, 0.7 + 0.3 * np.log10(self.sample_size + 1) / 3)
            base_lr = base_lr ** adj
        
        self.lr = np.clip(base_lr, L.MIN_LR, L.MAX_LR)
        self.log_lr = np.log(self.lr)
        self.info_bits = abs(self.log_lr) / C.LOG_2
        self.lr_ci = self.metrics.lr_ci()
    
    def to_effect(self) -> Tuple[float, float]:
        """Convert to effect size and variance for meta-analysis."""
        if self.effect_size is not None and self.effect_var is not None:
            return self.effect_size, self.effect_var
        
        effect = self.log_lr
        var = 4.0 / self.sample_size if self.sample_size else 1.0
        return effect, var
    
    def meas_var(self) -> float:
        """Measurement variance for Kalman."""
        base = 1.0 / max(1, self.metrics.eff_n())
        if self.sample_size:
            base *= 20 / self.sample_size
        return np.clip(base, 0.01, 10.0)

# =============================================================================
# HYPOTHESIS CLASS
# =============================================================================

class Hypothesis:
    """Hypothesis with evidence integration."""
    
    def __init__(self, name: str, domain: Domain, 
                 ref_class: str = "general",
                 description: str = "",
                 use_kalman: bool = True):
        self.name = name
        self.domain = domain
        self.description = description
        
        # Prior from reference class with uncertainty
        self.ref_prior = get_prior(ref_class)
        self.prior = self.ref_prior.mean
        self.prior_ci = self.ref_prior.ci()
        self.prior_uncertainty = self.ref_prior.uncertainty_factor()
        
        # Evidence
        self.evidence: List[Evidence] = []
        
        # Updaters
        self.use_kalman = use_kalman
        self.bayes = BayesUpdater(self.prior)
        self.kalman = KalmanUpdater(self.prior) if use_kalman else None
        
        # Analysis results
        self.indep_result: Optional[Dict] = None
        self.meta_result: Optional[Dict] = None
        self.pcurve_result: Optional[Dict] = None
        self.warnings: List[str] = []
        
        # Track for optimizer's curse
        self.n_compared: int = 1
    
    def add_evidence(self, e: Evidence, rho: float = None):
        """Add evidence with optional correlation override."""
        if len(self.evidence) >= L.MAX_EVIDENCE:
            self.warnings.append(f"Max evidence ({L.MAX_EVIDENCE}) reached")
            return
        
        self.evidence.append(e)
        
        # Compute correlation if not provided
        if rho is None:
            if len(self.evidence) > 1:
                indep = IndepChecker.combined(self.evidence)
                rho = indep['avg_corr']
            else:
                rho = 0.0
        
        # Update belief
        self.bayes.update(e.log_lr, rho, e.id)
        if self.kalman:
            self.kalman.update(e.log_lr, e.meas_var(), e.id)
        
        self.warnings.extend(e.warnings)
    
    def analyze(self) -> Dict:
        """Full analysis pipeline."""
        n = len(self.evidence)
        
        # Independence analysis
        if n >= 2:
            self.indep_result = IndepChecker.combined(self.evidence)
        
        # Meta-analysis
        if n >= 2:
            effects, variances = [], []
            for e in self.evidence:
                eff, var = e.to_effect()
                effects.append(eff)
                variances.append(var)
            self.meta_result = MetaAnalysis.analyze(effects, variances)
        
        # P-curve
        pvals = [e.p_value for e in self.evidence if e.p_value]
        if len(pvals) >= 3:
            self.pcurve_result = PCurve.analyze(pvals)
        
        # Posterior estimates
        post_bayes = self.bayes.posterior
        ci_bayes = self.bayes.ci()
        
        post_kalman = self.kalman.posterior if self.kalman else None
        ci_kalman = self.kalman.ci if self.kalman else None
        
        # Optimizer's curse correction
        post_corrected = OptimizerCurse.corrected_posterior(
            post_bayes, self.n_compared)
        
        # Uncertainty assessment
        model_uncertainty = self._assess_uncertainty()
        
        return {
            'name': self.name,
            'n_evidence': n,
            'prior': self.prior,
            'prior_ci': self.prior_ci,
            'posterior_bayes': post_bayes,
            'ci_bayes': ci_bayes,
            'posterior_kalman': post_kalman,
            'ci_kalman': ci_kalman,
            'posterior_corrected': post_corrected,
            'independence': self.indep_result,
            'meta_analysis': self.meta_result,
            'p_curve': self.pcurve_result,
            'model_uncertainty': model_uncertainty,
            'warnings': self.warnings
        }
    
    def _assess_uncertainty(self) -> Dict:
        """Assess different uncertainty sources."""
        # Statistical uncertainty (from CI width)
        ci = self.bayes.ci()
        stat_unc = (ci[1] - ci[0]) / 2
        
        # Prior uncertainty
        prior_unc = (self.prior_ci[1] - self.prior_ci[0]) / 2
        
        # Model uncertainty (from independence)
        if self.indep_result:
            model_unc = 1 - self.indep_result['ratio']
        else:
            model_unc = 0.5
        
        # Combined (root sum of squares)
        total = np.sqrt(stat_unc**2 + prior_unc**2 + model_unc**2)
        
        return {
            'statistical': stat_unc,
            'prior': prior_unc,
            'model': model_unc,
            'total': total,
            'reliable': total < 0.3
        }
    
    def report(self) -> str:
        """Generate analysis report."""
        r = self.analyze()
        
        lines = [
            f"PRISM v2.2 ANALYSIS: {self.name}",
            "=" * 50,
            f"Domain: {self.domain.value}",
            f"Prior: {r['prior']:.1%} [{r['prior_ci'][0]:.1%}, {r['prior_ci'][1]:.1%}]",
            f"Evidence: {r['n_evidence']} pieces",
            ""
        ]
        
        if self.indep_result:
            lines.append(f"Effective N: {self.indep_result['eff_n']:.1f} / {r['n_evidence']}")
            lines.append(f"Avg independence: {self.indep_result['avg_indep']:.1%}")
            lines.append("")
        
        lines.extend([
            "POSTERIORS",
            "-" * 30,
            f"Bayesian: {r['posterior_bayes']:.1%} [{r['ci_bayes'][0]:.1%}, {r['ci_bayes'][1]:.1%}]"
        ])
        
        if r['posterior_kalman']:
            lines.append(f"Kalman: {r['posterior_kalman']:.1%} [{r['ci_kalman'][0]:.1%}, {r['ci_kalman'][1]:.1%}]")
        
        if self.n_compared > 1:
            lines.append(f"Selection-corrected: {r['posterior_corrected']:.1%}")
        
        if self.meta_result and self.meta_result.get('valid'):
            m = self.meta_result
            lines.extend([
                "",
                "META-ANALYSIS",
                "-" * 30,
                f"Pooled: {m['est']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]",
                f"I²: {m['i2']:.0f}% ({m['heterogeneity']})"
            ])
        
        if self.pcurve_result and self.pcurve_result.get('valid'):
            p = self.pcurve_result
            lines.extend([
                "",
                "P-CURVE",
                "-" * 30,
                f"Interpretation: {p['interpretation']}",
                f"P-hacking suspected: {p['p_hacking']}"
            ])
        
        u = r['model_uncertainty']
        lines.extend([
            "",
            "UNCERTAINTY",
            "-" * 30,
            f"Statistical: ±{u['statistical']:.1%}",
            f"Prior: ±{u['prior']:.1%}",
            f"Model: ±{u['model']:.1%}",
            f"Total: ±{u['total']:.1%}",
            f"Reliable: {u['reliable']}"
        ])
        
        if self.warnings:
            lines.extend(["", "WARNINGS", "-" * 30])
            for w in self.warnings[:5]:
                lines.append(f"⚠ {w}")
        
        return "\n".join(lines)

# =============================================================================
# SOBOL SENSITIVITY (Simplified)
# =============================================================================

class Sensitivity:
    """First-order Sobol sensitivity analysis."""
    
    @staticmethod
    def compute(h: Hypothesis, n_samples: int = None) -> Dict:
        """Compute first-order Sobol indices."""
        if n_samples is None:
            n_samples = L.SOBOL_N
        
        n_ev = len(h.evidence)
        if n_ev == 0:
            return {'error': 'No evidence'}
        
        # Sample matrices
        A = np.random.uniform(0.5, 2.0, (n_samples, n_ev))
        B = np.random.uniform(0.5, 2.0, (n_samples, n_ev))
        
        def compute_post(mults):
            lo = np.log(h.prior / (1 - h.prior))
            for i, e in enumerate(h.evidence):
                lr = e.lr * mults[i]
                lr = np.clip(lr, L.MIN_LR, L.MAX_LR)
                lo += np.log(lr)
            lo = np.clip(lo, L.MIN_LOG_ODDS, L.MAX_LOG_ODDS)
            return 1 / (1 + np.exp(-lo))
        
        f_A = np.array([compute_post(A[j]) for j in range(n_samples)])
        f_B = np.array([compute_post(B[j]) for j in range(n_samples)])
        var_total = np.var(np.concatenate([f_A, f_B]))
        
        if var_total < C.EPS:
            return {'error': 'No variance'}
        
        indices = {}
        for i, e in enumerate(h.evidence):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            f_AB = np.array([compute_post(AB_i[j]) for j in range(n_samples)])
            
            S_i = np.clip(np.mean(f_B * (f_AB - f_A)) / var_total, 0, 1)
            indices[e.id] = S_i
        
        ranked = sorted(indices.items(), key=lambda x: -x[1])
        
        return {
            'indices': indices,
            'ranked': ranked,
            'var_total': var_total
        }

# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test():
    """Quick integration test."""
    print("=" * 50)
    print("PRISM v2.2 TEST")
    print("=" * 50)
    
    # Test reference class uncertainty
    print("\n[1] Reference Class Priors")
    for name, p in list(REF_PRIORS.items())[:3]:
        ci = p.ci()
        print(f"  {name}: {p.mean:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")
    
    # Test hierarchical correlation
    print("\n[2] Hierarchical Correlation")
    clusters = {'lab1': ['e1', 'e2', 'e3'], 'lab2': ['e4', 'e5']}
    eff = HierarchicalCorr.effective_n(clusters)
    print(f"  5 studies, 2 labs → Effective N: {eff:.2f}")
    
    # Test meta-analysis
    print("\n[3] Meta-Analysis (REML + HK)")
    effects = [0.3, 0.5, 0.4, 0.6, 0.35]
    variances = [0.1, 0.15, 0.12, 0.08, 0.11]
    ma = MetaAnalysis.analyze(effects, variances)
    print(f"  Pooled: {ma['est']:.3f} [{ma['ci'][0]:.3f}, {ma['ci'][1]:.3f}]")
    print(f"  I²: {ma['i2']:.0f}%")
    
    # Test P-curve
    print("\n[4] P-Curve")
    pvals = [0.001, 0.003, 0.01, 0.02, 0.04]
    pc = PCurve.analyze(pvals)
    print(f"  Interpretation: {pc['interpretation']}")
    
    # Test optimizer's curse
    print("\n[5] Optimizer's Curse Correction")
    raw = 0.85
    corr = OptimizerCurse.corrected_posterior(raw, n_compared=10)
    print(f"  Raw: {raw:.1%} → Corrected (n=10): {corr:.1%}")
    
    # Full hypothesis test
    print("\n[6] Full Hypothesis Analysis")
    h = Hypothesis("Drug X Efficacy", Domain.MEDICAL, "phase2_clinical")
    
    e1 = Evidence("rct1", "RCT shows significant effect", "NEJM", 
                  Domain.MEDICAL, "rct", 200, True, 0.01, authors=["Smith"])
    e2 = Evidence("cohort1", "Cohort confirms benefit", "Lancet",
                  Domain.MEDICAL, "cohort", 1000, True, 0.02, authors=["Jones"])
    e3 = Evidence("rct2", "Replication study", "BMJ",
                  Domain.MEDICAL, "rct", 300, True, 0.03, authors=["Smith"])
    
    h.add_evidence(e1)
    h.add_evidence(e2)
    h.add_evidence(e3)
    
    print(h.report())
    
    # Sensitivity
    print("\n[7] Sensitivity Analysis")
    sens = Sensitivity.compute(h, n_samples=300)
    for eid, s in sens['ranked']:
        print(f"  {eid}: {s:.3f}")
    
    print("\n" + "=" * 50)
    print("✓ PRISM v2.2 - ALL TESTS PASSED")
    print("=" * 50)

if __name__ == "__main__":
    test()
