"""
ENHANCED ANALYTICAL PROTOCOL WITH VISUAL MAPPING
=================================================
Combines:
- Process tracing (timeline of analysis steps)
- Content visualization (mechanism maps, causal graphs)
- Multiplicative fatal-flaw scoring (one bad score kills it)
- Multi-hypothesis comparison
- Sensitivity analysis (what assumptions matter most)

When you run an analysis, it generates:
1. Full event trace
2. Visual mechanism map
3. Sensitivity analysis
4. Comparison dashboard (if multiple hypotheses)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
import json

# =============================================================================
# CORE TYPES
# =============================================================================

class EvidenceDomain(Enum):
    MEDICAL = "medical"
    BUSINESS = "business"
    POLICY = "policy"
    TECHNOLOGY = "technology"
    GENERAL = "general"

class NodeType(Enum):
    """Types of nodes in mechanism maps"""
    CAUSE = "cause"           # Root cause / driver
    MECHANISM = "mechanism"   # Causal pathway
    OUTCOME = "outcome"       # Desired result
    BLOCKER = "blocker"       # What could prevent success
    ASSUMPTION = "assumption" # Untested belief
    EVIDENCE = "evidence"     # Supporting data
    INTERVENTION = "intervention"  # Where we act

class EdgeType(Enum):
    """Types of relationships in mechanism maps"""
    CAUSES = "causes"
    PREVENTS = "prevents"
    ENABLES = "enables"
    COMPENSATES = "compensates"  # Counteracts
    REQUIRES = "requires"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"

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

EVIDENCE_QUALITY = {
    EvidenceDomain.MEDICAL: {
        'meta_analysis': 1.0, 'rct': 0.85, 'cohort': 0.7,
        'case_control': 0.6, 'expert_opinion': 0.3, 'anecdote': 0.15
    },
    EvidenceDomain.BUSINESS: {
        'multi_company_analysis': 0.9, 'controlled_experiment': 0.8,
        'benchmark': 0.7, 'case_study': 0.5, 'expert_opinion': 0.4, 'anecdote': 0.2
    },
    EvidenceDomain.GENERAL: {
        'rigorous_study': 0.8, 'good_study': 0.6, 'weak_study': 0.4,
        'expert_opinion': 0.3, 'anecdote': 0.15
    }
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
    confidence: float = 0.5  # How sure are we this exists/matters
    description: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.node_type.value,
            'confidence': self.confidence,
            'description': self.description
        }

@dataclass
class MechanismEdge:
    """A relationship between nodes"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5  # How strong is this relationship
    evidence_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.edge_type.value,
            'strength': self.strength
        }

class MechanismMap:
    """Causal mechanism map for visualizing HOW something works"""
    
    def __init__(self):
        self.nodes: Dict[str, MechanismNode] = {}
        self.edges: List[MechanismEdge] = []
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
                'strength': edge.strength
            })
    
    def get_critical_path(self) -> List[str]:
        """Find the path from causes to outcomes with lowest confidence"""
        # Simple implementation - find nodes on path to outcome
        causes = [n for n in self.nodes.values() if n.node_type == NodeType.CAUSE]
        outcomes = [n for n in self.nodes.values() if n.node_type == NodeType.OUTCOME]
        
        if not causes or not outcomes:
            return []
        
        # Return nodes sorted by confidence (weakest links)
        all_nodes = list(self.nodes.values())
        return [n.id for n in sorted(all_nodes, key=lambda x: x.confidence)[:5]]
    
    def get_blockers(self) -> List[MechanismNode]:
        """Get all blocker nodes"""
        return [n for n in self.nodes.values() if n.node_type == NodeType.BLOCKER]
    
    def get_assumptions(self) -> List[MechanismNode]:
        """Get all assumption nodes"""
        return [n for n in self.nodes.values() if n.node_type == NodeType.ASSUMPTION]
    
    def to_dict(self) -> Dict:
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges]
        }
    
    def overall_confidence(self) -> float:
        """Multiplicative confidence - weakest link matters most"""
        if not self.nodes:
            return 0.0
        confidences = [n.confidence for n in self.nodes.values()]
        # Geometric mean (multiplicative) - penalizes low values heavily
        return float(np.exp(np.mean(np.log(np.array(confidences) + 0.01))))

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
# EVIDENCE
# =============================================================================

@dataclass
class Evidence:
    id: str
    content: str
    source: str
    quality: float
    date: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    study_design: Optional[str] = None
    supports_node_ids: List[str] = field(default_factory=list)
    
    def get_quality(self) -> float:
        if self.study_design:
            hierarchy = EVIDENCE_QUALITY.get(self.domain, EVIDENCE_QUALITY[EvidenceDomain.GENERAL])
            return hierarchy.get(self.study_design, self.quality)
        return self.quality

# =============================================================================
# DIMENSIONAL SCORING (ADDITIVE + MULTIPLICATIVE)
# =============================================================================

@dataclass
class DimensionScore:
    """Score for a single dimension with both value and criticality"""
    name: str
    value: float  # 0-1
    weight: float  # For additive scoring
    is_fatal_below: float = 0.3  # If below this, it's a fatal flaw
    rationale: str = ""
    
    @property
    def is_fatal_flaw(self) -> bool:
        return self.value < self.is_fatal_below

class ScoringSystem:
    """Dual scoring: additive (weighted sum) AND multiplicative (fatal flaw)"""
    
    def __init__(self):
        self.dimensions: Dict[str, DimensionScore] = {}
        self.tracer: Optional[AnalysisTracer] = None
    
    def set_dimension(self, name: str, value: float, weight: float = 0.2, 
                      fatal_threshold: float = 0.3, rationale: str = ""):
        self.dimensions[name] = DimensionScore(
            name=name, value=value, weight=weight,
            is_fatal_below=fatal_threshold, rationale=rationale
        )
        
        # Check for fatal flaw
        if value < fatal_threshold and self.tracer:
            self.tracer.log(TraceEventType.FATAL_FLAW_DETECTED, "L4", {
                'dimension': name,
                'value': value,
                'threshold': fatal_threshold,
                'rationale': rationale
            })
    
    def additive_score(self) -> float:
        """Traditional weighted sum"""
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions.values())
        weighted_sum = sum(d.value * d.weight for d in self.dimensions.values())
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def multiplicative_score(self) -> float:
        """Geometric mean - fatal flaws dominate"""
        if not self.dimensions:
            return 0.0
        values = [d.value for d in self.dimensions.values()]
        # Add small epsilon to avoid log(0)
        return float(np.exp(np.mean(np.log(np.array(values) + 0.01))))
    
    def fatal_flaws(self) -> List[DimensionScore]:
        """Get all dimensions that are fatal flaws"""
        return [d for d in self.dimensions.values() if d.is_fatal_flaw]
    
    def combined_score(self) -> Tuple[float, str]:
        """
        Combined scoring logic:
        - If ANY fatal flaw exists, score is capped at 0.3
        - Otherwise, use weighted average of additive and multiplicative
        """
        flaws = self.fatal_flaws()
        additive = self.additive_score()
        multiplicative = self.multiplicative_score()
        
        if flaws:
            # Fatal flaw caps the score
            score = min(0.3, multiplicative)
            reason = f"Fatal flaw in: {', '.join(f.name for f in flaws)}"
        else:
            # Blend both approaches
            score = 0.6 * additive + 0.4 * multiplicative
            reason = "No fatal flaws"
        
        return score, reason
    
    def sensitivity_analysis(self) -> List[Dict]:
        """
        Calculate how much each dimension affects the final score.
        Returns list sorted by impact (highest first).
        """
        if not self.dimensions:
            return []
        
        base_score, _ = self.combined_score()
        sensitivities = []
        
        for name, dim in self.dimensions.items():
            # What if this dimension was 0.1 higher?
            original = dim.value
            dim.value = min(1.0, original + 0.1)
            high_score, _ = self.combined_score()
            
            # What if 0.1 lower?
            dim.value = max(0.0, original - 0.1)
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
# MAIN ANALYSIS ELEMENT
# =============================================================================

@dataclass
class AnalysisElement:
    """Enhanced foundation element with mechanism mapping"""
    name: str
    domain: EvidenceDomain = EvidenceDomain.GENERAL
    
    # Core dimensions
    what: Optional[str] = None
    why: Optional[str] = None
    how: Optional[str] = None
    measure: Optional[str] = None
    
    # Confidence tracking
    what_confidence: float = 0.0
    why_confidence: float = 0.0
    how_confidence: float = 0.0
    measure_confidence: float = 0.0
    
    # Enhanced components
    mechanism_map: MechanismMap = field(default_factory=MechanismMap)
    scoring: ScoringSystem = field(default_factory=ScoringSystem)
    evidence: List[Evidence] = field(default_factory=list)
    tracer: Optional[AnalysisTracer] = None
    
    def __post_init__(self):
        self.mechanism_map = MechanismMap()
        self.scoring = ScoringSystem()
    
    def set_tracer(self, tracer: AnalysisTracer):
        self.tracer = tracer
        self.mechanism_map.tracer = tracer
        self.scoring.tracer = tracer
    
    def set_what(self, value: str, confidence: float):
        self.what = value
        self.what_confidence = confidence
        self.scoring.set_dimension('definition_clarity', confidence, weight=0.15)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1", 
                {'dimension': 'WHAT', 'confidence': confidence})
    
    def set_why(self, value: str, confidence: float):
        self.why = value
        self.why_confidence = confidence
        self.scoring.set_dimension('justification_strength', confidence, weight=0.20)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                {'dimension': 'WHY', 'confidence': confidence})
    
    def set_how(self, value: str, confidence: float):
        self.how = value
        self.how_confidence = confidence
        self.scoring.set_dimension('mechanism_validity', confidence, weight=0.25, fatal_threshold=0.25)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                {'dimension': 'HOW', 'confidence': confidence})
    
    def set_measure(self, value: str, confidence: float):
        self.measure = value
        self.measure_confidence = confidence
        self.scoring.set_dimension('measurability', confidence, weight=0.15)
        if self.tracer:
            self.tracer.log(TraceEventType.FOUNDATION_SET, "L1",
                {'dimension': 'MEASURE', 'confidence': confidence})
    
    def add_evidence(self, evidence: Evidence):
        self.evidence.append(evidence)
        quality = evidence.get_quality()
        
        # Update evidence dimension score
        if self.evidence:
            avg_quality = np.mean([e.get_quality() for e in self.evidence])
            # Diminishing returns
            n = len(self.evidence)
            effective_quality = avg_quality * (1 - 0.5 * np.exp(-n/3))
            self.scoring.set_dimension('evidence_quality', effective_quality, weight=0.15)
        
        if self.tracer:
            self.tracer.log(TraceEventType.EVIDENCE_ADDED, "L1", {
                'id': evidence.id, 'quality': quality, 'source': evidence.source
            })
    
    def add_mechanism_node(self, node: MechanismNode) -> str:
        return self.mechanism_map.add_node(node)
    
    def add_mechanism_edge(self, edge: MechanismEdge):
        self.mechanism_map.add_edge(edge)
    
    def set_feasibility(self, technical: float, economic: float, timeline: float):
        """Set feasibility dimensions"""
        self.scoring.set_dimension('technical_feasibility', technical, weight=0.15, fatal_threshold=0.2)
        self.scoring.set_dimension('economic_viability', economic, weight=0.10, fatal_threshold=0.2)
        self.scoring.set_dimension('timeline_realism', timeline, weight=0.10)
    
    def set_risk(self, execution_risk: float, external_risk: float):
        """Set risk dimensions (inverted - higher is better/safer)"""
        self.scoring.set_dimension('execution_safety', 1.0 - execution_risk, weight=0.10)
        self.scoring.set_dimension('external_resilience', 1.0 - external_risk, weight=0.10)

# =============================================================================
# ADVERSARIAL TESTER
# =============================================================================

@dataclass
class Criticism:
    content: str
    severity: float
    cycle: str
    dimension_affected: str = ""
    resolved: bool = False
    response: str = ""

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
        self.tracer.log(TraceEventType.ITERATION_START, "L2", {'iteration': self.iteration})
        
        # Check mechanism map
        assumptions = self.element.mechanism_map.get_assumptions()
        for assumption in assumptions:
            if assumption.confidence < 0.7:
                c = Criticism(
                    f"Untested assumption: '{assumption.label}' (conf: {assumption.confidence:.2f})",
                    severity=0.7, cycle="assumption_test", dimension_affected="mechanism_validity"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", {
                    'content': c.content, 'severity': c.severity, 'cycle': c.cycle
                })
        
        # Check for blockers
        blockers = self.element.mechanism_map.get_blockers()
        for blocker in blockers:
            if blocker.confidence > 0.5:  # Likely blocker
                c = Criticism(
                    f"Potential blocker: '{blocker.label}' (likelihood: {blocker.confidence:.2f})",
                    severity=0.8, cycle="pre_mortem", dimension_affected="execution_safety"
                )
                new.append(c)
                self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", {
                    'content': c.content, 'severity': c.severity, 'cycle': c.cycle
                })
        
        # Check fatal flaws
        flaws = self.element.scoring.fatal_flaws()
        for flaw in flaws:
            c = Criticism(
                f"Fatal flaw: {flaw.name} = {flaw.value:.2f} (below {flaw.is_fatal_below})",
                severity=0.95, cycle="fatal_flaw", dimension_affected=flaw.name
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", {
                'content': c.content, 'severity': c.severity, 'cycle': c.cycle
            })
        
        # Check evidence gaps
        if len(self.element.evidence) < 2:
            c = Criticism(
                "Insufficient evidence (< 2 sources)",
                severity=0.6, cycle="evidence_check", dimension_affected="evidence_quality"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", {
                'content': c.content, 'severity': c.severity, 'cycle': c.cycle
            })
        
        # Check weak links in mechanism
        weak_nodes = [n for n in self.element.mechanism_map.nodes.values() if n.confidence < 0.5]
        for node in weak_nodes[:3]:  # Top 3 weakest
            c = Criticism(
                f"Weak mechanism link: '{node.label}' (conf: {node.confidence:.2f})",
                severity=0.5, cycle="mechanism_analysis", dimension_affected="mechanism_validity"
            )
            new.append(c)
            self.tracer.log(TraceEventType.CRITICISM_GENERATED, "L2", {
                'content': c.content, 'severity': c.severity, 'cycle': c.cycle
            })
        
        self.criticisms.extend(new)
        return new
    
    def unresolved_critical(self, threshold: float = 0.7) -> List[Criticism]:
        return [c for c in self.criticisms if not c.resolved and c.severity >= threshold]
    
    def consistency_score(self) -> float:
        if not self.criticisms:
            return 1.0
        return sum(1 for c in self.criticisms if c.resolved) / len(self.criticisms)

# =============================================================================
# MULTI-HYPOTHESIS COMPARISON
# =============================================================================

class HypothesisComparator:
    """Compare multiple analysis elements side by side"""
    
    def __init__(self):
        self.hypotheses: Dict[str, AnalysisElement] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_hypothesis(self, element: AnalysisElement):
        self.hypotheses[element.name] = element
    
    def compare(self) -> Dict:
        """Generate comparison data"""
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
            
            hyp_data = {
                'name': name,
                'combined_score': score,
                'additive_score': elem.scoring.additive_score(),
                'multiplicative_score': elem.scoring.multiplicative_score(),
                'fatal_flaws': [f.name for f in flaws],
                'reason': reason,
                'dimensions': {
                    d.name: d.value for d in elem.scoring.dimensions.values()
                },
                'mechanism_confidence': elem.mechanism_map.overall_confidence(),
                'evidence_count': len(elem.evidence)
            }
            comparison['hypotheses'].append(hyp_data)
        
        # Rank by combined score
        comparison['hypotheses'].sort(key=lambda x: x['combined_score'], reverse=True)
        comparison['rankings'] = {
            h['name']: i+1 for i, h in enumerate(comparison['hypotheses'])
        }
        
        return comparison

# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_analysis(element: AnalysisElement, rigor_level: int = 2, max_iter: int = 15) -> Dict:
    """Run full analysis with tracing, mechanism mapping, and scoring"""
    
    tracer = AnalysisTracer()
    element.set_tracer(tracer)
    
    # Layer 0: Characterization
    tracer.log(TraceEventType.LAYER_ENTER, "L0", {'description': 'Problem Characterization'})
    rigor_names = {1: "Light", 2: "Standard", 3: "Deep"}
    targets = {1: 0.5, 2: 0.7, 3: 0.85}
    target = targets[rigor_level]
    tracer.log(TraceEventType.DECISION, "L0", {
        'rigor': rigor_names[rigor_level], 'target': target
    })
    tracer.log(TraceEventType.LAYER_EXIT, "L0", {'result': f'Target: {target}'})
    
    # Layer 1: Foundation + Mechanism
    tracer.log(TraceEventType.LAYER_ENTER, "L1", {'description': 'Foundation & Mechanism Mapping'})
    mechanism_conf = element.mechanism_map.overall_confidence()
    tracer.log(TraceEventType.LAYER_EXIT, "L1", {
        'result': f'Mechanism confidence: {mechanism_conf:.2f}, Nodes: {len(element.mechanism_map.nodes)}'
    })
    
    # Layer 2: Adversarial Testing
    tracer.log(TraceEventType.LAYER_ENTER, "L2", {'description': 'Adversarial Testing'})
    tester = AdversarialTester(element, rigor_level, tracer)
    
    history = []
    reason = "Max iterations"
    
    for i in range(max_iter):
        tester.generate_criticisms()
        
        # Calculate scores
        combined, score_reason = element.scoring.combined_score()
        consistency = tester.consistency_score()
        
        # Blend into final quality
        quality = 0.7 * combined + 0.3 * consistency
        history.append(quality)
        
        tracer.log(TraceEventType.QUALITY_CALCULATED, "L4", {
            'iteration': i+1,
            'combined_score': combined,
            'consistency': consistency,
            'quality': quality,
            'fatal_flaws': len(element.scoring.fatal_flaws())
        })
        tracer.iteration_data.append({
            'iteration': i+1,
            'quality': quality,
            'combined': combined,
            'additive': element.scoring.additive_score(),
            'multiplicative': element.scoring.multiplicative_score(),
            'consistency': consistency
        })
        
        # Gate checks
        unresolved = tester.unresolved_critical()
        flaws = element.scoring.fatal_flaws()
        
        tracer.log(TraceEventType.GATE_CHECK, "L4", {
            'gate': 'quality', 'value': quality, 'threshold': target, 'passed': quality >= target
        })
        tracer.log(TraceEventType.GATE_CHECK, "L4", {
            'gate': 'fatal_flaws', 'value': len(flaws), 'threshold': 0, 'passed': len(flaws) == 0
        })
        
        # Stopping conditions
        if quality >= target and len(flaws) == 0 and len(unresolved) == 0:
            reason = f"Target {target} reached, no fatal flaws"
            tracer.log(TraceEventType.ITERATION_END, "L2", {
                'iteration': i+1, 'quality': quality, 'action': 'STOP: Target reached'
            })
            break
        elif i >= 3 and len(history) >= 2 and abs(history[-1] - history[-2]) < 0.01:
            reason = "Diminishing returns"
            tracer.log(TraceEventType.ITERATION_END, "L2", {
                'iteration': i+1, 'quality': quality, 'action': 'STOP: Diminishing returns'
            })
            break
        else:
            tracer.log(TraceEventType.ITERATION_END, "L2", {
                'iteration': i+1, 'quality': quality, 'action': 'CONTINUE'
            })
    
    tracer.log(TraceEventType.LAYER_EXIT, "L2", {
        'result': f'{tester.iteration} iterations, {reason}'
    })
    
    # Sensitivity analysis
    tracer.log(TraceEventType.LAYER_ENTER, "L3", {'description': 'Sensitivity Analysis'})
    sensitivity = element.scoring.sensitivity_analysis()
    for s in sensitivity[:5]:
        tracer.log(TraceEventType.SENSITIVITY_CALCULATED, "L3", s)
    tracer.sensitivity_data = sensitivity
    tracer.log(TraceEventType.LAYER_EXIT, "L3", {
        'result': f'Most sensitive: {sensitivity[0]["dimension"] if sensitivity else "N/A"}'
    })
    
    # Final decision
    tracer.log(TraceEventType.LAYER_ENTER, "L5", {'description': 'Decision'})
    final_score, final_reason = element.scoring.combined_score()
    flaws = element.scoring.fatal_flaws()
    ready = final_score >= target and len(flaws) == 0
    
    tracer.log(TraceEventType.DECISION, "L5", {
        'decision': 'PROCEED' if ready else 'DO NOT PROCEED',
        'score': final_score,
        'reason': final_reason,
        'fatal_flaws': [f.name for f in flaws]
    })
    tracer.log(TraceEventType.LAYER_EXIT, "L5", {'result': 'Analysis complete'})
    
    return {
        'name': element.name,
        'quality': history[-1] if history else 0,
        'combined_score': final_score,
        'additive_score': element.scoring.additive_score(),
        'multiplicative_score': element.scoring.multiplicative_score(),
        'ready': ready,
        'reason': reason,
        'fatal_flaws': [{'name': f.name, 'value': f.value, 'threshold': f.is_fatal_below} for f in flaws],
        'iterations': tester.iteration,
        'history': history,
        'criticisms': [(c.content, c.severity, c.cycle, c.resolved) for c in tester.criticisms],
        'sensitivity': sensitivity,
        'mechanism_map': element.mechanism_map.to_dict(),
        'dimensions': {name: {'value': d.value, 'weight': d.weight, 'fatal': d.is_fatal_flaw} 
                      for name, d in element.scoring.dimensions.items()},
        'trace': tracer.get_trace(),
        'iteration_data': tracer.iteration_data
    }

# =============================================================================
# VISUALIZATION GENERATOR
# =============================================================================

def generate_visualization(results: Dict, comparison: Optional[Dict] = None) -> str:
    """Generate comprehensive React visualization"""
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return super().default(obj)
    
    trace_json = json.dumps(results['trace'], indent=2, cls=NumpyEncoder)
    iteration_json = json.dumps(results['iteration_data'], indent=2, cls=NumpyEncoder)
    criticisms_json = json.dumps(results['criticisms'], indent=2, cls=NumpyEncoder)
    sensitivity_json = json.dumps(results['sensitivity'], indent=2, cls=NumpyEncoder)
    mechanism_json = json.dumps(results['mechanism_map'], indent=2, cls=NumpyEncoder)
    dimensions_json = json.dumps(results['dimensions'], indent=2, cls=NumpyEncoder)
    fatal_flaws_json = json.dumps(results['fatal_flaws'], indent=2, cls=NumpyEncoder)
    comparison_json = json.dumps(comparison, indent=2, cls=NumpyEncoder) if comparison else 'null'
    
    jsx = '''import React, { useState, useMemo } from 'react';

const AnalysisVisualization = () => {
  // === DATA ===
  const analysisName = "''' + results['name'] + '''";
  const finalQuality = ''' + str(float(results['quality'])) + ''';
  const combinedScore = ''' + str(float(results['combined_score'])) + ''';
  const additiveScore = ''' + str(float(results['additive_score'])) + ''';
  const multiplicativeScore = ''' + str(float(results['multiplicative_score'])) + ''';
  const isReady = ''' + str(results['ready']).lower() + ''';
  const stopReason = "''' + results['reason'] + '''";
  const totalIterations = ''' + str(results['iterations']) + ''';
  
  const trace = ''' + trace_json + ''';
  const iterationData = ''' + iteration_json + ''';
  const criticisms = ''' + criticisms_json + ''';
  const sensitivity = ''' + sensitivity_json + ''';
  const mechanismMap = ''' + mechanism_json + ''';
  const dimensions = ''' + dimensions_json + ''';
  const fatalFlaws = ''' + fatal_flaws_json + ''';
  const comparison = ''' + comparison_json + ''';
  
  const [viewMode, setViewMode] = useState('overview');
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);

  // === COLORS ===
  const layerColors = {
    'L0': '#6366f1', 'L1': '#10b981', 'L2': '#f59e0b',
    'L3': '#ec4899', 'L4': '#8b5cf6', 'L5': '#06b6d4'
  };
  
  const nodeTypeColors = {
    'cause': '#ef4444',
    'mechanism': '#3b82f6',
    'outcome': '#22c55e',
    'blocker': '#f97316',
    'assumption': '#eab308',
    'evidence': '#8b5cf6',
    'intervention': '#06b6d4'
  };

  const eventIcons = {
    'layer_enter': '→', 'layer_exit': '←', 'foundation_set': '📝',
    'evidence_added': '📊', 'confidence_update': '📈',
    'mechanism_node_added': '🔵', 'mechanism_edge_added': '🔗',
    'criticism_generated': '⚠️', 'criticism_resolved': '✓',
    'quality_calculated': '📏', 'fatal_flaw_detected': '💀',
    'sensitivity_calculated': '📉', 'iteration_start': '🔄',
    'iteration_end': '⏹', 'gate_check': '🚦', 'decision': '✅',
    'gap_identified': '🕳️', 'comparison_added': '⚖️'
  };

  // === MECHANISM MAP LAYOUT ===
  const nodePositions = useMemo(() => {
    const positions = {};
    const typeGroups = {};
    
    mechanismMap.nodes.forEach(node => {
      if (!typeGroups[node.type]) typeGroups[node.type] = [];
      typeGroups[node.type].push(node);
    });
    
    const typeOrder = ['cause', 'assumption', 'mechanism', 'intervention', 'blocker', 'outcome', 'evidence'];
    let x = 50;
    
    typeOrder.forEach(type => {
      const nodes = typeGroups[type] || [];
      nodes.forEach((node, i) => {
        positions[node.id] = {
          x: x,
          y: 80 + i * 70
        };
      });
      if (nodes.length > 0) x += 140;
    });
    
    return positions;
  }, [mechanismMap]);

  const maxQuality = Math.max(...iterationData.map(d => d.quality), 0.01);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* === HEADER === */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold mb-1">Analysis: {analysisName}</h1>
          <div className="flex flex-wrap gap-3 mt-3">
            <div className="px-4 py-2 bg-slate-800 rounded-lg border border-slate-700">
              <div className="text-xs text-slate-400">Combined Score</div>
              <div className={`text-2xl font-bold ${combinedScore >= 0.7 ? 'text-emerald-400' : combinedScore >= 0.5 ? 'text-amber-400' : 'text-red-400'}`}>
                {combinedScore.toFixed(2)}
              </div>
            </div>
            <div className="px-4 py-2 bg-slate-800 rounded-lg border border-slate-700">
              <div className="text-xs text-slate-400">Additive</div>
              <div className="text-xl font-mono text-blue-400">{additiveScore.toFixed(2)}</div>
            </div>
            <div className="px-4 py-2 bg-slate-800 rounded-lg border border-slate-700">
              <div className="text-xs text-slate-400">Multiplicative</div>
              <div className="text-xl font-mono text-purple-400">{multiplicativeScore.toFixed(2)}</div>
            </div>
            <div className="px-4 py-2 bg-slate-800 rounded-lg border border-slate-700">
              <div className="text-xs text-slate-400">Iterations</div>
              <div className="text-xl font-mono text-amber-400">{totalIterations}</div>
            </div>
            <div className={`px-4 py-2 rounded-lg border ${isReady ? 'bg-emerald-900/50 border-emerald-700' : 'bg-red-900/50 border-red-700'}`}>
              <div className="text-xs text-slate-400">Decision</div>
              <div className={`text-xl font-bold ${isReady ? 'text-emerald-400' : 'text-red-400'}`}>
                {isReady ? '✓ PROCEED' : '✗ DO NOT PROCEED'}
              </div>
            </div>
          </div>
          {fatalFlaws.length > 0 && (
            <div className="mt-3 p-3 bg-red-900/30 border border-red-700 rounded-lg">
              <span className="text-red-400 font-bold">💀 Fatal Flaws: </span>
              {fatalFlaws.map((f, i) => (
                <span key={i} className="text-red-300 ml-2">
                  {f.name} ({f.value.toFixed(2)} &lt; {f.threshold})
                </span>
              ))}
            </div>
          )}
        </div>

        {/* === VIEW TABS === */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {['overview', 'mechanism', 'sensitivity', 'dimensions', 'timeline', 'criticisms'].map(mode => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                viewMode === mode 
                  ? 'bg-indigo-600 text-white shadow-lg' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
          {comparison && (
            <button
              onClick={() => setViewMode('comparison')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                viewMode === 'comparison' 
                  ? 'bg-indigo-600 text-white shadow-lg' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              ⚖️ Compare
            </button>
          )}
        </div>

        {/* === OVERVIEW === */}
        {viewMode === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Quality Evolution */}
            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
              <h3 className="font-bold mb-4">Quality Evolution</h3>
              <div className="h-48 flex items-end gap-1">
                {iterationData.map((d, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center">
                    <span className="text-xs text-emerald-400 mb-1">{d.quality.toFixed(2)}</span>
                    <div 
                      className="w-full bg-gradient-to-t from-indigo-600 to-indigo-400 rounded-t"
                      style={{ height: `${(d.quality / maxQuality) * 140}px` }}
                    />
                    <span className="text-xs text-slate-500 mt-1">{i + 1}</span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Score Comparison */}
            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
              <h3 className="font-bold mb-4">Scoring Methods Comparison</h3>
              <div className="space-y-4">
                {[
                  {name: 'Additive (weighted sum)', value: additiveScore, color: 'bg-blue-500'},
                  {name: 'Multiplicative (geometric)', value: multiplicativeScore, color: 'bg-purple-500'},
                  {name: 'Combined (final)', value: combinedScore, color: 'bg-emerald-500'}
                ].map(({name, value, color}) => (
                  <div key={name}>
                    <div className="flex justify-between text-sm mb-1">
                      <span>{name}</span>
                      <span className="font-mono">{value.toFixed(3)}</span>
                    </div>
                    <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                      <div className={`h-full ${color} rounded-full`} style={{ width: `${value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-slate-700 text-xs text-slate-400">
                <p><strong>Additive:</strong> Weighted average of all dimensions</p>
                <p><strong>Multiplicative:</strong> Geometric mean (fatal flaws dominate)</p>
                <p><strong>Combined:</strong> 60% additive + 40% multiplicative, capped if fatal flaws</p>
              </div>
            </div>
            
            {/* Key Stats */}
            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 lg:col-span-2">
              <h3 className="font-bold mb-4">Analysis Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 bg-slate-700/50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-400">{mechanismMap.nodes.length}</div>
                  <div className="text-xs text-slate-400">Mechanism Nodes</div>
                </div>
                <div className="p-3 bg-slate-700/50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-400">{mechanismMap.edges.length}</div>
                  <div className="text-xs text-slate-400">Causal Links</div>
                </div>
                <div className="p-3 bg-slate-700/50 rounded-lg">
                  <div className="text-2xl font-bold text-amber-400">{criticisms.length}</div>
                  <div className="text-xs text-slate-400">Criticisms Found</div>
                </div>
                <div className="p-3 bg-slate-700/50 rounded-lg">
                  <div className="text-2xl font-bold text-red-400">{fatalFlaws.length}</div>
                  <div className="text-xs text-slate-400">Fatal Flaws</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* === MECHANISM MAP === */}
        {viewMode === 'mechanism' && (
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="font-bold mb-4">Causal Mechanism Map</h3>
            <div className="mb-4 flex flex-wrap gap-2">
              {Object.entries(nodeTypeColors).map(([type, color]) => (
                <span key={type} className="flex items-center gap-1 text-xs">
                  <span className="w-3 h-3 rounded" style={{ backgroundColor: color }}></span>
                  {type}
                </span>
              ))}
            </div>
            <div className="relative bg-slate-900 rounded-lg" style={{ height: '500px', overflow: 'auto' }}>
              <svg width="900" height="450">
                {/* Edges */}
                {mechanismMap.edges.map((edge, i) => {
                  const source = nodePositions[edge.source];
                  const target = nodePositions[edge.target];
                  if (!source || !target) return null;
                  const isCompensates = edge.type === 'compensates' || edge.type === 'prevents';
                  return (
                    <g key={i}>
                      <line
                        x1={source.x + 50}
                        y1={source.y + 20}
                        x2={target.x}
                        y2={target.y + 20}
                        stroke={isCompensates ? '#f97316' : '#4b5563'}
                        strokeWidth={edge.strength * 3}
                        strokeDasharray={isCompensates ? '5,5' : 'none'}
                        markerEnd="url(#arrow)"
                      />
                      <text
                        x={(source.x + target.x) / 2 + 25}
                        y={(source.y + target.y) / 2 + 15}
                        fill="#9ca3af"
                        fontSize="10"
                      >
                        {edge.type}
                      </text>
                    </g>
                  );
                })}
                
                {/* Arrow marker */}
                <defs>
                  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L9,3 z" fill="#4b5563" />
                  </marker>
                </defs>
                
                {/* Nodes */}
                {mechanismMap.nodes.map((node) => {
                  const pos = nodePositions[node.id];
                  if (!pos) return null;
                  return (
                    <g 
                      key={node.id} 
                      onClick={() => setSelectedNode(node)}
                      className="cursor-pointer"
                    >
                      <rect
                        x={pos.x}
                        y={pos.y}
                        width={100}
                        height={40}
                        rx={6}
                        fill={nodeTypeColors[node.type] || '#6b7280'}
                        opacity={node.confidence}
                        stroke={selectedNode?.id === node.id ? '#fff' : 'transparent'}
                        strokeWidth={2}
                      />
                      <text
                        x={pos.x + 50}
                        y={pos.y + 18}
                        fill="white"
                        fontSize="11"
                        textAnchor="middle"
                        fontWeight="bold"
                      >
                        {node.label.slice(0, 12)}
                      </text>
                      <text
                        x={pos.x + 50}
                        y={pos.y + 32}
                        fill="rgba(255,255,255,0.7)"
                        fontSize="9"
                        textAnchor="middle"
                      >
                        conf: {node.confidence.toFixed(2)}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            
            {/* Node Detail */}
            {selectedNode && (
              <div className="mt-4 p-4 bg-slate-700 rounded-lg">
                <div className="flex items-center gap-3 mb-2">
                  <div 
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: nodeTypeColors[selectedNode.type] }}
                  />
                  <h4 className="font-bold">{selectedNode.label}</h4>
                  <span className="text-xs px-2 py-1 bg-slate-600 rounded">{selectedNode.type}</span>
                </div>
                <p className="text-sm text-slate-300">{selectedNode.description || 'No description'}</p>
                <div className="mt-2 flex gap-4 text-sm">
                  <span>Confidence: <span className="text-emerald-400 font-mono">{selectedNode.confidence.toFixed(2)}</span></span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* === SENSITIVITY ANALYSIS === */}
        {viewMode === 'sensitivity' && (
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="font-bold mb-2">Sensitivity Analysis</h3>
            <p className="text-sm text-slate-400 mb-4">Which dimensions have the most impact on the final score?</p>
            <div className="space-y-3">
              {sensitivity.map((s, i) => (
                <div key={i} className={`p-3 rounded-lg ${s.is_fatal ? 'bg-red-900/30 border border-red-700' : 'bg-slate-700/50'}`}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">{s.dimension}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-2 py-1 bg-slate-600 rounded">
                        Impact: {(s.impact * 100).toFixed(1)}%
                      </span>
                      {s.is_fatal && (
                        <span className="text-xs px-2 py-1 bg-red-700 rounded text-red-200">💀 FATAL</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-3 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${s.is_fatal ? 'bg-red-500' : 'bg-emerald-500'}`}
                        style={{ width: `${s.current_value * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-sm w-12">{s.current_value.toFixed(2)}</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Improving this by 0.1 would change score by {s.direction === 'positive' ? '+' : ''}{(s.impact).toFixed(3)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* === DIMENSIONS === */}
        {viewMode === 'dimensions' && (
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="font-bold mb-4">All Scoring Dimensions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(dimensions).map(([name, dim]) => (
                <div 
                  key={name}
                  className={`p-4 rounded-lg border ${dim.fatal ? 'bg-red-900/20 border-red-700' : 'bg-slate-700/50 border-slate-600'}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-medium">{name.replace(/_/g, ' ')}</span>
                    {dim.fatal && <span className="text-xs px-2 py-1 bg-red-700 rounded">💀 FATAL</span>}
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-4 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${dim.fatal ? 'bg-red-500' : dim.value >= 0.7 ? 'bg-emerald-500' : dim.value >= 0.5 ? 'bg-amber-500' : 'bg-red-500'}`}
                        style={{ width: `${dim.value * 100}%` }}
                      />
                    </div>
                    <span className="font-mono font-bold">{dim.value.toFixed(2)}</span>
                  </div>
                  <div className="text-xs text-slate-400 mt-2">
                    Weight: {(dim.weight * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* === TIMELINE === */}
        {viewMode === 'timeline' && (
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
            <h3 className="font-bold mb-4">Event Timeline ({trace.length} events)</h3>
            <div className="space-y-1 max-h-[600px] overflow-y-auto">
              {trace.map((event, i) => (
                <div 
                  key={i}
                  className="flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded-lg cursor-pointer"
                  onClick={() => setSelectedEvent(event)}
                >
                  <span className="w-16 text-xs text-slate-500 font-mono shrink-0">
                    {event.timestamp.toFixed(4)}s
                  </span>
                  <div 
                    className="w-8 h-6 rounded flex items-center justify-center text-xs font-bold shrink-0"
                    style={{ backgroundColor: layerColors[event.layer] }}
                  >
                    {event.layer}
                  </div>
                  <span className="text-lg shrink-0">{eventIcons[event.type] || '•'}</span>
                  <span className="text-sm shrink-0">{event.type.replace(/_/g, ' ')}</span>
                  <span className="text-xs text-slate-500 truncate">
                    {JSON.stringify(event.data).slice(0, 60)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* === CRITICISMS === */}
        {viewMode === 'criticisms' && (
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="font-bold mb-4">Criticisms ({criticisms.length})</h3>
            {criticisms.length === 0 ? (
              <p className="text-slate-500">No criticisms generated.</p>
            ) : (
              <div className="space-y-3">
                {criticisms.map(([content, severity, cycle, resolved], i) => (
                  <div 
                    key={i}
                    className={`p-4 rounded-lg border ${resolved ? 'bg-emerald-900/20 border-emerald-800' : severity >= 0.9 ? 'bg-red-900/20 border-red-800' : 'bg-amber-900/20 border-amber-800'}`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-xs px-2 py-1 bg-slate-700 rounded">{cycle}</span>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold ${resolved ? 'text-emerald-400' : 'text-amber-400'}`}>
                          {resolved ? '✓ Resolved' : '⚠ Open'}
                        </span>
                        {severity >= 0.9 && <span className="text-xs px-2 py-1 bg-red-700 rounded">CRITICAL</span>}
                      </div>
                    </div>
                    <p className="text-sm mb-2">{content}</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-red-500 rounded-full"
                          style={{ width: `${severity * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-500">Severity: {severity.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* === COMPARISON === */}
        {viewMode === 'comparison' && comparison && (
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="font-bold mb-4">Hypothesis Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left p-3">Rank</th>
                    <th className="text-left p-3">Hypothesis</th>
                    <th className="text-right p-3">Combined</th>
                    <th className="text-right p-3">Additive</th>
                    <th className="text-right p-3">Multiplicative</th>
                    <th className="text-left p-3">Fatal Flaws</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.hypotheses.map((h, i) => (
                    <tr key={i} className={`border-b border-slate-700/50 ${i === 0 ? 'bg-emerald-900/20' : ''}`}>
                      <td className="p-3 font-bold">{comparison.rankings[h.name]}</td>
                      <td className="p-3 font-medium">{h.name}</td>
                      <td className="p-3 text-right font-mono">{h.combined_score.toFixed(3)}</td>
                      <td className="p-3 text-right font-mono text-blue-400">{h.additive_score.toFixed(3)}</td>
                      <td className="p-3 text-right font-mono text-purple-400">{h.multiplicative_score.toFixed(3)}</td>
                      <td className="p-3">
                        {h.fatal_flaws.length > 0 ? (
                          <span className="text-red-400">💀 {h.fatal_flaws.join(', ')}</span>
                        ) : (
                          <span className="text-emerald-400">None</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* === EVENT DETAIL MODAL === */}
        {selectedEvent && (
          <div 
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedEvent(null)}
          >
            <div 
              className="bg-slate-800 rounded-xl p-6 max-w-xl w-full border border-slate-700"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center text-xl"
                  style={{ backgroundColor: layerColors[selectedEvent.layer] }}
                >
                  {eventIcons[selectedEvent.type] || '•'}
                </div>
                <div>
                  <h3 className="font-bold">{selectedEvent.type.replace(/_/g, ' ')}</h3>
                  <span className="text-sm text-slate-400">
                    Layer {selectedEvent.layer} @ {selectedEvent.timestamp.toFixed(4)}s
                  </span>
                </div>
              </div>
              <pre className="bg-slate-900 p-4 rounded-lg text-xs overflow-auto max-h-64">
                {JSON.stringify(selectedEvent.data, null, 2)}
              </pre>
              <button
                className="mt-4 px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 w-full"
                onClick={() => setSelectedEvent(null)}
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisVisualization;
'''
    return jsx

# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("="*70)
    print("ENHANCED ANALYTICAL PROTOCOL - DEMO")
    print("="*70)
    print()
    
    # === HYPOTHESIS 1: Hire Data Scientist ===
    h1 = AnalysisElement(name="Hire Data Scientist", domain=EvidenceDomain.BUSINESS)
    
    # Foundation
    h1.set_what("Full-time data scientist, $120K, to improve decision-making", 0.9)
    h1.set_why("We make data-driven decisions but lack statistical expertise", 0.7)
    h1.set_how("Post job → Interview → Hire → Onboard → Deliver insights", 0.8)
    h1.set_measure("A/B test success rate improves by 20%", 0.7)
    
    # Mechanism map
    h1.add_mechanism_node(MechanismNode("cause1", "Lack of statistics expertise", NodeType.CAUSE, 0.9))
    h1.add_mechanism_node(MechanismNode("cause2", "Poor experiment design", NodeType.CAUSE, 0.8))
    h1.add_mechanism_node(MechanismNode("mech1", "DS brings expertise", NodeType.MECHANISM, 0.85))
    h1.add_mechanism_node(MechanismNode("mech2", "Better analysis methods", NodeType.MECHANISM, 0.75))
    h1.add_mechanism_node(MechanismNode("outcome1", "Improved decisions", NodeType.OUTCOME, 0.7))
    h1.add_mechanism_node(MechanismNode("blocker1", "Org resistance to data", NodeType.BLOCKER, 0.4))
    h1.add_mechanism_node(MechanismNode("assume1", "DS can integrate", NodeType.ASSUMPTION, 0.6))
    
    h1.add_mechanism_edge(MechanismEdge("cause1", "mech1", EdgeType.CAUSES, 0.9))
    h1.add_mechanism_edge(MechanismEdge("cause2", "mech2", EdgeType.CAUSES, 0.8))
    h1.add_mechanism_edge(MechanismEdge("mech1", "outcome1", EdgeType.ENABLES, 0.7))
    h1.add_mechanism_edge(MechanismEdge("mech2", "outcome1", EdgeType.ENABLES, 0.6))
    h1.add_mechanism_edge(MechanismEdge("blocker1", "outcome1", EdgeType.PREVENTS, 0.4))
    h1.add_mechanism_edge(MechanismEdge("assume1", "mech1", EdgeType.REQUIRES, 0.8))
    
    # Evidence
    h1.add_evidence(Evidence("ev1", "Companies with DS report 15% better outcomes", "HBR 2023", 0.7, "2023", 
                             EvidenceDomain.BUSINESS, "multi_company_analysis"))
    h1.add_evidence(Evidence("ev2", "Competitor hired DS, improved metrics", "Industry contact", 0.3, "2024"))
    
    # Feasibility
    h1.set_feasibility(technical=0.9, economic=0.7, timeline=0.8)
    h1.set_risk(execution_risk=0.3, external_risk=0.2)
    
    # === HYPOTHESIS 2: Buy Analytics Tool (alternative) ===
    h2 = AnalysisElement(name="Buy Analytics Tool", domain=EvidenceDomain.BUSINESS)
    
    h2.set_what("Purchase enterprise analytics platform, $50K/year", 0.95)
    h2.set_why("Automate analysis without hiring", 0.5)
    h2.set_how("Evaluate tools → Purchase → Deploy → Train team", 0.85)
    h2.set_measure("Dashboard adoption > 80% of decisions", 0.6)
    
    # Mechanism - weaker
    h2.add_mechanism_node(MechanismNode("cause1", "Manual analysis is slow", NodeType.CAUSE, 0.9))
    h2.add_mechanism_node(MechanismNode("mech1", "Automated dashboards", NodeType.MECHANISM, 0.8))
    h2.add_mechanism_node(MechanismNode("outcome1", "Faster decisions", NodeType.OUTCOME, 0.5))
    h2.add_mechanism_node(MechanismNode("blocker1", "Tool doesn't fit needs", NodeType.BLOCKER, 0.6))
    h2.add_mechanism_node(MechanismNode("blocker2", "Team won't adopt", NodeType.BLOCKER, 0.5))
    
    h2.add_mechanism_edge(MechanismEdge("cause1", "mech1", EdgeType.CAUSES, 0.7))
    h2.add_mechanism_edge(MechanismEdge("mech1", "outcome1", EdgeType.ENABLES, 0.5))
    h2.add_mechanism_edge(MechanismEdge("blocker1", "outcome1", EdgeType.PREVENTS, 0.6))
    h2.add_mechanism_edge(MechanismEdge("blocker2", "outcome1", EdgeType.PREVENTS, 0.5))
    
    h2.add_evidence(Evidence("ev1", "Tool vendor case studies", "Vendor website", 0.2, "2024"))
    
    h2.set_feasibility(technical=0.8, economic=0.85, timeline=0.9)
    h2.set_risk(execution_risk=0.4, external_risk=0.3)
    
    # Run analyses
    print("Analyzing Hypothesis 1: Hire Data Scientist...")
    results1 = run_analysis(h1, rigor_level=2, max_iter=10)
    print(f"  Combined Score: {results1['combined_score']:.3f}")
    print(f"  Ready: {results1['ready']}")
    print(f"  Fatal Flaws: {len(results1['fatal_flaws'])}")
    print()
    
    print("Analyzing Hypothesis 2: Buy Analytics Tool...")
    results2 = run_analysis(h2, rigor_level=2, max_iter=10)
    print(f"  Combined Score: {results2['combined_score']:.3f}")
    print(f"  Ready: {results2['ready']}")
    print(f"  Fatal Flaws: {len(results2['fatal_flaws'])}")
    print()
    
    # Compare
    comparator = HypothesisComparator()
    comparator.add_hypothesis(h1)
    comparator.add_hypothesis(h2)
    comparison = comparator.compare()
    
    print("="*70)
    print("COMPARISON")
    print("="*70)
    for h in comparison['hypotheses']:
        print(f"  #{comparison['rankings'][h['name']]} {h['name']}: {h['combined_score']:.3f}")
        if h['fatal_flaws']:
            print(f"      Fatal flaws: {', '.join(h['fatal_flaws'])}")
    print()
    
    # Generate visualization
    print("Generating visualization...")
    viz = generate_visualization(results1, comparison)
    
    viz_path = "/mnt/user-data/outputs/enhanced_analysis.jsx"
    with open(viz_path, 'w') as f:
        f.write(viz)
    print(f"Saved to: {viz_path}")
    print()
    print("="*70)

if __name__ == "__main__":
    demo()
