"""
SCIENTIFIC ANALYSIS: ALS TREATMENT PATHWAYS
============================================
Amyotrophic Lateral Sclerosis (ALS) is a fatal neurodegenerative disease
affecting motor neurons. Current treatments (riluzole, edaravone) only
modestly slow progression. No cure exists.

This analysis compares competing hypotheses for which biological pathway
should be prioritized for therapeutic development.

COMPETING HYPOTHESES:
1. Protein Aggregation (SOD1/TDP-43) - Target misfolded proteins
2. Glutamate Excitotoxicity - Reduce glutamate-induced neuron death
3. Neuroinflammation - Target microglial/astrocyte inflammation
4. Gene Therapy (ASO) - Silence disease-causing genes
5. Neurotrophic Support - Boost neuron survival factors

Author: Scientific Analysis System
Disease: ALS (Amyotrophic Lateral Sclerosis)
"""

import sys
sys.path.insert(0, '/mnt/user-data/outputs')

from enhanced_protocol_v2 import (
    AnalysisElement, MechanismNode, MechanismEdge, Evidence,
    EvidenceDomain, NodeType, EdgeType, CausalLevel,
    run_analysis, HypothesisComparator
)


def create_protein_aggregation_hypothesis():
    """
    HYPOTHESIS 1: Target Protein Aggregation (SOD1/TDP-43)
    
    Theory: Misfolded SOD1 and TDP-43 proteins aggregate in motor neurons,
    causing cellular dysfunction and death. Clearing or preventing these
    aggregates could slow or halt disease.
    
    Current status: Most validated in familial ALS (SOD1 mutations).
    Tofersen (ASO for SOD1) recently FDA approved for SOD1-ALS.
    """
    
    h = AnalysisElement(
        name="Protein Aggregation Targeting",
        domain=EvidenceDomain.MEDICAL
    )
    
    # Foundation
    h.set_what(
        "Develop therapies targeting SOD1/TDP-43 protein aggregation in motor neurons",
        confidence=0.85  # Well-defined target
    )
    h.set_why(
        "Protein aggregates are found in >97% of ALS patients; SOD1 mutations cause familial ALS",
        confidence=0.80  # Strong but not universal
    )
    h.set_how(
        "Antisense oligonucleotides (ASOs) to reduce SOD1 production, or small molecules to prevent aggregation",
        confidence=0.75  # Mechanism understood, delivery challenging
    )
    h.set_measure(
        "Reduction in CSF SOD1 levels, slowed functional decline (ALSFRS-R), extended survival",
        confidence=0.80  # Clear biomarkers exist
    )
    
    # Feasibility
    h.set_feasibility(
        technical=0.75,   # ASO delivery to CNS is proven but challenging
        economic=0.60,    # Very expensive ($200K+/year for Tofersen)
        timeline=0.70     # Already have approved therapy, can build on it
    )
    h.set_risk(
        execution_risk=0.35,   # CNS delivery, patient selection
        external_risk=0.25     # Regulatory path established
    )
    
    # Mechanism Map
    # Causes
    n1 = MechanismNode("sod1_mutation", "SOD1 gene mutation", NodeType.CAUSE,
                       description="Mutations in SOD1 cause protein misfolding")
    n1.confidence = 0.95  # Proven genetic cause
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("tdp43_aggregation", "TDP-43 cytoplasmic aggregation", NodeType.CAUSE,
                       description="TDP-43 mislocalizes and aggregates in >97% of ALS")
    n2.confidence = 0.90
    h.add_mechanism_node(n2)
    
    # Mechanism
    n3 = MechanismNode("protein_toxicity", "Aggregate-induced toxicity", NodeType.MECHANISM,
                       description="Aggregates cause proteostasis collapse, ER stress")
    n3.confidence = 0.75  # Correlation clear, causation debated
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("aso_intervention", "ASO reduces toxic protein", NodeType.INTERVENTION,
                       description="Antisense oligonucleotides reduce SOD1 mRNA/protein")
    n4.confidence = 0.85  # Proven to reduce SOD1
    h.add_mechanism_node(n4)
    
    # Outcome
    n5 = MechanismNode("neuron_survival", "Motor neuron survival", NodeType.OUTCOME,
                       description="Preserved motor neuron function and patient survival")
    n5.confidence = 0.60  # Modest clinical benefit shown
    h.add_mechanism_node(n5)
    
    # Blockers
    n6 = MechanismNode("sporadic_als", "Sporadic ALS complexity", NodeType.BLOCKER,
                       description="95% of ALS is sporadic with unknown cause")
    n6.confidence = 0.70  # Major limitation
    h.add_mechanism_node(n6)
    
    # Assumptions
    n7 = MechanismNode("aggregates_causal", "Aggregates are causal not just markers", NodeType.ASSUMPTION,
                       description="Assumes aggregates cause disease, not just correlate")
    n7.confidence = 0.65  # Debated in field
    h.add_mechanism_node(n7)
    
    # Edges with causal levels
    h.add_mechanism_edge(MechanismEdge(
        "sod1_mutation", "protein_toxicity", EdgeType.CAUSES, 0.90,
        causal_level=CausalLevel.COUNTERFACTUAL,  # Genetic causation proven
        confounding_risk=0.1
    ))
    h.add_mechanism_edge(MechanismEdge(
        "tdp43_aggregation", "protein_toxicity", EdgeType.CAUSES, 0.70,
        causal_level=CausalLevel.ASSOCIATION,  # Correlation, causation unclear
        confounding_risk=0.4
    ))
    h.add_mechanism_edge(MechanismEdge(
        "protein_toxicity", "neuron_survival", EdgeType.PREVENTS, 0.75,
        causal_level=CausalLevel.INTERVENTION,  # Animal model interventions
        confounding_risk=0.3
    ))
    h.add_mechanism_edge(MechanismEdge(
        "aso_intervention", "protein_toxicity", EdgeType.PREVENTS, 0.80,
        causal_level=CausalLevel.INTERVENTION,  # Clinical trial data
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "sporadic_als", "neuron_survival", EdgeType.PREVENTS, 0.60,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.3
    ))
    
    # Evidence
    h.add_evidence(Evidence(
        "ev1",
        "Tofersen Phase 3 trial: 60% reduction in plasma neurofilament light chain in SOD1-ALS",
        "NEJM 2022 (VALOR trial)",
        quality=0.85,
        date="2022",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev2",
        "TDP-43 pathology present in >97% of ALS cases at autopsy",
        "Journal of Neuropathology 2006",
        quality=0.80,
        date="2006",
        domain=EvidenceDomain.MEDICAL,
        study_design="cohort",
        causal_level=CausalLevel.ASSOCIATION,  # Observational
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev3",
        "SOD1 knockout mice don't develop ALS; gain-of-function toxicity",
        "Nature Neuroscience 2004",
        quality=0.75,
        date="2004",
        domain=EvidenceDomain.MEDICAL,
        study_design="controlled_experiment",
        causal_level=CausalLevel.INTERVENTION,  # Animal model
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev4",
        "Tofersen did not meet primary endpoint of ALSFRS-R change at 28 weeks",
        "NEJM 2022 (VALOR trial)",
        quality=0.85,
        date="2022",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=False  # Contradicting!
    ))
    
    # Utility scenarios
    h.add_scenario("Major breakthrough: Halts progression in SOD1-ALS, platform for other forms", 0.15, 1.0)
    h.add_scenario("Moderate success: Slows progression significantly in genetic ALS (~5%)", 0.35, 0.5)
    h.add_scenario("Limited success: Modest benefit in small subset, very expensive", 0.35, 0.2)
    h.add_scenario("Failure: Aggregates are downstream markers, not causal", 0.15, -0.3)
    
    return h


def create_glutamate_excitotoxicity_hypothesis():
    """
    HYPOTHESIS 2: Target Glutamate Excitotoxicity
    
    Theory: Excessive glutamate signaling causes calcium influx and 
    excitotoxic death of motor neurons. Reducing glutamate could protect neurons.
    
    Current status: Riluzole (glutamate release inhibitor) is FDA-approved
    but only extends survival by 2-3 months. Effect is modest.
    """
    
    h = AnalysisElement(
        name="Glutamate Excitotoxicity Targeting",
        domain=EvidenceDomain.MEDICAL
    )
    
    h.set_what(
        "Develop more effective glutamate modulators or AMPA/NMDA antagonists",
        confidence=0.80
    )
    h.set_why(
        "Motor neurons are vulnerable to excitotoxicity; riluzole provides modest benefit",
        confidence=0.65  # Mechanism is one of several
    )
    h.set_how(
        "Novel glutamate receptor antagonists, enhanced glutamate reuptake, calcium channel blockers",
        confidence=0.60  # Many trials have failed
    )
    h.set_measure(
        "Extended survival beyond riluzole alone, slowed ALSFRS-R decline",
        confidence=0.75
    )
    
    h.set_feasibility(
        technical=0.65,   # Many drugs exist but efficacy limited
        economic=0.80,    # Small molecules are cheaper
        timeline=0.60     # Long history of modest results
    )
    h.set_risk(
        execution_risk=0.50,   # High - many failures
        external_risk=0.30
    )
    
    # Mechanism map
    n1 = MechanismNode("glutamate_excess", "Excessive synaptic glutamate", NodeType.CAUSE)
    n1.confidence = 0.70
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("calcium_influx", "Pathological calcium entry", NodeType.MECHANISM)
    n2.confidence = 0.75
    h.add_mechanism_node(n2)
    
    n3 = MechanismNode("excitotoxicity", "Excitotoxic cell death", NodeType.MECHANISM)
    n3.confidence = 0.70
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("riluzole", "Riluzole blocks glutamate release", NodeType.INTERVENTION)
    n4.confidence = 0.85  # Well-proven mechanism
    h.add_mechanism_node(n4)
    
    n5 = MechanismNode("neuron_survival", "Motor neuron survival", NodeType.OUTCOME)
    n5.confidence = 0.45  # Only modest clinical benefit
    h.add_mechanism_node(n5)
    
    n6 = MechanismNode("downstream_effect", "Excitotoxicity may be secondary", NodeType.BLOCKER)
    n6.confidence = 0.55  # May not be primary driver
    h.add_mechanism_node(n6)
    
    # Edges
    h.add_mechanism_edge(MechanismEdge(
        "glutamate_excess", "calcium_influx", EdgeType.CAUSES, 0.80,
        causal_level=CausalLevel.INTERVENTION,
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "calcium_influx", "excitotoxicity", EdgeType.CAUSES, 0.75,
        causal_level=CausalLevel.INTERVENTION,
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "riluzole", "glutamate_excess", EdgeType.PREVENTS, 0.70,
        causal_level=CausalLevel.INTERVENTION,
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "excitotoxicity", "neuron_survival", EdgeType.PREVENTS, 0.60,
        causal_level=CausalLevel.ASSOCIATION,  # Correlation in ALS
        confounding_risk=0.4
    ))
    
    # Evidence
    h.add_evidence(Evidence(
        "ev1",
        "Riluzole extends survival by 2-3 months in meta-analysis",
        "Cochrane Review 2012",
        quality=0.90,
        date="2012",
        domain=EvidenceDomain.MEDICAL,
        study_design="meta_analysis",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev2",
        "Multiple AMPA/NMDA antagonist trials failed to show benefit",
        "Various trials 2000-2020",
        quality=0.80,
        date="2020",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=False  # Contradicting
    ))
    h.add_evidence(Evidence(
        "ev3",
        "CSF glutamate levels elevated in ALS patients vs controls",
        "Neurology 1990",
        quality=0.60,
        date="1990",
        domain=EvidenceDomain.MEDICAL,
        study_design="case_control",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    # Utility scenarios
    h.add_scenario("Breakthrough: New compound dramatically extends survival", 0.10, 1.0)
    h.add_scenario("Incremental: Modest improvement over riluzole", 0.25, 0.3)
    h.add_scenario("No improvement: Pathway is secondary, not primary", 0.50, 0.0)
    h.add_scenario("Wasted resources: Many more failed trials", 0.15, -0.4)
    
    return h


def create_neuroinflammation_hypothesis():
    """
    HYPOTHESIS 3: Target Neuroinflammation
    
    Theory: Activated microglia and astrocytes create toxic inflammatory
    environment that accelerates motor neuron death. Anti-inflammatory
    approaches could slow progression.
    
    Current status: Promising preclinical data, several trials ongoing.
    """
    
    h = AnalysisElement(
        name="Neuroinflammation Targeting",
        domain=EvidenceDomain.MEDICAL
    )
    
    h.set_what(
        "Develop therapies targeting microglial activation and astrocyte toxicity",
        confidence=0.75
    )
    h.set_why(
        "Reactive glia found in all ALS patients; non-cell autonomous death mechanism",
        confidence=0.75
    )
    h.set_how(
        "Modulate microglial phenotype, target complement pathway, reduce astrocyte-secreted toxins",
        confidence=0.55  # Multiple approaches, unclear which is best
    )
    h.set_measure(
        "Reduced inflammatory markers, slowed progression, extended survival",
        confidence=0.65  # Biomarkers less established
    )
    
    h.set_feasibility(
        technical=0.55,   # Targeting CNS inflammation is difficult
        economic=0.70,    # Some repurposed drugs possible
        timeline=0.50     # Earlier stage research
    )
    h.set_risk(
        execution_risk=0.55,
        external_risk=0.35
    )
    
    # Mechanism map
    n1 = MechanismNode("motor_neuron_stress", "Initial motor neuron stress", NodeType.CAUSE)
    n1.confidence = 0.70
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("microglial_activation", "Microglial activation (M1 phenotype)", NodeType.MECHANISM)
    n2.confidence = 0.80
    h.add_mechanism_node(n2)
    
    n3 = MechanismNode("astrocyte_toxicity", "Astrocyte-secreted toxic factors", NodeType.MECHANISM)
    n3.confidence = 0.75
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("inflammatory_cascade", "Pro-inflammatory cytokine cascade", NodeType.MECHANISM)
    n4.confidence = 0.80
    h.add_mechanism_node(n4)
    
    n5 = MechanismNode("neuron_death", "Accelerated motor neuron death", NodeType.OUTCOME)
    n5.confidence = 0.65
    h.add_mechanism_node(n5)
    
    n6 = MechanismNode("secondary_process", "Inflammation may be reactive not causal", NodeType.BLOCKER)
    n6.confidence = 0.50
    h.add_mechanism_node(n6)
    
    n7 = MechanismNode("immune_suppression_risk", "Over-suppression impairs repair", NodeType.BLOCKER)
    n7.confidence = 0.45
    h.add_mechanism_node(n7)
    
    # Edges
    h.add_mechanism_edge(MechanismEdge(
        "motor_neuron_stress", "microglial_activation", EdgeType.CAUSES, 0.75,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.4
    ))
    h.add_mechanism_edge(MechanismEdge(
        "microglial_activation", "inflammatory_cascade", EdgeType.CAUSES, 0.80,
        causal_level=CausalLevel.INTERVENTION,  # In vitro studies
        confounding_risk=0.3
    ))
    h.add_mechanism_edge(MechanismEdge(
        "astrocyte_toxicity", "neuron_death", EdgeType.CAUSES, 0.70,
        causal_level=CausalLevel.INTERVENTION,  # Co-culture experiments
        confounding_risk=0.35
    ))
    h.add_mechanism_edge(MechanismEdge(
        "inflammatory_cascade", "neuron_death", EdgeType.ENABLES, 0.65,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.4
    ))
    
    # Evidence
    h.add_evidence(Evidence(
        "ev1",
        "ALS astrocytes are toxic to motor neurons in co-culture; toxicity is non-cell autonomous",
        "Nature Neuroscience 2007",
        quality=0.80,
        date="2007",
        domain=EvidenceDomain.MEDICAL,
        study_design="controlled_experiment",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev2",
        "Masitinib (tyrosine kinase inhibitor) Phase 3: 27% slowing of functional decline",
        "Lancet Neurology 2021",
        quality=0.85,
        date="2021",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev3",
        "Multiple anti-inflammatory trials (minocycline, celecoxib) showed no benefit",
        "Various trials 2005-2015",
        quality=0.80,
        date="2015",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=False
    ))
    h.add_evidence(Evidence(
        "ev4",
        "PET imaging shows microglial activation correlates with disease progression",
        "Brain 2015",
        quality=0.70,
        date="2015",
        domain=EvidenceDomain.MEDICAL,
        study_design="cohort",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=True
    ))
    
    # Utility scenarios
    h.add_scenario("Breakthrough: Slows progression by >50%", 0.15, 1.0)
    h.add_scenario("Significant benefit: 25-50% slowing (like Masitinib)", 0.25, 0.6)
    h.add_scenario("Modest benefit: <25% slowing", 0.30, 0.2)
    h.add_scenario("No benefit: Inflammation is secondary", 0.30, -0.2)
    
    return h


def create_gene_therapy_hypothesis():
    """
    HYPOTHESIS 4: Gene Therapy (Beyond ASOs)
    
    Theory: Use viral vectors (AAV) or other gene delivery methods to 
    provide neuroprotective genes, silence toxic genes, or correct
    mutations directly.
    
    Current status: Early clinical trials, very promising preclinical data.
    """
    
    h = AnalysisElement(
        name="Gene Therapy (AAV-based)",
        domain=EvidenceDomain.MEDICAL
    )
    
    h.set_what(
        "Develop AAV-mediated gene therapies for neuroprotection or gene silencing",
        confidence=0.80
    )
    h.set_why(
        "Gene therapy can provide durable, one-time treatment; success in SMA shows path",
        confidence=0.70
    )
    h.set_how(
        "Intrathecal AAV delivery of protective genes or silencing constructs",
        confidence=0.60  # CNS delivery challenging
    )
    h.set_measure(
        "Gene expression, biomarkers (NfL), functional outcomes, survival",
        confidence=0.75
    )
    
    h.set_feasibility(
        technical=0.50,   # Major delivery challenges
        economic=0.40,    # Extremely expensive ($1M+ per treatment)
        timeline=0.45     # 5-10 years to approval
    )
    h.set_risk(
        execution_risk=0.60,   # High technical risk
        external_risk=0.40     # Regulatory uncertainty
    )
    
    # Mechanism map
    n1 = MechanismNode("genetic_defect", "Genetic mutation (SOD1, C9orf72, etc)", NodeType.CAUSE)
    n1.confidence = 0.90  # Well-established for familial
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("aav_delivery", "AAV vector delivers therapeutic gene", NodeType.INTERVENTION)
    n2.confidence = 0.70
    h.add_mechanism_node(n2)
    
    n3 = MechanismNode("gene_silencing", "Toxic gene silenced or corrected", NodeType.MECHANISM)
    n3.confidence = 0.65
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("neuroprotection", "Sustained neuroprotection", NodeType.MECHANISM)
    n4.confidence = 0.55
    h.add_mechanism_node(n4)
    
    n5 = MechanismNode("motor_neuron_preservation", "Motor neuron preservation", NodeType.OUTCOME)
    n5.confidence = 0.50  # Unproven in ALS yet
    h.add_mechanism_node(n5)
    
    n6 = MechanismNode("delivery_barrier", "Blood-brain barrier limits delivery", NodeType.BLOCKER)
    n6.confidence = 0.75
    h.add_mechanism_node(n6)
    
    n7 = MechanismNode("immune_response", "Immune response to AAV vector", NodeType.BLOCKER)
    n7.confidence = 0.60
    h.add_mechanism_node(n7)
    
    n8 = MechanismNode("sporadic_limitation", "Gene therapy may not apply to sporadic ALS", NodeType.ASSUMPTION)
    n8.confidence = 0.70
    h.add_mechanism_node(n8)
    
    # Edges
    h.add_mechanism_edge(MechanismEdge(
        "genetic_defect", "gene_silencing", EdgeType.REQUIRES, 0.80,
        causal_level=CausalLevel.COUNTERFACTUAL,
        confounding_risk=0.2
    ))
    h.add_mechanism_edge(MechanismEdge(
        "aav_delivery", "gene_silencing", EdgeType.ENABLES, 0.70,
        causal_level=CausalLevel.INTERVENTION,
        confounding_risk=0.3
    ))
    h.add_mechanism_edge(MechanismEdge(
        "gene_silencing", "neuroprotection", EdgeType.CAUSES, 0.65,
        causal_level=CausalLevel.INTERVENTION,  # Animal models
        confounding_risk=0.35
    ))
    h.add_mechanism_edge(MechanismEdge(
        "delivery_barrier", "aav_delivery", EdgeType.PREVENTS, 0.60,
        causal_level=CausalLevel.COUNTERFACTUAL,
        confounding_risk=0.2
    ))
    
    # Evidence
    h.add_evidence(Evidence(
        "ev1",
        "Zolgensma (AAV-SMN1) cures SMA in infants; proof of concept for motor neuron disease",
        "NEJM 2017",
        quality=0.90,
        date="2017",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev2",
        "AAV-mediated SOD1 silencing extends survival 50% in mouse model",
        "Molecular Therapy 2018",
        quality=0.65,
        date="2018",
        domain=EvidenceDomain.MEDICAL,
        study_design="controlled_experiment",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev3",
        "No human trial data for AAV gene therapy in ALS yet",
        "ClinicalTrials.gov 2024",
        quality=0.70,
        date="2024",
        domain=EvidenceDomain.MEDICAL,
        study_design="expert_opinion",
        causal_level=CausalLevel.ASSOCIATION,
        supports_hypothesis=False  # Lack of human evidence
    ))
    
    # Utility scenarios
    h.add_scenario("Transformative: One-time cure for genetic ALS", 0.10, 1.0)
    h.add_scenario("Significant: Major benefit for genetic subtypes", 0.20, 0.7)
    h.add_scenario("Limited: Works but only in small genetic subset", 0.30, 0.3)
    h.add_scenario("Technical failure: Delivery/safety issues halt development", 0.40, -0.5)
    
    return h


def create_neurotrophic_hypothesis():
    """
    HYPOTHESIS 5: Neurotrophic Factor Support
    
    Theory: Motor neurons die due to lack of trophic support. Providing
    growth factors (BDNF, GDNF, VEGF) could sustain dying neurons.
    
    Current status: Multiple failed trials; delivery remains challenging.
    """
    
    h = AnalysisElement(
        name="Neurotrophic Factor Therapy",
        domain=EvidenceDomain.MEDICAL
    )
    
    h.set_what(
        "Deliver neurotrophic factors (GDNF, BDNF, VEGF) to support motor neurons",
        confidence=0.75
    )
    h.set_why(
        "Motor neurons depend on trophic support; factors are reduced in ALS",
        confidence=0.60  # Evidence mixed
    )
    h.set_how(
        "Gene therapy delivery, protein infusion, small molecule mimetics",
        confidence=0.45  # Major delivery problems
    )
    h.set_measure(
        "Increased motor neuron survival markers, slowed functional decline",
        confidence=0.65
    )
    
    h.set_feasibility(
        technical=0.35,   # Poor - delivery very difficult
        economic=0.50,
        timeline=0.40
    )
    h.set_risk(
        execution_risk=0.70,   # Very high - many failures
        external_risk=0.35
    )
    
    # Mechanism map (simplified for brevity)
    n1 = MechanismNode("trophic_deficit", "Reduced neurotrophic support", NodeType.CAUSE)
    n1.confidence = 0.55
    h.add_mechanism_node(n1)
    
    n2 = MechanismNode("factor_delivery", "Therapeutic factor delivery", NodeType.INTERVENTION)
    n2.confidence = 0.40
    h.add_mechanism_node(n2)
    
    n3 = MechanismNode("neuron_survival", "Enhanced motor neuron survival", NodeType.OUTCOME)
    n3.confidence = 0.35  # Poor clinical translation
    h.add_mechanism_node(n3)
    
    n4 = MechanismNode("delivery_failure", "Factors don't reach motor neurons", NodeType.BLOCKER)
    n4.confidence = 0.75
    h.add_mechanism_node(n4)
    
    h.add_mechanism_edge(MechanismEdge(
        "trophic_deficit", "neuron_survival", EdgeType.PREVENTS, 0.50,
        causal_level=CausalLevel.ASSOCIATION,
        confounding_risk=0.5
    ))
    h.add_mechanism_edge(MechanismEdge(
        "factor_delivery", "neuron_survival", EdgeType.ENABLES, 0.40,
        causal_level=CausalLevel.INTERVENTION,
        confounding_risk=0.4
    ))
    h.add_mechanism_edge(MechanismEdge(
        "delivery_failure", "factor_delivery", EdgeType.PREVENTS, 0.70,
        causal_level=CausalLevel.COUNTERFACTUAL,
        confounding_risk=0.2
    ))
    
    # Evidence
    h.add_evidence(Evidence(
        "ev1",
        "BDNF, CNTF, IGF-1 trials all failed to show clinical benefit",
        "Multiple trials 1995-2010",
        quality=0.85,
        date="2010",
        domain=EvidenceDomain.MEDICAL,
        study_design="rct",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=False  # Major contradicting evidence
    ))
    h.add_evidence(Evidence(
        "ev2",
        "VEGF shows benefit in SOD1 mouse model",
        "Nature Medicine 2004",
        quality=0.70,
        date="2004",
        domain=EvidenceDomain.MEDICAL,
        study_design="controlled_experiment",
        causal_level=CausalLevel.INTERVENTION,
        supports_hypothesis=True
    ))
    h.add_evidence(Evidence(
        "ev3",
        "Poor CNS penetration of protein factors limits efficacy",
        "Review 2015",
        quality=0.75,
        date="2015",
        domain=EvidenceDomain.MEDICAL,
        study_design="expert_opinion",
        causal_level=CausalLevel.COUNTERFACTUAL,
        supports_hypothesis=False
    ))
    
    # Utility scenarios
    h.add_scenario("Breakthrough: New delivery method works", 0.05, 1.0)
    h.add_scenario("Modest benefit with gene therapy delivery", 0.15, 0.4)
    h.add_scenario("Continued failure to translate", 0.60, -0.1)
    h.add_scenario("Abandon pathway after more failures", 0.20, -0.4)
    
    return h


def run_als_analysis():
    """Run complete analysis of ALS treatment pathways"""
    
    print("=" * 70)
    print("SCIENTIFIC ANALYSIS: ALS TREATMENT PATHWAY PRIORITIZATION")
    print("=" * 70)
    print()
    print("Amyotrophic Lateral Sclerosis (ALS) is a fatal neurodegenerative disease")
    print("with no cure. This analysis evaluates which biological pathway should be")
    print("prioritized for therapeutic development investment.")
    print()
    
    # Create all hypotheses
    hypotheses = {
        'protein': create_protein_aggregation_hypothesis(),
        'glutamate': create_glutamate_excitotoxicity_hypothesis(),
        'inflammation': create_neuroinflammation_hypothesis(),
        'gene_therapy': create_gene_therapy_hypothesis(),
        'neurotrophic': create_neurotrophic_hypothesis()
    }
    
    # Run analyses
    results = {}
    comparator = HypothesisComparator()
    
    for name, hyp in hypotheses.items():
        print(f"Analyzing: {hyp.name}...")
        results[name] = run_analysis(hyp, rigor_level=3, max_iter=10)
        comparator.add_hypothesis(hyp)
    
    # Compare
    comparison = comparator.compare()
    
    # Print detailed results
    print()
    print("=" * 70)
    print("DETAILED ANALYSIS RESULTS")
    print("=" * 70)
    
    for hyp_data in comparison['hypotheses']:
        name = hyp_data['name']
        rank = comparison['rankings'][name]
        
        print(f"\n{'='*60}")
        print(f"#{rank}: {name}")
        print(f"{'='*60}")
        
        print(f"\n📊 SCORES:")
        print(f"   Bayesian Score:     {hyp_data['bayesian_score']:.3f}")
        print(f"   Combined Score:     {hyp_data['combined_score']:.3f}")
        
        print(f"\n🎯 EPISTEMIC STATE:")
        print(f"   Credence:           {hyp_data['credence']:.3f}")
        ci = hyp_data['confidence_interval']
        print(f"   95% CI:             [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        print(f"\n💰 DECISION THEORY:")
        print(f"   Expected Utility:   {hyp_data['expected_utility']:.3f}")
        print(f"   Certainty Equiv:    {hyp_data['certainty_equivalent']:.3f}")
        print(f"   Value of Info:      {hyp_data['value_of_information']:.3f}")
        
        print(f"\n🔗 CAUSAL ANALYSIS:")
        print(f"   Mechanism Conf:     {hyp_data['mechanism_confidence']:.3f}")
        print(f"   Avg Causal Str:     {hyp_data['average_causal_strength']:.3f}")
        
        print(f"\n📚 EVIDENCE:")
        print(f"   Evidence Count:     {hyp_data['evidence_count']}")
        print(f"   Total Bits:         {hyp_data['total_evidence_bits']:.2f}")
        
        if hyp_data['fatal_flaws']:
            print(f"\n💀 FATAL FLAWS:")
            for flaw in hyp_data['fatal_flaws']:
                print(f"   • {flaw}")
        
        if hyp_data['biases_detected']:
            print(f"\n⚠️  BIASES DETECTED:")
            for bias in hyp_data['biases_detected']:
                print(f"   • {bias}")
    
    # Final ranking
    print("\n")
    print("=" * 70)
    print("FINAL RANKING: WHICH PATHWAY TO PRIORITIZE")
    print("=" * 70)
    
    print("\n┌─────┬─────────────────────────────────┬─────────┬─────────┬─────────┐")
    print("│Rank │ Pathway                         │ Score   │ EU      │ VOI     │")
    print("├─────┼─────────────────────────────────┼─────────┼─────────┼─────────┤")
    
    for hyp_data in comparison['hypotheses']:
        rank = comparison['rankings'][hyp_data['name']]
        name = hyp_data['name'][:31].ljust(31)
        score = f"{hyp_data['combined_score']:.3f}".ljust(7)
        eu = f"{hyp_data['expected_utility']:.3f}".ljust(7)
        voi = f"{hyp_data['value_of_information']:.3f}".ljust(7)
        print(f"│ {rank}   │ {name} │ {score} │ {eu} │ {voi} │")
    
    print("└─────┴─────────────────────────────────┴─────────┴─────────┴─────────┘")
    
    # Recommendations
    top = comparison['hypotheses'][0]
    second = comparison['hypotheses'][1]
    bottom = comparison['hypotheses'][-1]
    
    print("\n")
    print("=" * 70)
    print("SCIENTIFIC RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
🥇 PRIMARY RECOMMENDATION: {top['name']}
   Score: {top['combined_score']:.3f} | Expected Utility: {top['expected_utility']:.3f}
   
   Rationale: This pathway has the strongest combination of:
   - Validated causal mechanisms (causal strength: {top['average_causal_strength']:.2f})
   - Quality evidence base ({top['evidence_count']} sources, {top['total_evidence_bits']:.1f} bits)
   - Favorable risk/reward profile (EU: {top['expected_utility']:.2f})
   
🥈 SECONDARY RECOMMENDATION: {second['name']}  
   Score: {second['combined_score']:.3f} | Expected Utility: {second['expected_utility']:.3f}
   
   Worth parallel investigation due to complementary mechanisms.

⚠️  CAUTION: {bottom['name']}
   Score: {bottom['combined_score']:.3f} | Expected Utility: {bottom['expected_utility']:.3f}
   
   Historical failures and delivery challenges suggest deprioritization
   unless breakthrough delivery technology emerges.

📋 KEY INSIGHTS:
   1. Protein aggregation targeting benefits from recent Tofersen approval,
      providing validated regulatory pathway and biomarkers.
   
   2. Neuroinflammation shows promise (Masitinib data) but causal role
      vs. secondary effect remains uncertain.
   
   3. Gene therapy has transformative potential but high technical risk
      and limited human data warrant cautious investment.
   
   4. Value of Information analysis suggests continued research needed
      for all pathways - none has sufficient evidence for certainty.
""")
    
    return comparison, results


if __name__ == "__main__":
    comparison, results = run_als_analysis()
