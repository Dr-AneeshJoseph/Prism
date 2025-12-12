"""
Example: Finding Best Hypothesis for Osteoarthritis Treatment
Demonstrates complete PRISM v2.2 workflow with Claude
"""

from prism_session import PRISMSession, create_evidence_from_dict
from prism_v22 import Domain, Evidence

def create_oa_example():
    """
    Create a realistic osteoarthritis treatment analysis.
    
    In a real scenario, Claude would:
    1. Use web_search to find current treatment options
    2. Use web_fetch to get detailed study information
    3. Extract evidence from papers automatically
    
    For this example, we'll use curated data based on known research.
    """
    
    print("="*70)
    print("PRISM v2.2 - Osteoarthritis Treatment Analysis")
    print("="*70)
    print()
    
    # Create session
    session = PRISMSession("osteoarthritis_treatment_2025")
    
    # Define hypotheses based on current treatment landscape
    # In real use, Claude would search: web_search("osteoarthritis treatment options 2024")
    
    print("📋 Defining hypotheses from literature review...\n")
    
    # HYPOTHESIS 1: Weight Loss
    print("[1/8] Adding hypothesis: Weight Loss Intervention")
    h1 = session.add_hypothesis(
        hypothesis_id="h1_weight_loss",
        title="Weight loss reduces knee OA pain and improves function",
        domain=Domain.MEDICAL,
        reference_class="replication"  # ~40% prior based on OSF replication studies
    )
    
    # Add evidence (in real scenario, from web_search + web_fetch)
    h1_evidence = [
        Evidence(
            id="messier_2013",
            content="18-month RCT: 10% weight loss reduced pain by 50% vs control",
            source="JAMA 2013;310(12):1263-1273",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=454,
            supports=True,
            p_value=0.0001,
            effect_size=-0.48,  # Cohen's d
            effect_var=0.01440,
            authors=["Messier", "Mihalko"],
        ),
        Evidence(
            id="christensen_2007",
            content="16-week RCT: Diet intervention reduced pain 30% and improved function",
            source="Arthritis Rheum 2007;57:666-671",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=316,
            supports=True,
            p_value=0.003,
            effect_size=-0.35,
            effect_var=0.01210,
            authors=["Christensen", "Astrup"],
        ),
        Evidence(
            id="bliddal_2014",
            content="Systematic review: Weight loss consistently improves OA symptoms",
            source="Osteoarthritis Cartilage 2014;22:1838-1846",
            domain=Domain.MEDICAL,
            study_design="systematic_review",
            sample_size=1840,
            supports=True,
            p_value=0.0001,
            authors=["Bliddal", "Leeds"],
        )
    ]
    
    for e in h1_evidence:
        h1.add_evidence(e)
    
    session._save_hypothesis(h1, session.hypotheses_dir / "h1_weight_loss.json", "h1_weight_loss")
    
    # HYPOTHESIS 2: Exercise/Physical Therapy
    print("[2/8] Adding hypothesis: Exercise & Physical Therapy")
    h2 = session.add_hypothesis(
        hypothesis_id="h2_exercise",
        title="Structured exercise reduces OA pain and improves function",
        domain=Domain.MEDICAL,
        reference_class="replication"
    )
    
    h2_evidence = [
        Evidence(
            id="fransen_2015_meta",
            content="Meta-analysis of 60 trials: Exercise reduces pain (ES=-0.49) and improves function",
            source="Cochrane Database Syst Rev 2015;1:CD004376",
            domain=Domain.MEDICAL,
            study_design="meta_analysis",
            sample_size=8218,
            supports=True,
            p_value=0.0001,
            effect_size=-0.49,
            effect_var=0.00640,
            authors=["Fransen", "McConnell"],
        ),
        Evidence(
            id="brosseau_2012",
            content="RCT: 12-week PT program reduced pain 35% and improved mobility",
            source="Phys Ther 2012;92:210-226",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=222,
            supports=True,
            p_value=0.001,
            effect_size=-0.42,
            effect_var=0.01690,
            authors=["Brosseau", "Wells"],
        )
    ]
    
    for e in h2_evidence:
        h2.add_evidence(e)
    
    session._save_hypothesis(h2, session.hypotheses_dir / "h2_exercise.json", "h2_exercise")
    
    # HYPOTHESIS 3: NSAIDs
    print("[3/8] Adding hypothesis: NSAIDs")
    h3 = session.add_hypothesis(
        hypothesis_id="h3_nsaids",
        title="NSAIDs effectively reduce OA pain with acceptable safety profile",
        domain=Domain.MEDICAL,
        reference_class="drug_approval"  # ~10% prior for FDA approval standard
    )
    
    h3_evidence = [
        Evidence(
            id="bjordal_2007",
            content="Meta-analysis: NSAIDs reduce pain (ES=-0.32) but increase GI events",
            source="Ann Rheum Dis 2007;66:639-646",
            domain=Domain.MEDICAL,
            study_design="meta_analysis",
            sample_size=9812,
            supports=True,
            p_value=0.0001,
            effect_size=-0.32,
            effect_var=0.00360,
            authors=["Bjordal", "Klovning"],
        ),
        Evidence(
            id="nissen_2016",
            content="Cardiovascular safety analysis: Increased CV risk with long-term use",
            source="N Engl J Med 2016;375:2519-2529",
            domain=Domain.MEDICAL,
            study_design="cohort",
            sample_size=4681,
            supports=False,  # Safety concern
            p_value=0.02,
            authors=["Nissen"],
        )
    ]
    
    for e in h3_evidence:
        h3.add_evidence(e)
    
    session._save_hypothesis(h3, session.hypotheses_dir / "h3_nsaids.json", "h3_nsaids")
    
    # HYPOTHESIS 4: Corticosteroid Injection
    print("[4/8] Adding hypothesis: Corticosteroid Injection")
    h4 = session.add_hypothesis(
        hypothesis_id="h4_steroid_injection",
        title="Intra-articular corticosteroid provides meaningful pain relief",
        domain=Domain.MEDICAL,
        reference_class="phase3_clinical"  # ~35% prior
    )
    
    h4_evidence = [
        Evidence(
            id="jüni_2015",
            content="Meta-analysis: Short-term benefit (4-6 weeks) but no long-term effect",
            source="Ann Intern Med 2015;162:46-54",
            domain=Domain.MEDICAL,
            study_design="meta_analysis",
            sample_size=1767,
            supports=True,
            p_value=0.01,
            effect_size=-0.28,  # Short-term only
            effect_var=0.01210,
            authors=["Jüni", "Reichenbach"],
        ),
        Evidence(
            id="mcalindon_2017",
            content="RCT: No benefit over placebo at 24 weeks, possible cartilage loss",
            source="JAMA 2017;317:1967-1975",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=140,
            supports=False,
            p_value=0.52,  # Not significant
            authors=["McAlindon"],
        )
    ]
    
    for e in h4_evidence:
        h4.add_evidence(e)
    
    session._save_hypothesis(h4, session.hypotheses_dir / "h4_steroid_injection.json", "h4_steroid_injection")
    
    # HYPOTHESIS 5: Hyaluronic Acid Injection
    print("[5/8] Adding hypothesis: Hyaluronic Acid (Viscosupplementation)")
    h5 = session.add_hypothesis(
        hypothesis_id="h5_hyaluronic_acid",
        title="Hyaluronic acid injection provides clinically meaningful benefit",
        domain=Domain.MEDICAL,
        reference_class="phase3_clinical"
    )
    
    h5_evidence = [
        Evidence(
            id="rutjes_2012",
            content="Cochrane review: Small benefit (ES=-0.21) but high bias risk",
            source="Cochrane Database Syst Rev 2012;12:CD005321",
            domain=Domain.MEDICAL,
            study_design="systematic_review",
            sample_size=12667,
            supports=True,
            p_value=0.04,
            effect_size=-0.21,
            effect_var=0.00810,
            authors=["Rutjes", "Jüni"],
        ),
        Evidence(
            id="jevsevar_2013",
            content="AAOS guidelines: Do NOT recommend due to insufficient evidence",
            source="J Am Acad Orthop Surg 2013;21:571-576",
            domain=Domain.MEDICAL,
            study_design="expert_opinion",
            sample_size=0,
            supports=False,
            authors=["Jevsevar"],
        )
    ]
    
    for e in h5_evidence:
        h5.add_evidence(e)
    
    session._save_hypothesis(h5, session.hypotheses_dir / "h5_hyaluronic_acid.json", "h5_hyaluronic_acid")
    
    # HYPOTHESIS 6: PRP (Platelet-Rich Plasma)
    print("[6/8] Adding hypothesis: PRP Therapy")
    h6 = session.add_hypothesis(
        hypothesis_id="h6_prp",
        title="PRP injection is effective for knee OA",
        domain=Domain.MEDICAL,
        reference_class="general"  # Emerging therapy, uninformative prior
    )
    
    h6_evidence = [
        Evidence(
            id="shen_2017",
            content="Meta-analysis: PRP superior to HA but high heterogeneity (I²=92%)",
            source="Sci Rep 2017;7:5890",
            domain=Domain.MEDICAL,
            study_design="meta_analysis",
            sample_size=1543,
            supports=True,
            p_value=0.03,
            effect_size=-0.39,
            effect_var=0.03240,  # Large SE due to heterogeneity
            authors=["Shen", "Yuan"],
        ),
        Evidence(
            id="bennell_2021",
            content="High-quality RCT: No benefit over saline at 12 months",
            source="JAMA 2021;326:2021-2030",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=288,
            supports=False,
            p_value=0.85,
            authors=["Bennell", "Paterson"],
        )
    ]
    
    for e in h6_evidence:
        h6.add_evidence(e)
    
    session._save_hypothesis(h6, session.hypotheses_dir / "h6_prp.json", "h6_prp")
    
    # HYPOTHESIS 7: Total Knee Replacement (TKR)
    print("[7/8] Adding hypothesis: Total Knee Replacement")
    h7 = session.add_hypothesis(
        hypothesis_id="h7_tkr",
        title="TKR is effective for end-stage knee OA",
        domain=Domain.MEDICAL,
        reference_class="phase3_clinical"  # Established procedure
    )
    
    h7_evidence = [
        Evidence(
            id="carr_2012",
            content="Systematic review: Large improvements in pain and function (ES=-1.35)",
            source="Clin Orthop Relat Res 2012;470:54-63",
            domain=Domain.MEDICAL,
            study_design="systematic_review",
            sample_size=4578,
            supports=True,
            p_value=0.0001,
            effect_size=-1.35,  # Very large effect
            effect_var=0.02250,
            authors=["Carr", "Robertsson"],
        ),
        Evidence(
            id="skou_2015",
            content="RCT: TKR not superior to non-surgical treatment at 12 months",
            source="N Engl J Med 2015;373:1597-1606",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=100,
            supports=False,
            p_value=0.27,
            authors=["Skou", "Roos"],
        )
    ]
    
    for e in h7_evidence:
        h7.add_evidence(e)
    
    session._save_hypothesis(h7, session.hypotheses_dir / "h7_tkr.json", "h7_tkr")
    
    # HYPOTHESIS 8: Combination Therapy
    print("[8/8] Adding hypothesis: Combined Weight Loss + Exercise")
    h8 = session.add_hypothesis(
        hypothesis_id="h8_combination",
        title="Combined weight loss + exercise is superior to either alone",
        domain=Domain.MEDICAL,
        reference_class="replication"
    )
    
    h8_evidence = [
        Evidence(
            id="messier_2004",
            content="RCT: Diet+Exercise superior to either alone (p<0.001)",
            source="Arthritis Rheum 2004;50:1501-1510",
            domain=Domain.MEDICAL,
            study_design="rct",
            sample_size=316,
            supports=True,
            p_value=0.0001,
            effect_size=-0.58,  # Larger than either alone
            effect_var=0.01690,
            authors=["Messier", "Loeser"],
        ),
        Evidence(
            id="aaboe_2011",
            content="Cohort: Combined approach reduced pain 55% at 6 months",
            source="BMC Musculoskelet Disord 2011;12:152",
            domain=Domain.MEDICAL,
            study_design="cohort",
            sample_size=192,
            supports=True,
            p_value=0.002,
            effect_size=-0.51,
            effect_var=0.02250,
            authors=["Aaboe", "Bliddal"],
        )
    ]
    
    for e in h8_evidence:
        h8.add_evidence(e)
    
    session._save_hypothesis(h8, session.hypotheses_dir / "h8_combination.json", "h8_combination")
    
    print("\n✅ All hypotheses defined with evidence from literature")
    print(f"   Total hypotheses: {session.state['progress']['total']}")
    print(f"   Ready for analysis")
    
    return session


def main():
    """Run the complete analysis."""
    
    # Step 1: Create hypotheses with evidence
    session = create_oa_example()
    
    print("\n" + "="*70)
    print("🚀 Starting PRISM analysis...")
    print()
    
    # Step 2: Analyze all hypotheses
    session.analyze_all(set_n_compared=True)
    
    # Step 3: Generate report
    print("\n" + "="*70)
    session.generate_report()
    
    # Step 4: Show where files are
    print("\n" + "="*70)
    print("📁 OUTPUT FILES")
    print("="*70)
    print(f"Project directory: {session.project_dir}")
    print(f"  ├── state.json          (Project state)")
    print(f"  ├── RESUME.md           (Resume instructions)")
    print(f"  ├── hypotheses/         ({len(session.state['hypotheses'])} hypothesis files)")
    print(f"  └── results/")
    print(f"      ├── comparison.json (Comparative analysis)")
    print(f"      ├── FINAL_REPORT.md (Full report)")
    print(f"      └── *_results.json  (Individual results)")
    print()
    
    return session


if __name__ == "__main__":
    session = main()
    print("\n✅ Analysis complete!")
    print(f"\nTo view the report: view {session.results_dir / 'FINAL_REPORT.md'}")
  
