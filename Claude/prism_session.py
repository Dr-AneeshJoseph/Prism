"""
PRISM Session Manager - Claude-Optimized Wrapper
Adds checkpointing, resumability, and project management to PRISM v2.2
Author: Dr. Aneesh Joseph + Claude
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

# Import core PRISM v2.2
# Note: The file is named prism_v2_2.py (with underscores)
from prism_v2_2 import (
    Hypothesis, Evidence, Domain, StudyType,
    get_prior, Sensitivity, REF_PRIORS
)


class PRISMSession:
    """
    Manages stateful, resumable PRISM analyses.
    Designed for single-Claude completion, but checkpoints for safety.
    """
    
    def __init__(self, project_name: str, base_dir: str = "/mnt/user-data/outputs"):
        self.project_name = project_name
        self.project_dir = Path(base_dir) / f"prism_{project_name}"
        
        # Create directory structure
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.hypotheses_dir = self.project_dir / "hypotheses"
        self.results_dir = self.project_dir / "results"
        self.evidence_dir = self.project_dir / "evidence"
        
        for d in [self.hypotheses_dir, self.results_dir, self.evidence_dir]:
            d.mkdir(exist_ok=True)
        
        self.state_file = self.project_dir / "state.json"
        self.resume_file = self.project_dir / "RESUME.md"
        
        # Load or initialize state
        self.state = self._load_or_create_state()
        
        # Track tokens (approximate)
        self.token_estimate = 0
    
    def _load_or_create_state(self) -> Dict:
        """Load existing state or create new project state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                print(f"📂 Loaded existing project: {self.project_name}")
                print(f"   Created: {state['created']}")
                print(f"   Progress: {state['progress']['completed']}/{state['progress']['total']} hypotheses")
                return state
        
        print(f"🆕 Creating new PRISM project: {self.project_name}")
        return {
            'project_name': self.project_name,
            'created': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'version': '2.2',
            'hypotheses': {},
            'progress': {
                'total': 0,
                'completed': 0,
                'current': None,
                'status': 'initialized'
            },
            'comparison': None
        }
    
    def add_hypothesis(self, 
                      hypothesis_id: str,
                      title: str,
                      domain: Domain,
                      reference_class: str = "general",
                      evidence_list: Optional[List[Evidence]] = None) -> Hypothesis:
        """
        Add a new hypothesis to the project.
        Returns the Hypothesis object for further configuration.
        """
        
        # Create hypothesis
        h = Hypothesis(title, domain, reference_class)
        h.id = hypothesis_id  # Set custom ID
        
        # Add any provided evidence
        if evidence_list:
            for e in evidence_list:
                h.add_evidence(e)
        
        # Register in state
        self.state['hypotheses'][hypothesis_id] = {
            'title': title,
            'domain': domain.value,
            'reference_class': reference_class,
            'status': 'pending',
            'evidence_count': len(evidence_list) if evidence_list else 0,
            'added': datetime.now().isoformat()
        }
        self.state['progress']['total'] += 1
        
        # Save hypothesis data
        h_file = self.hypotheses_dir / f"{hypothesis_id}.json"
        self._save_hypothesis(h, h_file, hypothesis_id)
        
        self._save_state()
        
        print(f"   ✓ Added hypothesis: {hypothesis_id}")
        print(f"     Prior: {h.prior:.1%} (from {reference_class})")
        print(f"     Evidence: {len(h.evidence)} pieces")
        
        return h
    
    def _save_hypothesis(self, h: Hypothesis, filepath: Path, hypothesis_id: str = None):
        """Save hypothesis to JSON."""
        # Use provided ID or extract from filepath
        if hypothesis_id is None:
            hypothesis_id = filepath.stem
        
        data = {
            'id': hypothesis_id,
            'name': h.name,
            'domain': h.domain.value,
            'prior': h.prior,
            'prior_ci': list(h.prior_ci),
            'n_compared': h.n_compared,
            'use_kalman': h.use_kalman,
            'evidence': [
                {
                    'id': e.id,
                    'content': e.content,
                    'source': e.source,
                    'domain': e.domain.value,
                    'study_design': e.study_design,
                    'sample_size': e.sample_size,
                    'supports': e.supports,
                    'p_value': e.p_value,
                    'effect_size': e.effect_size,
                    'effect_var': e.effect_var,
                    'authors': e.authors,
                } for e in h.evidence
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_hypothesis(self, hypothesis_id: str) -> Hypothesis:
        """Load hypothesis from saved JSON."""
        h_file = self.hypotheses_dir / f"{hypothesis_id}.json"
        
        with open(h_file) as f:
            data = json.load(f)
        
        # Get reference class from state
        ref_class = self.state['hypotheses'][hypothesis_id]['reference_class']
        
        # Create hypothesis
        h = Hypothesis(data['name'], Domain(data['domain']), ref_class)
        h.n_compared = data.get('n_compared', 1)
        h.use_kalman = data.get('use_kalman', True)
        
        # Reconstruct evidence
        for e_data in data['evidence']:
            e = Evidence(
                id=e_data['id'],
                content=e_data['content'],
                source=e_data['source'],
                domain=Domain(e_data['domain']),
                study_design=e_data['study_design'],
                sample_size=e_data['sample_size'],
                supports=e_data['supports'],
                p_value=e_data.get('p_value'),
                effect_size=e_data.get('effect_size'),
                effect_var=e_data.get('effect_var'),
                authors=e_data.get('authors', []),
            )
            h.add_evidence(e)
        
        return h
    
    def analyze_hypothesis(self, hypothesis_id: str) -> Dict:
        """
        Analyze one hypothesis completely.
        This is designed to complete in one go but checkpoints for safety.
        """
        
        print(f"\n{'='*70}")
        print(f"🔬 ANALYZING: {hypothesis_id}")
        print(f"{'='*70}\n")
        
        # Update state - mark as current
        self.state['progress']['current'] = hypothesis_id
        self.state['progress']['status'] = f'analyzing_{hypothesis_id}'
        self._save_state()
        
        # Load hypothesis
        h = self._load_hypothesis(hypothesis_id)
        h_meta = self.state['hypotheses'][hypothesis_id]
        
        print(f"Hypothesis: {h.name}")
        print(f"Domain: {h.domain.value}")
        print(f"Prior: {h.prior:.1%} (from {h_meta['reference_class']})")
        print(f"Evidence: {len(h.evidence)} pieces")
        print(f"Using Kalman: {h.use_kalman}")
        if h.n_compared > 1:
            print(f"Comparing: {h.n_compared} hypotheses (optimizer's curse will be applied)")
        print()
        
        # Run PRISM analysis
        print("🔄 Running PRISM v2.2 analysis...")
        try:
            results = h.analyze()
            
            # Print results
            print("\n📊 RESULTS")
            print("-" * 40)
            print(f"Prior: {results['prior']:.1%} [{results['prior_ci'][0]:.1%}, {results['prior_ci'][1]:.1%}]")
            print(f"Bayesian Posterior: {results['posterior_bayes']:.1%} [{results['ci_bayes'][0]:.1%}, {results['ci_bayes'][1]:.1%}]")
            
            if results['posterior_kalman']:
                print(f"Kalman Posterior: {results['posterior_kalman']:.1%} [{results['ci_kalman'][0]:.1%}, {results['ci_kalman'][1]:.1%}]")
            
            if results['posterior_corrected'] != results['posterior_bayes']:
                print(f"Corrected Posterior: {results['posterior_corrected']:.1%} (optimizer's curse adjusted)")
            
            # Independence
            if results['independence']:
                indep = results['independence']
                print(f"\nIndependence: {indep['avg_indep']:.1%} (effective N: {indep['eff_n']:.1f}/{indep['n']})")
            
            # Meta-analysis
            if results['meta_analysis'] and results['meta_analysis'].get('valid'):
                ma = results['meta_analysis']
                print(f"\nMeta-Analysis:")
                print(f"  Pooled: {ma['est']:.3f} [{ma['ci'][0]:.3f}, {ma['ci'][1]:.3f}]")
                print(f"  I²: {ma['i2']:.0f}% ({ma['heterogeneity']})")
            
            # P-curve
            if results['p_curve'] and results['p_curve'].get('valid'):
                pc = results['p_curve']
                print(f"\nP-Curve: {pc['interpretation']}")
                if pc['p_hacking']:
                    print("  ⚠️ P-hacking suspected!")
            
            # Uncertainty
            unc = results['model_uncertainty']
            print(f"\nUncertainty Breakdown:")
            print(f"  Statistical: ±{unc['statistical']:.1%}")
            print(f"  Prior: ±{unc['prior']:.1%}")
            print(f"  Model: ±{unc['model']:.1%}")
            print(f"  Total: ±{unc['total']:.1%}")
            print(f"  Reliable: {unc['reliable']}")
            
            # Warnings
            if results['warnings']:
                print(f"\n⚠️ Warnings:")
                for w in results['warnings'][:5]:
                    print(f"  - {w}")
            
            # Save results
            results_file = self.results_dir / f"{hypothesis_id}_results.json"
            
            # Convert numpy types for JSON
            results_json = self._convert_for_json(results)
            
            with open(results_file, 'w') as f:
                json.dump(results_json, f, indent=2)
            
            # Update state
            h_meta['status'] = 'completed'
            h_meta['results_file'] = str(results_file)
            h_meta['posterior_bayes'] = results['posterior_bayes']
            h_meta['posterior_corrected'] = results['posterior_corrected']
            h_meta['evidence_count'] = len(h.evidence)
            self.state['progress']['completed'] += 1
            self.state['progress']['current'] = None
            self._save_state()
            
            # Write resume instructions
            self._write_resume_instructions()
            
            print(f"\n✅ Analysis complete: {hypothesis_id}")
            print(f"   Results saved: {results_file.name}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error analyzing {hypothesis_id}: {e}")
            h_meta['status'] = 'error'
            h_meta['error'] = str(e)
            self._save_state()
            raise
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    def analyze_all(self, set_n_compared: bool = True) -> Dict[str, Dict]:
        """
        Analyze all pending hypotheses.
        If set_n_compared=True, applies optimizer's curse correction.
        """
        
        pending = [h_id for h_id, h in self.state['hypotheses'].items() 
                  if h['status'] == 'pending']
        
        if not pending:
            print("No pending hypotheses to analyze.")
            return {}
        
        print(f"\n{'='*70}")
        print(f"🚀 PRISM v2.2 BATCH ANALYSIS")
        print(f"{'='*70}")
        print(f"Hypotheses to analyze: {len(pending)}")
        
        # Set n_compared for optimizer's curse
        if set_n_compared:
            n_total = len(self.state['hypotheses'])
            print(f"Optimizer's curse correction: n={n_total}")
            
            # Update all hypotheses with n_compared
            for h_id in self.state['hypotheses']:
                h_file = self.hypotheses_dir / f"{h_id}.json"
                with open(h_file) as f:
                    h_data = json.load(f)
                h_data['n_compared'] = n_total
                with open(h_file, 'w') as f:
                    json.dump(h_data, f, indent=2)
        
        results = {}
        for i, h_id in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] ", end="")
            try:
                results[h_id] = self.analyze_hypothesis(h_id)
            except Exception as e:
                print(f"Failed: {e}")
                results[h_id] = {'error': str(e)}
        
        # Generate comparison
        self._generate_comparison()
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Completed: {self.state['progress']['completed']}/{self.state['progress']['total']}")
        
        return results
    
    def _generate_comparison(self):
        """Generate comparative analysis of all completed hypotheses."""
        
        completed = {h_id: h for h_id, h in self.state['hypotheses'].items() 
                    if h['status'] == 'completed'}
        
        if len(completed) < 2:
            return
        
        # Rank by corrected posterior
        ranked = sorted(
            completed.items(),
            key=lambda x: x[1].get('posterior_corrected', 0),
            reverse=True
        )
        
        comparison = {
            'generated': datetime.now().isoformat(),
            'n_hypotheses': len(completed),
            'ranking': [
                {
                    'rank': i + 1,
                    'hypothesis_id': h_id,
                    'title': h['title'],
                    'posterior_corrected': h.get('posterior_corrected'),
                    'posterior_bayes': h.get('posterior_bayes'),
                }
                for i, (h_id, h) in enumerate(ranked)
            ],
            'best': ranked[0][0] if ranked else None,
        }
        
        self.state['comparison'] = comparison
        self._save_state()
        
        # Save comparison file
        comp_file = self.results_dir / "comparison.json"
        with open(comp_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n📊 COMPARATIVE ANALYSIS")
        print("-" * 40)
        for item in comparison['ranking']:
            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(item['rank'], '  ')
            print(f"{medal} {item['hypothesis_id']}: {item['posterior_corrected']:.1%}")
        print(f"\n🏆 Best: {comparison['best']}")
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report."""
        
        lines = [
            f"# PRISM Analysis Report: {self.project_name}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"**Version:** PRISM v2.2",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Comparison summary
        if self.state.get('comparison'):
            comp = self.state['comparison']
            best_id = comp['best']
            best_h = self.state['hypotheses'][best_id]
            
            lines.extend([
                f"**Best Hypothesis:** {best_h['title']}  ",
                f"**Posterior Probability:** {best_h['posterior_corrected']:.1%}",
                "",
                "### Ranking",
                "",
                "| Rank | Hypothesis | Posterior |",
                "|------|------------|-----------|",
            ])
            
            for item in comp['ranking']:
                lines.append(f"| {item['rank']} | {item['title'][:40]}... | {item['posterior_corrected']:.1%} |")
            
            lines.extend(["", "---", ""])
        
        # Individual hypothesis details
        lines.extend([
            "## Detailed Results",
            "",
        ])
        
        for h_id, h_meta in self.state['hypotheses'].items():
            if h_meta['status'] != 'completed':
                continue
            
            lines.extend([
                f"### {h_id}: {h_meta['title']}",
                "",
                f"**Domain:** {h_meta['domain']}  ",
                f"**Reference Class:** {h_meta['reference_class']}  ",
                f"**Evidence Count:** {h_meta['evidence_count']}",
                "",
            ])
            
            # Load detailed results
            results_file = h_meta['results_file']
            with open(results_file) as f:
                results = json.load(f)
            
            lines.extend([
                "**Results:**",
                "",
                f"- **Prior:** {results['prior']:.1%} [{results['prior_ci'][0]:.1%}, {results['prior_ci'][1]:.1%}]",
                f"- **Bayesian Posterior:** {results['posterior_bayes']:.1%} [{results['ci_bayes'][0]:.1%}, {results['ci_bayes'][1]:.1%}]",
            ])
            
            if results['posterior_kalman']:
                lines.append(f"- **Kalman Posterior:** {results['posterior_kalman']:.1%} [{results['ci_kalman'][0]:.1%}, {results['ci_kalman'][1]:.1%}]")
            
            if results['posterior_corrected'] != results['posterior_bayes']:
                lines.append(f"- **Corrected Posterior:** {results['posterior_corrected']:.1%} (optimizer's curse adjusted)")
            
            # Meta-analysis
            if results['meta_analysis'] and results['meta_analysis'].get('valid'):
                ma = results['meta_analysis']
                lines.extend([
                    "",
                    "**Meta-Analysis:**",
                    "",
                    f"- Pooled Effect: {ma['est']:.3f} [{ma['ci'][0]:.3f}, {ma['ci'][1]:.3f}]",
                    f"- I²: {ma['i2']:.0f}% ({ma['heterogeneity']})"
                ])
            
            # P-curve
            if results['p_curve'] and results['p_curve'].get('valid'):
                pc = results['p_curve']
                lines.extend([
                    "",
                    "**P-Curve Analysis:**",
                    "",
                    f"- Interpretation: {pc['interpretation']}",
                    f"- P-hacking suspected: {pc['p_hacking']}"
                ])
            
            # Uncertainty
            unc = results['model_uncertainty']
            lines.extend([
                "",
                "**Uncertainty Breakdown:**",
                "",
                f"- Statistical: ±{unc['statistical']:.1%}",
                f"- Prior: ±{unc['prior']:.1%}",
                f"- Model: ±{unc['model']:.1%}",
                f"- Total: ±{unc['total']:.1%}",
                f"- Reliable: {unc['reliable']}"
            ])
            
            # Warnings
            if results['warnings']:
                lines.extend(["", "**⚠️ Warnings:**", ""])
                for w in results['warnings']:
                    lines.append(f"- {w}")
            
            lines.extend(["", "---", ""])
        
        # Methodology
        lines.extend([
            "## Methodology",
            "",
            "This analysis used PRISM v2.2 (Protocol for Rigorous Investigation of Scientific Mechanisms).",
            "",
            "**Key Features:**",
            "",
            "- Bayesian updating with reference class priors",
            "- Hierarchical correlation correction (within-cluster ρ=0.6, between-cluster ρ=0.2)",
            "- REML meta-analysis with Hartung-Knapp adjustment",
            "- P-curve publication bias detection",
            "- Kalman filtering for temporal evidence integration",
            "- Optimizer's curse correction for multiple hypothesis comparison",
            "- Uncertainty decomposition (statistical + prior + model)",
            "",
            "**Reference:** Dr. Aneesh Joseph, PRISM v2.2 Scientific Guide (December 2025)",
            ""
        ])
        
        report_text = "\n".join(lines)
        
        # Save report
        report_file = self.results_dir / "FINAL_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n📄 Full report generated: {report_file.name}")
        
        return report_text
    
    def _save_state(self):
        """Save current state to JSON."""
        self.state['last_modified'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _write_resume_instructions(self):
        """Write human-readable resume file."""
        
        pending = [h_id for h_id, h in self.state['hypotheses'].items() 
                  if h['status'] == 'pending']
        completed = [h_id for h_id, h in self.state['hypotheses'].items() 
                    if h['status'] == 'completed']
        
        lines = [
            f"# PRISM Analysis Resume Point",
            f"## Project: {self.project_name}",
            f"## Last Updated: {self.state['last_modified']}",
            "",
            "---",
            "",
            "## Progress",
            "",
            f"- **Total Hypotheses:** {self.state['progress']['total']}",
            f"- **Completed:** {self.state['progress']['completed']}",
            f"- **Pending:** {len(pending)}",
            f"- **Status:** {self.state['progress']['status']}",
            "",
            "## To Resume Analysis",
            "",
            "```python",
            "from prism_session import PRISMSession",
            "",
            f"session = PRISMSession('{self.project_name}')",
            "session.analyze_all()  # Continue with pending hypotheses",
            "session.generate_report()  # Generate final report",
            "```",
            "",
        ]
        
        if completed:
            lines.extend([
                "## Completed Hypotheses",
                "",
            ])
            for h_id in completed:
                h_meta = self.state['hypotheses'][h_id]
                post = h_meta.get('posterior_corrected', h_meta.get('posterior_bayes', 0))
                lines.append(f"- **{h_id}** ({post:.1%}): {h_meta['title']}")
            lines.append("")
        
        if pending:
            lines.extend([
                "## Pending Hypotheses",
                "",
            ])
            for h_id in pending:
                h_meta = self.state['hypotheses'][h_id]
                lines.append(f"- **{h_id}**: {h_meta['title']}")
            lines.append("")
        
        lines.extend([
            "## Next Steps",
            "",
        ])
        
        if pending:
            lines.extend([
                "1. Complete analysis of pending hypotheses",
                "2. Generate comparative analysis",
                "3. Create final report",
            ])
        else:
            lines.extend([
                "1. ✅ All hypotheses analyzed",
                "2. Generate final report with `session.generate_report()`",
            ])
        
        with open(self.resume_file, 'w') as f:
            f.write("\n".join(lines))
    
    def resume(self):
        """Resume analysis from last checkpoint."""
        current = self.state['progress']['current']
        
        if current and self.state['hypotheses'][current]['status'] != 'completed':
            print(f"⚠️  Resuming incomplete analysis: {current}")
            self.analyze_hypothesis(current)
        
        return self.analyze_all()


def create_evidence_from_dict(data: Dict) -> Evidence:
    """Helper to create Evidence from dictionary."""
    return Evidence(
        id=data['id'],
        content=data['content'],
        source=data['source'],
        domain=Domain(data['domain']) if isinstance(data['domain'], str) else data['domain'],
        study_design=data['study_design'],
        sample_size=data['sample_size'],
        supports=data['supports'],
        p_value=data.get('p_value'),
        effect_size=data.get('effect_size'),
        effect_var=data.get('effect_var'),
        authors=data.get('authors', []),
    )


if __name__ == "__main__":
    print("PRISM Session Manager v2.2")
    print("Use: from prism_session import PRISMSession")
