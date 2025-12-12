# PRISM Claude Integration

This directory contains files for using PRISM with Claude AI.

---

## What's in This Directory

```
Claude/
├── SKILL.md              # Skill file that Claude reads
├── prism_session.py      # Session management wrapper
└── README_CLAUDE.md      # This file
```

---

## Setup (One-Time Installation)

### Step 1: Install the Skill

**Option A: From Mobile (Android/iOS)**
1. Download `SKILL.md` from this directory
2. Start a new Claude chat
3. Upload the `SKILL.md` file
4. Say: "Install this as a user skill for PRISM at /mnt/skills/user/prism/SKILL.md"
5. Done! Future chats will automatically know about PRISM

**Option B: From Web**
1. Download `SKILL.md` from this directory
2. Start a new Claude chat
3. Upload the file
4. Say: "Save this to /mnt/skills/user/prism/SKILL.md"

### Step 2: Verify Installation

In any new Claude chat, say:
```
"Do you know about PRISM?"
```

Claude should respond with knowledge of PRISM v2.2 methodology.

---

## How to Use

### Easiest Way: Just Ask Claude

```
You: "Use PRISM to find the best treatment for insomnia"

Claude will automatically:
1. Read the SKILL.md file
2. Search the web for studies
3. Extract evidence
4. Run PRISM analysis
5. Present ranked results
```

### With Your Own Data

```
You: "I have evidence about three hypotheses for treating anxiety.
     Use PRISM to analyze them."
     
     [Provide your data]

Claude will:
1. Create a PRISM session
2. Format your evidence
3. Run the analysis
4. Give you results
```

### Custom Analysis

```
You: "Use PRISM to compare these business strategies:
     1. Expand to Asia
     2. Focus on North America
     3. Digital-first approach
     
     Search for evidence about success rates."

Claude will search and analyze automatically.
```

---

## Features

### Automatic Web Search
Claude will search for:
- Recent clinical trials
- Systematic reviews
- Meta-analyses
- Expert guidelines

### Evidence Extraction
Claude extracts:
- Study design
- Sample size
- Effect sizes
- P-values
- Authors and sources

### Complete Analysis
Claude runs:
- Bayesian updating
- Meta-analysis
- Publication bias detection
- Uncertainty quantification
- Optimizer's curse correction

### Professional Reports
Claude generates:
- Executive summary
- Ranked hypotheses
- Detailed results
- Uncertainty breakdown
- Downloadable markdown reports

---

## Advanced Usage

### Resume Interrupted Analysis

If analysis stops midway:
```
You: "Continue the PRISM analysis we started earlier"

Claude will load the saved state and continue.
```

### Add New Evidence

```
You: "Add this new study to hypothesis H1 in our diabetes analysis:
     [study details]"

Claude will update and re-analyze.
```

### Custom Reference Classes

```
You: "Use a custom prior of 25% for this hypothesis based on
     pilot study results"

Claude can adjust priors as needed.
```

---

## Example Workflows

### Medical Research
```
You: "Use PRISM to find the best treatment for Type 2 diabetes"

Claude: [Searches recent trials, analyzes 8 treatments]

Result:
🏆 Best: Metformin + SGLT2 inhibitor (84.3%)
🥈 GLP-1 agonists (79.1%)
🥉 Insulin therapy (71.5%)
...
```

### Business Strategy
```
You: "Should we expand internationally or focus on domestic market?
     Use PRISM to analyze evidence."

Claude: [Searches case studies, analyzes outcomes]

Result:
🏆 Staged international expansion (76.8%)
🥈 Domestic market deepening (68.2%)
...
```

### Policy Analysis
```
You: "What's the most effective intervention for reducing homelessness?"

Claude: [Searches policy studies, analyzes interventions]

Result:
🏆 Housing First + support services (88.5%)
🥈 Rapid rehousing (74.3%)
...
```

---

## Mobile Usage (Android/iOS App)

### Important Notes for Mobile

1. **Files are temporary** - Download any reports you want to keep
2. **No persistent storage** - Each chat is independent
3. **Skill persists** - Once installed, SKILL.md works across all chats
4. **Use cloud storage** - For long-term project tracking

### Recommended Mobile Workflow

**For one-time analyses:**
```
1. Ask Claude to run PRISM
2. Get results
3. Download the report
4. Done!
```

**For ongoing projects:**
```
1. Start analysis in one chat
2. Download state files Claude creates
3. In next chat, upload state files
4. Claude continues from where you left off
```

---

## Troubleshooting

### "Claude doesn't know about PRISM"
- Re-install SKILL.md
- Verify it's at `/mnt/skills/user/prism/SKILL.md`
- Start a fresh chat

### "Analysis seems wrong"
- Check the evidence Claude extracted
- Verify study designs are correct
- Ask Claude to explain the calculation

### "Need to modify the methodology"
- Update SKILL.md in this directory
- Re-upload to Claude
- Future chats will use new methodology

---

## Files in Your Chat

When Claude runs PRISM, it creates:

```
/mnt/user-data/outputs/prism_{project}/
├── RESUME.md              # Human-readable status
├── state.json             # Project state
├── hypotheses/            # Hypothesis data
│   ├── h1_*.json
│   └── h2_*.json
└── results/
    ├── FINAL_REPORT.md    # Main report (download this!)
    ├── comparison.json    # Rankings
    └── *_results.json     # Detailed results
```

**Download FINAL_REPORT.md** - That's your main output!

---

## Updating to New Versions

When PRISM gets updated (e.g., to v2.3):

1. Download new `SKILL.md` from this directory
2. Upload to Claude
3. Say: "Update the PRISM skill with this new version"
4. Done!

---

## Getting Help

### In Claude
```
"How do I use PRISM to analyze [my specific case]?"
```

### On GitHub
- Open an issue
- Check existing discussions
- See examples/ directory

---

## Technical Details

### What the Skill Contains
- PRISM v2.2 methodology
- Statistical methods (Bayesian, meta-analysis, etc.)
- Best practices for evidence extraction
- Checkpointing and resumability protocols
- Token usage optimization

### How It Works
1. You mention PRISM in chat
2. Claude automatically reads `/mnt/skills/user/prism/SKILL.md`
3. Claude follows the methodology in the skill
4. Claude uses `prism_v2_2.py` for calculations
5. Results are saved and presented to you

---

## Notes for Developers

If you're modifying PRISM:

1. **Update prism_v2_2.py** - The core Python code
2. **Update SKILL.md** - The Claude instructions
3. **Keep them synchronized** - Same version numbers
4. **Test with Claude** - Before committing
5. **Update this README** - Document changes

---

## Version Compatibility

| PRISM Version | Skill Version | Compatible |
|---------------|---------------|------------|
| v2.2 | v2.2 | ✅ Yes |
| v2.1 | v2.2 | ✅ Yes |
| v2.2 | v2.1 | ⚠️ Limited |

**Recommendation:** Keep skill and code versions matched.

---

## Support

For questions about:
- **PRISM methodology** → See main README.md
- **Claude integration** → Ask Claude directly
- **Bug reports** → GitHub issues
- **Feature requests** → GitHub discussions

---

**Happy analyzing with PRISM and Claude! 🚀**
