# PRISM Quick Start for Mobile Users (Android/iOS)

**For Claude App Users** - Simple setup in 5 minutes

---

## What You Need

1. Claude mobile app (Android or iOS)
2. Your GitHub account
3. 5 minutes

---

## Step 1: Get the Files (Right Now, From This Chat)

I'll present all the files you need. Tap each one to download:

### Core Files
- `prism_v22.py` - The main PRISM code
- `SKILL.md` - Instructions for Claude

### Supporting Files  
- `prism_session.py` - Session manager
- `README.md` - Documentation
- `README_CLAUDE.md` - Claude usage guide

**Download all these files to your phone now.**

---

## Step 2: Upload to GitHub

### Using GitHub Mobile App

1. Open GitHub app
2. Go to your repository: `Dr-AneeshJoseph/Prism`
3. Tap the **"+"** button
4. Select **"Upload file"**
5. Choose the files you just downloaded
6. Add commit message: "Add PRISM v2.2 with Claude integration"
7. Tap **"Commit"**

**Suggested folder structure:**
```
Prism/
├── prism_v22.py          (root)
├── README.md             (root)
└── claude/
    ├── SKILL.md          (create claude folder first)
    ├── prism_session.py
    └── README_CLAUDE.md
```

### Using GitHub Website (Mobile Browser)

1. Go to `github.com/Dr-AneeshJoseph/Prism`
2. Tap menu → "Upload files"
3. Select files from your phone
4. Commit

---

## Step 3: Install PRISM Skill in Claude

**This is the KEY step for mobile users!**

1. **Start a NEW Claude chat**
2. **Tap the attachment icon** (📎)
3. **Select the `SKILL.md` file** you downloaded
4. **Type this message:**
   ```
   Please save this as a user skill at 
   /mnt/skills/user/prism/SKILL.md
   
   This is the PRISM v2.2 methodology that you should 
   use whenever I mention PRISM.
   ```
5. **Done!** Claude confirms it's installed

---

## Step 4: Test It!

**In the SAME chat or a NEW chat, say:**

```
Use PRISM to find the best treatment for insomnia.
Search for recent studies and give me a full analysis.
```

**Claude will:**
1. ✅ Automatically read the SKILL you installed
2. ✅ Search the web for studies
3. ✅ Run PRISM analysis
4. ✅ Give you ranked results

**If it works:** You're all set! 🎉

**If Claude says "I don't know what PRISM is":**
- The skill didn't install properly
- Repeat Step 3 in a new chat

---

## Step 5: Use PRISM Anytime

### For Quick Analyses

**Just ask in plain English:**
```
"Use PRISM to analyze the best way to reduce anxiety"
"Compare these 3 business strategies using PRISM"
"What's the most effective intervention for obesity?"
```

### With Your Own Data

```
"I have data from 5 studies about meditation. 
Use PRISM to analyze whether meditation reduces stress.

Study 1: RCT, n=200, p=0.01, effect size = -0.45
Study 2: Meta-analysis, n=1500, p=0.001, effect = -0.52
[etc]"
```

### Getting Reports

When analysis completes:
1. Claude will show summary
2. Claude will create `FINAL_REPORT.md`
3. **Tap the report to download it**
4. Save to your phone or cloud storage

---

## Mobile-Specific Tips

### ✅ DO:
- Download reports immediately (they expire when chat ends)
- Use cloud storage (Google Drive, Dropbox) for long-term storage
- Start fresh chats for new projects
- Copy important results to notes

### ❌ DON'T:
- Expect files to persist between chats (they don't)
- Try to access `/home/claude/` directly (mobile can't do this)
- Upload huge datasets (use summaries instead)

---

## Workflow for Ongoing Projects

### Option 1: One Chat Per Analysis (Simple)
```
Chat 1: Analyze diabetes treatments → Download report → Done
Chat 2: Analyze anxiety interventions → Download report → Done
```

### Option 2: Multi-Session Projects (Advanced)
```
Chat 1: Start analysis
        → Download all state files Claude creates
        
[Later, new chat]

Chat 2: Upload state files
        → Say "Continue the PRISM analysis"
        → Claude resumes from where you left off
```

---

## Troubleshooting on Mobile

### Problem: "Files not found"
**Solution:** Files only exist during the chat. Download them before ending.

### Problem: "Can't run Python code"
**Solution:** You don't run it yourself. Claude runs it on its computer automatically.

### Problem: "Lost my analysis"
**Solution:** Download reports immediately. If lost, start over (takes ~5 minutes).

### Problem: "SKILL not working"
**Solution:** 
1. Start new chat
2. Re-upload SKILL.md
3. Ask Claude to reinstall it

---

## What You Can Do on Mobile

### ✅ EVERYTHING:
- Run PRISM analyses
- Get research recommendations
- Compare hypotheses
- Generate reports
- Search for evidence
- Extract data from papers
- Update priors
- Re-analyze with new evidence

### ⚠️ Limitations:
- Can't edit Python code directly (ask Claude to edit instead)
- Files don't persist (download what you need)
- Large datasets difficult (use summaries)

---

## Real Example: Mobile Workflow

**You:** *[Opens Claude app]*

**You:** *[Taps new chat]*

**You:** 
```
Use PRISM to find the best diet for weight loss.
Search for recent meta-analyses and systematic reviews.
```

**Claude:** 
```
[Searches web]
[Creates PRISM session]
[Analyzes 6 diet approaches]

🏆 Best Diet: Mediterranean diet
   Posterior: 86.4%
   
📋 Ranking:
   🥇 Mediterranean: 86.4%
   🥈 Low-carb: 78.2%
   🥉 Intermittent fasting: 72.1%
   ...
   
[Full report: FINAL_REPORT.md]
```

**You:** *[Taps FINAL_REPORT.md]*

**You:** *[Downloads to phone]*

**You:** *[Uploads to Google Drive for later]*

**Done! Total time: 3 minutes** ✅

---

## Quick Commands

**Start analysis:**
```
"Use PRISM to analyze [topic]"
```

**With specific hypotheses:**
```
"Use PRISM to compare:
1. Hypothesis A
2. Hypothesis B
3. Hypothesis C"
```

**Add custom evidence:**
```
"In our previous PRISM analysis, add this new study:
[study details]"
```

**Get explanation:**
```
"Explain how PRISM calculated that posterior probability"
```

**Modify parameters:**
```
"Re-run the analysis with a 30% prior instead of 40%"
```

---

## Sharing Results

### From Mobile:

**PDF Reports:**
1. Download markdown report
2. Use any markdown-to-PDF app
3. Share via email/messaging

**Screenshots:**
1. Screenshot Claude's summary
2. Share directly

**Cloud Storage:**
1. Download reports
2. Upload to Drive/Dropbox
3. Share link

---

## Summary: Mobile Usage in 3 Steps

```
┌─────────────────────────────────────────┐
│  1. INSTALL (Once)                      │
│     Upload SKILL.md to Claude           │
│     ↓                                   │
│  2. USE (Anytime)                       │
│     "Use PRISM to analyze [topic]"      │
│     ↓                                   │
│  3. DOWNLOAD (Each time)                │
│     Tap report → Save to phone          │
└─────────────────────────────────────────┘
```

**That's it! You're ready to use PRISM on mobile!** 📱

---

## Next Steps

1. ✅ Download the files I present (scroll up in chat)
2. ✅ Upload to your GitHub
3. ✅ Install SKILL.md in Claude (Step 3 above)
4. ✅ Test with a simple question
5. ✅ Start using for real research!

**Questions? Just ask me!**
