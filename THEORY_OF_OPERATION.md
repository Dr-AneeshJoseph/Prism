# Theory of Operation: PRISM v1.0

## 1. Epistemic State & Bayesian Safety
PRISM v1.0 departs from naive Bayesian implementations by strictly bounding the confidence intervals to prevent "false certainty" and floating-point instability.

### 1.1 Log-Odds Representation
To ensure numerical stability during iterative updates, all credence values ($p$) are converted to log-odds ($L$) for internal calculation:

$$L = \ln\left(\frac{p}{1-p}\right)$$

Updates are performed additively in log-space using the evidence Likelihood Ratio ($LR$):

$$L_{new} = L_{old} + \ln(LR)$$

### 1.2 The "Sincerity" Clamps
To prevent the system from becoming infinitely certain (which breaks future updates), the system enforces non-negotiable hard caps defined in `SafetyLimits`:

1.  **Credence Cap:** $p \in [0.01, 0.99]$. The system is mathematically incapable of being 100% certain.
2.  **Log-Odds Cap:** $L \in [-10.0, 10.0]$.
3.  **Likelihood Ratio Clamp:** Evidence strength is clamped to $LR \in [0.01, 100.0]$ to prevent a single piece of evidence from dominating the entire graph.

---

## 2. Evidence Quality & Causal Boosting
PRISM resolves the "RCT Illusion" (where a weak RCT outscores strong observational data) by separating **Study Design** from **Causal Level**.

### 2.1 The Boost Formula
Evidence quality is not a static lookup. It is calculated dynamically:

$$Q_{effective} = Q_{base} \times (1 + B_{causal}) \times M_{sample}$$

Where:
* $Q_{base}$: The inherent rigor of the study design (e.g., Cohort = 0.75, Meta-Analysis = 0.95).
* $B_{causal}$: A boost factor, **not** a discount.
    * *Association:* +0%
    * *Intervention:* +15%
    * *Counterfactual:* +5%
* $M_{sample}$: Modifier based on sample size (e.g., $n < 50$ incurs a 40% penalty).

This ensures that a large ($n > 10,000$) Observational study ($0.55 \times 1.10$) can compete with a small Experimental study.

---

## 3. Risk-Aware Utility Modeling (CRRA)
Standard Expected Utility (EU) fails to account for "Ruin," where a negative outcome is unacceptable regardless of the potential upside. PRISM implements **Constant Relative Risk Aversion (CRRA)**.

### 3.1 Handling Negative Utility
Standard CRRA functions ($U(x) = \frac{x^{1-\gamma}}{1-\gamma}$) fail when $x \le 0$. PRISM implements a Shifted CRRA to handle negative utility scenarios (e.g., bankruptcy):

$$U_{CRRA}(x) = \frac{(x + S)^{1-\gamma}}{1-\gamma} - S_{adjust}$$

Where:
* $\gamma$ (Gamma): The risk aversion coefficient ($0=$ Neutral, $1=$ Logarithmic, $>2=$ High Aversion).
* $S$: A shift factor equal to $|min(U_{scenarios})| + 1$, ensuring the base is always positive.

### 3.2 Certainty Equivalent
The system outputs the "Certainty Equivalent" (CE)—the guaranteed value the user should accept over the gamble:

$$CE = \left( E[U_{CRRA}] \times (1-\gamma) \right)^{\frac{1}{1-\gamma}} - S$$

---

## 4. Evidence Independence (The Echo Chamber Filter)
To prevent "Information Cascades," PRISM calculates an Independence Score ($I$) for every pair of evidence pieces.

### 4.1 Scoring Logic
The score $I \in [0, 1]$ is a discount factor applied to the evidence bits. It is derived by penalizing:
* **Shared Authors:** 30% penalty.
* **Shared Citations:** 40% penalty.
* **Shared Data:** 60% penalty.
* **Content Similarity (Jaccard):**
    $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
    If $J > 0.5$, a 30% penalty is applied.

---

## 5. Calibration & Cold Starts
To ensure the system "knows what it doesn't know," PRISM tracks the Expected Calibration Error (ECE).

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)|$$

If $N < 20$ (Cold Start), the system flags the confidence interval as "Unreliable" and artificially widens the bounds to reflect epistemic uncertainty.
