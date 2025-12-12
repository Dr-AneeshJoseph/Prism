# Theory of Operation: PRISM v2.2

**Author:** Dr. Aneesh Joseph  
**Version:** 2.2 | December 2025

---

## 1. Epistemic State & Bayesian Safety

PRISM v2.2 uses a Bayesian framework with strict numerical safeguards to prevent overconfidence and ensure stable computation.

### 1.1 Log-Odds Representation

All credence values ($p$) are converted to log-odds ($L$) for numerical stability:

$$L = \ln\left(\frac{p}{1-p}\right)$$

Updates are performed additively using the evidence Likelihood Ratio ($LR$):

$$L_{new} = L_{old} + \ln(LR)$$

### 1.2 Safety Clamps

To prevent infinite certainty (which breaks future updates), hard caps are enforced:

1. **Credence Cap:** $p \in [0.01, 0.99]$
2. **Log-Odds Cap:** $L \in [-10.0, 10.0]$
3. **Likelihood Ratio Clamp:** $LR \in [0.01, 100.0]$

---

## 2. Reference Class Priors with Uncertainty

### 2.1 Beta Distribution Priors

Unlike v1.0's point estimates, v2.2 uses **Beta distributions** to represent prior uncertainty:

$$\text{Prior} \sim \text{Beta}(\alpha, \beta)$$

Where $\alpha$ and $\beta$ are derived from historical success/failure counts with Jeffrey's prior:

$$\alpha = \text{successes} + 0.5$$
$$\beta = \text{failures} + 0.5$$

### 2.2 Available Reference Classes

| Reference Class | Mean Prior | 95% CI | Source |
|-----------------|------------|--------|--------|
| `phase2_clinical` | 15% | [8%, 24%] | FDA 2000-2020 |
| `phase3_clinical` | 35% | [26%, 45%] | FDA 2000-2020 |
| `drug_approval` | 10% | [5%, 18%] | FDA 2000-2020 |
| `startup_5yr` | 10% | [5%, 18%] | CB Insights |
| `replication` | 40% | [31%, 50%] | OSF 2015 |
| `general` | 50% | [32%, 68%] | Uninformative |

### 2.3 Prior Uncertainty Factor

The uncertainty factor reflects effective sample size:

- $n_{eff} < 10$: High uncertainty (factor = 2.0)
- $n_{eff} < 50$: Moderate uncertainty (factor = 1.5)
- $n_{eff} < 200$: Low uncertainty (factor = 1.2)
- $n_{eff} \geq 200$: Minimal uncertainty (factor = 1.0)

---

## 3. Hierarchical Correlation Correction

### 3.1 The "Deadly Product" Problem

Naive Bayesian updating multiplies likelihood ratios:

$$\text{Posterior odds} = \text{Prior odds} \times LR_1 \times LR_2 \times \ldots \times LR_n$$

This assumes **independence**. When evidence is correlated (same lab, same authors, same methodology), this leads to exponential overconfidence.

### 3.2 Hierarchical Structure

PRISM v2.2 clusters evidence by:
- **Author overlap**
- **Study design**
- **Source/journal**

Correlation parameters:
- **Within-cluster:** $\rho_{within} = 0.6$
- **Between-cluster:** $\rho_{between} = 0.2$

### 3.3 Design Effect Adjustment

The Design Effect (DEFF) inflates variance:

$$DEFF = 1 + (n - 1) \times \rho$$

Effective sample size:

$$n_{eff} = \frac{n}{DEFF}$$

Adjusted log-likelihood ratio:

$$\ln(LR_{adj}) = \frac{\ln(LR)}{\sqrt{DEFF}}$$

---

## 4. Meta-Analysis (REML + Hartung-Knapp)

### 4.1 REML Estimation

Restricted Maximum Likelihood (REML) estimates between-study variance $\tau^2$:

$$\hat{\tau}^2_{REML} = \arg\max_{\tau^2} \left[ -\frac{1}{2} \sum_i \ln(v_i + \tau^2) - \frac{1}{2} \sum_i \frac{(y_i - \hat{\theta})^2}{v_i + \tau^2} \right]$$

### 4.2 Hartung-Knapp Adjustment

Standard random-effects CI underestimates uncertainty. The Hartung-Knapp adjustment uses a t-distribution:

$$CI = \hat{\theta} \pm t_{n-1, 1-\alpha/2} \times \sqrt{q_{HK} \times \widehat{Var}(\hat{\theta})}$$

Where:

$$q_{HK} = \frac{\sum_i w_i (y_i - \hat{\theta})^2}{n - 1}$$

### 4.3 Heterogeneity (I²)

$$I^2 = \max\left(0, \frac{Q - (k-1)}{Q} \times 100\%\right)$$

Interpretation:
- $I^2 < 25\%$: Low heterogeneity
- $25\% \leq I^2 < 75\%$: Moderate heterogeneity
- $I^2 \geq 75\%$: High heterogeneity

---

## 5. P-Curve Analysis for Publication Bias

### 5.1 Motivation

Publication bias inflates effect sizes. P-curve tests whether significant p-values show evidential value or p-hacking.

### 5.2 Method

For significant p-values ($p < 0.05$), compute PP-values:

$$PP_i = \frac{p_i}{\alpha}$$

Under the null (no true effect), PP-values are uniform. Under a true effect, they are right-skewed (more small p-values).

### 5.3 Interpretation

- **Right-skewed (binomial test):** Evidential value present
- **Left-skewed:** P-hacking suspected
- **Flat:** Inconclusive

---

## 6. Kalman Filtering for Temporal Evidence

### 6.1 State-Space Model

For evidence arriving over time, Kalman filtering provides optimal updating:

**Prediction:**
$$\hat{L}_{t|t-1} = \hat{L}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + Q$$

**Update:**
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R_t}$$
$$\hat{L}_{t|t} = \hat{L}_{t|t-1} + K_t \times \ln(LR_t)$$
$$P_{t|t} = (1 - K_t) \times P_{t|t-1}$$

Where:
- $Q$: Process variance (drift in true credence)
- $R_t$: Measurement variance (evidence uncertainty)
- $K_t$: Kalman gain

---

## 7. Optimizer's Curse Correction

### 7.1 The Problem

When selecting the "best" hypothesis from $n$ candidates, the winner's estimate is biased upward:

$$E[\hat{\theta}_{max} - \theta_{max}] \approx \sigma \sqrt{2 \ln n}$$

### 7.2 Correction

The corrected posterior shrinks toward the prior:

$$L_{corrected} = L_{raw} - \sigma \sqrt{2 \ln n}$$

Default $\sigma = 0.15$ based on typical posterior standard deviation.

---

## 8. Independence Detection

### 8.1 Multi-Factor Scoring

Evidence independence is assessed via:

1. **TF-IDF Semantic Similarity (40%):** Content overlap
2. **Author Overlap (35%):** Shared authors (Jaccard index)
3. **Source Overlap (25%):** Same journal/source

### 8.2 Effective Sample Size

Combined correlation:

$$\rho_{avg} = 0.4 \times \rho_{semantic} + 0.35 \times \rho_{author} + 0.25 \times \rho_{source}$$

$$n_{eff} = \frac{n}{1 + (n-1) \times \rho_{avg}}$$

---

## 9. Uncertainty Decomposition

### 9.1 Sources of Uncertainty

1. **Statistical Uncertainty:** Width of credible interval
2. **Prior Uncertainty:** Width of reference class prior CI
3. **Model Uncertainty:** Based on evidence independence ratio

### 9.2 Total Uncertainty

$$\sigma_{total} = \sqrt{\sigma_{stat}^2 + \sigma_{prior}^2 + \sigma_{model}^2}$$

### 9.3 Reliability Threshold

Analysis is flagged as "unreliable" if:

$$\sigma_{total} > 0.30$$

---

## 10. Evidence Quality Weighting

### 10.1 Study Design Likelihood Ratios

| Study Type | LR+ (supports) | LR- (contradicts) |
|------------|----------------|-------------------|
| Meta-analysis | 5.4 | 0.19 |
| Systematic review | 4.8 | 0.21 |
| RCT | 2.3 | 0.43 |
| Cohort | 1.9 | 0.53 |
| Case-control | 1.5 | 0.67 |
| Observational | 1.2 | 0.83 |
| Expert opinion | 1.1 | 0.91 |

### 10.2 Sample Size Adjustment

Larger samples provide stronger evidence:

$$LR_{adj} = LR^{f(n)}$$

Where $f(n) = \min(1.2, 0.7 + 0.3 \times \log_{10}(n+1) / 3)$

---

## 11. Diagnostic Metrics (Beta-Distributed)

Sensitivity and specificity are modeled as Beta distributions:

$$\text{Sensitivity} \sim \text{Beta}(\alpha_s, \beta_s)$$
$$\text{Specificity} \sim \text{Beta}(\alpha_p, \beta_p)$$

This allows Monte Carlo estimation of LR confidence intervals.

---

## 12. Risk-Aware Utility (Inherited from v1.0)

PRISM supports Constant Relative Risk Aversion (CRRA) for decision-making:

$$U_{CRRA}(x) = \frac{(x + S)^{1-\gamma}}{1-\gamma}$$

Where:
- $\gamma$: Risk aversion coefficient
- $S$: Shift factor for negative utilities

---

## 13. Summary of v2.2 Improvements

| Feature | v1.0 | v2.2 |
|---------|------|------|
| Prior | Point estimate | Beta distribution with CI |
| Correlation | Manual adjustment | Hierarchical clustering |
| Meta-analysis | DerSimonian-Laird | REML + Hartung-Knapp |
| Publication bias | Not included | P-curve analysis |
| Multiple comparison | Not included | Optimizer's curse correction |
| Temporal evidence | Not included | Kalman filtering |
| Independence | Jaccard similarity | TF-IDF + Author + Source |

---

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. Patterson, H. D., & Thompson, R. (1971). Recovery of inter-block information when block sizes are unequal. *Biometrika*, 58(3), 545-554.
3. Hartung, J., & Knapp, G. (2001). A refined method for the meta-analysis of controlled clinical trials. *Statistics in Medicine*, 20(24), 3875-3889.
4. Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014). P-curve: A key to the file-drawer. *Journal of Experimental Psychology: General*, 143(2), 534.
5. Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263-291.
6. Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. *Science*, 349(6251), aac4716.

---

**PRISM v2.2 - Rigorous Bayesian Hypothesis Evaluation**

*Copyright © 2025 Dr. Aneesh Joseph*
