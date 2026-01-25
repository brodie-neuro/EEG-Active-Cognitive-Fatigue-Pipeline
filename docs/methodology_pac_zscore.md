# Why Surrogate Z-Scores for PAC?

You asked for a deep dive into the **Surrogate Z-Score** method for Phase-Amplitude Coupling (PAC). This is a critical methodological choice.

## 1. The Problem with Raw PAC
Raw PAC values (Modulation Index) are sensitive to:
1.  **Overall power differences:** A subject with naturally high gamma power will have higher raw PAC, even if the *coupling* isn't stronger.
2.  **Noise floor:** 1/f noise creates spurious "coupling" just by chance.
3.  **Signal length:** Shorter trials produce biased PAC values.

If you just used raw Modulation Index, you couldn't tell if an increase in PAC was real or just because the person's gamma power increased (which happens with muscle tension!).

## 2. The Surrogate Solution (How It Works)
We create a "Chance Distribution" for every single trial.

*   **Real Signal:** Phase (Theta) from Trial 1 + Amplitude (Gamma) from Trial 1.
    *   *Calculate PAC_real.*
*   **Surrogate (Fake) Signal:** Phase (Theta) from Trial 1 + Amplitude (Gamma) from **Trial 5** (randomly swapped).
    *   *Calculate PAC_fake.*
    *   *Repeat 200 times.*

This breaks the temporal relationship. Any PAC found in the surrogates is purely chance.

## 3. The Z-Score Calculation
We compare the Real PAC to the distribution of 200 Fake PACs:

$$ Z = \frac{\text{PAC}_{real} - \text{Mean}(\text{PAC}_{fake})}{\text{Deg}(\text{PAC}_{fake})} $$

*   **Z = 0:** No coupling (same as chance).
*   **Z = 2.0:** Significant coupling (2 standard deviations above chance).

## 4. Is This Robust? (Yes, Very)
*   **Normalization:** It puts everyone on the same scale (Standard Deviations).
*   **Correction:** It automatically subtracts the bias caused by high gamma power (muscle). If muscle increases gamma power, it increases *both* Real and Fake PAC equally, so the Z-score cancels it out.
*   **Statistics:** It gives you a metric that is directly interpretable as "strength of coupling above chance."

## Conclusion
Use **Modulation Index (Tort et al., 2010)** as the core metric, but **ALWAYS** apply Surrogate Z-Scoring. It is the gold standard for comparing PAC across conditions/groups.
