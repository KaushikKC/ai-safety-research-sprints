# Predictions: logged before any model run

**Logged:** 2026-10-30, before downloading the model or extracting any activations.
**Author note:** Claude drafted these from PLAN.md, and the researcher approved them as the pre-registration for this run. Nothing below may be edited after results exist. Revisions go in a new dated section.

**Model:** Qwen2.5-1.5B-Instruct (28 layers). The probe is logistic regression on mean-pooled response-token residual activations.

| # | Prediction | Confidence |
|---|---|---|
| P1 | In-distribution held-out AUC on Set A (instructed pairs, teacher-forced) is > 0.95 at the best layer. The best layer is in the middle third (layers 10–18). | 70% |
| P2 | The layer-0 (embedding) probe is near chance (< 0.6) on Set A, because the response tokens are identical across classes. On Set B it is clearly above chance (> 0.75), because the response tokens differ. | 80% |
| P3 | A probe trained on Set A transfers to Set B (on-policy lies about cities) with AUC at least 0.10 lower than Set A's in-distribution AUC, but still > 0.7. | 55% |
| P4 | Transfer from Set A to the roleplaying scenarios (teacher-forced honest vs deceptive completions, no instruction to lie) is weak: best-layer AUC < 0.75. | 60% |
| P5 | The Set-A probe scores "complied-honestly" responses (deceive instruction present, but the true answer given) higher than honest responses. Their AUC vs honest is > 0.65, meaning the probe partly reads the instruction. | 65% |
| P6 | The Set-A probe separates false from true Geometry-of-Truth city statements, with no deception framing, at AUC > 0.7. In other words, "deception" is entangled with "falsehood". | 50% |

---
## Addendum: 2026-10-30, after pilots, still before any probe was trained
Qwen2.5-1.5B-Instruct would not lie on command (`notes/pilot_compliance.md`), so Set B is now **teacher-forced**.
Its cells are honest prompt + true answer, deceive prompt + false answer, deceive prompt + true answer ("complied-honestly"), and honest prompt + false answer ("honest mistake").
P2, P3 and P5 are kept as written and now refer to this Set B. "On-policy" in P3 no longer applies.
No prediction text or confidence was changed.
