# 04 · Generalization and evaluation: overfitting, skew, imbalance, calibration

## Overfitting

**Symptom:** e.g. 99% on training data, 70% on new data.

The model memorized the training examples, noise included, instead of learning the general pattern. Like a student who memorized last year's exam answers: perfect on those questions, lost on new ones. You detect it by comparing training performance to **held-out** data the model never trained on, before deployment.

**Helps:** dropout, weight decay, more training data, early stopping, simpler models.

## Training-serving skew (a different problem)

The model sees different inputs in production than it saw in training. The model may be fine; the data going in does not match.

- **Symptom:** looks good offline (including on held-out data) but worse live, even on the same kind of data.
- **Typical causes:** a feature computed one way in training and another in production (different time windows, units, null handling); stale or late-arriving data at serving time; a preprocessing step that exists in only one of the two paths.
- **Fixes:** share the same feature code between training and serving; log live features and compare their distributions to the training data.

| | Overfitting | Training-serving skew |
|---|---|---|
| Offline held-out performance | Bad | Good |
| Cause | Model memorized noise | Pipeline feeds different inputs |
| Fix | Regularize, more data, early stop | Align feature code, monitor distributions |

## Class imbalance

**Problem:** one class is rare. With 10 frauds in 1,000 transactions, a model that says "not fraud" every time scores 99% accuracy and catches nothing. Training struggles because the loss is an average, so the rare class barely moves it.

**Fixes work at four levels:**

1. **Measure the right thing.** Use **precision** (of the transactions flagged, how many were truly fraud?), **recall** (of all real fraud, how many were caught?), F1, and PR-AUC instead of accuracy.
2. **Change the data.** Undersample the majority, oversample the minority, or create synthetic examples (SMOTE).
3. **Change the loss.** Class weights (a fraud mistake counts, say, 50x more) or focal loss (shrinks the loss on easy examples so training focuses on hard, rare ones).
4. **Change the decision.** The model outputs a probability and you pick a cutoff. Lowering it raises recall and lowers precision. Choose by the business cost of each mistake.

**Catch:** resampling and weighting distort predicted probabilities, which leads to calibration.

## Calibration

A weather forecaster who says "70% chance of rain" on 100 days should see rain on about 70 of them. A model is **calibrated** when its predicted probabilities behave the same way.

![Reliability diagram](../diagrams/11-reliability-diagram.svg)

- **Not the same as ranking quality.** AUC only checks order. A model can rank perfectly while saying 0.99 for everything it likes: great AUC, terrible calibration.
- **When it matters:** whenever the number itself is used, not just the order: ad pricing on click probability, expected-cost decisions, combining scores from several models. For pure ranking it matters much less.
- **What breaks it:** neural nets tend to be overconfident; imbalance fixes (downsampling the majority 10x makes probabilities run too high); drift between training and live data.
- **How to check:** a reliability diagram (above), or the **Brier score**: the average squared gap between predicted probability and the actual 0/1 outcome (lower is better).
- **How to fix:** keep the model, learn a small correction on held-out data:
  - **Platt scaling:** fit an S-shaped (logistic) curve from score to true probability.
  - **Isotonic regression:** flexible, step-like mapping; needs more data.
  - **Temperature scaling:** divide the raw outputs by one number T > 1, softening overconfidence without changing the ranking.

Next: [05 · Fine-tuning](05-fine-tuning.md)
