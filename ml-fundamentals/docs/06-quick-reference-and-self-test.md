# 06 · Quick reference and self-test

## Cheat sheet

| Concept | One line |
|---|---|
| Feature | An input the model sees |
| Label | The answer to predict (for an LLM: the next token in the text) |
| Loss | A number for how wrong the prediction was |
| Batch / step / epoch | Group of examples / one weight update / one full pass over the data |
| Training | Features + labels adjust the weights |
| Inference | Features only, frozen weights, one forward pass |
| MLP | Stacked layers of weighted sums + nonlinearity; every unit connects to the next layer |
| Nonlinearity (ReLU) | Without it, stacked layers collapse to one linear function |
| Batch norm | Normalize one feature across the batch (CNNs) |
| Layer norm | Normalize one example across its features (transformers) |
| SGD | Step downhill by the current slope |
| Momentum | Keep part of the previous direction |
| Adam | Momentum + per-weight step size |
| Dropout | Randomly switch off units in training; off at inference |
| Weight decay | Pull weights toward zero to fight overfitting |
| Residual connection | Add a layer's input to its output; helps deep nets train |
| Overfitting | Great on training data, worse on held-out data |
| Training-serving skew | Model is fine, but production inputs differ from training inputs |
| Class imbalance | Rare positive class; accuracy misleads, use precision/recall/PR-AUC |
| Calibration | Predicted probabilities match observed frequencies |
| Fine-tuning | Continue training a pretrained model on a task-specific dataset, gently |
| LoRA | Train small adapters; keep the base weights frozen |

## Common mix-ups

- **Layer norm vs batch norm:** layer norm looks at a single example across its features; batch norm looks at one feature across the batch.
- **Dropout at inference:** off. Leaving it on makes the same input give different answers.
- **LLM labels:** the next token in the training text. User queries only appear at inference.
- **Overfitting vs skew:** overfitting shows up on held-out data offline; skew shows up only in production.
- **Norm layers** smooth activations inside the network, not the raw input features.
- **Optimizers** are update rules for weights; the learning rate is a setting they use, and the schedule changes it over time.
- **Adam's per-weight step** is driven by gradient magnitude: large gradients get damped, small ones boosted.
- **AUC vs calibration:** AUC measures ranking only; a model can rank well and still output badly scaled probabilities.

## Self-test

Answer out loud in 2-3 minutes each, then check against the notes.

1. In a house-price table with square feet, bedrooms, age and price: which is the label and which are the features?
2. During inference, which of these still happen: computing the loss, updating weights, running features through the model?
3. What does batch norm normalize over, what does layer norm normalize over, and which do transformers use? Why does batch norm break at batch size 1?
4. Adam is "momentum plus what"? Why does that help on a narrow valley?
5. What does dropout do during training, and at inference? What goes wrong if it stays on in production?
6. A model scores 99% on training data and 70% on new data. Name the problem and two fixes.
7. Why does an MLP need a nonlinearity? What happens to ten stacked layers without one?
8. In an LLM, what is the label and where does it come from?
9. A fraud model looks good offline but is worse live on the same kind of transactions. Overfitting or skew? Why?
10. Match each to its job: batch/layer norm, Adam/SGD/momentum, dropout/weight decay, residual connection. Jobs: keep signals flowing through deep nets; decide how weights are updated; keep activations well-scaled; reduce overfitting.
11. Is the loss computed at every step? What are a batch, a step and an epoch?
12. Why is accuracy misleading under class imbalance? Name four fixes at four levels.
13. What does it mean for a model to be calibrated? How can a model have high AUC and bad calibration? Name three ways to fix it.
14. What is fine-tuning? Name three approaches and the main risk.

## Topics worth studying next

- **Bias-variance tradeoff:** diagnosing each; regularization (L1 vs L2, dropout, early stopping).
- **Loss functions:** MSE vs cross-entropy; why logistic regression uses cross-entropy; the gradient sketch.
- **Evaluation metrics:** precision/recall, ROC-AUC vs PR-AUC (and when each misleads under imbalance), NDCG/MRR for ranking, calibration.
- **Data problems:** leakage, train/validation/test splits, training-serving skew.
- **Backprop and optimization:** chain-rule intuition, vanishing/exploding gradients, ReLU vs sigmoid/tanh.
- **Trees vs neural nets:** random forests vs gradient boosting; when each wins on tabular data.
- **Embeddings and two-tower models:** negative sampling, contrastive loss, approximate nearest neighbor retrieval.
- **Transformers:** attention (queries, keys, values), why layer norm and residuals are there, positional information.
- **Offline vs online evaluation:** why offline metrics can disagree with A/B results, position bias, cold start, feedback loops.

Practice structure for explaining any concept: the intuition in one sentence, a tiny example, then the main failure mode.
