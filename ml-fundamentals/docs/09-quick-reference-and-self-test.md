# 09 · Quick reference and self-test

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
| Transformer | Stack of blocks; each block = attention (mix across tokens) + MLP (refine each token), with layer norm and residuals |
| Query / Key / Value | Query = what I look for; Key = what I offer for matching; Value = the content passed on |
| Attention | Score query against all keys (q·k/√d), softmax into weights, take the weighted sum of values |
| Multi-head attention | Several attentions in parallel, each with its own Q, K, V weights |
| Causal mask | Blocks looking at future tokens; lets decoder LLMs train on all positions in parallel |
| Position encoding / RoPE | Injects token order, since attention alone ignores it |
| KV cache | Stores past keys and values during generation so they are not recomputed |
| Logits | Raw scores over the vocabulary, before softmax |
| Temperature | Divide logits by T before softmax: T < 1 sharper, T > 1 flatter, T → 0 greedy; ranking unchanged |
| Top-k / top-p | Sample only from the k most likely tokens / the smallest set adding up to probability p |
| Linear regression | Weighted sum of features; fit by minimizing squared error (closed form or gradient descent) |
| Logistic regression | Linear score through a sigmoid; log-odds are linear; trained with cross-entropy; linear boundary |
| Cross-entropy | Negative log-likelihood; gradient w.r.t. the score is (p − y); punishes confident wrong answers |
| Decision tree split | Greedy choice of the question that most reduces impurity (Gini, entropy, variance) |
| Random forest | Many deep trees on bootstrap samples, averaged: cuts variance |
| GBDT | Shallow trees in sequence, each fit to the residuals (negative gradient), summed with a learning rate: cuts bias |
| LambdaMART | GBDT trained with a ranking loss; a long-standing search-ranking workhorse |
| Bias / variance | Error from too-simple vs too-sensitive models; compare training and validation error to tell which |
| L2 / L1 | Penalize w² (shrinks smoothly) / \|w\| (drives weights to exactly zero) |
| Vanishing gradient | Many derivatives below 1 multiplied together; fixed by ReLU/GELU, residuals, normalization, good init |
| Precision / recall / F1 | Of flagged, how many right / of real, how many caught / harmonic mean of the two |
| ROC-AUC vs PR-AUC | Ranking quality over all thresholds; prefer PR-AUC when positives are rare |
| NDCG / MRR | Graded relevance discounted by position / reciprocal rank of the first relevant result |
| Data leakage | Information at training time that won't exist at prediction time |

## Common mix-ups

- **Layer norm vs batch norm:** layer norm looks at a single example across its features; batch norm looks at one feature across the batch.
- **Dropout at inference:** off. Leaving it on makes the same input give different answers.
- **LLM labels:** the next token in the training text. User queries only appear at inference.
- **Overfitting vs skew:** overfitting shows up on held-out data offline; skew shows up only in production.
- **Norm layers** smooth activations inside the network, not the raw input features.
- **Optimizers** are update rules for weights; the learning rate is a setting they use, and the schedule changes it over time.
- **Adam's per-weight step** is driven by gradient magnitude: large gradients get damped, small ones boosted.
- **AUC vs calibration:** AUC measures ranking only; a model can rank well and still output badly scaled probabilities.
- **Attention vs MLP:** attention mixes information across tokens; the MLP transforms each token separately.
- **Temperature:** changes the spread of the probabilities, never the ranking, and does not change the model's weights.
- **Training vs generation:** training is parallel over all positions (causal mask); generation is one token at a time.
- **Logistic regression is a classifier**, not a regressor; it models the log-odds linearly.
- **Random forest vs GBDT:** forest = parallel deep trees averaged (variance); GBDT = sequential shallow trees each correcting the last (bias).
- **GBDT cannot extrapolate** beyond the target range it saw in training, and needs early stopping to avoid overfitting.
- **Weight decay vs L2:** the same for SGD, different for Adam; that is why AdamW exists.
- **Accuracy vs PR-AUC:** under heavy imbalance, accuracy and ROC-AUC can look great while precision is poor.
- **Validation vs test:** tune on validation, report on test once; fit preprocessing on training data only.
- **Loss vs metric:** the loss must be differentiable; the metric is what you report.

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
15. Explain attention with queries, keys and values. Why divide by √d, and why multiple heads?
16. What are the parts of a transformer block and what does each do? Why layer norm instead of batch norm?
17. What does the causal mask do, and how does it let an LLM train on next tokens in parallel?
18. What does temperature do to a model's output probabilities? How does it relate to temperature scaling in calibration?
19. Compare linear and logistic regression. Why is logistic regression trained with cross-entropy, and what is its gradient?
20. Explain GBDT in 60 seconds. How does it differ from a random forest, and when would you prefer each?
21. Diagnose from the numbers: training error 2%, validation error 15%. What do you try? What if both were 15%?
22. L1 vs L2, and weight decay vs L2 with Adam.
23. Precision, recall, F1, ROC-AUC and PR-AUC: define each and say when ROC-AUC misleads.
24. What is data leakage? Give three examples and how to prevent them.

## Topics worth studying next

These are not yet covered in these notes.

- **CNNs and RNNs/LSTMs:** convolutions, pooling, why they suit images and sequences; how transformers replaced RNNs.
- **Embeddings and two-tower retrieval in depth:** negative sampling, contrastive loss, approximate nearest neighbor search, hard negatives.
- **A/B testing statistics:** randomization, p-values, power and sample size, guardrail metrics, novelty effects, interference.
- **Recommendation and ranking systems:** candidate generation vs ranking, cold start, diversity, exploration vs exploitation, bandits.
- **Interpretability:** feature importance, SHAP, partial dependence, attention is not explanation.
- **Generative models beyond LLMs:** VAEs, GANs, diffusion at a conceptual level.
- **Efficient attention and long context:** FlashAttention, sparse attention, grouped-query attention.
- **Production ML:** drift detection, monitoring, feature stores, retraining cadence, shadow deployments.
- **Probability and statistics refreshers:** Bayes' rule, maximum likelihood vs MAP, the central limit theorem, confidence intervals.

Practice structure for explaining any concept: the intuition in one sentence, a tiny example, then the main failure mode.
