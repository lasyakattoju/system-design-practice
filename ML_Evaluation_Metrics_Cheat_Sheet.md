# ML Model Evaluation: Metrics Cheat Sheet & Interview Q&A

A reference for how machine learning scientists evaluate model performance: what each metric means, how it is computed, when to use it, and a real-world tech-industry example. Includes a scenario-to-metric guide and spoken-style interview answers.

## Contents

1. [Metrics reference](#metrics-reference)
   - [Classification](#classification)
   - [Regression](#regression)
   - [Ranking / Recommendation](#ranking--recommendation)
   - [NLP / Generation](#nlp--generation)
   - [NER / Sequence Labeling](#ner--sequence-labeling)
   - [Clustering](#clustering)
   - [Validation Methodology](#validation-methodology)
   - [Diagnostics](#diagnostics)
   - [Robustness](#robustness)
   - [Online Evaluation](#online-evaluation)
   - [Operational](#operational)
2. [Which metric when](#which-metric-when)
3. [Interview Q&A](#interview-qa)
   - [Classification metrics](#classification-metrics)
   - [Calibration](#calibration)
   - [Regression metrics](#regression-metrics)
   - [Ranking metrics](#ranking-metrics)
   - [NLP / LLM evaluation](#nlp--llm-evaluation)
   - [NER / structured extraction](#ner--structured-extraction)
   - [Validation methodology](#validation-methodology-1)
   - [Diagnostics](#diagnostics-1)
   - [Online evaluation](#online-evaluation-1)
   - [Operations & monitoring](#operations--monitoring)
   - [Putting it together](#putting-it-together)

---

## Metrics reference

### Classification

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Accuracy** | Fraction of all predictions that are correct. | `(TP + TN) / (TP + TN + FP + FN)` | Use when classes are balanced and every error costs the same. Misleading on imbalanced data: always predicting 'not fraud' gives 99.9% accuracy at a 0.1% fraud rate. | Photo-tagging classifier in a gallery app where the classes (dog, beach, food, etc.) are roughly evenly distributed. |
| **Precision** | Of the items the model predicted positive, the fraction that are truly positive. | `TP / (TP + FP)` | Optimize when false positives are costly. Trades off against recall as you raise the threshold. | Email spam filter: marking a legitimate email (job offer, invoice) as spam is worse than letting a little spam through. |
| **Recall (sensitivity, TPR)** | Of the actual positives, the fraction the model found. | `TP / (TP + FN)` | Optimize when false negatives are costly. Trades off against precision. | Content-moderation system catching harmful posts, or payments fraud detection where a missed fraud means direct financial loss. |
| **F1 score** | Harmonic mean of precision and recall: one number that balances both. | `2 * P * R / (P + R). F-beta weights recall (beta>1) or precision (beta<1) more.` | Good single number for imbalanced problems. Ignores true negatives and hides which of P or R is weak, so always report both too. | Support-ticket auto-routing: a misrouted ticket and a ticket that never gets routed both slow down resolution. |
| **ROC-AUC** | Probability that the model scores a random positive higher than a random negative. Area under the TPR vs FPR curve across all thresholds. | `Area under the curve of TPR (recall) vs FPR (FP / (FP + TN)). 0.5 = random, 1.0 = perfect.` | Threshold-independent, good for comparing models. Can look overly optimistic when positives are very rare because FPR is diluted by the huge negative class. | Comparing candidate credit-risk models at a fintech before the business picks an approval cutoff. |
| **PR-AUC (average precision)** | Area under the precision vs recall curve. Summarizes the precision/recall tradeoff over all thresholds. | `Area under precision vs recall curve (or average precision). Random-model baseline equals the positive rate.` | Prefer over ROC-AUC when positives are rare, since it focuses on performance on the positive class. | Fraud detection at a payments company with ~0.1% fraud rate, where ROC-AUC can look great while precision is poor. |
| **Log loss (cross-entropy)** | Measures the quality of predicted probabilities. Heavily penalizes confident wrong predictions. | `-(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]. Lower is better.` | Use when the probability itself is consumed downstream, not just the label. A single very confident mistake can dominate the score. | Click-through-rate prediction in an ads auction, where the predicted probability feeds directly into the bid. |
| **Calibration (reliability curve)** | Whether predicted probabilities match observed frequencies: among everything scored 0.8, about 80% should be positive. | `Bin predictions by predicted probability, plot mean predicted probability vs actual positive rate per bin. The diagonal is perfect calibration.` | A model can have high AUC and still be badly calibrated. Fix with Platt scaling or isotonic regression. Essential whenever the score is used as a real probability. | '70% chance this user will book' scores used to decide how much to spend on a retargeting offer; medical or insurance risk scores. |
| **ECE (Expected Calibration Error)** | Average gap between model confidence and actual accuracy, weighted across confidence bins. | `sum over bins of (n_bin / N) * \|accuracy_bin - confidence_bin\|` | Single-number summary of the reliability curve. Sensitive to the number and type of bins. | Voice assistant intent classifier: checking that '90% confident' really means correct 90% of the time before deciding when to fall back to a human or ask a clarifying question. |
| **Brier score** | Mean squared error between predicted probability and the actual 0/1 outcome. Captures both calibration and sharpness. | `(1/N) * sum (p - y)^2. Range 0 to 1, lower is better.` | Proper scoring rule, less punishing of extreme errors than log loss. Compare against a baseline that always predicts the base rate. | Predicting the probability a flight will be delayed in a travel app, where users see the probability. |

### Regression

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **MAE (Mean Absolute Error)** | Average absolute difference between prediction and truth, in the target's own units. | `(1/N) * sum \|y - y_hat\|` | Easy to explain, robust to outliers. Treats all errors linearly. | ETA prediction in ride-hailing or food delivery: 'off by 3.5 minutes on average'. |
| **RMSE (Root Mean Squared Error)** | Square root of the average squared error. Penalizes large errors more than small ones. | `sqrt( (1/N) * sum (y - y_hat)^2 )` | Use when large misses are disproportionately bad. Sensitive to outliers. Always >= MAE. | Cloud capacity or energy demand forecasting, where a big under-prediction causes an outage. |
| **R-squared (coefficient of determination)** | Fraction of the variance in the target that the model explains. | `1 - SS_residual / SS_total. 1 = perfect, 0 = no better than predicting the mean, can be negative.` | Does not tell you about bias or whether predictions are good in absolute terms. Always rises when you add features on training data (use adjusted R-squared or validation data). | Home-price estimation model: how much of the variation in sale prices is captured by property features. |
| **MAPE (Mean Absolute Percentage Error)** | Average error expressed as a percentage of the true value. | `(100/N) * sum \|y - y_hat\| / \|y\|` | Scale-free, good for comparing across items. Undefined or explodes when true values are near zero, and penalizes over-forecasts more than under-forecasts. | SKU-level sales forecasting across thousands of products in an e-commerce retailer with very different volumes. |
| **Quantile (pinball) loss** | Loss used to train and evaluate predictions of a specific percentile of the distribution instead of the mean. | `For quantile tau: max( tau * (y - y_hat), (tau - 1) * (y - y_hat) )` | Use when over- and under-prediction have different costs, or when you need prediction intervals. | Predicting the 90th-percentile delivery time to power an 'arrives by' promise; setting inventory safety stock. |

### Ranking / Recommendation

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **NDCG@k** | Quality of the top-k ordering. Rewards putting highly relevant items at higher positions and supports graded relevance. | `DCG@k = sum over i=1..k of rel_i / log2(i + 1). NDCG = DCG / ideal DCG (range 0 to 1).` | Standard search/ranking metric. Needs graded relevance labels. Position discount means errors at the top matter most. | Hotel or marketplace search ranking where labels are graded (booking > click > impression) and the best result should be at position 1, not 8. |
| **MAP (Mean Average Precision)** | Mean across queries of average precision, which rewards retrieving all relevant items early. | `AP = average of precision@i at each rank i where a relevant item appears. MAP = mean of AP over queries.` | Binary relevance only. Good when each query has multiple relevant items. | Enterprise document or code search where each query has several relevant results. |
| **MRR (Mean Reciprocal Rank)** | Average of 1 / (rank of the first relevant result). Focuses on how quickly the first right answer appears. | `(1/Q) * sum 1 / rank_first_relevant` | Use when one correct answer is what matters. Ignores everything after the first hit. | Typeahead/autocomplete and FAQ-bot retrieval: did the right suggestion show up at or near the top? |
| **Recall@k** | Fraction of all relevant items that appear in the top k results. | `(# relevant items in top k) / (total # relevant items)` | Key for retrieval stages: if a good item doesn't make the shortlist, later rankers can't recover it. | Candidate-generation (e.g., two-tower embedding retrieval) stage of a recommender feeding a heavier ranking model. |
| **Hit rate@k** | Fraction of users or queries for which at least one relevant item appears in the top k. | `(# queries with >=1 relevant item in top k) / (# queries)` | Simple and interpretable, common in next-item prediction. Does not care about position within k or how many hits. | Streaming service: did the next video the user watched appear in the top 10 recommendations? |
| **Coverage** | Share of the catalog (or users) that the system ever recommends or serves. | `(# distinct items recommended) / (# items in catalog)` | Detects popularity bias and ignored inventory. High accuracy with low coverage means the model keeps showing the same few items. | Marketplace or travel platform making sure long-tail listings and new hosts get exposure. |
| **Diversity** | How different the items in one recommendation list are from each other. | `Average pairwise dissimilarity within a list (intra-list distance), using embeddings, categories, etc.` | Prevents near-duplicate lists. Often traded off against relevance. | Travel search results page that mixes property types and price tiers instead of ten near-identical hotels; news feed variety. |
| **Novelty** | How unfamiliar or non-obvious the recommended items are to the user (often how unpopular they are). | `e.g., mean of -log2(popularity(item)) over recommended items, or share of items the user hasn't seen.` | Encourages discovery. Too much novelty hurts relevance, so track alongside accuracy metrics. | Music discovery playlist that should avoid songs the listener already knows. |

### NLP / Generation

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Perplexity** | How surprised a language model is by real text. Lower = the model assigns higher probability to the actual text. | `exp( average negative log-likelihood per token )` | Only comparable between models with the same tokenizer and dataset. Does not measure helpfulness or factuality. | Tracking language-model pretraining progress or comparing two LMs on the same held-out corpus. |
| **BLEU** | N-gram overlap between generated text and reference text, precision-oriented, with a brevity penalty. | `Geometric mean of modified n-gram precisions (1 to 4-grams) * brevity penalty.` | Weak proxy: rewards matching wording, not meaning. Poor at the sentence level, better as a corpus-level benchmark. | Machine translation system evaluation (Google Translate-style benchmarks). |
| **ROUGE (ROUGE-1/2/L)** | N-gram (or longest-common-subsequence) overlap with reference text, recall-oriented. | `ROUGE-N recall = overlapping n-grams / n-grams in reference. ROUGE-L uses longest common subsequence.` | Weak proxy like BLEU: rewards wording overlap, misses paraphrases and factual errors. | News article or meeting-notes summarization. |
| **BERTScore** | Similarity between generated and reference text using contextual embeddings, so paraphrases score well. | `Match each token to its most similar token (cosine similarity of embeddings) in the other text; aggregate into precision, recall, F1.` | Captures meaning better than word overlap but depends on the embedding model and still needs a reference. | Evaluating paraphrase generation or summaries where wording differs but meaning matches. |
| **Exact Match (EM)** | 1 if the predicted answer string equals the reference answer (after normalization), else 0. | `Fraction of examples with prediction == gold after lowercasing, punctuation and article removal.` | Strict: 'NYC' vs 'New York City' counts as wrong. | Reading-comprehension QA benchmarks like SQuAD; text-to-SQL execution/answer matching. |
| **Token-level F1 (QA)** | Partial-credit score based on token overlap between predicted and gold answers. | `Precision and recall over shared tokens; F1 = harmonic mean, averaged over examples.` | Softer than exact match. Still a lexical measure. | Extractive QA over documents, reported next to EM in SQuAD-style evaluation. |
| **LLM-as-judge / human ratings** | A person or another model scores outputs on quality, helpfulness, correctness, tone, or compares two outputs head-to-head. | `Rubric scores (e.g., 1-5) or pairwise preference win-rate; aggregate with agreement checks.` | Needed when there is no single right answer. Judge models have biases (position, verbosity, self-preference), so calibrate against human labels and measure inter-rater agreement. | Evaluating chatbot or assistant response quality; side-by-side preference tests between two model versions. |
| **Groundedness / faithfulness** | Whether every claim in the generated answer is supported by the provided source or retrieved context. | `(# claims supported by context) / (# claims in answer), judged by humans or an NLI/LLM verifier.` | Separates 'sounds right' from 'is supported by the sources'. Does not check whether the source itself is correct. | RAG customer-support assistant answering strictly from a company's help-center documents. |
| **Hallucination rate** | How often the model states unsupported or false information. | `(# responses containing unsupported/false claims) / (# responses), via human review or automated checks.` | Define what counts (unsupported vs false) up front. Often tracked per risk tier of question. | Medical, legal, or customer-support bots where an invented fact has real consequences. |

### NER / Sequence Labeling

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Entity-level Precision / Recall / F1** | An entity counts as correct only if both its span (boundaries) and its type match exactly. Precision, recall and F1 are computed over entities, not tokens. | `P = correct entities / predicted entities; R = correct entities / gold entities; F1 = harmonic mean.` | Stricter than token-level accuracy: getting 'New York' right but 'City' wrong still fails the entity. Use libraries like seqeval for BIO tags. | Extracting location, dates, amenities and party size from a query like 'pet-friendly cabin near Lake Tahoe next weekend' to drive search filters; extracting drugs and conditions from clinical notes. |

### Clustering

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Silhouette score** | How well each point fits its own cluster compared with the nearest other cluster. Range -1 to 1. | `For each point: (b - a) / max(a, b), a = mean distance to own cluster, b = mean distance to nearest other cluster. Average over points.` | Unsupervised (no labels needed). Favors compact, convex clusters. | Choosing how many customer segments to use for marketing campaigns. |
| **Davies-Bouldin index** | Average, over clusters, of the similarity between a cluster and its most similar other cluster. Lower is better. | `(1/k) * sum over i of max over j != i of (s_i + s_j) / d_ij; s = cluster scatter, d = distance between centroids.` | Cheap, unsupervised. Assumes roughly spherical clusters. | Comparing k-means runs when grouping support tickets or embeddings with no ground-truth labels. |
| **Adjusted Rand Index (ARI)** | Agreement between the discovered clusters and ground-truth labels, corrected for chance. | `Pair-counting agreement between two partitions, adjusted so random labeling scores ~0 and perfect agreement scores 1.` | Needs ground-truth labels. Can be negative if worse than chance. | Checking whether clusters of product listings recover known product categories. |

### Validation Methodology

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Train / validation / test split** | Train fits the model, validation tunes hyperparameters and picks models, test gives a final unbiased estimate. | `Typical splits 70/15/15 or 80/10/10. Test set is used once at the end.` | Repeatedly peeking at the test set leaks information and inflates results. | Standard build of any production model: tune on validation, report test once before launch. |
| **k-fold / stratified cross-validation** | Split data into k folds, train on k-1 and evaluate on the held-out fold, rotate, then average. Stratified keeps class proportions in every fold. | `Average the metric over k folds (commonly k=5 or 10) and report mean +/- std.` | More stable estimate on small data. Expensive for large models, and invalid if data is temporal or grouped (use time/group splits). | Medical imaging or other small labeled datasets; stratified folds so rare disease cases appear in every fold. |
| **Time-based split** | Train on the past and evaluate on the future, never the reverse. | `Choose a cutoff date (or use rolling/expanding windows): train before, validate/test after.` | Prevents leakage of future information. Random splits on temporal data give overly optimistic numbers. | Demand forecasting, fraud detection, and ranking models trained on logs: train on weeks 1-8, test on week 9. |
| **Group split** | Keep all rows belonging to the same entity (user, session, hotel, patient) on the same side of the split. | `Group k-fold / group shuffle split keyed on the entity ID.` | Prevents the model from memorizing an entity and being tested on it. | Recommender or search logs: split by user so one user's history isn't in both train and test. |
| **Baselines** | Simple reference models or heuristics your model must beat to prove real value. | `Examples: majority class, popularity, last value, current production model, logistic regression.` | A fancy model that doesn't beat a simple baseline isn't worth its complexity and serving cost. | Showing a new ranking model beats both 'sort by popularity' and the current production ranker before shipping. |
| **Bootstrap confidence intervals** | Estimate uncertainty in a metric by resampling the test set with replacement many times. | `Resample test set B times (e.g., 1000), compute the metric each time, take the 2.5th and 97.5th percentiles.` | Tells you whether a difference could be noise. Works for almost any metric. | Putting error bars on offline NDCG or F1 before deciding whether a model change is meaningful. |
| **Paired significance tests** | Statistical tests comparing two models on the same examples (paired t-test, Wilcoxon signed-rank, McNemar's test). | `Compute per-example differences (or disagreement counts) between models and test whether the mean difference is zero.` | Paired tests are more powerful than comparing two independent CIs because they control for example difficulty. | Deciding if a +0.3% offline NDCG gain is real before spending an A/B test slot on it. |

### Diagnostics

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Learning curves** | Plots of training and validation performance vs training set size (or training epochs). | `Train on increasing subsets; plot train and validation score for each size.` | Reveals whether more data would help (gap closing) or whether the model is limited (both curves flat). | Deciding whether to invest in labeling more data or switch to a bigger model. |
| **Bias / variance (under/overfitting)** | Bias is error from overly simple assumptions (underfitting). Variance is sensitivity to the specific training data (overfitting). | `High bias: poor on train and validation. High variance: great on train, poor on validation (large gap).` | Guides the fix: bias -> richer model/features; variance -> more data, regularization, simpler model. | Diagnosing why a ranking model's offline NDCG is much lower on validation than training. |
| **Confusion matrix** | Table of predicted vs actual classes showing counts of TP, FP, TN, FN (or per class in multiclass). | `Rows = actual class, columns = predicted class (or vice versa); cell = count.` | Shows which classes are being confused, which aggregate metrics hide. | Intent classification in a virtual assistant: seeing that 'cancel booking' is constantly confused with 'change booking'. |
| **Slice-based evaluation** | Computing metrics separately for subgroups (segments) of the data. | `Group the evaluation set by attribute (region, device, language, new vs returning user, rare class) and compute the metric per group.` | Overall averages hide weak spots. Check slice sizes for statistical reliability. | Finding that a search model performs well overall but poorly for new users, non-English queries, or a specific market. |
| **Fairness checks** | Comparing error rates or outcomes across demographic or protected groups. | `Demographic parity (equal positive rate), equal opportunity (equal TPR), equalized odds (equal TPR and FPR) across groups.` | Metrics can conflict, so choose the one that matches the harm you care about. Requires group labels or proxies. | Loan-approval or hiring-screen models: comparing false-rejection rates across groups. |
| **Feature importance / SHAP** | Which inputs drive predictions, globally and for an individual prediction. SHAP assigns each feature a contribution based on Shapley values. | `Permutation importance, tree-based importance, or SHAP values per prediction (sum of contributions = prediction - baseline).` | Great for debugging and explanations. Correlated features can split or mask importance. | Explaining why a loan was denied, or discovering that a suspicious leaky feature dominates a fraud model. |
| **Ablations** | Remove or disable one component or feature at a time and measure the change in performance. | `Retrain/evaluate with component removed; compare against the full model.` | Shows what each piece actually contributes and whether it justifies its cost. | Testing whether a new embedding or feature is worth its extra serving latency in a ranking model. |
| **Threshold analysis** | Studying how precision, recall, and other metrics change as the decision cutoff moves. | `Sweep the threshold, plot precision/recall/F1/cost vs threshold (PR or ROC curve).` | The default 0.5 is rarely optimal; choose the threshold from business constraints. | Choosing a fraud-alert cutoff that balances fraud caught against the number of cases analysts can review per day. |
| **Cost-sensitive evaluation** | Weighing errors by their real-world cost instead of treating them equally. | `Expected cost = FP * cost_FP + FN * cost_FN (+ other terms); pick model or threshold that minimizes it.` | Connects ML metrics to business impact. Needs reasonable cost estimates. | Weighting a missed fraud ($500 loss) far above a false alarm ($5 manual review). |

### Robustness

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Distribution shift / OOD tests** | Evaluating performance on data that differs from the training distribution (new time period, region, or population). | `Hold out data from a different period/domain/segment; compare metrics with in-distribution performance.` | Reveals brittleness before it appears in production. | A travel-demand or search model trained on pre-pandemic behavior evaluated on post-pandemic behavior. |
| **Noise / adversarial / edge-case tests** | Stress-testing with corrupted, hostile, or rare inputs. | `Inject typos, missing values, slang, adversarial perturbations, prompt injections; test rare or boundary cases.` | Finds failure modes that average-case test sets miss. | Search or chat systems handling typos, slang, very long queries, or prompt-injection attempts. |
| **Behavioral tests (CheckList)** | Unit-test-style checks of model behavior: minimum functionality tests, invariance tests (output shouldn't change), directional expectation tests (output should change in a known direction). | `Write templated test cases per capability, e.g. swap names (INV), add negation (DIR), and measure failure rate.` | Good for catching systematic bugs and bias that aggregate metrics hide. | Sentiment model: swapping 'John' for 'Aisha' must not change the label, but adding 'not' to 'good' should flip it. |
| **Uncertainty quantification** | Measuring how confident the model is and whether it knows when it doesn't know. | `Ensembles, MC dropout, calibrated probabilities, conformal prediction intervals; evaluate via calibration and selective-prediction curves.` | Enables deferring to humans or fallbacks when the model is unsure. | Self-driving or medical-triage systems handing off to a human when confidence is low. |

### Online Evaluation

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **A/B test** | Randomly split live traffic between control (current model) and treatment (new model) and compare business metrics. | `Randomize users, run for a pre-determined duration/sample size, test difference in metrics for significance.` | The gold standard for causal impact. Needs enough traffic, a fixed analysis plan, and attention to novelty effects and sample-ratio mismatch. | Launching a new ranking model to 5% of traffic and measuring bookings per visitor. |
| **CTR (click-through rate)** | Fraction of impressions that receive a click. | `clicks / impressions` | Easy and fast signal but can be gamed (clickbait) and doesn't equal satisfaction. | Ad or search-result ranking quality signal. |
| **Conversion rate** | Fraction of users or sessions that complete a goal (purchase, booking, sign-up). | `conversions / visitors (or sessions)` | Closer to business value than CTR but slower and noisier. | Checkout or booking funnel optimization; recommendation widgets driving purchases. |
| **Revenue metrics** | Money generated per user or session (e.g., revenue per visitor, ARPU). | `total revenue / # users (or visitors, sessions)` | Highly skewed with heavy tails, so need large samples or variance-reduction techniques. | Pricing, ads, or recommendation models where the goal is incremental revenue. |
| **Retention** | Fraction of users who return after a given period. | `users active on day/week N / users in the starting cohort` | Long-horizon metric that captures lasting value, but takes time to measure. | Streaming or subscription personalization where long-term engagement matters more than clicks. |
| **Guardrail metrics** | Metrics that must not get worse even if the primary metric improves. | `Define thresholds up front (e.g., latency p99 < X ms, error rate not up, no drop in customer-service contacts).` | Prevents shipping a model that wins on one metric at the expense of experience or reliability. | A more accurate but heavier model that must not push page latency or error rates up. |
| **Interleaving** | Mix results from two rankers into a single list shown to the same user and see which one's results get more clicks. | `e.g., team-draft interleaving: rankers alternate picking items; credit clicks to the ranker that contributed the clicked item.` | Needs far less traffic than A/B and is highly sensitive for ranking changes. Measures preference, not business metrics. | Netflix-style or search-engine-style comparison of two rankers when traffic is too limited for a full A/B test. |
| **Shadow deployment** | The new model runs on live traffic in parallel but its outputs are not shown to users. | `Log new model predictions next to production, compare offline and check latency/errors.` | Safe validation of real-traffic behavior and performance. Gives no user-response data. | Validating a new model's outputs and latency on production traffic before any user sees them. |
| **Canary rollout** | Release to a small percentage of traffic first and watch for regressions before ramping up. | `Ramp 1% -> 5% -> 25% -> 100% with automated monitoring and rollback criteria.` | Limits blast radius of failures. Not a substitute for a properly powered A/B test. | Releasing a new model to 1% of users and monitoring errors and key metrics before going wide. |
| **Off-policy / counterfactual evaluation** | Estimating how a new policy or model would have performed using logged data from the old policy, without deploying it. | `Inverse propensity scoring (IPS), self-normalized IPS, doubly robust estimators using logged propensities.` | Needs logged propensities and exploration/overlap in the logging policy. High variance when the policies differ a lot. | Estimating a new recommender's performance from logs of the current one without risking a live test. |

### Operational

| Metric | Meaning | How it is computed | When to use / watch out | Real-world tech example |
|---|---|---|---|---|
| **Latency (p50 / p95 / p99)** | Response time for the median request and for the slowest 5% and 1% of requests. | `Percentiles of request-latency distribution.` | Averages hide tail latency, which is what users feel and what often breaks SLAs. | Real-time search or typeahead where a slow tail ruins the user experience. |
| **Throughput** | How many requests or predictions the system handles per unit time. | `Requests (or predictions) per second (QPS) at an acceptable latency.` | Determines hardware and capacity planning. | Sizing infrastructure for peak traffic such as Black Friday or a holiday travel surge. |
| **Memory / cost per inference** | Resources and money consumed per prediction. | `GPU/CPU time, memory footprint, and dollars per 1000 predictions.` | A more accurate model may not be affordable or fit on-device. | Deciding whether an LLM is affordable at millions of calls per day, or whether a model fits on a phone. |
| **Data drift** | The distribution of input features changes over time versus training. | `Compare feature distributions using PSI, KS test, or divergence measures.` | Often an early warning sign even before labels arrive. | A new app version changes how a feature is logged, shifting model inputs. |
| **Concept drift** | The relationship between inputs and the target changes, so the same inputs now mean something different. | `Monitor performance against delayed labels; compare recent vs historical error.` | Data drift doesn't always cause concept drift and vice versa. Needs labels or proxies to detect. | User preferences shift seasonally, or fraudsters adapt their behavior so old patterns stop working. |
| **Performance decay / retraining triggers** | Monitored drops in model quality that signal it's time to retrain or investigate. | `Track key metrics (e.g., weekly AUC) with alert thresholds; schedule or trigger retraining on drift or decay.` | Combine scheduled retraining with metric- and drift-based triggers. | An alert fires when weekly AUC falls below a threshold, triggering an automated retraining pipeline. |

---

## Which metric when

| Scenario | Metrics to reach for | Why |
|---|---|---|
| **Rare-event detection (fraud, abuse)** | PR-AUC, precision, recall, F1 at a chosen threshold, cost-sensitive evaluation | Accuracy and ROC-AUC look inflated when positives are rare. |
| **Probability used downstream (ads, risk, bidding)** | Log loss, Brier score, calibration curve, ECE | The probability value must be trustworthy, not just the ranking. |
| **Search / recommendation ranking** | NDCG@k, MRR, MAP, Recall@k (retrieval stage), then coverage/diversity/novelty | Top positions matter most; retrieval and ranking stages need different metrics. |
| **Forecasting a number (ETA, demand, price)** | MAE, RMSE, MAPE, quantile loss for percentiles | Choose by how costly large vs small errors and over vs under-prediction are. |
| **Translation / summarization** | BLEU, ROUGE, BERTScore, plus human or LLM-judge ratings | Overlap metrics are weak proxies; add semantic and human evaluation. |
| **RAG / LLM assistant** | Groundedness, hallucination rate, answer correctness (EM/F1), retrieval recall@k, LLM/human judge | Evaluate retrieval and generation separately, then end to end. |
| **Entity extraction (NER)** | Entity-level precision, recall, F1 | Span and type must both match. |
| **Unlabeled clustering** | Silhouette, Davies-Bouldin; ARI if labels exist | Pair with a downstream usefulness check. |
| **Comparing two models offline** | Bootstrap CIs, paired tests, slice-based evaluation | Show the gain is real and not concentrated in one segment. |
| **Deciding whether to ship** | A/B test with primary + guardrail metrics; interleaving, shadow, canary beforehand | Offline metrics are proxies; online results decide. |
| **Temporal or per-user data** | Time-based split, group split | Random splits leak the future or the same user. |
| **Model degrading in production** | Data drift (PSI/KS), concept drift, performance decay alerts | Detect early, retrain on triggers. |

---

## Interview Q&A

Answers are written in a spoken style so you can practice saying them aloud. Each question lists a likely follow-up.

### Classification metrics

#### Q1. Why wouldn't you use accuracy for a fraud detection model?

Because fraud is extremely rare, maybe one in a thousand transactions. A model that predicts 'not fraud' every single time would be 99.9% accurate and catch nothing, so accuracy tells me almost nothing about whether the model is useful. Instead I'd look at precision and recall on the fraud class, and PR-AUC to summarize across thresholds. Then I'd pick the operating threshold using the actual cost of a missed fraud versus the cost of reviewing a false alarm.

> **Likely follow-up:** What precision and recall would you target? (Answer: derive from costs and analyst capacity, not a fixed number.)

#### Q2. Explain precision and recall, and when you'd favor one over the other.

Precision asks: of everything I flagged as positive, how much was actually positive. Recall asks: of everything that was actually positive, how much did I catch. I favor precision when false alarms are expensive, like a spam filter hiding a real email. I favor recall when misses are expensive, like cancer screening or abuse detection. They trade off as I move the threshold, so in practice I choose the threshold based on the business cost of each error.

> **Likely follow-up:** How do you combine them? (F1 or F-beta; or choose a threshold by cost.)

#### Q3. ROC-AUC versus PR-AUC: when do you use each?

ROC-AUC measures how well the model ranks a random positive above a random negative, and it's threshold independent. It's great when classes are reasonably balanced. When positives are very rare, ROC-AUC can look excellent because the false positive rate is diluted by the huge number of negatives, even if precision is poor. PR-AUC focuses only on the positive class, so it's much more sensitive in rare-event problems like fraud or abuse. I usually report both and lean on PR-AUC for imbalanced problems.

> **Likely follow-up:** What's the random baseline for each? (ROC-AUC 0.5; PR-AUC equals the positive rate.)

#### Q4. What is F1, and what are its limitations?

F1 is the harmonic mean of precision and recall, so it only gets high when both are high. It's a convenient single number for imbalanced problems. The limitations are that it hides which of the two is weak, it assumes precision and recall matter equally, it ignores true negatives, and it depends on a single threshold. So I report precision and recall alongside it, and use F-beta if one side matters more.

> **Likely follow-up:** When would you use F2 or F0.5? (F2 weights recall more, F0.5 weights precision more.)

### Calibration

#### Q5. What is calibration and why does it matter?

A model is calibrated if its predicted probabilities match reality: among all cases where it says 0.8, about 80% should actually be positive. It matters whenever the probability is used for a decision, like bidding in an ad auction, setting a risk price, or deciding when to defer to a human. A model can have a great AUC and still be badly calibrated, because AUC only cares about ranking. I check it with a reliability curve and ECE or Brier score, and I fix it with Platt scaling or isotonic regression on a held-out set.

> **Likely follow-up:** Platt scaling vs isotonic? (Platt: parametric, good for small data; isotonic: flexible, needs more data, can overfit.)

#### Q6. Log loss versus Brier score: what's the difference?

Both are proper scoring rules that evaluate predicted probabilities. Log loss penalizes confident wrong predictions extremely heavily, because the penalty grows without bound as a wrong prediction approaches certainty. Brier score is just mean squared error on the probability, so it's bounded and gentler on extreme mistakes. I'd use log loss when confident errors are especially dangerous and the probability feeds directly into downstream decisions, and Brier when I want a more robust, interpretable measure.

> **Likely follow-up:** Which is more interpretable to stakeholders? (Brier, bounded 0 to 1, compare to the base-rate baseline.)

### Regression metrics

#### Q7. MAE versus RMSE: how do you choose?

MAE is the average absolute error, so it treats every unit of error the same and is robust to outliers. RMSE squares the errors before averaging, so it punishes large misses much more. I choose RMSE when big errors are disproportionately costly, like under-forecasting capacity and causing an outage. I choose MAE when I want a typical-error number that's easy to explain, like 'ETA is off by about three minutes on average'.

> **Likely follow-up:** What does it mean if RMSE is much larger than MAE? (Heavy-tailed errors; a few large misses dominate.)

#### Q8. What are the problems with MAPE?

MAPE divides by the true value, so it blows up or becomes undefined when actuals are near zero. It also isn't symmetric: it penalizes over-forecasts more than under-forecasts, which can bias models toward under-predicting. It's still useful for comparing across items with very different scales, like thousands of SKUs. For intermittent or low-volume series I'd use something like WAPE or MASE instead.

> **Likely follow-up:** What's WAPE? (Sum of absolute errors divided by sum of actuals; avoids per-item blowups.)

#### Q9. When would you use quantile loss?

When the cost of over-predicting and under-predicting is asymmetric, or I need an interval rather than a point estimate. For example, to promise 'arrives by 6pm' I want the 90th percentile of delivery time, not the mean, because being late is worse than being early. Quantile loss, also called pinball loss, trains and evaluates the model on a chosen percentile. I also check coverage: a 90th percentile prediction should be exceeded about 10% of the time.

> **Likely follow-up:** How do you evaluate prediction intervals? (Coverage and interval width; conformal prediction.)

### Ranking metrics

#### Q10. Explain NDCG and why it's used for ranking.

NDCG measures the quality of the top k results. Each result has a relevance grade, and its gain is discounted by its position, so putting a highly relevant item at rank one is worth much more than at rank eight. I divide by the ideal ordering's score so it's normalized between zero and one and comparable across queries. It's popular in search because it handles graded relevance, like booking being more valuable than a click, and because users mostly look at the top few results.

> **Likely follow-up:** What are its weaknesses? (Needs graded labels, position discount is an assumption, labels are biased by past rankers.)

#### Q11. NDCG versus MRR versus MAP: when do you use each?

MRR cares only about the rank of the first relevant result, so it's ideal when there's essentially one right answer, like QA or autocomplete. MAP uses binary relevance and rewards retrieving all relevant items early, which fits queries with several relevant results. NDCG supports graded relevance and a position discount, which fits most search and recommendation ranking. So the choice depends on whether there's one right answer, many binary relevant answers, or graded relevance.

> **Likely follow-up:** Which would you use for typeahead? (MRR, plus success rate at k and keystrokes saved.)

#### Q12. How do you evaluate a multi-stage recommender or search system?

I evaluate each stage on what it's responsible for. The retrieval or candidate generation stage is judged on recall at k, because if a good item doesn't make the shortlist, no later model can recover it. The ranking stage is judged on NDCG, MRR or MAP over that shortlist. Then I evaluate the end-to-end system with online metrics like conversion and engagement, plus diversity and coverage. Evaluating only end to end makes it hard to tell which stage to fix.

> **Likely follow-up:** What k do you choose for retrieval recall? (Match the size of the candidate set passed to the ranker.)

#### Q13. Beyond accuracy, what else do you measure in a recommender?

Coverage, which is the share of the catalog that ever gets recommended, to detect popularity bias and ignored inventory. Diversity, meaning how different the items are within one list, so users don't see ten near-duplicates. Novelty, meaning how unfamiliar the items are, to support discovery. These often trade off against relevance, so I track them together and set guardrails, rather than optimizing accuracy alone.

> **Likely follow-up:** How do you measure diversity? (Average pairwise distance between items in a list, using embeddings or categories.)

### NLP / LLM evaluation

#### Q14. Why are BLEU and ROUGE considered weak, and what would you use instead?

They measure n-gram overlap with a reference, so they reward matching wording rather than matching meaning. A correct paraphrase can score poorly and a fluent but wrong answer can score well. They're still fine as cheap regression benchmarks, for example in translation or summarization. For better signal I'd add embedding-based metrics like BERTScore, task-specific metrics like factual consistency, and human or LLM-as-judge evaluation.

> **Likely follow-up:** What does BERTScore do differently? (Matches tokens by embedding similarity rather than exact words.)

#### Q15. How would you evaluate a RAG (retrieval-augmented generation) system?

I'd separate retrieval from generation and then evaluate end to end. For retrieval I measure recall at k and ranking quality: did the right passages come back. For generation I measure groundedness, meaning every claim is supported by the retrieved context, plus answer correctness against references and hallucination rate. Then I do end-to-end evaluation with a labeled question set, human review on a sample, and LLM-as-judge at scale. I also slice by question type and difficulty, and monitor in production with user feedback.

> **Likely follow-up:** If answers are wrong, how do you tell if it's retrieval or generation? (Check whether the gold passage was retrieved; if yes, it's generation.)

#### Q16. What are the pitfalls of LLM-as-judge?

Judge models have biases: they favor longer answers, favor the answer shown first, and often favor outputs from their own model family. They can also be inconsistent and miss subtle factual errors. To mitigate, I use clear rubrics, randomize order or evaluate both orders, use pairwise comparisons, and calibrate the judge against a human-labeled sample, reporting agreement. I treat it as a scalable proxy and keep periodic human evaluation as the ground truth.

> **Likely follow-up:** How would you validate the judge? (Compute agreement with human labels, e.g., Cohen's kappa or correlation, on a sample.)

#### Q17. How do you evaluate a hallucination problem?

First I define what counts: unsupported by the provided context, versus factually false in the world. Then I build a test set with questions where the answer is in the context, absent from the context, and adversarial ones. I measure the rate of unsupported or false claims using human review and automated verifiers such as an NLI model or an LLM checking each claim against the source. I also track whether the model appropriately abstains when it doesn't know, since lowering hallucinations by refusing everything isn't a win.

> **Likely follow-up:** How do you balance hallucination reduction with helpfulness? (Track abstention rate and answer coverage alongside hallucination rate.)

### NER / structured extraction

#### Q18. How do you evaluate an NER model, and why is entity-level different from token-level?

Token-level accuracy counts each word separately, which is generous: getting 'New' right and 'York' wrong still earns partial credit and the many non-entity tokens inflate the score. Entity-level evaluation counts an entity correct only if both the span and the type match exactly. I compute precision, recall and F1 over entities, usually per entity type, and look at the confusion between types and boundary errors. For downstream use, like converting a query into search filters, I also measure end-to-end impact on result quality.

> **Likely follow-up:** What about partial matches? (Report strict and relaxed/partial-overlap F1 variants.)

#### Q19. How would you evaluate a natural-language search system that turns queries into structured filters and ranked results?

In layers. First the understanding layer: entity extraction and intent accuracy, with entity-level F1 per type, and filter-level precision and recall because a wrong filter silently removes good results. Second the retrieval and ranking layer: recall at k and NDCG against graded relevance labels, sliced by query type, like short, long, ambiguous or multi-constraint queries. Third online: A/B test on conversion and engagement with guardrails such as latency and zero-result rate. I'd also keep a fixed regression set of tricky queries to catch breakages.

> **Likely follow-up:** How do you get labels? (Click/booking logs with debiasing, plus human-annotated query sets for hard cases.)

### Validation methodology

#### Q20. What is data leakage and how do you prevent it?

Leakage is when information that wouldn't be available at prediction time sneaks into training or evaluation, making offline results look better than reality. Common sources are random splits on time-series data, the same user or session appearing in both train and test, features computed using future data, and preprocessing like scaling fit on the whole dataset. I prevent it with time-based splits, group splits keyed on user or entity, point-in-time correct feature pipelines, and fitting preprocessing only on the training fold.

> **Likely follow-up:** How would you detect leakage? (Suspiciously high performance, one feature dominating importance, a big offline-online gap.)

#### Q21. How do you know a model improvement is real and not noise?

I put uncertainty on the metric. I bootstrap the test set to get confidence intervals, and for comparing two models on the same data I use paired tests, like a paired bootstrap or Wilcoxon, or McNemar's test for classifiers, because pairing controls for example difficulty. I also check the improvement across slices and random seeds. Even then, an offline gain is a hypothesis; the confirmation is an online A/B test.

> **Likely follow-up:** Why paired rather than independent comparison? (Removes variance from example difficulty, more statistical power.)

#### Q22. How do you choose between train/validation/test splits and cross-validation?

With plenty of data I use a simple train, validation and test split: tune on validation and touch the test set once at the end. With small data I use k-fold cross-validation, stratified for classification so rare classes appear in every fold. If the data is temporal I use rolling or expanding time-based splits, and if there are repeated entities I use group k-fold. The key is that the split must mirror how the model will be used in production.

> **Likely follow-up:** What's nested cross-validation for? (Unbiased estimate when also tuning hyperparameters.)

#### Q23. Why do you need baselines?

Because a metric number alone is meaningless without context. A baseline like majority class, popularity, last value, or the current production model tells me how much value the new model adds. If a complex model barely beats a simple heuristic, it may not justify its serving cost and maintenance. I always compare against both a trivial baseline and the incumbent system.

### Diagnostics

#### Q24. How do you diagnose overfitting versus underfitting?

I compare training and validation performance, ideally with learning curves. If both are poor, it's high bias, so underfitting: I'd add model capacity or better features. If training is great but validation is much worse, it's high variance, so overfitting: I'd add data, regularize, simplify the model or use early stopping. Learning curves also tell me whether more data would help: if the curves are converging and flat, more data won't, but if there's still a gap it may.

> **Likely follow-up:** What do you do if validation is better than training? (Check for leakage, regularization/dropout effects, or distribution differences.)

#### Q25. Overall metrics look good, but users complain. What do you do?

I'd slice the evaluation. Averages hide weak spots, so I'd break results down by segment like new versus returning users, region, device, language, query type and rare classes, and look at error clusters and the worst examples. I'd also check whether the offline test set matches real traffic. Then I'd add those slices to a permanent evaluation suite, so the failure is caught before future releases.

> **Likely follow-up:** How do you choose slices? (Business-critical segments, high-volume segments, and known risk areas; check slice sizes.)

#### Q26. How do you evaluate fairness in a model?

I compare error rates and outcomes across groups, using definitions that fit the harm. Demographic parity compares positive rates, equal opportunity compares true positive rates, and equalized odds compares both true and false positive rates. These can't all be satisfied at once, so I pick based on the use case, such as equal opportunity for loan approvals so qualified applicants are treated equally. I also look at calibration across groups and mitigate with data, thresholds or constraints, noting the tradeoffs.

> **Likely follow-up:** Why can't all fairness metrics be satisfied? (Impossibility results when base rates differ across groups.)

#### Q27. How do you pick a decision threshold?

I don't default to 0.5. I sweep thresholds and look at the precision-recall tradeoff, then pick based on cost: expected cost equals false positives times their cost plus false negatives times theirs. I also account for operational constraints, such as how many cases a review team can handle per day. If the base rate shifts, I revisit the threshold, and I check the probabilities are calibrated first.

> **Likely follow-up:** What if costs are uncertain? (Show cost curves across plausible cost ratios and pick a robust threshold.)

#### Q28. What are feature importance and SHAP, and what are the pitfalls?

Feature importance tells me which inputs drive the model. Permutation importance shuffles a feature and measures the performance drop, and SHAP assigns each feature a contribution to each individual prediction based on Shapley values, so I get both local and global explanations. Pitfalls: correlated features can split or hide importance, importance describes the model not causality, and it's sensitive to the data distribution used. I use it for debugging, finding leaky features, and explanations, not as proof of cause.

> **Likely follow-up:** How do you debug a feature that looks too important? (Check for leakage or target proxies.)

### Online evaluation

#### Q29. Offline metrics improved but the A/B test showed no gain. Why?

Several possible reasons. Offline labels may be biased by the old system, since logged clicks only exist for items the old ranker showed. The offline metric may not align with the business goal, like NDCG improving while conversion doesn't. There may be distribution shift between the offline set and live traffic, or the effect may be too small or the test underpowered. There could also be latency or experience regressions offsetting gains. I'd check the metric alignment, run slice analysis, verify the experiment's power and sample ratio, and consider interleaving or counterfactual evaluation for ranking.

> **Likely follow-up:** How would you prevent this next time? (Choose offline metrics that correlate with online results; validate that correlation historically.)

#### Q30. How would you design an A/B test for a new ranking model?

I'd start with a hypothesis and one primary metric tied to the goal, like bookings per visitor, plus secondary metrics and guardrails such as latency, error rate and zero-result rate. I'd randomize at the user level, compute the required sample size and duration from the minimum detectable effect and variance, and run for full weekly cycles to avoid day-of-week bias. I'd check for sample ratio mismatch and novelty effects, fix the analysis plan up front to avoid peeking, and analyze by key segments. Then I'd decide based on the primary metric and guardrails together.

> **Likely follow-up:** What is sample ratio mismatch? (Observed traffic split differs from the intended split, indicating a bug.)

#### Q31. What is interleaving and when would you use it instead of an A/B test?

Interleaving mixes results from two rankers into a single list for the same user, and credits clicks to whichever ranker contributed the clicked item. Because each user sees both rankers, it removes between-user variance and needs far less traffic to detect a preference, often orders of magnitude less. I'd use it for fast iteration on ranking changes when traffic is limited. The downside is it measures preference rather than business outcomes, so I'd confirm winners with an A/B test.

> **Likely follow-up:** What are its limitations? (Only for ranking changes, doesn't capture downstream metrics like revenue.)

#### Q32. What is off-policy evaluation and when is it useful?

It estimates how a new policy or model would perform using logged data from the old policy, without deploying it. The classic approach is inverse propensity scoring: reweight each logged outcome by how likely the new policy would be to take that action versus the logging policy. It's useful when live tests are risky or expensive, such as for recommenders. It requires logged propensities and enough overlap between policies, and it can have high variance, so doubly robust estimators are often used.

> **Likely follow-up:** What breaks it? (No exploration in the logging policy, so the new policy takes actions never logged.)

#### Q33. Explain shadow deployment and canary rollout. Why use them?

In a shadow deployment the new model receives live traffic and produces predictions, but users never see them. I use it to validate outputs, latency and errors under real load without any user risk. In a canary rollout I release to a small percentage of users, like 1%, with monitoring and automatic rollback, then ramp up. Shadow checks safety and behavior, the canary checks real-world impact on a limited blast radius, and the A/B test measures the causal effect.

> **Likely follow-up:** Can shadow mode measure business impact? (No; users don't see its output, so there's no response data.)

### Operations & monitoring

#### Q34. How do you detect and handle model drift in production?

I distinguish data drift, where input distributions change, from concept drift, where the relationship between inputs and target changes. Data drift I monitor with statistics like PSI or KS tests on features and predictions, since it needs no labels. Concept drift I detect via performance on delayed labels or proxy metrics. I set alerts with thresholds and retrain on a schedule plus on triggers, and I investigate the root cause first, since drift can come from a pipeline or logging bug rather than the real world.

> **Likely follow-up:** How do you monitor when labels are delayed? (Prediction distribution, proxy metrics, and periodic labeled samples.)

#### Q35. When might you ship a less accurate model?

When the cost, latency or reliability tradeoffs outweigh the accuracy gain. If a heavier model improves NDCG by 0.2% but adds 80 ms of tail latency, it can hurt conversion and cost more to serve. Real-time systems like typeahead have strict p99 latency budgets, and on-device or high-volume use cases have tight cost and memory limits. I'd quantify the tradeoff, using ablations and A/B tests with latency as a guardrail, and consider distillation or caching to get most of the gain cheaply.

> **Likely follow-up:** How would you reduce a model's cost while keeping quality? (Distillation, quantization, caching, cascades.)

### Putting it together

#### Q36. Walk me through how you'd evaluate a new model end to end before launch.

I'd go in stages. First, define success: the business goal, the primary metric and guardrails, and the cost of different errors. Second, set up a sound offline evaluation: correct splits to avoid leakage, strong baselines, the right metrics, confidence intervals and significance tests. Third, go deeper with error analysis, slice-based and fairness checks, calibration, robustness and edge-case tests. Fourth, validate operationally: latency, throughput and cost. Fifth, de-risk with shadow deployment and a canary, then run an A/B test with guardrails. Finally, monitor drift and decay after launch, with retraining triggers.

> **Likely follow-up:** Which of these steps would you never skip? (Leakage-safe splits, baselines, slice analysis, and an online test.)

#### Q37. How do you handle class imbalance in evaluation and training?

For evaluation, I avoid accuracy and use precision, recall, PR-AUC and F1, evaluated on a test set that reflects the true class distribution, and I'm careful to calibrate and pick thresholds on that distribution. For training, options include class weights, resampling like oversampling the minority or undersampling the majority, and focal loss. If I resample for training, the predicted probabilities become miscalibrated, so I recalibrate before using them as probabilities. And I use stratified splits so rare classes show up in every fold.

> **Likely follow-up:** Why not evaluate on a rebalanced test set? (It misstates real-world precision; metrics must reflect production prevalence.)
