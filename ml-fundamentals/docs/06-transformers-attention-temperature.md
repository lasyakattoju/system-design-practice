# 06 · Transformers, attention and temperature

Running example: an LLM completing "The cat sat on the ..." with the next token.

This chapter builds on earlier ones: the MLP ([02](02-model-families.md)), layer norm and residual connections ([03](03-training-components.md)), and next-token training ([02](02-model-families.md)).

## 1. Why transformers exist

Language needs context. In "The animal didn't cross the street because **it** was tired", the word "it" means "animal", and only the rest of the sentence tells you so. A model has to let each word look at the other words.

Older sequence models (RNNs) read one token at a time and squeezed everything seen so far into a single running summary. That is slow (no parallelism) and forgetful over long text. A **transformer** instead lets every token look directly at every other token in one step (that is **attention**), and all tokens are processed in parallel, which suits GPUs.

## 2. The big picture: from text to a next-token prediction

![Transformer pipeline](../diagrams/12-transformer-pipeline.svg)

1. **Tokens:** the text is split into tokens (words or word pieces) and each is mapped to an integer ID.
2. **Embeddings:** each ID is looked up in a learned table and becomes a vector (a list of numbers). Similar tokens end up with similar vectors (see the embedding space in [02](02-model-families.md)).
3. **Position information** is added so the model knows the order (section 7).
4. **N transformer blocks** are stacked. Each block lets the token vectors exchange information (attention) and then refines each vector on its own (MLP). Small models stack a dozen or so blocks, large ones many dozens.
5. **Output scores (logits):** the final vector at the last position is converted to one score for every token in the vocabulary.
6. **Softmax:** turns scores into probabilities that sum to 1. One token is then chosen (section 10).

## 3. Attention, in detail

Goal: update each token's vector using information from the tokens that matter to it.

Each token vector is turned into three new vectors by three learned weight matrices:

| Vector | Think of it as | Role |
|---|---|---|
| **Query (Q)** | "What am I looking for?" | Used by the token that is doing the looking |
| **Key (K)** | "What do I offer, so I can be matched?" | Used by every token that might be looked at |
| **Value (V)** | "The information I pass on if I am picked" | What actually gets mixed into the result |

A search analogy: the query is your search text, keys are page titles you match against, and values are the page contents you read.

![Attention step by step](../diagrams/14-attention-steps.svg)

**The four steps for one token (say "it"):**

1. Compute its query.
2. **Score** it against the key of every token: the dot product q · k (large when they match), divided by √d, where d is the key vector size. The division keeps scores from growing with d, which would otherwise make the softmax too spiky.
3. **Softmax** the scores into weights that sum to 1.
4. Output the **weighted sum of the values**. This is the updated vector for "it". It now carries information from the tokens it attended to.

The same process runs for every token at once. Weights are learned in training, not hand-written.

![Attention weights for "it"](../diagrams/15-attention-weights.svg)

### A tiny worked example

One query and three tokens, with 2-number vectors (d = 2).

| Token | Key | Value | q · k | Scaled (÷ √2) | Weight (softmax) |
|---|---|---|---|---|---|
| 1 | [1, 0] | [2, 0] | 1 | 0.707 | 0.401 |
| 2 | [0, 1] | [0, 4] | 0 | 0.000 | 0.198 |
| 3 | [1, 1] | [1, 1] | 1 | 0.707 | 0.401 |

Query q = [1, 0]. Weights sum to 1.

Output = 0.401 × [2, 0] + 0.198 × [0, 4] + 0.401 × [1, 1] = **[1.203, 1.192]**.

Tokens 1 and 3 matched the query best, so together they contribute about 80% of the result.

## 4. Multi-head attention

One attention pattern is limiting: a token may need to track several relationships at once (what a pronoun refers to, which verb goes with which subject, nearby words). **Multi-head attention** runs several attentions in parallel, each **head** with its own Q, K, V weights. The head outputs are concatenated and mixed with one more learned matrix back to the original vector size.

![Multi-head attention](../diagrams/17-multi-head-attention.svg)

Each head works in a smaller slice of the vector, so total cost stays similar to a single big head.

## 5. The causal mask (how LLMs train on next tokens)

An LLM that generates text must not peek at future tokens. A **causal mask** blocks each token from attending to anything after it: the scores for future positions are set to minus infinity before softmax, so their weights become zero.

![Causal mask](../diagrams/16-causal-mask.svg)

This makes training efficient. Given a 1,000-token passage, the model predicts the next token at **every position in a single parallel pass**, and the loss averages over all of them. Each position's label is simply the next token in the text.

## 6. The transformer block: attention plus MLP, wrapped for stability

![One transformer block](../diagrams/13-transformer-block.svg)

- **Attention** moves information **between** tokens.
- **MLP** (the same fully connected network from chapter 2) refines **each token on its own**. In large models it holds most of the parameters.
- **Layer norm** keeps the numbers well-scaled (it works per example, which is why transformers use it instead of batch norm).
- **Residual connections** add each sub-layer's input back to its output, so signals and gradients flow easily through dozens of blocks.

A common layout is "pre-norm": layer norm comes before each sub-layer, as drawn. Original transformers put it after; both appear in practice.

## 7. Position information

Attention treats its input as an unordered set: shuffle the tokens and the scores for each pair stay the same. "Dog bites man" and "Man bites dog" would look alike. So models inject order:

- **Positional encodings** added to the embeddings (fixed sine/cosine patterns or learned vectors), or
- **Rotary embeddings (RoPE)**, which rotate the query and key vectors by an amount that depends on position, so scores depend on the distance between tokens. Common in modern LLMs.

## 8. Three transformer layouts

| Layout | Attention sees | Typical use |
|---|---|---|
| **Encoder-only** (e.g. BERT) | All tokens both ways | Understanding tasks: classification, search embeddings |
| **Decoder-only** (e.g. GPT-style LLMs) | Earlier tokens only (causal) | Text generation; most chat LLMs |
| **Encoder-decoder** (e.g. T5, the original transformer) | Encoder reads the input fully; decoder generates while attending to it | Translation, summarization |

## 9. Training vs inference for a generative LLM

- **Training:** the whole text goes in at once, with the causal mask. Every position predicts its next token in parallel. Loss is cross-entropy against the true next tokens, and weights update by backprop (see [01](01-core-concepts.md)).
- **Inference (generation):** the model works **one token at a time**. It predicts a distribution for the next token, one token is chosen, it is appended to the input, and the loop repeats until a stop token. Weights are frozen.
- **KV cache:** past tokens' keys and values never change, so they are stored and reused instead of recomputed at each step. This is a big speedup, and it is why long contexts use lots of memory.
- **Cost:** every token attends to every other, so attention compute grows with the **square** of the sequence length. This is why very long context windows are expensive and why many efficiency tricks exist.

## 10. Temperature

At the last step the model outputs **logits** (raw scores), one per vocabulary token. Softmax converts them to probabilities. **Temperature (T)** divides the logits first:

> probability = softmax(logits / T)

![Temperature](../diagrams/18-temperature.svg)

Example with logits for "mat", "floor", "sofa", "moon" = 2.0, 1.0, 0.5, -1.5:

| Token | T = 0.5 | T = 1 | T = 2 |
|---|---|---|---|
| mat | 84.3% | 61.7% | 44.4% |
| floor | 11.4% | 22.7% | 26.9% |
| sofa | 4.2% | 13.8% | 21.0% |
| moon | 0.1% | 1.9% | 7.7% |

- **T < 1:** sharper, the top token gets even more probability. More deterministic and repeatable.
- **T = 1:** the model's own probabilities.
- **T > 1:** flatter, unlikely tokens get a real chance. More varied, more creative, more error-prone.
- **T → 0:** approaches **greedy decoding**: always pick the highest-probability token.
- **The ranking never changes.** Only the spread does.

### Related sampling controls

- **Greedy:** always take the top token.
- **Top-k:** sample only from the k most likely tokens.
- **Top-p (nucleus):** sample only from the smallest set of tokens whose probabilities add up to p (e.g. 0.9).
- These are often combined with temperature. Use low temperature for factual or code tasks, higher for brainstorming and creative writing.

### Same idea as calibration's "temperature scaling"

In [04](04-generalization-and-evaluation.md), temperature scaling divides a classifier's logits by one number T to fix overconfident probabilities. It is the same operation. There T is fitted on held-out data so the probabilities are honest; here T is a knob you set at generation time to trade off consistency against variety.

## 11. Common mix-ups

- **Attention vs MLP:** attention mixes information **across** tokens; the MLP transforms **each** token separately.
- **Q, K, V** are all produced from the same token vectors by different learned matrices. Queries and keys decide **where to look**; values carry **what is read**.
- **Attention weights** come from softmax, so they are positive and sum to 1 for each token.
- **Causal mask** is what lets a decoder-only model train on all positions in parallel without cheating.
- **Temperature** changes sampling, not the model's weights, and it does not change which token is most likely.
- **Temperature 0** means greedy decoding; it makes output repeatable, not necessarily correct.
- **Training** is parallel over positions; **generation** is sequential, one token per step.

## 12. Self-test

1. Why can't an MLP alone handle "it" referring to "animal"? What does attention add?
2. What are queries, keys and values? Which two decide where to look, and which carries the content?
3. Walk through the four steps of attention for one token. Why divide by √d?
4. Why multiple heads?
5. What does the causal mask do, and why does it make training on next tokens parallel?
6. What does each of attention, MLP, layer norm and residual connections do inside a block?
7. Why does a transformer need position information?
8. Encoder-only vs decoder-only vs encoder-decoder: what does attention see in each?
9. What is the KV cache, and why does attention cost grow quadratically with sequence length?
10. What does temperature do to the probabilities? What happens at T < 1, T > 1 and T → 0? Does it change the ranking?
11. How does temperature in text generation relate to temperature scaling for calibration?
12. When would you pick low vs high temperature? What do top-k and top-p add?

Next: [07 · Quick reference and self-test](07-quick-reference-and-self-test.md)
