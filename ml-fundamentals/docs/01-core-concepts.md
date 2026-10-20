# 01 · Core concepts: features, labels, training, inference

Running example: predict a house's price.

## Features and label

- **Features (X)**: the inputs the model sees. Here: square feet, bedrooms, age.
- **Label (y)**: the answer you want the model to predict. Here: price.
- Each row is one **training example**: features in, label alongside.

![Features and label](../diagrams/01-features-and-label.svg)

## What happens during training

Training is a loop that repeats over many batches of examples:

1. **Forward pass:** feed the features through the model to get a **prediction**.
2. **Loss:** compare the prediction with the true label. The loss is a number measuring how wrong the guess was.
3. **Backward pass:** work out how each weight contributed to the error (gradients).
4. **Update:** nudge the weights to reduce the loss.

![The training loop](../diagrams/02-training-loop.svg)

Try it: [`demos/training-demo.html`](../demos/training-demo.html) fits a line to house-price data one step at a time. Raise the learning rate toward 1.0 and the line overshoots instead of settling.

### Vocabulary: batch, step, epoch

| Term | Meaning |
|---|---|
| **Batch (mini-batch)** | A small group of examples processed together (e.g. 256). The loss is computed once per batch, as the average error over the batch. |
| **Step (iteration)** | One forward pass, loss, backward pass and weight update on one batch. |
| **Epoch** | One full sweep through the whole training set. |

With 1 million examples and batch size 256, one epoch is about 3,900 steps. Ten epochs is about 39,000 steps. Large models reach millions of steps. This is tractable because GPUs process a whole batch in parallel.

Every so often the loss is also computed on a held-out **validation set** (no weight updates). That is how you watch for overfitting and decide when to stop.

### How long does training take? (rough ranges)

- Small models on tabular data: seconds to minutes.
- Typical production ranking/recommendation models: hours to a few days on one or a few GPUs.
- Large vision or speech models: days on multiple GPUs.
- Large language models: weeks to months on thousands of GPUs.
- Fine-tuning an existing model: minutes to hours.

## What happens during inference

Inference is using the trained model on new data:

- Features go in, a prediction comes out.
- **Weights are frozen.** There is no label (the true price is unknown, which is why you are asking the model), so there is no loss and no update.
- It is a single forward pass.

![Inference](../diagrams/03-inference.svg)

## One-line summary

Training uses features plus labels to adjust the weights. Inference uses features alone and the weights already learned to produce a prediction.

Next: [02 · Model families](02-model-families.md)
