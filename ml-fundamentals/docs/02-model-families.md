# 02 · Model families

The loop from [01](01-core-concepts.md) (guess, measure error, adjust) is the core of supervised learning. Other model families change one or more pieces of it.

| Family | Training signal | What gets adjusted |
|---|---|---|
| Neural network (MLP) | Labels, through a loss | Weights |
| Decision tree / tree ensembles | Labels | The questions and splits |
| Embeddings / unsupervised | None, just patterns in the data | Point positions |
| Language model | The next token in the text | Weights |
| Reinforcement learning | Reward from the environment | The agent's behavior |

Inference is always the same: frozen parameters, features in, prediction out.

## Neural networks and the MLP

**MLP = multilayer perceptron**, the plain, classic neural network.

- Each unit takes outputs from the previous layer, multiplies each by a weight, sums them, and passes the sum through a **nonlinearity** (e.g. ReLU).
- A layer is a row of such units. An MLP stacks several layers, and every unit connects to every unit in the next layer (hence "fully connected" or "dense").
- **The nonlinearity is essential.** Without it, any number of stacked layers collapses into a single linear function (a weighted sum of weighted sums is still a weighted sum). With it, the network can bend to fit complex patterns.
- Training uses **backpropagation** (chain rule) plus gradient descent to nudge every weight after each guess.

![MLP](../diagrams/04-neural-network-mlp.svg)

### Where MLPs are used

- **Tabular data on their own:** fraud detection, churn, credit scoring, pricing. Gradient-boosted trees often match or beat them on small tabular datasets, so teams try both.
- **Recommendation and ranking:** models like Wide & Deep and DLRM turn user and item features into embeddings, concatenate them, and pass them through an MLP that outputs a click or relevance score.
- **Heads on bigger models:** a CNN or text encoder extracts features, then a small MLP turns them into the answer.
- **Inside transformers:** every layer has an MLP block after attention. In LLMs these blocks hold roughly two-thirds of the parameters.
- **Smaller roles:** value/policy networks in RL, small networks in diffusion models and neural rendering.

**Limit:** an MLP treats input as a flat list of numbers, so it does not know neighboring pixels are related or that word order matters. CNNs and transformers add that structure.

## Tree-based models

No weights to nudge. Training builds a tree by choosing the yes/no questions (splits) that best separate the labels. Random forests average many trees; gradient boosting builds trees sequentially, each correcting the previous ones.

![Decision tree](../diagrams/05-decision-tree.svg)

## Embeddings and unsupervised learning

There is no label column. The model finds structure in the features alone. An **embedding** maps items (words, products, users) to points in space so that similar items sit close together. This underlies clustering, semantic retrieval and two-tower recommenders.

![Embedding space](../diagrams/06-embedding-space.svg)

## Language models (LLMs)

The labels come from the text itself (**self-supervised learning**):

- **Features:** the tokens so far, e.g. "The cat sat on the".
- **Label:** the token that actually came next in the training text, here "mat".

Every position in every document is a training example, so trillions of labeled examples exist without hand labeling. A **token** is a word or a piece of a word. User queries only appear at inference, after training.

![LLM next-token prediction](../diagrams/07-llm-next-token.svg)

## Reinforcement learning

There is no fixed label table. An agent acts in an environment, receives a new state and a **reward**, and adjusts its behavior to earn more reward. The loop is guess, get feedback, update, but the feedback is a score instead of the correct answer.

![Reinforcement learning](../diagrams/08-reinforcement-learning.svg)

Next: [03 · Training components](03-training-components.md)
