# A Tiny Multilayer Perceptron, Built from Scratch

A small, hands-on implementation of a **multilayer perceptron (MLP)** using Python and NumPy—no deep-learning framework, automatic differentiation, or advanced mathematics required.

The goal is to make the basic mechanics of a neural network understandable with high-school-level mathematics: matrix multiplication, functions, derivatives, and the chain rule. As a classic test, the network learns the **XOR problem**, showing how adding hidden layers with non-linear activations overcomes a limitation of a single-layer perceptron.

## Why this project?

In *Perceptrons* (1969), Marvin Minsky and Seymour Papert examined important limitations of perceptron models. One key distinction is between a **single-layer perceptron** and a network with **hidden layers**.

A single linear decision boundary cannot separate the XOR patterns. A multilayer network can: its hidden units transform the inputs into a representation that makes the problem separable. This project demonstrates that idea with a compact network you can inspect line by line.

The book is often discussed in broad terms as a critique of neural networks. The narrower point demonstrated here is more useful: the classic limitations of a single-layer perceptron do not automatically apply to neural networks that include hidden layers and non-linear activation functions.

## The network

The default model has the architecture **2 → 5 → 5 → 2**:

```text
      2 inputs       hidden layer 1      hidden layer 2       2 outputs
    ┌─────────┐      ┌────────────┐      ┌────────────┐      ┌──────────┐
    │   x₁    │ ───▶ │            │ ───▶ │            │ ───▶ │ output 1 │
    │   x₂    │      │  5 × ReLU  │      │  5 × ReLU  │      │ output 2 │
    └─────────┘      └────────────┘      └────────────┘      └──────────┘
                                                               sigmoid
```

- **Hidden layers:** ReLU activation, `max(0, x)`.
- **Output layer:** sigmoid activation, which maps each output to the range 0–1.
- **Weights:** initialized with scaled random values; biases start at zero.
- **Learning:** full-batch gradient descent with gradients calculated by backpropagation.
- **Loss:** mean, over samples, of the sum of squared errors across the two outputs.

For each layer, the forward pass follows the same simple pattern:

\[
z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}, \qquad
a^{(l)} = f^{(l)}(z^{(l)})
\]

Backpropagation applies the chain rule to determine how each weight and bias contributed to the final error, then updates those parameters in the direction that reduces the loss.

## The XOR task

The model receives two binary inputs and produces a **two-value, one-hot-style output**:

| Input 1 | Input 2 | Target output | Meaning |
|:---:|:---:|:---:|---|
| 0 | 0 | `[1, 0]` | Same |
| 0 | 1 | `[0, 1]` | Different |
| 1 | 0 | `[0, 1]` | Different |
| 1 | 1 | `[1, 0]` | Same |

In other words, the first output represents “same” and the second represents “different.” The larger output indicates the model's predicted class.

## Requirements

- Python 3.10 or newer
- NumPy

Install NumPy with:

```bash
python -m pip install numpy
```

## Run it

Save the implementation as `main.py`, then run:

```bash
python main.py
```

On the first run, the script initializes a model, trains it on the four XOR examples, and saves the learned parameters to `model.npz`.

On later runs, if `model.npz` exists, the script loads the saved parameters instead of training again. To train a fresh model, delete `model.npz` before running the script.

## Example output

One successful run produced:

```text
python main.py
no model exists, starting training...
loss before: 1.0327818499158088
loss after (99.98871350098898% improvement): 0.00011656491327681893
[0.99478143 0.01071981]
[0.00774901 0.99307674]
[0.00785911 0.99239498]
[0.99040086 0.00209468]
```

The four prediction vectors correspond, in order, to inputs `[0, 0]`, `[0, 1]`, `[1, 0]`, and `[1, 1]`. The model has learned to give a high first output for equal inputs and a high second output for different inputs.

Your exact numbers may differ because the model uses random weight initialization unless a seed is supplied. The output above is an example from one run, not a guaranteed result.

## What's implemented?

Everything essential to this small MLP lives in the `MLP` class:

- `feedforward` — computes the network's output and stores intermediate values needed for backpropagation.
- `train` — trains on the supplied dataset using gradient descent and backpropagation.
- `evaluate` — calculates the loss for a dataset.
- `save` and `load` — store and restore weights and biases using a NumPy `.npz` file.

The code also defines the sigmoid and ReLU activation functions and their derivatives directly. This keeps the relationship between the forward pass, derivatives, and parameter updates visible rather than hiding it behind a framework.

## Scope and limitations

This is an educational demonstration, not a general-purpose machine-learning library. It is intentionally small: the example uses four training samples, full-batch gradient descent, and a fixed activation-function arrangement (ReLU in hidden layers and sigmoid at the output). There is no train/test split, batching, or general model-configuration system.

The implementation is best used as a starting point for experimenting: change the hidden-layer sizes, inspect the intermediate activations, vary the learning rate, or supply a fixed seed to make an experiment reproducible.

## Further reading

- Marvin Minsky and Seymour Papert, *Perceptrons: An Introduction to Computational Geometry* (1969).
- The XOR problem as a simple example of a function that is not linearly separable.

---

**The takeaway:** a single-layer perceptron is limited to a linear decision boundary. A small MLP with hidden non-linear layers can learn a richer decision boundary—and solve XOR.