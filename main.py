from pathlib import Path
from optimizers.sgd import SGD
from optimizers.adam import Adam
from mlp import MLP
import numpy as np

# XOR
PATH = "model.npz"

model_file = Path(PATH)
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]])

if model_file.exists():
    nn = MLP.load(PATH)
    print(f"loaded model at {PATH}")
    loss = nn.evaluate(X, Y)
    print(f"loss: {loss:.6f}")
else:
    nn = MLP()
    print("no model exists, starting training...")
    before_loss = nn.evaluate(X, Y)
    print(f"loss before: {before_loss:.6f}")
    opt = Adam(nn.parameters())
    nn.train(X, Y, opt, epochs=5000)

    # nn.save(PATH)
    loss = nn.evaluate(X, Y)
    print(f"loss after ({(before_loss-loss)*100/before_loss:.6f}% improvement): {loss:.6f}")

print(nn.feedforward(np.array([0, 0]))[0].squeeze())
print(nn.feedforward(np.array([0, 1]))[0].squeeze())
print(nn.feedforward(np.array([1, 0]))[0].squeeze())
print(nn.feedforward(np.array([1, 1]))[0].squeeze())