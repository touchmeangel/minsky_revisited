from pathlib import Path
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return (x > 0).astype(float)

class MLP:
    def __init__(self, sizes=(2, 5, 5, 2), seed=None) -> None:
        rng = np.random.default_rng(seed)
        self.weights = [rng.standard_normal((i, o)) * np.sqrt(2 / i)
                        for i, o in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(o) for o in sizes[1:]]
        hidden = len(sizes) - 2
        self.func = [relu] * hidden + [sigmoid]
        self.dfunc = [derivative_relu] * hidden + [derivative_sigmoid]
        
    def feedforward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        a = X
        zs = []
        activations = [X]
        for W, b, f in zip(self.weights, self.biases, self.func):
            z = a @ W + b
            a = f(z)
            zs.append(z)
            activations.append(a)
        return a, zs, activations
    
    def _evaluate(self, out: np.ndarray, Y: np.ndarray) -> float:
        return float(np.mean(np.sum((out - Y) ** 2, axis=1)))
        
    def evaluate(self, data: np.ndarray) -> float:
        X, Y = data[:, :2], data[:, 2:]          # inputs (N, 2), targets (N, 2)
        out = self.feedforward(X)[0]
        return self._evaluate(out, Y)
    
    # sq = np.array([[1., 3.],
    #            [2., 2.]])    # 2 samples, 2 outputs

    # np.mean(sq)                       # 8 / 4 = 2.0  -> divides by N * K (samples * outputs)
    # np.mean(np.sum(sq, axis=1))       # (4 + 4) / 2 = 4.0  -> divides by N only
    
    def train(self, data: np.ndarray, step: float = 0.1, epochs: int = 5000, log_every: int = 500):
        X, Y = data[:, :2], data[:, 2:]
        N = X.shape[0]
        for epoch in range(epochs):
            # sigmoid(aW + b)-y)**2 = 2(sigmoid(aW + b)-y)) * sigmoid'(aW + b) * a^L-1
            # ∂C/∂a = 2(sigmoid(aW + b)-y))
            # ∂a/∂z = sigmoid'(aW + b)
            # ∂z/∂W = a^L-1
            # ∂z/∂b = 1
            # W, b
            
            # ∂z/∂a^L-1 = W
            # ∂a^L-1/∂z^L-1 = relu'(z^L-1)
            # ∂z^L-1/∂W^L-1 = a^L-2
            # W^L-1
            out, zs, activations = self.feedforward(X)
            if epoch % log_every == 0:
                print(epoch, self._evaluate(out, Y))
            delta = 2 * (out - Y) * self.dfunc[-1](zs[-1]) 
            for i in reversed(range(len(self.weights))):
                grad_W = ((activations[i].T @ delta) / N)
                grad_b = delta.mean(axis=0)
                if i > 0:
                    delta = self.dfunc[i-1](zs[i-1]) * (delta @ self.weights[i].T)
                self.weights[i] -= step * grad_W
                self.biases[i] -= step * grad_b
            
    def save(self, path: str) -> None:
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays) 
    
    @staticmethod
    def load(path: str) -> "MLP":
        with np.load(path) as f:
            n = len(f.files) // 2
            weights = [f[f"W{i}"] for i in range(n)]
            biases = [f[f"b{i}"] for i in range(n)]
        sizes = (weights[0].shape[0], *[W.shape[1] for W in weights])
        model = MLP(sizes)
        model.weights, model.biases = weights, biases
        return model
# XOR
PATH = "model.npz"

model_file = Path(PATH)
t_data = np.array([[0, 0, 1, 0],
                   [0, 1, 0, 1],
                   [1, 0, 0, 1],
                   [1, 1, 1, 0]], dtype=float)

if model_file.exists():
    nn = MLP.load(PATH)
    print(f"loaded model at {PATH}")
    loss = nn.evaluate(t_data)
    print(f"loss: {loss}")
else:
    nn = MLP()
    print("no model exists, starting training...")
    before_loss = nn.evaluate(t_data)
    print(f"loss before: {before_loss}")
    nn.train(t_data, step=0.1, epochs=50000)
    nn.save(PATH)
    loss = nn.evaluate(t_data)
    print(f"loss after ({(before_loss-loss)*100/before_loss}% improvement): {loss}")

print(nn.feedforward(np.array([0, 0]))[0])
print(nn.feedforward(np.array([0, 1]))[0])
print(nn.feedforward(np.array([1, 0]))[0])
print(nn.feedforward(np.array([1, 1]))[0])