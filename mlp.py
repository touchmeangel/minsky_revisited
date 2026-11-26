from optimizers.base import Optimizer
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

EPS = 1e-12

class MLP:
    def __init__(self, sizes=(2, 5, 5, 1), seed=None) -> None:
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

    def _evaluate_cross_entropy(self, out, Y):
        out = np.clip(out, EPS, 1 - EPS)
        ce = -Y*np.log(out) - (1-Y)*np.log1p(-out)
        return float(np.mean(np.sum(ce, axis=1)))
    
    def _derivative_cross_entropy(self, z_last, Y):
        a = np.clip(self.func[-1](z_last), EPS, 1 - EPS)
        dC_da = -Y/a + (1-Y)/(1-a)
        return dC_da * self.dfunc[-1](z_last)
    
    def _evaluate_mse(self, out: np.ndarray, Y: np.ndarray) -> float:
        return float(np.mean(np.sum((out - Y) ** 2, axis=1)))

    def _derivative_mse(self, z_last: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return 2 * (self.func[-1](z_last) - Y) * self.dfunc[-1](z_last)
        
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> float:
        out = self.feedforward(X)[0]
        return self._evaluate_cross_entropy(out, Y)
    
    def parameters(self) -> list[np.ndarray]:
        return self.weights + self.biases # combining everything into one list
    
    # sq = np.array([[1., 3.],
    #            [2., 2.]])    # 2 samples, 2 outputs

    # np.mean(sq)                       # 8 / 4 = 2.0  -> divides by N * K (samples * outputs)
    # np.mean(np.sum(sq, axis=1))       # (4 + 4) / 2 = 4.0  -> divides by N only
    
    def gradients(self, zs: list[np.ndarray], activations: list[np.ndarray], Y: np.ndarray) -> list[np.ndarray]:
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
        N = activations[0].shape[0]
        L = len(self.weights)

        grad_W = [np.zeros_like(self.weights[i]) for i in range(L)]
        grad_b = [np.zeros_like(self.biases[i]) for i in range(L)]

        delta = self._derivative_cross_entropy(zs[-1], Y)
        for i in reversed(range(L)):
            grad_W[i] = (activations[i].T @ delta) / N
            grad_b[i] = delta.mean(axis=0)

            if i > 0:
                delta = (self.dfunc[i-1](zs[i-1]) * (delta @ self.weights[i].T))

        return grad_W + grad_b

    def train(self, X: np.ndarray, Y: np.ndarray, optimizer: Optimizer, epochs: int = 5000, log_every: int = 500):
        N = X.shape[0]
        for epoch in range(epochs):
            out, zs, activations = self.feedforward(X)
            if epoch % log_every == 0:
                print(epoch, self._evaluate_cross_entropy(out, Y))

            grads = self.gradients(zs, activations, Y)
            optimizer.step(grads)

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
