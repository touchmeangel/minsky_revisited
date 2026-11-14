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
    def __init__(self) -> None:
        # input 2x3 -> 3x3 -> 3x2
        self.weights = [np.random.randn(2, 3), np.random.randn(3, 3), np.random.randn(3, 2)]
        self.biases = [np.random.randn(3), np.random.randn(3), np.random.randn(2)]
        self.func = [relu, relu, sigmoid]
        self.dfunc = [derivative_relu, derivative_relu, derivative_sigmoid]
        
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
    
    def train(self, data: np.ndarray, step: float = 0.01):
        X, Y = data[:, :2], data[:, 2:]
        N = X.shape[0]
        out, zs, activations = self.feedforward(X)
        loss = self._evaluate(out, Y)
        print(f"loss: {loss}")
        # sigmoid(aW + b)-y)**2 = 2(sigmoid(aW + b)-y)) * sigmoid'(aW + b) * a^L-1
        # ∂C/∂a = 2(sigmoid(aW + b)-y))
        # ∂a/∂z = sigmoid'(aW + b)
        # ∂z/∂W = a^L-1
        # W
        
        # ∂z/∂a^L-1 = W
        # ∂a^L-1/∂z^L-1 = relu'(z^L-1)
        # ∂z^L-1/∂W^L-1 = a^L-2
        # W^L-1
        delta = 2 * (out - Y) * self.dfunc[-1](zs[-1]) 
        for i in reversed(range(len(self.weights)-1)): # HIGHLY UNOPTIMISED
            self.weights[i+1] -= step * ((activations[i+1].T @ delta) / N)
            delta = self.dfunc[i](zs[i]) * (self.weights[i+1] @ delta.T).T
            loss = self.evaluate(t_data)
            print(f"loss: {loss}")
        
# XOR
nn = MLP()
t_data = np.array([[0, 0, 1, 0],
                #    [0, 1, 0, 1],
                #    [1, 0, 0, 1],
                #    [1, 1, 1, 0]
                   ], dtype=float)

nn.train(t_data)