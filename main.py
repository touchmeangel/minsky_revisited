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
        
    def feedforward(self, X: np.ndarray) -> np.ndarray:
        a = X                                    # shape (N, 2), the activation
        for W, b, f in zip(self.weights, self.biases, self.func):
            a = f(a @ W + b)
        return a    
        
    def evaluate(self, data: np.ndarray) -> float:
        X, Y = data[:, :2], data[:, 2:]          # inputs (N, 2), targets (N, 2)
        out = self.feedforward(X)
        return float(np.mean(np.sum((out - Y) ** 2, axis=1)))
    
    # sq = np.array([[1., 3.],
    #            [2., 2.]])    # 2 samples, 2 outputs

    # np.mean(sq)                       # 8 / 4 = 2.0  -> divides by N * K (samples * outputs)
    # np.mean(np.sum(sq, axis=1))       # (4 + 4) / 2 = 4.0  -> divides by N only
    
    def train(self, data: np.ndarray):
        loss = nn.evaluate(t_data)
        print(f"loss: {loss}")

# XOR
nn = MLP()
t_data = np.array([[0, 0, 1, 0],
                   [0, 1, 0, 1],
                   [1, 0, 0, 1],
                   [1, 1, 1, 0]], dtype=float)

nn.train(t_data)