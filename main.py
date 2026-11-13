import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# XOR
class MLP:
    def __init__(self) -> None:
        # input 2x10 -> 10x10 -> 10x2
        self.weights = [np.random.randn(2, 10), np.random.randn(10, 10), np.random.randn(10, 1)]
        self.biases = [np.random.randn(10), np.random.randn(10), np.random.randn(1)]
        self.func = [relu, relu, sigmoid]
        
    def feedforward(self, a: float, b: float) -> float:
        c = np.array([[a, b]])
        
        for weights, biases, f in zip(self.weights, self.biases, self.func):
            c = f((c @ weights) + biases)
            
        return c[0]
        
    def evaluate(self, a: float, b: float, y: float) -> float:
        return (self.feedforward(a, b) - y)**2
    
    # def train():
        
        
nn = MLP()
r = nn.feedforward(0.0, 0.0)

t_data = []

# [0.0, 0.0] = 0.0
t_data.append(nn.evaluate(0.0, 0.0, 0.0))
# [0.0, 1.0] = 1.0
t_data.append(nn.evaluate(0.0, 1.0, 1.0))
# [1.0, 0.0] = 1.0
t_data.append(nn.evaluate(1.0, 0.0, 1.0))
# [1.0, 1.0] = 0.0
t_data.append(nn.evaluate(1.0, 1.0, 0.0))

print(f"r: {r}, avg cost: {np.average(t_data)}")