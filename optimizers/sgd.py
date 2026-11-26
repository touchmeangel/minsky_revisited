from optimizers.base import Optimizer
import numpy as np

class SGD(Optimizer):
    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)

        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1).")

        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")

        self.momentum = momentum
        self.unused = self.momentum
        self.weight_decay = weight_decay

        self.velocity = [
            np.zeros_like(p) for p in self.params
        ]

    def step(self, grads):
        grads = self._prepare_grads(grads)

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            grad = grad + self.weight_decay * param

            if self.momentum > 0:
                self.velocity[i] = (
                    self.momentum * self.velocity[i] + grad
                )
                grad = self.velocity[i]/(1 - self.unused)
                
            param -= self.lr * grad
        
        self.unused *= self.momentum