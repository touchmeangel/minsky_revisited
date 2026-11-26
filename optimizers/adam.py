from optimizers.base import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)

        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        if not 0 <= self.b1 < 1 or not 0 <= self.b2 < 1:
            raise ValueError("Betas must be in [0, 1).")

        if eps <= 0 or weight_decay < 0:
            raise ValueError("Invalid epsilon or weight decay.")

        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

        self.unused_m = self.b1
        self.unused_v = self.b2

    def step(self, grads):
        grads = self._prepare_grads(grads)

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            grad = grad + self.weight_decay * param

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad

            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grad**2

            m_corrected = self.m[i] / (1 - self.unused_m)
            v_corrected = self.v[i] / (1 - self.unused_v)

            param -= self.lr * m_corrected/(np.sqrt(v_corrected) + self.eps)
            
        self.unused_m *= self.b1
        self.unused_v *= self.b2