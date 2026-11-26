import numpy as np

class Optimizer:
    def __init__(self, params: list[np.ndarray], lr: float):
        self.params = list(params)
        self.lr = lr

        if not self.params:
            raise ValueError("Optimizer received no parameters.")

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

    def _prepare_grads(self, grads: list[np.ndarray]):
        """Verifies the shape is correct"""
        grads = list(grads)

        if len(grads) != len(self.params):
            raise ValueError("Number of gradients must match number of parameters.")

        result = []

        for param, grad in zip(self.params, grads):
            grad = np.asarray(grad, dtype=param.dtype)

            if grad.shape != param.shape:
                raise ValueError(
                    f"Gradient shape {grad.shape} does not match "
                    f"parameter shape {param.shape}."
                )

            result.append(grad)

        return result

    def step(self, grads: list[np.ndarray]):
        raise NotImplementedError