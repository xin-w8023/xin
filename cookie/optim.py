from cookie import Tensor
from cookie.layers import Layer
from cookie.model import Model

class Optimizer:
    def step(self, grad: Tensor) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, model: Model, learning_rate: float=1e-3) -> None:
        self.model = model
        self.lr = learning_rate

    def step(self, grad: Tensor) -> None:
        self.model.backward(grad)
        for layer in self.model.layer_list:
            if isinstance(layer, Layer):
                for name in layer.parameters:
                    layer.parameters[name] -= self.lr * layer.grads[name]