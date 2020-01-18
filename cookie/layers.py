from typing import Dict, Callable

import numpy as np
from cookie.tensor import Tensor


class Layer:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.parameters: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size})'

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def param_grad(self) -> [Dict[str, Tensor], Dict[str, Tensor]]:
        return self.parameters, self.grads


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Linear,self).__init__(input_size, output_size)
        self.parameters['weight'] = np.random.randn(input_size, output_size)
        self.parameters['bias'] = np.random.rand(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.parameters['weight'] + self.parameters['bias']

    def backward(self, grad: Tensor) -> Tensor:
        """
        y = x@w + b
        dy/dx = w^T
        dy/dw = x^T
        dy/db = 1
        z = f(y)
        dz/dx = dz/dy*dy/dx = grad @ w^T
        dz/dw = dz/dy*dy/dw = x^T @ grad
        dz/db = dz/dy*dy/db = grad * 1
        :param grad: upstream grad flow
        :return: grad through this layer
        """
        self.grads['weight'] = self.inputs.T @ grad
        self.grads['bias'] = np.sum(grad, axis=0)
        return grad @ self.parameters['weight'].T

    __call__ = forward


func = Callable[[Tensor], Tensor]

class Activation:
    def __init__(self, f: func, f_prime: func) -> None:
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    __call__ = forward

class Tanh(Activation):

    def __init__(self) -> None:
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            y = tanh(x)
            return 1 - y ** 2

        super(Tanh, self).__init__(tanh, tanh_prime)



class Sigmoid(Activation):
    def __init__(self) -> None:
        def sigmoid(x: Tensor) -> Tensor:
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x: Tensor) -> Tensor:
            y = sigmoid(x)
            return y * (1 - y)

        super(Sigmoid, self).__init__(sigmoid, sigmoid_prime)


class Relu(Activation):
    def __init__(self, inplace: bool=True) -> None:
        self.inplace = inplace
        self.mask = None
        def relu(x: Tensor) -> Tensor:
            self.mask = np.ones_like(x)
            if self.inplace:
                self.mask[x<0] = 0
                self.mask[x>1] = 0
                x[x<0] = 0
                x[x>1] = 1
                return x
            else:
                self.mask[x < 0] = 0
                self.mask[x > 1] = 0
                return self.mask*x
        def relu_prime(x: Tensor) -> Tensor:
            return self.mask
        super().__init__(relu, relu_prime)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(inplace={self.inplace})'