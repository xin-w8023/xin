from .tensor import Tensor
from .common import add_indent

class Model:
    def __init__(self, layer_list: list=None) -> None:
        self.layer_list = layer_list if layer_list is not None else list()

    def __repr__(self) -> str:
        main_str = f'{self.__class__.__name__} (\n'
        for layer in self.layer_list:
            main_str += f'  {add_indent(repr(layer))},\n'
        main_str += ')'
        return main_str

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layer_list:
            inputs = layer(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layer_list):
            grad = layer.backward(grad)
        return grad

    __call__ = forward