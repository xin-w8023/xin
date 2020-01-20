from typing import Any

import numpy as np

class TensorBase(object):
    @property
    def data(self):
        raise NotImplementedError

class Tensor(TensorBase):
    def __init__(self, data:Any, size:Any=None, grad: Any=None) -> None:
        self.__data = np.array(data).reshape(size) if size is not None else np.array(data)
        self.__grad = grad
        self.__size = self.__data.shape

    @property
    def size(self):
        return self.__size

    @property
    def data(self) -> Any:
        return self.__data

    @property
    def grad(self) -> Any:
        return self.__grad

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def __str__(self) -> str:
        main_str = f'{self.__class__.__name__}('
        main_str += str(self.data)
        main_str += ')'
        return main_str

    def __matmul__(self, other:TensorBase) -> TensorBase:
        return Tensor(np.dot(self.data, other.data))

    def __mul__(self, other:TensorBase) -> TensorBase:
        return Tensor(self.data*other.data)

    __module__ = 'cookie'

if __name__ == '__main__':
    tensor1 = Tensor([[1, 2, 3], [1, 2, 3]], size=(6, 1))
    tensor2 = Tensor([1])
    print(tensor1.__matmul__(tensor2))
