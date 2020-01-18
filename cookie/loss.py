
from cookie.tensor import Tensor


class Loss:

    def forward(self, prediction: Tensor, target: Tensor) -> float:
        raise NotImplementedError

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

class MSE(Loss):

    def forward(self, prediction: Tensor, target: Tensor) -> float:
        return ((prediction-target) ** 2).sum()

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return 2*(prediction-target)

    __call__ = forward