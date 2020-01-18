from typing import Iterable

from cookie import Tensor
import cookie

class Batch:
    def __init__(self, feat: Tensor, label: Tensor):
        self.__feat = feat
        self.__label = label

    @property
    def feat(self) -> Tensor:
        return self.__feat

    @property
    def label(self) -> Tensor:
        return self.__label


class DataLoader:
    def __init__(self, feat: Tensor, label: Tensor, batch_size: int=32, shuffle: bool=True) -> None:
        self.feat = feat
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.feat)

    def __call__(self) -> Batch:
        idx = cookie.arange(self.num_samples)
        if self.shuffle:
            cookie.shuffle(idx)
        starts = cookie.arange(0, self.num_samples, self.batch_size)
        for s in starts:
            yield Batch(self.feat[idx][s:s+self.batch_size], self.label[idx][s:s+self.batch_size])
