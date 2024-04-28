from typing import Any
from torchmetrics import Accuracy
from torch import Tensor,device

class MBAAcc:

    def __init__(self, **kwargs: Any) -> None:
        self.acc = Accuracy(**kwargs)

    def __call__(self, outputs : tuple[Tensor,Tensor,Tensor], y : Tensor):
        return self.acc(outputs[1], y)
    
    def to(self, device : device):
        self.acc = self.acc.to(device)
        return self
