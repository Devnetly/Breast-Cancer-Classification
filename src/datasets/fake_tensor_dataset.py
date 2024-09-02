import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class FakeTensorDataset(Dataset):
    """
        A fake dataset that generates random tensors and labels.
    """

    def __init__(self,
        shape : tuple,  
        length : int,
        num_classes : int,
        tensor_transform : Optional[Callable] = None,    
        label_transform : Optional[Callable] = None     
    ) -> None:
        super().__init__()

        self.shape = shape
        self.length = length
        self.num_classes = num_classes
        self.tensor_transform = tensor_transform
        self.label_transform = label_transform
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx : int) -> tuple[torch.Tensor,int]:

        tensor = torch.rand(self.shape)
        label = torch.randint(0, self.num_classes, (1,)).item()

        if self.tensor_transform is not None:
            tensor = self.tensor_transform(tensor)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return tensor, label