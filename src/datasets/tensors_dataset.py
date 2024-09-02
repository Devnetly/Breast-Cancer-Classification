import torch
import sys
sys.path.append('../..')
from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable,Optional
from src.utils import get_metadata  

class TensorDataset(Dataset):

    def __init__(self,
        root : str,
        tensor_transform : Optional[Callable] = None,
        label_transform : Optional[Callable] = None,
        split : Optional[str] = None,
        output_type : str = "auto"
    ) -> None:

        super().__init__()

        self.root = root
        self.tensor_transform = tensor_transform
        self.label_transform = label_transform
        self.split = split
        self.output_type = output_type

        self.metadata = get_metadata()

        self.classes : list = self.metadata["type"].unique().tolist()

        if self.split is not None:
            self.metadata = self.metadata[self.metadata["split"] == self.split]
    
    def __len__(self) -> int:
        return self.metadata.shape[0]
    
    def __getitem__(self, idx : int) -> tuple[Tensor,int]:

        row = self.metadata.iloc[idx]

        path = row['path']

        tensor = torch.load(path)

        if self.output_type == "dense" and tensor.is_sparse:
            tensor = tensor.to_dense()

        if self.output_type == "sparse" and not tensor.is_sparse:
            raise ValueError("Output type is sparse but tensor is dense")
        
        if self.output_type == "values" and tensor.is_sparse:
            tensor = tensor.coalesce().values()
        
        if self.output_type == "auto" and tensor.is_sparse:
            tensor = tensor.coalesce().values()

        label = self.classes.index(row['type'])

        if self.tensor_transform is not None:
            tensor = self.tensor_transform(tensor)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return tensor,label