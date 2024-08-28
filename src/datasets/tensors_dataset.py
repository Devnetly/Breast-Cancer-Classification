import os
import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable,Optional

class TensorDataset(Dataset):

    def __init__(self,
        root : str,
        tensor_transform : Optional[Callable] = None,
        label_transform : Optional[Callable] = None    ,
        split : Optional[str] = None     
    ) -> None:

        super().__init__()

        self.root = root
        self.tensor_transform = tensor_transform
        self.label_transform = label_transform
        self.split = split

        self.metadata = self.get_metadata()

        self.classes : list = self.metadata["type"].unique().tolist()

        if self.split is not None:
            self.metadata = self.metadata[self.metadata["split"] == self.split]

    def get_metadata(self) -> pd.DataFrame:

        metadata = {
            "name" : [],
            "split" : [],
            "type" : [],
            "subtype" : []
        }

        for split in os.listdir(self.root):

            split_path = os.path.join(self.root, split)

            for type in os.listdir(split_path):

                type_path = os.path.join(split_path, type)

                for subtype in os.listdir(type_path):

                    subtype_path = os.path.join(type_path, subtype)
                    names = os.listdir(subtype_path)

                    metadata["name"].append(names)
                    metadata["split"].append([split] * len(names))
                    metadata["type"].append([type] * len(names))
                    metadata["subtype"].append([subtype] * len(names))

        metadata = pd.DataFrame(metadata)

        return metadata
    
    def __len__(self) -> int:
        return self.metadata.shape[0]
    
    def __getitem__(self, idx : int) -> tuple[Tensor,int]:

        row = self.metadata.iloc[idx]

        path = os.path.join(
            self.root,
            row['split'],
            row['type'],
            row['subtype'],
            row['name']
        )

        tensor = torch.load(path)

        label = self.classes.index(row['type'])

        if self.tensor_transform is not None:
            tensor = self.tensor_transform(tensor)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return tensor,label