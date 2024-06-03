import os
import torch
from torch.utils.data import Dataset
from typing import Callable,Optional

class TensorDataset(Dataset):

    def __init__(self, root : str, transform : Optional[Callable] = None) -> None:

        super().__init__()

        self.root = root
        self.transform = transform
        self.classes,self.classes_to_idx,self.tensors = self.find_classes()

    def get_labels(self):
        return [item[1] for item in self.tensors]

    def find_classes(self) -> tuple[list[str], dict[str,int], list[tuple[str,int]]]:

        classes = os.listdir(self.root)
        classes = list(filter(lambda x : os.path.isdir(os.path.join(self.root, x)), classes))
        classes = sorted(classes)

        classes_to_idx = {}
        tensors = []

        for i,class_ in enumerate(classes):

            classes_to_idx[class_] = i

            class_path = os.path.join(self.root, class_)

            for type_ in os.listdir(class_path):

                type_path = os.path.join(class_path, type_)

                type_tensors = os.listdir(type_path)
                type_tensors = list(map(lambda x : (os.path.join(type_path, x), i), type_tensors))

                tensors.extend(type_tensors)

        return classes,classes_to_idx,tensors
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:

        path = self.tensors[index][0]
        label = self.tensors[index][1]
        tensor = torch.load(path)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label
    
    def __len__(self) -> int:
        return len(self.tensors)
