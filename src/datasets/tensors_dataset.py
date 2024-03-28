import os
import torch
from torch.utils.data import Dataset

class TensorDataset(Dataset):

    def __init__(self, root : str) -> None:

        super().__init__()

        self.root = root
        self.classes,self.classes_to_idx,self.tensors = self.find_classes()

    def find_classes(self) -> tuple[list[str], dict[str,int], list[tuple[str,int]]]:

        classes = os.listdir(self.root)
        classes = list(filter(lambda x : os.path.isdir(os.path.join(self.root, x)), classes))
        classes = sorted(classes)

        classes_to_idx = {}
        tensors = []

        for i,class_ in enumerate(classes):

            classes_to_idx[class_] = i

            class_path = os.path.join(self.root, class_)

            class_tensors = os.listdir(class_path)
            class_tensors = list(map(lambda x : (os.path.join(class_path, x), i), class_tensors))

            tensors.extend(class_tensors)

        return classes,classes_to_idx,tensors
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:

        path = self.tensors[index][0]
        label = self.tensors[index][1]
        tensor = torch.load(path)
        tensor = torch.unsqueeze(tensor, dim=0)

        return tensor, label
    
    def __len__(self) -> int:
        return len(self.tensors)
