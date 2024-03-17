from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import ImageFolder
from typing import Any

class BalancedSampler(WeightedRandomSampler):
    
    def __init__(self, 
        dataset : ImageFolder,
        num_samples: int,
        replacement: bool = True,
        generator: Any | None = None
    ):

        weights = {}

        for _,label in dataset.imgs:
            weights[label] += 1

        for key in weights.keys():
            weights[key] = 1 / weights[key]

        super().__init__(
            weights=weights.values(),
            num_samples=num_samples,
            generator=generator,
            replacement=replacement
        )