from torchvision.datasets import ImageFolder 

class RoIDataset(ImageFolder):

    def __getitem__(self, index: int):
        x,y = super().__getitem__(index)
        path = self.imgs[index][0]
        return path, x, y