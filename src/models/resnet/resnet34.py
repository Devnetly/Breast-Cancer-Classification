from .resnet import ResNet,models

class ResNet34(ResNet):

    def __init__(self, 
        n_classes: int, 
        dropout_rate: float = 0.0,
        weights : str = 'IMAGENET1K_V1',
        depth : int = 2
    ) -> None:
        resnet = models.resnet34(weights=weights)
        super().__init__(resnet, n_classes, dropout_rate, depth)
