from .resnet import ResNet,models

class ResNet50(ResNet):

    def __init__(self, 
        n_classes: int, 
        dropout_rate: float = 0.0,
        weights : str = 'IMAGENET1K_V1'
    ) -> None:
        resnet = models.resnet50(weights=weights)
        super().__init__(resnet, n_classes, dropout_rate)