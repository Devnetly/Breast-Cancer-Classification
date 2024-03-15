import torch
from torch import nn
from torchvision import models

class ResNet34(nn.Module):

    def __init__(self, *, n_classes : int, layers_to_freeze : int = 5):

        super().__init__()

        self.resnet34 = models.resnet34(weights='IMAGENET1K_V1')

        layers_names = self.get_layers_names()[:layers_to_freeze]

        for name, param in self.resnet34.named_parameters():
            if any([name.split('.')[0] == freezed_layer for freezed_layer in layers_names]): param.requires_grad = False

        self.resnet34.fc = nn.Linear(in_features=self.resnet34.fc.in_features,out_features=n_classes)

    def forward(self, X) -> torch.Tensor:
        return self.resnet34(X)

    def get_layers_names(self) -> list[str]:

        parameters = dict(self.resnet34.named_parameters()).keys()

        layers = []

        for parameter in parameters:
            layer_name = parameter.split('.')[0]

            if layer_name not in layers:
                layers.append(layer_name)

        return layers
