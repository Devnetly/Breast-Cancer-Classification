import torch
from torch import nn
from torchvision import models

class ResNet18(nn.Module):
  """
    - A custom class that makes fine tuning resnet18 easier.
  """

  def __init__(self,*,
    n_classes : int,
    layers_to_freeze : int = 5
  ):
    """
      - n_classes : the number of classes,a positive integer.
      - layers_to_freeze : the number of the layers to freeze, a positive integer.
    """

    super().__init__()

    self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')

    # freeze the desired layers
    layers_names = self.get_layers_names()[:layers_to_freeze]

    for name, param in self.resnet18.named_parameters():
      if any([name.split('.')[0] == trainable_layer for trainable_layer in layers_names]):
        param.requires_grad = False

    # override the last layer because
    # we don't necessarly have the same
    # number of classes.
    self.resnet18.fc = nn.Linear(in_features=self.resnet18.fc.in_features,out_features=n_classes)

  def forward(self, X) -> torch.Tensor:
    return self.resnet18(X)

  def get_layers_names(self) -> list[str]:

    parameters = dict(self.resnet18.named_parameters()).keys()

    layers = []

    for parameter in parameters:
      layer_name = parameter.split('.')[0]
      if layer_name not in layers:
        layers.append(layer_name)

    return layers