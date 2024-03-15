import torch
from torch import nn
from torchvision import models

class ResNet34(nn.Module):
  """
    - A custom class that makes fine tuning resnet18 easier.
  """

  def __init__(self,*,
    n_classes : int,
  ) -> None:
    """
      - n_classes : the number of classes,a positive integer.
    """

    super().__init__()

    self.resnet34 = models.resnet34(weights='IMAGENET1K_V1')

    # override the last layer because
    # we don't necessarly have the same
    # number of classes.
    self.resnet34.fc = nn.Linear(in_features=self.resnet34.fc.in_features,out_features=n_classes)

    for param in self.resnet34.layer3.parameters():
      param.requires_grad = True

    for param in self.resnet34.layer4.parameters():
        param.requires_grad = True

    for param in self.resnet34.fc.parameters():
        param.requires_grad = True

  def forward(self, X) -> torch.Tensor:
    return self.resnet34(X)
