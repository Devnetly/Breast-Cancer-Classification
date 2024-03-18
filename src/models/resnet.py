import torch
import re
from collections import OrderedDict
from typing import Mapping
from torchvision import models
from torch import nn,Tensor
from typing import Any

class ResNet(nn.Module):
    """
        A modified version of pytorch implementation for the resnet familly
        of architectures,it frezzez all the layers exepct the last tow blocks
        fully connected layer,and adds dropout after the average pooling layer
        and modifies the out_features of the last fully connected layer.
    """
    
    def __init__(self, 
        resnet : models.ResNet,
        n_classes : int,
        dropout_rate : float = 0.0,
    ) -> None:
        """
            The constructor of ResNet class.

            Arguments :
            - resnet : the base resnet (for example : resnet18,resnet34,resnet50),of type : `torchvision.models.ResNet`.
            - n_classes : the number of distinct classes,a positive integer.
            - dropout_rate : the dropout rate of the dropout layer.

            Retuns :
            - None.
        """
        
        super().__init__()

        self.resnet = resnet
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

        self._freeze()

    def _freeze(self) -> None:

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x : Tensor) -> Tensor:

        y = self.resnet.conv1(x)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)

        y = self.resnet.layer1(y)
        y = self.resnet.layer2(y)
        y = self.resnet.layer3(y)
        y = self.resnet.layer4(y)

        y = self.resnet.avgpool(y)
        y = torch.flatten(y, 1)

        y = self.dropout(y)

        y = self.resnet.fc(y)

        return y
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        
        ### create a new dictionary with the correct keys
        ### to be compatible with legacy weights
        new_dict = OrderedDict()

        for key, value in state_dict.items():
            new_key = re.sub(r"resnet\d\d", "resnet", key)
            new_dict[new_key] = value

        return super().load_state_dict(new_dict, strict, assign)