import torch
from torch import nn,Tensor
from typing import Literal


class MBALoss(nn.Module):

    def __init__(self, branches_count : int,device : Literal['cpu', 'cuda'] = 'cpu'):

        super().__init__()

        self.ce1 = nn.CrossEntropyLoss()
        self.ce2 = nn.CrossEntropyLoss()

        self.branches_count = branches_count
        self.device = device

    def forward(self, x : tuple[Tensor,Tensor,Tensor], y : Tensor) -> torch.Tensor:

        sub_preds,preds,att = x

        l0 = self.ce1(preds, y)
        l1 = self.ce1(sub_preds, y.repeat_interleave(self.branches_count))
        l2 = torch.tensor(0.0).to(self.device)

        att = torch.softmax(att, dim=-1)

        for i in range(self.branches_count):
            for j in range(i+1, self.branches_count):
                l2 += torch.cosine_similarity(att[:, i], att[:, j], dim=-1).mean()

        l2 = l2 / (self.branches_count * (self.branches_count - 1))

        loss = l0 + l1 + l2

        return loss
