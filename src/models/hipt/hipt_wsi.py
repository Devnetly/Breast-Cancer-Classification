# This file was adapted from the repository at https://github.com/mahmoodlab/HIPT
# Date accessed: 2024-05-10

import torch
import torch.nn.functional as F
from torch import nn

class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout : float = 0, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class HIPT_WSI(nn.Module):

    def __init__(self,dropout=0.25, n_classes=3):

        super(HIPT_WSI, self).__init__()
        size = [384, 192, 192]

        ### Global Aggregation
        
        self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout))
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192, nhead=3, dim_feedforward=192, dropout=dropout, activation='relu'
            ), 
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=dropout, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[1], n_classes)

    def forward(self, x):    

        x = x.squeeze()
        h_4096 = self.global_phi(x)
        h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
        A_4096, h_4096 = self.global_attn_pool(h_4096)  
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
        h_path = torch.mm(A_4096, h_4096)
        h_WSI = self.global_rho(h_path)

        logits = self.classifier(h_WSI)
        
        return logits