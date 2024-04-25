import torch
from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self,dim : int):

        super().__init__()

        self.fc1 = nn.Linear(in_features=dim,out_features=dim, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim,out_features=dim, bias=False)
        self.relu2 = nn.ReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        y = self.fc1(x)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)

        return y

class DimReduction(nn.Module):

    def __init__(self,
        n_channels : int,
        m_dim : int,
        n_residual_blocks : int = 0
    ) -> None:
        
        super().__init__()

        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock(dim=m_dim) for _ in range(n_residual_blocks)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        y = self.fc1(x)
        y = self.relu1(y)
        y = self.residual_blocks(y)

        return y
    
class AttentionGated(nn.Module):

    def __init__(self,
        l : int = 512,
        d : int = 128,
        k : int = 1
    ):

        super().__init__()

        self.attention_v = nn.Sequential(
            nn.Linear(l, d),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(l, d),
            nn.Sigmoid()
        )

        self.att_weights = nn.Linear(d, k)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        v = self.attention_v(x)
        u = self.attention_u(x)

        a = self.att_weights(u * v)
        a = torch.transpose(input=a, dim0=1, dim1=0)

        return a


class ShallowClassifier(nn.Module):

    def __init__(self, 
        in_features : int,
        n_classes : int,
        dropout_rate : float = 0
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(in_features=in_features,out_features=n_classes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        y = self.dropout(x)
        y = self.fc(y)

        return y

class STKIM(nn.Module):

    def __init__(self,
        branches_count : int = 5,
        mask_rate : float = 0.6,
        k : int = 10,
    ) -> None:
        
        super().__init__()

        self.branches_count = branches_count
        self.mask_rate = mask_rate
        self.k = k

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        if self.training and self.k > 0: # use stkim only during training

            k, n = x.shape
            n_pacthes_to_mask = min(self.k, n)
            _,indices = torch.topk(x, k=n_pacthes_to_mask)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,int(n_pacthes_to_mask * self.mask_rate)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(x.device)
            random_mask.scatter_(-1, masked_indices, 0)
            x = x.masked_fill(random_mask == 0, -1e9)

        return x

class MultiBranchAttention(nn.Module):

    def __init__(self,
        d_features : int, # 512 for resnet18,384 for vit
        d_inner : int, # 256 for resnet18,128 for vit
        n_classes : int = 3,
        d : int = 128,
        dropout_rate : int = 0,
        branches_count : int = 5,
        mask_rate : float = 0.6,
        k : int = 10,
    ) -> None:
        
        super().__init__()

        self.d_features = d_features
        self.d_inner = d_inner
        self.n_classes = n_classes
        self.d = d
        self.dropout_rate = dropout_rate
        self.branches_count = branches_count
        self.mask_rate = mask_rate
        self.k = k

        self.dim_reduction = DimReduction(d_features,d_inner)
        self.attention_gated = AttentionGated(d_inner, d, branches_count)
        
        self.classifier = nn.Sequential(*[ShallowClassifier(d_inner,n_classes,dropout_rate) for _ in range(branches_count)])

        self.stkim = STKIM(branches_count=branches_count,mask_rate=mask_rate,k=k)

        self.slide_classifier = ShallowClassifier(in_features=d_inner, n_classes=n_classes, dropout_rate=dropout_rate)

    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        
        x = x[0]

        x = self.dim_reduction(x)
        a = self.attention_gated(x)

        a = self.stkim(a)

        a_out = a

        a = torch.nn.functional.softmax(a, dim=1)
        afeat = torch.mm(a, x)

        outputs = []

        for i,head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))

        outputs = torch.stack(outputs, dim=0)

        bag_a = torch.nn.functional.softmax(a_out, dim=1).mean(dim = 0, keepdim=True)
        bag_feat = torch.mm(bag_a, x)
        class_ = self.slide_classifier(bag_feat)

        a_out = torch.unsqueeze(a_out, dim=0)

        return outputs,class_,a_out