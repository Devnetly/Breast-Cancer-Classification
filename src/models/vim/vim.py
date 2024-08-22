import torch
from vision_mamba import Vim

class VisionMamba(torch.nn.Module):
    def __init__(self, device, dim=256,  # Dimension of the transformer model
    heads=8,  # Number of attention heads
    dt_rank=32,  # Rank of the dynamic routing matrix
    dim_inner=256,  # Inner dimension of the transformer model
    d_state=256,  # Dimension of the state vector
    num_classes=1000,  # Number of output classes
    image_size=224,  # Size of the input image
    patch_size=16,  # Size of each image patch
    channels=3,  # Number of input channels
    dropout=0.1,  # Dropout rate
    depth=12,  # Depth of the transformer model
    ):
        super().__init__()
        self.model = Vim(
            dim=dim,  # Dimension of the transformer model
            heads=heads,  # Number of attention heads
            dt_rank=dt_rank,  # Rank of the dynamic routing matrix
            dim_inner=dim_inner,  # Inner dimension of the transformer model
            d_state=d_state,  # Dimension of the state vector
            num_classes=num_classes,  # Number of output classes
            image_size=image_size,  # Size of the input image
            patch_size=patch_size,  # Size of each image patch
            channels=channels,  # Number of input channels
            dropout=dropout,  # Dropout rate
            depth=depth,  # Depth of the transformer model
        )
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        return self.model.forward(x)

    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        return img, img.shape[2], img.shape[3]