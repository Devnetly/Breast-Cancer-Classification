import cv2
import torchstain
import torch
from torchvision.transforms import ToTensor,Compose,Lambda

class StainNormalizer:

    def __init__(
        self,
        template_img_src : str
    ) -> None:
        
        template_img = cv2.imread(template_img_src)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

        T = Compose([
            ToTensor(),
            Lambda(lambda x : x * 255)
        ])

        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.fit(T(template_img))

    def __call__(self, img : torch.Tensor) -> torch.Tensor:
        norm, _, _ = self.normalizer.normalize(img)
        norm = norm.permute(2,0,1)
        return norm