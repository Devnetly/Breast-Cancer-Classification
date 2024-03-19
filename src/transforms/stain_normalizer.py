import cv2
import torchstain
import torch
from torchvision.transforms import ToTensor,Compose,Lambda

class StainNormalizer:
    """Stain normalizer class using torchstain library."""

    AVAILABLE_METHODS = ['macenko', 'reinhar']

    def __init__(
        self,
        template_img_src : str,
        method : str = 'macenko'
    ) -> None:
        """Initialize the stain normalizer.

            Args:
                template_img_src (str): Path to the template image.
                method (str, optional): Stain normalization method. Defaults to 'macenko'.
        """
        
        if method not in StainNormalizer.AVAILABLE_METHODS:
            raise Exception(f"{method} is not in one of {'.'.join(StainNormalizer.AVAILABLE_METHODS)}")
        
        template_img = cv2.imread(template_img_src)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

        T = Compose([
            ToTensor(),
            Lambda(lambda x : x * 255)
        ])

        self.normalizer = self._create(type_=method)
        self.normalizer.fit(T(template_img))

    def __call__(self, img : torch.Tensor) -> torch.Tensor:
        """Normalize an image tensor.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Normalized image tensor.
        """

        norm, _, _ = self.normalizer.normalize(img)
        norm = norm.permute(2,1,0)
        return norm
    
    def _create(self, type_ : str):
        """Create a stain normalizer instance based on the specified type.

        Args:
            type_ (str): Stain normalization method.

        Returns:
            torchstain.normalizers: Stain normalizer instance.
        """

        if type_ == "macenko":
            return torchstain.normalizers.MacenkoNormalizer(backend='torch')
        elif type_ == "reinhar":
            return torchstain.normalizers.ReinhardNormalizer(backend='torch')
        else:
            raise Exception(f"{type_} is not supported")