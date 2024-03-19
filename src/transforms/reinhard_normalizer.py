import cv2
import numpy as np
from PIL import Image

class ReinhardNotmalizer:

    def __init__(
        self,
        template_img_src : str
    ) -> None:
        
        template_img = cv2.imread(template_img_src)
        self.template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
        
        self.mean, self.std = self._get_mean_std(self.template_img)

    def __call__(self, img : Image.Image) -> Image.Image:

        np_img = np.asanyarray(img)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2LAB)

        mean, std = self._get_mean_std(np_img)
        
        np_img = (np_img - mean) * (self.std / std) + self.mean

        np_img = np.round(np_img, 2)

        np_img = np.where(np_img < 0, 0, np_img)
        np_img = np.where(np_img > 255, 255, np_img)

        np_img = np_img.astype(np.uint8)
        
        np_img = cv2.cvtColor(np_img, cv2.COLOR_LAB2RGB)

        pil_img = Image.fromarray(np_img, mode='RGB')

        return pil_img


    def _get_mean_std(self, x):

        mean, std = cv2.meanStdDev(x)

        mean = np.around(np.squeeze(mean), decimals=2)
        std = np.around(np.squeeze(std), decimals=2)

        return mean, std