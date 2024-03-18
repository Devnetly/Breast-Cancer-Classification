from PIL import Image
import random

class KRandomRotation:
    """
        performs a random rotation : the degress are either : 0,90,180 or 270
    """

    def __init__(self) -> None:
        pass

    def __call__(self, img : Image.Image) -> Image.Image:
        """
            ### rotates an  image.

            Arguments : 
            - 
        """
        k = int(4 * random.random())
        return img.rotate(angle= k * 90)