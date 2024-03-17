from PIL import Image
import random

class KRandomRotation:
    """
        performs a random rotation : the degress are either : 0,90,180 or 270
    """

    def __init__(self, probas : list[float]) -> None:
        """
            #### the constructor of the KRandomRotation class.

            Arguments :
            - probas : a list of float of length 4 representing, the probabilities of each degree.

            Returns:
            - None.
        """
        if len(probas) != 4:
            raise Exception(f"Expected length of probas is 4, found {len(probas)}")
        
        self.probas = probas

    def __call__(self, img : Image.Image) -> Image.Image:
        """
            ### rotates an  image.

            Arguments : 
            - 
        """
        k = random.choices(population=[0,1,2,3], weights=self.probas)[0]
        return img.rotate(angle= k * 90)