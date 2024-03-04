from PIL import Image
import math
import numpy as np

class ImageResizer:
    """
    A class to resize the image to the next multiple of 224, so that the images can be divided into 224x224 patches later.
    """

    def __init__(self):
        pass

    def get_new_dimensions(width : int, height : int, patch_height : int = 224, patch_width : int = 224):
        """
        Get the new dimensions of the image after resizing.

        Parameters:
        - width: The width of the image.
        - height: The height of the image.
        - patch_height: The height of the patch.
        - patch_width: The width of the patch.

        Returns:
        - new_height: The new height of the image.
        - new_width: The new width of the image.
        """
    
        width_coef = int(np.round(width / patch_width).astype(np.int32))
        height_coef = int(np.round(height / patch_height).astype(np.int32))

        new_width = width_coef * patch_width
        new_height = height_coef * patch_height

        return new_width, new_height

    def __call__(self, image):
        """
        Resize the given image to the next multiple of 224.

        Parameters:
        - image: an image of type pillow.

        Returns:
        - resized_image: The resized image of type pillow.
        """

        width, height = image.size

        new_width, new_height = ImageResizer.get_new_dimensions(width, height)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        return resized_image
    
# Example usage
"""
image = Image.open('path/to/image.png')
print("original image: ", image.size)
resizer = ImageResizer()
resized_image = resizer.__call__(image)
print(resized_image.size)
"""