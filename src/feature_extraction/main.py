import os
import dotenv
import torch
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
sys.path.append('../..')
from src.models import ResNet34
from torchvision.transforms import ToTensor
from torch import nn, Tensor
from tqdm import tqdm
from itertools import product


# Set the path for OpenSlide library (Windows only)
OPENSLIDE_PATH = dotenv.get_key(dotenv.find_dotenv(), "OPENSLIDE_PATH")
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide import open_slide
        from openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide import open_slide
    from openslide.deepzoom import DeepZoomGenerator



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 224
# Path to the folder containing WSI files
WSI_FOLDER = dotenv.get_key(dotenv.find_dotenv(), "WSI_FOLDER")
# Path to save the output grid-based feature maps
GFE_FOLDER = dotenv.get_key(dotenv.find_dotenv(), "GFE_FOLDER")


# Function to extract feature maps from a WSI
def get_vector(model: nn.Module, PATCH: Tensor):

    """
    Extracts feature vectors from patches using the ResNet34 model.

    Args:
       model (nn.Module): The ResNet34 model instance.
       PATCH (Tensor): The input patch tensor.

    Returns:
       Tensor: The extracted feature vector.
    """

    y = model.conv1(PATCH)
    y = model.bn1(y)
    y = model.relu(y)
    y = model.maxpool(y)
    y = model.layer1(y)
    y = model.layer2(y)
    y = model.layer3(y)
    y = model.layer4(y)
    y = model.avgpool(y)
    y = torch.flatten(y)
    return y


model = ResNet34(n_classes=3).to(DEVICE)
# Set the model to evaluation mode
model.eval()

to_tensor = ToTensor()


for filename in os.listdir(WSI_FOLDER):
    if filename.endswith('.svs'):
        slide_path = os.path.join(WSI_FOLDER, filename)
        
        print(f'Processing: {filename}')


        slide = open_slide(slide_path)
        print("Slide:", slide_path)
        print("Dimensions:", slide.level_dimensions)
        
        # Create a DeepZoomGenerator object to extract tiles from the slide
        tiles = DeepZoomGenerator(slide, tile_size=PATCH, overlap=0, limit_bounds=False)
        # Get the number of tiles in the WSI based on the slide dimensions
        W, H = tiles.level_tiles[tiles.level_count - 1]
        print("Tiles:", W, H)
        with torch.inference_mode():
            grid = []
            for h in tqdm(range(H)):
                row = []
                for w in range(W):
                    tile = tiles.get_tile(level=tiles.level_count - 1, address=(w, h))
                    tile = to_tensor(tile).to(DEVICE)
                    # Get a tile from the slide and convert it to a tensor on the specified device
                    vec = get_vector(model.resnet, tile.unsqueeze(0))
                    # Extract the feature vector for the tile
                    row.append(vec)
                # Append the feature vectors to a row, then append the row to the grid
                row_tensor = torch.stack(row, dim=0)
                grid.append(row_tensor)

            # Stack the rows of the grid into a 3D tensor representing the grid-based feature map
            G = torch.stack(grid, dim=0)
        
        # Save the grid-based feature map as a PyTorch tensor file
        torch.save(G, os.path.join(GFE_FOLDER, filename + '.pth'))