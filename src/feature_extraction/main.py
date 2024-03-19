import os
import dotenv
import torch
import sys
from models import ResNet
from torchvision.transforms import ToTensor
from tqdm import tqdm
sys.path.append('../..')


### ugly but necessary
try:
    OPENSLIDE_PATH = dotenv.get_key(dotenv.find_dotenv(), "OPENSLIDE_PATH")
except Exception as e:
    print("Error setting OpenSlide path:", str(e))


if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide import open_slide
        from openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide import open_slide
    from openslide.deepzoom import DeepZoomGenerator

def get_vector(
    model : torch.nn.Module,
    patch : torch.Tensor
) -> torch.Tensor:
        
    y = model.conv1(patch)
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

def transform_wsis(
    model : ResNet,
    root : str,
    patch_size : int,
    tensors_folder : str
) -> None:
    
    to_tensor = ToTensor()

    for file in os.listdir(root):

        filename = os.path.join(root, file)
    
        slide = open_slide(filename=filename)
        tiles = DeepZoomGenerator(slide, tile_size=patch_size, overlap=0, limit_bounds=False)
        W, H = tiles.level_tiles[tiles.level_count - 1]

        model.eval()

        with torch.inference_mode():

            grid = []

            for h in tqdm(range(H)):

                row = []

                for w in range(W):

                    tile = tiles.get_tile(level=tiles.level_count - 1, address=(w, h))
                    tile = to_tensor(tile).to()

                    # Get a tile from the slide and convert it to a tensor on the specified device
                    vec = get_vector(model.resnet, tile.unsqueeze(0))

                    # Extract the feature vector for the tile
                    row.append(vec)

                    # Append the feature vectors to a row, then append the row to the grid
                    row_tensor = torch.stack(row, dim=0)
                    grid.append(row_tensor)
            
                # Stack the rows of the grid into a 3D tensor representing the grid-based feature map
                row_tensor = torch.stack(row, dim=0)
                grid.append(row_tensor)
            
            # Stack the rows of the grid into a 3D tensor representing the grid-based feature map
            G = torch.stack(grid, dim=0)
            G = G.permute(2, 0, 1)

            name,_ = os.path.splitext(filename)

            torch.save(G, os.path.join(tensors_folder, name + '.pth'))

def main(args):

    """
    arguments : 
        - the path of the folder containing the wsi
        - the path of the folder to save the wsis
        - the patch size (default : 224)
        - the model : (resnet18,34 or 50)
        - the path to the model's weights
    """

    # load the model weights
    # call the transform_wsis function

    pass

if __name__ == '__main__':
    # parse the args
    # call the main function
    pass