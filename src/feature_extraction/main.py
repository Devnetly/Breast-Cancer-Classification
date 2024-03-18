import sys
import torch
sys.path.append('../..')
from src.models import ResNet34
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from torchvision.transforms import ToTensor
from torch import nn,Tensor
from tqdm import tqdm
from itertools import product

wsi_path = "/home/abdelnour/Documents/4eme_anne/S2/projet/data/wsi/BRACS_1003728.svs"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device = {device}")

patch = 224

model = ResNet34(n_classes=3).to(device)

def get_vector(model : nn.Module, patch : Tensor):

    y = model.conv1(patch)
    y = model.bn1(y)
    y = model.layer1(y)
    y = model.layer2(y)
    y = model.layer3(y)
    y = model.layer4(y)
    y = model.avgpool(y)
    y = torch.squeeze(y)

    return y

slide = open_slide(wsi_path)
print(slide.level_dimensions)
tiles = DeepZoomGenerator(slide,tile_size=patch,overlap=0,limit_bounds=False)

W, H = tiles.level_tiles[tiles.level_count - 1]

print(W,H)

to_tensor = ToTensor()

model.eval()

batch_size = 128

vecs = []

batch = []

i = 0

with torch.inference_mode():

    for w,h in tqdm(list(product(range(W - 1),range(H - 1)))):

        if len(batch) < batch_size:
            tile = tiles.get_tile(level=tiles.level_count-1,address=(w,h))
            tile = to_tensor(tile)
            batch.append(tile)
        else:
            batch = torch.stack(batch).to(device)
            vec = get_vector(model.resnet34,batch)
            vecs.append(vec)
            batch = []