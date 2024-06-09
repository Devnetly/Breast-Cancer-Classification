import os
import torch
import sys
import argparse
from torch import nn
from torchvision.models.resnet import ResNet
from torchvision.transforms import ToTensor,Resize,Compose,Normalize
from tqdm.tk import tqdm
sys.path.append('../')
from src.models import ResNet,ResNet18,ResNet34,HIPT_4K
from src.utils import load_model_from_folder
from src.datasets import WSIDataset
from torch.utils.data import DataLoader
import argparse
import tkinter as tk

def create_transforms(model : nn.Module,patch_size : int = 224):

    if isinstance(model, ResNet):
        return Compose([
            Resize(size=(patch_size,patch_size)),
            ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif isinstance(model, HIPT_4K):
        return Compose([
            Resize(size=(patch_size,patch_size)),
            ToTensor(), 
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    else:
        raise Exception(f"{model.__class__.__name__} is not supported.")


def transform_wsis(
    model : nn.Module,
    source_path : str,
    patch_size : int,
    destination_folder : str,
    device : str,
    prefetch_factor : int = 2,
    app : tk.Tk = None
) -> None:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """model = None

    model = ResNet18(n_classes=3)
    load_model_from_folder(model=model, weights_folder="D:\\AIDS\\S2\\Project\\Breast Cancer Detection\\Breast-Cancer-Detection\\models\\resnet18",verbose=True)
    model.resnet.fc = nn.Identity()
    model = model.to(device)"""

    model = HIPT_4K(device).to(device)

    load_model_from_folder(
        model=model, 
        weights_folder="./weights", 
        verbose=True,
        weights_id="hipt.pt"
    )

    processed_wsis = []

    transform = create_transforms(model, patch_size=patch_size)

    model.eval()

    with torch.inference_mode():

            basename = os.path.basename(source_path)

            if app:  # Check if the app is provided
                app.update_progress(f"Processing {source_path} : \n")

            print(f"Processing {source_path} : \n")

            dataset = WSIDataset(wsi_path=source_path,patch_size=patch_size,transform=transform)
            loader = DataLoader(dataset=dataset,batch_size=4, num_workers=0, prefetch_factor=None)
            #process dummy.pth instead of the actual image for testing
            
            matrix = [[[] for _ in range(dataset.width)] for _ in range(dataset.height)]

            for tiles, ws, hs in tqdm(loader, tk_parent=app):
                for i,(tile,w,h) in enumerate(zip(tiles,ws,hs)):
                    tile = tile.unsqueeze(0).to(device)
                    vector = model(tile).squeeze().cpu()
                    matrix[h][w] = vector

                    """for i, (w, h) in enumerate(zip(ws, hs)):
                        matrix[h][w] = vectors[i]"""

                    if app:  # Check if the app is provided
                        app.update_progress(f"Processed {i+1}/{len(loader)} batches")

            matrix = [torch.stack(row, dim=0) for row in matrix]
            matrix = torch.stack(matrix, dim=0)

            path_to_save = destination_folder
            path_to_save,_ = os.path.splitext(path_to_save)
            path_to_save = os.path.join(destination_folder, f"{os.path.basename(path_to_save)}{os.path.extsep}pth")

            if app: 
                app.update_progress(f"\n --- Whole slide image was saved to path : {path_to_save} -- \n")  # Call the update_progress function

            pardir = os.path.dirname(path_to_save)

            processed_wsis.append(basename)

            if not os.path.exists(pardir):
                os.makedirs(pardir, exist_ok=True)

            matrix = matrix.permute(2, 0, 1)
            torch.save(matrix, f=path_to_save)

            del matrix


def main():
    parser = argparse.ArgumentParser(description="Transform whole slide images (WSIs) into feature vectors.")
    parser.add_argument("--source_path", type=str, help="Path to the source WSI file.")
    parser.add_argument("--destination_folder", type=str, help="Path to the destination folder for saving the feature vectors.")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patches to extract from the WSI.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of samples to prefetch from the data loader.")

    args = parser.parse_args()

    model = HIPT_4K(args.device)

    load_model_from_folder(
        model=model, 
        weights_folder="./weights/hipt.pt", 
        verbose=True
    )
    # model.resnet.fc = nn.Identity()
    # model.to(args.device)

    transform_wsis(
        model=model,
        source_path=args.source_path,
        patch_size=args.patch_size,
        destination_folder="./output",
        device=args.device,
        prefetch_factor=args.prefetch_factor
    )

if __name__ == "__main__":
    main()