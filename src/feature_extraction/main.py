import torch
import pandas as pd
import os
import sys
import numpy as np
import time
import timm
sys.path.append('../..')
from torch import nn,Tensor
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataclasses  import dataclass
from src.utils import get_metadata,load_model_from_folder
from src.datasets import WSIDataset
from typing import Optional
from tqdm import tqdm
from torchvision import transforms as T

@dataclass
class FeatureExtractionArgs:

    in_path : str
    out_path : str
    coords_folder : str
    train : bool
    test : bool
    val : bool
    patch_size : int
    model : str
    model_weights : str
    metadata_path : str
    n : int
    batch_size : int
    num_workers : int
    prefetch_factor : int

def gbfe(
    model : nn.Module,
    loader : DataLoader,
    device : torch.device
) -> Tensor:

    patch_size = loader.dataset.patch_size
    embed_size = model(torch.zeros(1,3,patch_size,patch_size).to(device)).shape[1]
    matrix = torch.zeros((loader.dataset.height, loader.dataset.width, embed_size))

    for tiles, ws, hs in tqdm(loader):

        tiles = tiles.to(device)

        with torch.inference_mode():
            features = model(tiles)

        for i,(w, h) in enumerate(zip(ws,hs)):

            w = int(w.item() / patch_size)
            h = int(h.item() / patch_size)
            
            matrix[h,w,:] = features[i].to('cpu')

    return matrix

def fe_with_patch_selection(
    model : nn.Module,
    loader : DataLoader,
    device : torch.device
) -> Tensor:
    
    vector = []

    for tiles, _, _ in tqdm(loader):

        tiles = tiles.to(device)

        with torch.inference_mode():
            vectors = model(tiles).to('cpu')

        vector.append(vectors)

    vector = torch.cat(vector)

    return vector

def feature_extract(
    model : nn.Module,
    root : str,
    coords_folder : Optional[str],
    patch_size : int,
    tensors_folder : str,
    metdata_file : str,
    n : int,
    device : torch.device,
    batch_size : int,
    num_workers : int,
    prefetch_factor : int,
    train : bool,
    test : bool,
    val : bool
) -> None:
    
    if not train and not test and not val:
        raise ValueError("At least one of train, test, or val must be True.")

    ### Load the metadata
    if os.path.exists(metdata_file):
        metadata = pd.read_csv(metdata_file)
    else:
        metadata = get_metadata(root)
        metadata["time"] = np.zeros(metadata.shape[0])
        metadata["processed"] = np.zeros(metadata.shape[0]).astype(bool)

    ### Select paths
    splits = {
        "train" : train,
        "test" : test,
        "val" : val
    }

    mask = (metadata["split"].map(splits)) & (~metadata["processed"])
    metadata_subset = metadata[mask].head(n)

    print(f"Processing {len(metadata_subset)} images.")

    ### Preprocessing
    transform = T.Compose([
        T.Resize((patch_size,patch_size)),
        T.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ### Feature Extraction Loop
    for _,row in metadata_subset.iterrows():

        print(f"\nProcessing {row["path"]}")

        ### Dataset creation
        coords_path = os.path.join(
            coords_folder,
            row["split"],
            row["type"],
            row["subtype"],
            "patches",
            os.path.splitext(row["name"])[0] + '.h5'
        ) if coords_folder is not None else None

        dataset = WSIDataset(
            wsi_path = row["path"],
            patch_size = patch_size,
            transform=transform,
            coords_path = coords_path
        )

        ### Dataloader
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor
        )

        ### Feature Extraction
        tensor = None

        tic = time.time()

        if coords_folder is None:
            tensor = gbfe(model,dataloader,device)
        else:
            tensor = fe_with_patch_selection(model,dataloader,device)

        toc = time.time()

        ### Save tensor
        tensor_path = os.path.join(
            tensors_folder,
            row["split"],
            row["type"],
            row["subtype"],
            os.path.splitext(row["name"])[0] + '.pt'
        )

        folder_path = os.path.dirname(tensor_path)
        os.makedirs(folder_path,exist_ok=True)

        torch.save(tensor, f=tensor_path)

        del tensor

        print(f"Tensor was saved to path : {tensor_path} in {(toc - tic):.2f} seconds.")

        ### Mark Tensor as processed
        idx = metadata.index[metadata["name"] == row["name"]].to_list()[0]
        metadata.loc[idx,"processed"] = True
        metadata.loc[idx,"time"] = toc - tic

        ### Save metadata
        metadata.to_csv(metdata_file,index=False)

def main(args : FeatureExtractionArgs):

    ### Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Load model
    model = timm.create_model(args.model,pretrained=False,num_classes=0).eval().to(device)
    load_model_from_folder(model,weights_folder=args.model_weights)

    ### Start Feature Extraction
    feature_extract(
        model=model,
        root=args.in_path,
        coords_folder=args.coords_folder,
        patch_size=args.patch_size,
        tensors_folder=args.out_path,
        metdata_file=args.metadata_path,
        n=args.n,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        train=args.train,
        test=args.test,
        val=args.val
    )

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--coords-folder', type=str, required=False)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model-weights', type=str, required=True)
    parser.add_argument('--metadata-path', type=str, required=True)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=None)

    args = parser.parse_args()

    main(args)