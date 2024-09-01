import numpy as np
import h5py
import os
import pandas as pd
import torch
import warnings
import random
import json
from torch import nn

def load_json(filename : str) -> dict:
        
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def seed_everything(seed : int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_coords(filename : str) -> np.ndarray:
    
    ds_arr = None

    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()] 

    return ds_arr

def get_metadata(root) -> pd.DataFrame:

    metadata = {
        "name" : [],
        "split" : [],
        "type" : [],
        "subtype" : [],
        "path" : []
    }

    for split in os.listdir(root):

        split_path = os.path.join(root, split)

        for type in os.listdir(split_path):

            type_path = os.path.join(split_path, type)

            for subtype in os.listdir(type_path):

                subtype_path = os.path.join(type_path, subtype)
                names = os.listdir(subtype_path)

                metadata["name"].extend(names)
                metadata["split"].extend([split] * len(names))
                metadata["type"].extend([type] * len(names))
                metadata["subtype"].extend([subtype] * len(names))
                metadata["path"].extend([os.path.join(root,split,type,subtype,name) for name in names])

    metadata = pd.DataFrame(metadata)

    return metadata

def load_model_from_folder(
    model: nn.Module,
    weights_folder : str,
    weights_id : str = None,
    verbose : bool = False
) -> None:
    
    if not weights_id:

        available_weights = os.listdir(weights_folder)
        available_weights = filter(lambda x : x.endswith('.pt') or x.endswith('.pth'), available_weights)
        available_weights = list(available_weights)

        if len(available_weights) > 0:

            available_weights.sort()
            weights = available_weights[-1]

            if verbose:
                print(f"loading weights with name : {weights}")

            weights = os.path.join(weights_folder, weights)

            state_dict = torch.load(weights)

            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            msg = model.load_state_dict(state_dict, strict=False)

            print(msg)

        else:
            warnings.warn('no weights are available,keeping random weights.')

    else:

        weights_path = os.path.join(weights_folder, weights_id)

        if not os.path.exists(weights_path):
            raise Exception(f'no such a weights file : {weights_path}')
        
        state_dict = torch.load(weights_path)
        msg = model.load_state_dict(state_dict, strict=False)
        
        print(msg)