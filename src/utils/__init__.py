import pandas as pd
import os
import torch
import warnings
import dotenv
from torch import nn
from .evaluation_utils import compute_metrics,create_df,get_roi_name,load_history_from_folder,make_metric,predict

def history2df(history : dict) -> pd.DataFrame:
    
    dfs = []

    for key in history.keys():
        df = pd.DataFrame(history[key])
        df["split"] = [key for _ in range(len(df))]
        dfs.append(df)

    return pd.concat(dfs)


def load_model_from_folder(
    model: nn.Module,
    weights_folder : str,
    weights_id : str = None,
    verbose : bool = False
) -> None:
    
    if weights_id is None:

        available_weights = os.listdir(weights_folder)
        available_weights = filter(lambda x : x.endswith('.pt'), available_weights)
        available_weights = list(available_weights)

        if len(available_weights) > 0:

            available_weights.sort()
            weights = available_weights[-1]

            if verbose:
                print(f"loading weights with name : {weights}")

            weights = os.path.join(weights_folder, weights)

            state_dict = torch.load(weights)
            model.load_state_dict(state_dict)
        else:
            warnings.warn('no weights are available,keeping random weights.')

    else:

        weights_path = os.path.join(weights_folder, weights_id)

        if not os.path.exists(weights_path):
            raise Exception(f'no such a weights file : {weights_path}')
        
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

def load_envirement_variables() -> tuple[str, str, str]:

    PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), "HISTORIES_DIR")
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")
    DATA_DIR = dotenv.get_key(dotenv.find_dotenv(), "DATA_DIR")

    return PATCHES_DIR,HISTORIES_DIR,MODELS_DIR,DATA_DIR
