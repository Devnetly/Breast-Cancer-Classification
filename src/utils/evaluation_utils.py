import os
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

def load_history_from_folder(path : str) -> pd.DataFrame:

    files = os.listdir(path)
    files.sort()

    p = 0
    dfs = []

    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        df["epoch"] = df["epoch"] + p
        dfs.append(df)
        p += len(df["epoch"].value_counts())
        
    return pd.concat(dfs)

def get_roi_name(patch_name : str) -> str:
    name, ext = os.path.splitext(patch_name)
    roi_name = '_'.join(name.split('_')[:-1])
    return roi_name + ext

def create_df(paths : list[str],labels : list[int], y_hat : Tensor) -> pd.DataFrame:

    df = pd.DataFrame()
    df["patch_name"] = [os.path.basename(path) for path in paths]
    df["label"] = labels

    df["roi"] = df["patch_name"].apply(get_roi_name)

    df["benign"] = y_hat[:,0].tolist()
    df["atypical"] = y_hat[:,1].tolist()
    df["malignant"] = y_hat[:,2].tolist()

    df['predicted_label'] = torch.argmax(y_hat, dim=1).tolist()

    return df

def make_metric(metric,**kwargs): 

    def _metric(y, y_hat):
        kwargs["y_true"] = y
        kwargs["y_pred"] = y_hat
        return metric(**kwargs)
    
    return _metric

def compute_metrics(
    metrics : dict, 
    y : np.ndarray, 
    y_hat : np.ndarray
) -> pd.Series:

    results = {}

    for name, metric in metrics.items():
        results[name] = metric(y, y_hat)

    return pd.Series(results)

def predict(
    model : nn.Module,
    dataloader : DataLoader,
    device = torch.device
) -> tuple[pd.DataFrame, list[str]]:
    
    Y = []
    paths = []
    labels = []

    model.eval()
    
    with torch.inference_mode():
        
        for path,x,y in tqdm(dataloader):

            x,y = x.to(device),y.to(device)

            y_hat = model(x)
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)

            Y.append(y_hat.cpu())
            paths.extend(path)
            labels.extend(y.cpu().tolist())

    return paths,torch.vstack(Y),labels