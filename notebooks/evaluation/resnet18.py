#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import sys
import dotenv
import os
sys.path.append('../..')
from src.models import ResNet34,ResNet18
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose,ToTensor
from src.transforms import LabelMapper,make_patches,ImageResizer
from typing import Any, Tuple
from src.utils import load_model_from_folder
from tqdm import tqdm


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)


# In[3]:


model = ResNet34(n_classes=3).to(device)


# In[4]:


PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")
TRAIN_DIR = os.path.join(PATCHES_DIR, "train")
VAL_DIR = os.path.join(PATCHES_DIR, "val")

print(MODELS_DIR)
print(PATCHES_DIR)
print(TRAIN_DIR)
print(VAL_DIR)


# In[5]:


weights_folder = os.path.join(MODELS_DIR, "resnet34")
load_model_from_folder(model, weights_folder, verbose=True)


# In[6]:


summary(model, input_size=(3,224,224), device=device.type)


# In[7]:


label_mapper = LabelMapper({
    0:0, # 0 is the label for benign (BY)
    1:0, 
    2:0,
    3:1, # 1 is the label for atypical (AT)
    4:1,
    5:2, # 2 is the label for malignant (MT)
    6:2,
})


# In[8]:


class RoIDataset(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        x,y = super().__getitem__(index)
        path = self.imgs[index][0]
        basename = os.path.basename(path)
        name,extention = os.path.splitext(basename)
        original_roi = '_'.join(name.split('_')[:-1]) + extention

        return basename,original_roi,x,y


# In[9]:


dataset = ImageFolder(
    root=VAL_DIR,
    target_transform=label_mapper,
    transform=Compose([
        ImageResizer(),
        ToTensor()
    ])
)


# In[10]:


def collate_fn(batch : list[tuple[torch.Tensor,int]]) -> tuple[torch.Tensor,torch.Tensor,list[int]]:

    X = []
    Y = []
    splits = []

    for x, y in batch:
        patches = make_patches(x, 224, 224, return_tensor=True)
        X.append(patches)
        Y.append(y)
        splits.append(len(patches))

    X = torch.concat(X).type(torch.float)
    Y = torch.Tensor(Y).type(torch.long)

    return X, Y, splits


# In[11]:


loader = DataLoader(dataset=dataset, batch_size=128)


# In[12]:


class SoftVoter(nn.Module):

    def __init__(self, base : nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x : torch.Tensor, splits : list[int]) -> torch.Tensor:

        y_hat = self.base(x)
        y_hat = nn.functional.softmax(y_hat, dim=1)
        y_hat = torch.stack([t.mean(dim=0) for t in torch.split(y_hat, splits)])

        return y_hat


# In[13]:


voter = SoftVoter(base=model)


# In[14]:


def predict(
    voter : nn.Module,
    dataloader : DataLoader
):
    
    Y = []

    for x, y, splits in tqdm(dataloader):
        x = x.to(device)
        y_hat = voter(x, splits)
        Y.extend(y_hat.tolist())

    return y


# In[15]:


def predict2(
    model : nn.Module,
    dataloader : DataLoader
) -> pd.DataFrame:
    
    result = {
        "patch_name" : [],
        "roi_name" : [],
        "benign" : [],
        "atypical" : [],
        "malignant" : [],
        "label" : []
    }

    model.eval()
    
    with torch.inference_mode():
        
        for x,y in tqdm(dataloader):

            x,y = x.to(device),y.to(device)

            y_hat = model(x)
            y_hat = torch.nn.functional.softmax(x, dim=1)

    return pd.DataFrame(result)


# In[ ]:


result = predict2(model, loader)

