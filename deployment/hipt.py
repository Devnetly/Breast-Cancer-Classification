import torch
import sys
import dotenv
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.tk import tqdm
sys.path.append('..')
from src.models import AttentionModel,HIPT_WSI

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HIPT_WSI(dropout=0.35).to(device).eval()

"""model = AttentionModel(
    num_classes=3,
    dropout=0.2,
    filters_in=512,
    filters_out=64
)"""

state_dict = torch.load("weights/1716450175.7177708.pt", map_location=device)
msg = model.load_state_dict(state_dict=state_dict)
print(msg)

def Predict(tensor):
    
    with torch.inference_mode():
        tensor = tensor.permute(dims=(1,2,0)).reshape(-1, tensor.shape[0]).unsqueeze(0)
        tensor = tensor.to(device)
        y = model(tensor)
        y = torch.nn.functional.softmax(y, dim=1)
    
    return y.cpu()