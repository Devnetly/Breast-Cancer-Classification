import torch
import sys
import dotenv
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.tk import tqdm
sys.path.append('..')
from src.models import AttentionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AttentionModel(
    num_classes=3,
    dropout=0.2,
    filters_in=512,
    filters_out=64
)
model.load_state_dict(torch.load("1713023191.1225462_best_weights.pt", map_location=device))

def Predict(tensor):
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    y = model(tensor)
    y = torch.nn.functional.softmax(y, dim=1)
    
    return y