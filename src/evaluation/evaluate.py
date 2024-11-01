# Imports
import torch
import numpy as np
import sys
import dotenv
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch import Tensor
sys.path.append('../..')
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor
from torchvision.datasets import ImageFolder
from src.transforms import LabelMapper
from src.utils import load_model_from_folder,load_history_from_folder,predict,create_df,make_metric,compute_metrics,seed_everything
from src.datasets import RoIDataset
from src.training.train_feature_extractor import load_json,Config,create_model,create_dataloaders
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score,precision_score

# Ser up seaborn theme
sns.set_theme(palette="husl",style="darkgrid")

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EXPIREMENT_NAME = "vit_resnet"
BATCH_SIZE = 512
NUM_WORKERS = 6
PREFETCH_FACTOR = 2

# Reproducibility
seed_everything(SEED)

# Load environment variables
PATCHES_FOLDER = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_FOLDER")
EXPERIMENTS_FOLDER = dotenv.get_key(dotenv.find_dotenv(), "EXPERIMENTS_FOLDER")

# Paths

TRAIN_DIR = os.path.join(PATCHES_FOLDER, "train")
VAL_DIR = os.path.join(PATCHES_FOLDER, "val")
TEST_DIR = os.path.join(PATCHES_FOLDER, "test")

EXPIREMENT_PATH = os.path.join(EXPERIMENTS_FOLDER,EXPIREMENT_NAME)
CONFIG_PATH = os.path.join(EXPIREMENT_PATH,"config.json")
HISTORY_PATH = os.path.join(EXPIREMENT_PATH,"history.csv")
CHECKPOINTS_FOLDER = os.path.join(EXPIREMENT_PATH,"checkpoints")
EVALUATION_RESULTS_FOLDER = os.path.join(EXPIREMENT_PATH,"evaluation_results")

os.makedirs(EVALUATION_RESULTS_FOLDER,exist_ok=True)

# Load config
config = Config(**load_json(CONFIG_PATH))

# Load model
model = create_model(config).to(DEVICE).eval()
load_model_from_folder(model,CHECKPOINTS_FOLDER)

# Graphs
history = pd.read_csv(HISTORY_PATH)
metrics = set(history.columns) - set(["epoch","time","split"])

for metric in metrics:
    ax = sns.lineplot(data=history,x="epoch",y=metric,hue="split")
    fig_path = os.path.join(EVALUATION_RESULTS_FOLDER,f"{metric}.png")
    ax.figure.savefig(fig_path)

# Predictions for each patch

loaders = create_dataloaders(config,BATCH_SIZE,NUM_WORKERS,PREFETCH_FACTOR)
test_loader : DataLoader[ImageFolder] = loaders[2]

def predict(
    model: nn.Module,
    loader: DataLoader[ImageFolder],
) -> tuple[Tensor,Tensor]:
    
    Y = []
    labels = []

    model.eval().to(DEVICE)
    
    with torch.inference_mode():
        
        for x,y in tqdm(loader):

            x,y = x.to(DEVICE),y.to(DEVICE)

            y_hat = model(x)

            Y.append(y_hat)
            labels.extend(y.tolist())

    return torch.cat(Y),torch.tensor(labels)

y_hat,labels = predict(model,test_loader)
paths = [path for path,_ in test_loader.dataset.samples]
prediction_df = create_df(paths,labels,y_hat)

prediction_df.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"predictions.csv"),index=False)

# Evaluation

metrics= {
    "accuracy" : accuracy_score,
    "precision_macro" : make_metric(precision_score, average="macro"),
    "precision_micro" : make_metric(precision_score, average="micro"),
    "recall_macro" : make_metric(recall_score, average="macro"),
    "recall_micro" : make_metric(recall_score, average="micro"),
    "f1_macro" : make_metric(f1_score, average="macro"),
    "f1_micro" : make_metric(f1_score, average="micro")
}

# Patch level
patch_metrics = compute_metrics(metrics, prediction_df["label"], prediction_df["predicted_label"])
patch_metrics.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"patch_metrics.csv"))

cm = confusion_matrix(prediction_df["label"], prediction_df["predicted_label"])
ax = sns.heatmap(data=cm,annot=True,fmt=',d',cmap='Blues')
ax.figure.savefig(os.path.join(EVALUATION_RESULTS_FOLDER,"patch_level_confusion_matrix.png"))

# ROI level : soft voting

soft_df = prediction_df[["label","roi","benign","atypical","malignant"]] \
    .groupby(by=["label","roi"]) \
    .mean() \
    .reset_index()

soft_df["predicted_label"] = np.argmax(soft_df[['benign','atypical','malignant']].values, axis=1)
soft_df.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"soft_voting.csv"))

soft_voting_metrics = compute_metrics(metrics, soft_df["label"], soft_df["predicted_label"])
soft_voting_metrics.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"soft_voting_metrics.csv"))

cm = confusion_matrix(soft_df["label"], soft_df["predicted_label"])
ax = sns.heatmap(data=cm,annot=True,fmt=',d',cmap='Blues')
ax.figure.savefig(os.path.join(EVALUATION_RESULTS_FOLDER,"soft_voting_confusion_matrix.png"))

# ROI level : hard voting

def count(x:np.ndarray) -> float:
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]

hard_df = prediction_df[["label","roi","predicted_label"]] \
    .groupby(by=["label","roi"]) \
    .agg(count) \
    .reset_index()

hard_df.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"hard_voting.csv"))

hard_voting_metrics = compute_metrics(metrics, hard_df["label"], hard_df["predicted_label"])
hard_voting_metrics.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"hard_voting_metrics.csv"))

cm = confusion_matrix(hard_df["label"], hard_df["predicted_label"])
ax = sns.heatmap(data=cm,annot=True,fmt=',d',cmap='Blues')
ax.figure.savefig(os.path.join(EVALUATION_RESULTS_FOLDER,"hard_voting_confusion_matrix.png"))