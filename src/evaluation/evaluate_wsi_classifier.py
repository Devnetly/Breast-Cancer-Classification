import torch
import sys
import dotenv
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../..')
from src.models import AttentionModel
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from src.utils import load_model_from_folder,seed_everything,load_json
from src.datasets import TensorDataset
from tqdm.notebook import tqdm
from torchmetrics import F1Score,Accuracy,AUROC,Metric,Precision,Recall
from argparse import ArgumentParser
from src.training.train_wsi_classifier import Config,create_model
from sklearn.metrics import confusion_matrix

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    return parser.parse_args()

args = get_args()

# Ser up seaborn theme
sns.set_theme(palette="husl",style="darkgrid")

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EXPIREMENT_NAME = args.experiment
BATCH_SIZE = 512
NUM_WORKERS = 6
PREFETCH_FACTOR = 2

# Reproducibility
seed_everything(SEED)

# Load environment variables
EXPERIMENTS_FOLDER = dotenv.get_key(dotenv.find_dotenv(), "EXPERIMENTS_FOLDER")

# Paths
EXPIREMENT_PATH = os.path.join(EXPERIMENTS_FOLDER,EXPIREMENT_NAME)
CONFIG_PATH = os.path.join(EXPIREMENT_PATH,"config.json")
HISTORY_PATH = os.path.join(EXPIREMENT_PATH,"history.csv")
CHECKPOINTS_FOLDER = os.path.join(EXPIREMENT_PATH,"checkpoints")
EVALUATION_RESULTS_FOLDER = os.path.join(EXPIREMENT_PATH,"evaluation_results")

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

# Predictions

datatset = val_dataset = TensorDataset(
    root=config.data_dir,
    tensor_transform=Lambda(lambda x: x),
    split='val',
    output_type='auto'
)

loader = DataLoader(dataset=datatset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)

def predict(
    loader : DataLoader,
    model : AttentionModel
) -> tuple[torch.Tensor,torch.Tensor]:
    
    model.eval()

    Y_hat = []
    Y = []

    with torch.inference_mode():

        for x,y in tqdm(loader):

            x,y = x.to(DEVICE),y.to(DEVICE)

            y_hat = model(x)
            y_hat = torch.squeeze(y_hat)

            y = torch.squeeze(y).item()

            Y.append(y)
            Y_hat.append(y_hat)
    
    Y_hat = torch.vstack(Y_hat)
    Y = torch.tensor(Y)
    Y = Y.squeeze()
    Y = Y.to(DEVICE)

    return Y_hat,Y

Y_hat,Y = predict(loader,model)

# Evaluation

metrics = {
    "accuracy" : Accuracy(task='multiclass',num_classes=3).to(DEVICE),
    "f1_score" : F1Score(task='multiclass',num_classes=3,average='macro').to(DEVICE),
    "auc" : AUROC(task='multiclass',num_classes=3).to(DEVICE),
    "precision" : Precision(task='multiclass',num_classes=3,average='macro').to(DEVICE),
    "recall" : Recall(task='multiclass',num_classes=3,average='macro').to(DEVICE)
}

def compute_metrics(metrics : dict[str,Metric], y : torch.Tensor, y_hat : torch.Tensor) -> pd.Series:

    result = {}

    for name, metric in metrics.items():
        score = metric(y_hat, y)
        result[name] = score.item()

    return  pd.Series(result)

metrics = compute_metrics(metrics,Y,Y_hat)
metrics.to_csv(os.path.join(EVALUATION_RESULTS_FOLDER,"metrics.csv"))

# Confusion matrix
cm = confusion_matrix(Y.cpu().numpy(),Y_hat.argmax(dim=1).cpu().numpy())
ax = sns.heatmap(cm,annot=True,fmt=',d',cmap='Blues')
ax.figure.savefig(os.path.join(EVALUATION_RESULTS_FOLDER,"confusion_matrix.png"))
