import torch
import argparse
import sys
import torchmetrics
import time
import os
sys.path.append('../..')
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.models import AttentionModel
from src.datasets import TensorDataset
from src.trainer import Trainer
from src.utils import history2df,load_model_from_folder
from helpers import load_envirement_variables

class DEFAULTS:
    DROPOUT = 0.2
    FILTERS_IN = 512
    FILTERS_OUT = 64
    LEARNING_RATE = 0.00001
    WEIGHT_DEACY = 1e-3
    EPOCHS = 10

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def create_loader(
    train_dir : str,
    val_dir : str
) -> tuple[DataLoader, DataLoader]:
    
    train_data = TensorDataset(root=train_dir)
    val_data = TensorDataset(root=val_dir)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)

    return train_loader,val_loader

def main(args):
    
    _,HISTORIES_DIR,MODELS_DIR,_ = load_envirement_variables()

    histories_folder = os.path.join(HISTORIES_DIR,args.histories_folder)
    weights_folder = os.path.join(MODELS_DIR,args.weights_folder)

    if not os.path.exists(histories_folder):
        raise Exception(f'no such a folder {histories_folder}')
    
    if not os.path.exists(weights_folder):
        raise Exception(f'no such a folder {weights_folder}')

    model = AttentionModel(
        num_classes=GLOBAL.NUM_CLASSES,
        dropout=args.dropout,
        filters_in=args.filters_in,
        filters_out=args.filters_out
    ).to(GLOBAL.DEVICE)

    load_model_from_folder(model=model,weights_folder=weights_folder,verbose=True)

    train_dir = "/home/abdelnour/Documents/4eme_anne/S2/projet/data/tensors"
    val_dir = "/home/abdelnour/Documents/4eme_anne/S2/projet/data/tensors"

    train_loader, val_loader = create_loader(train_dir,val_dir)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=DEFAULTS.WEIGHT_DEACY)

    loss = torch.nn.CrossEntropyLoss()

    accuracy = torchmetrics \
        .Accuracy(num_classes=GLOBAL.NUM_CLASSES, task='multiclass') \
        .to(GLOBAL.DEVICE)

    trainer = Trainer() \
        .set_optimizer(optimizer=optimizer) \
        .set_loss(loss) \
        .set_device(GLOBAL.DEVICE) \
        .add_metric(name="accuracy", metric=accuracy)
    
    trainer.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs
    )

    # Save the history
    t = time.time()

    history2df(trainer.history).to_csv(
        os.path.join(histories_folder,f"{t}.csv"), 
        index=False
    )

    # Save the model
    torch.save(model.state_dict(),os.path.join(weights_folder,f"{t}.pt"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights-folder", type=str, required=True)
    parser.add_argument("--histories-folder", type=str, required=True)
    parser.add_argument("--dropout", type=float, default=DEFAULTS.DROPOUT)
    parser.add_argument("--filters-in", type=int, default=DEFAULTS.FILTERS_IN)
    parser.add_argument("--filters-out", type=int, default=DEFAULTS.FILTERS_OUT)
    parser.add_argument('--learning-rate', type=float, default=DEFAULTS.LEARNING_RATE)
    parser.add_argument('--weight-decay', type=float, default=DEFAULTS.WEIGHT_DEACY)
    parser.add_argument('--epochs', type=int, default=DEFAULTS.EPOCHS)

    args = parser.parse_args()

    main(args)