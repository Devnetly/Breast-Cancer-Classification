import torch
import argparse
import sys
import torchmetrics
import dotenv
import time
import os
sys.path.append('../../..')
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.models import AttentionModel
from src.datasets import TensorDataset
from src.trainer import Trainer
from src.utils import history2df,load_model_from_folder,load_envirement_variables
from src.transforms import Pipeline,Transpose,Flip,LeftShift,RightShift,UpShift,DownShift
from torchvision.transforms import Lambda,RandomChoice

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

def create_loaders(
    train_dir : str,
    val_dir : str,
    num_workers : int,
    prefetch_factor : int
) -> tuple[DataLoader, DataLoader]:
        
    train_transform = Pipeline(transfroms=[
        Lambda(lambd=lambda x : torch.permute(x, dims=(1, 2, 0))),
        RandomChoice(transforms=[
            Pipeline([
                Transpose(dim0=0,dim1=1),
                Flip(dims=(1,))
            ]),
            Transpose(dim0=0,dim1=1),
            Pipeline([
                Transpose(dim0=0,dim1=1),
                Flip(dims=(0,))
            ]),
            Flip(dims=(0,)),
            Flip(dims=(1,)),
            LeftShift(shift=3),
            RightShift(shift=3),
            UpShift(shift=3),
            DownShift(shift=3),
        ]),
        Lambda(lambd=lambda x : torch.permute(x, dims=(2, 0, 1))),
        Lambda(lambd=lambda x : torch.unsqueeze(x, dim=0)),
    ])

    val_transform = Lambda(lambd=lambda x : torch.unsqueeze(x, dim=0))
    
    train_data = TensorDataset(root=train_dir,transform=train_transform)
    val_data = TensorDataset(root=val_dir,transform=val_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,num_workers=num_workers,prefetch_factor=prefetch_factor)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True,num_workers=num_workers,prefetch_factor=prefetch_factor)

    return train_loader,val_loader

def main(args):
    
    _,HISTORIES_DIR,MODELS_DIR,_ = load_envirement_variables()
    GFE_FOLDER = dotenv.get_key(dotenv.find_dotenv(), "GFE_FOLDER")

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

    train_dir = os.path.join(GFE_FOLDER, 'train')
    val_dir = os.path.join(GFE_FOLDER, 'val')

    train_loader, val_loader = create_loaders(train_dir,val_dir,args.num_workers,args.prefetch_factor)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=DEFAULTS.WEIGHT_DEACY)

    loss = torch.nn.CrossEntropyLoss()

    accuracy = torchmetrics \
        .Accuracy(num_classes=GLOBAL.NUM_CLASSES, task='multiclass') \
        .to(GLOBAL.DEVICE)
    
    f1_score = torchmetrics \
        .F1Score(num_classes=GLOBAL.NUM_CLASSES,task='multiclass',average='macro') \
        .to(GLOBAL.DEVICE)

    trainer = Trainer() \
        .set_optimizer(optimizer=optimizer) \
        .set_loss(loss) \
        .set_device(GLOBAL.DEVICE) \
        .add_metric(name="accuracy", metric=accuracy) \
        .add_metric(name="f1_score", metric=f1_score) \
        .set_save_best_weights(True) \
        .set_score_metric("f1_score")
    
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
    torch.save(trainer.best_weights,os.path.join(weights_folder,f"{t}_best_weights.pt"))


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
    parser.add_argument('--num-workers', type=int,default=0)
    parser.add_argument('--prefetch-factor', type=int)

    args = parser.parse_args()

    main(args)