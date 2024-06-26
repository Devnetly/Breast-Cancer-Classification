import torch
import argparse
import sys
import torchmetrics
import dotenv
import time
import os
sys.path.append('../../..')
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.models import AttentionModel,MultiBranchAttention,HIPT_WSI
from src.losses import MBALoss
from src.datasets import TensorDataset
from src.trainer import Trainer
from src.utils import history2df,load_model_from_folder,load_envirement_variables
from src.transforms import Pipeline,Transpose,Flip,LeftShift,RightShift,UpShift,DownShift
from torchvision.transforms import Lambda,RandomChoice
from src.metrics import MBAAcc
from src.schedulers import CosineScheduler
from torchsampler import ImbalancedDatasetSampler
from pprint import pprint

class DEFAULTS:
    DROPOUT = 0.2
    FILTERS_IN = 512
    FILTERS_OUT = 64
    LEARNING_RATE = 0.00001
    WEIGHT_DEACY = 1e-3
    EPOCHS = 10
    MASK_RATE = 0.6
    K = 10
    BRANCHES_COUNT = 5
    D = 128
    LAST_EPOCH = 0
    SAMPLER = "random"
    MIL_LR = 0.000005
    DECAY_ALPHA = 3
    DATA_AUGMENTATION = True

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def create_loaders(
    model : str,
    train_dir : str,
    val_dir : str,
    num_workers : int,
    prefetch_factor : int,
    sampler : str,
    data_augmentation : bool = True
) -> tuple[DataLoader, DataLoader]:
    
    train_transform,val_transform = None,None
        
    if model == "ABNN" or model == "HIPT":
        
        if data_augmentation:
            train_transform = [
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
            ]
        else:
            train_transform = []

        if model == "ABNN":
            train_transform.append(Lambda(lambd=lambda x : torch.unsqueeze(x, dim=0)))
            val_transform = Lambda(lambd=lambda x : torch.unsqueeze(x, dim=0))
        else:
            train_transform.append(Lambda(lambd=lambda x : torch.permute(x, dims=(1,2,0)).reshape(-1, x.shape[0])))
            val_transform = Lambda(lambd=lambda x : torch.permute(x, dims=(1,2,0)).reshape(-1, x.shape[0]))


        train_transform = Pipeline(train_transform)

    train_data = TensorDataset(root=train_dir,transform=train_transform)
    val_data = TensorDataset(root=val_dir,transform=val_transform)

    train_loader = None

    if sampler == "random":
        train_loader = DataLoader(dataset=train_data, batch_size=1,shuffle=True,num_workers=num_workers,prefetch_factor=prefetch_factor)
    elif sampler == "balanced":
        sampler = ImbalancedDatasetSampler(dataset=train_data)
        train_loader = DataLoader(dataset=train_data, batch_size=1,sampler=sampler,num_workers=num_workers,prefetch_factor=prefetch_factor)
    else:
        raise Exception(f"{sampler} is not supported.")

    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True,num_workers=num_workers,prefetch_factor=prefetch_factor)

    return train_loader,val_loader

def create_model(args) -> tuple[nn.Module,nn.Module]:

    model,loss = None,None
    
    if args.model == "ABNN":

        model = AttentionModel(
            num_classes=GLOBAL.NUM_CLASSES,
            dropout=args.dropout,
            filters_in=args.filters_in,
            filters_out=args.filters_out
        ).to(GLOBAL.DEVICE)

        loss = torch.nn.CrossEntropyLoss()

    elif args.model == "ACMIL":

        if args.features in ["vit"]:
            d_features,d_inner = 384,128
        elif args.features in ['resnet18','resnet34']:
            d_features,d_inner = 512,256
        elif args.features in ['resnet50']:
            d_features,d_inner = 2048,384
        
        model = MultiBranchAttention(
            d_features=d_features,
            d_inner=d_inner,
            mask_rate=args.mask_rate,
            k=args.k,
            branches_count=args.branches_count,
            dropout_rate=args.dropout,
            d=args.d
        ).to(GLOBAL.DEVICE)

        loss = MBALoss(branches_count=model.branches_count,device=GLOBAL.DEVICE)

    elif args.model == "HIPT":

        model = HIPT_WSI(
            dropout=args.dropout,
            n_classes=GLOBAL.NUM_CLASSES
        ).to(GLOBAL.DEVICE)

        loss = torch.nn.CrossEntropyLoss()

    else:
        raise Exception(f"{args.model} is not a valid model name.")
    
    return model,loss

def main(args):

    print(f"Starting training with args : \n")
    pprint(args)
    print()
    
    _,HISTORIES_DIR,MODELS_DIR,_ = load_envirement_variables()
    DATA_DIR = dotenv.get_key(dotenv.find_dotenv(), "DATA_DIR")

    tensors_folder = os.path.join(DATA_DIR, args.tensors)    
    histories_folder = os.path.join(HISTORIES_DIR,args.histories_folder)
    weights_folder = os.path.join(MODELS_DIR,args.weights_folder)
    best_weights_folder = os.path.join(MODELS_DIR,args.weights_folder, "best_weights")

    if not os.path.exists(histories_folder):
        raise Exception(f'no such a folder {histories_folder}')
    
    if not os.path.exists(weights_folder):
        raise Exception(f'no such a folder {weights_folder}')
    
    if not os.path.exists(best_weights_folder):
        os.mkdir(best_weights_folder)

    model,loss = create_model(args)

    load_model_from_folder(model=model,weights_folder=weights_folder,verbose=True)

    train_dir = os.path.join(tensors_folder, 'train')
    val_dir = os.path.join(tensors_folder, 'val')

    print(f"sampler = {args.sampler} \n")

    train_loader, val_loader = create_loaders(args.model,train_dir,val_dir,args.num_workers,args.prefetch_factor,args.sampler,args.data_augmentation)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=DEFAULTS.WEIGHT_DEACY)

    if args.model in ["ABNN","HIPT"]:
        accuracy = torchmetrics \
            .Accuracy(num_classes=GLOBAL.NUM_CLASSES, task='multiclass') \
            .to(GLOBAL.DEVICE)
    else:
        accuracy = MBAAcc(num_classes=GLOBAL.NUM_CLASSES, task='multiclass').to(GLOBAL.DEVICE)

    scheduler = None

    if args.use_lr_decay:

        print("\n Using LR decay \n")
        print(f'Creating schedulers with last_epoch = {args.last_epoch} \n')

        scheduler = CosineScheduler(
            optimizer=optimizer,
            lr=args.learning_rate,
            num_steps_per_epoch=len(train_loader),
            last_epoch=args.last_epoch,
            alpha=args.decay_alpha,
            min_lr=args.min_lr
        )
    print(f"Starting training with device = {GLOBAL.DEVICE} \n")
    trainer = Trainer(save_weight_every=args.save_weights_every,weights_folder=weights_folder) \
        .set_optimizer(optimizer=optimizer) \
        .set_loss(loss) \
        .set_device(GLOBAL.DEVICE) \
        .add_metric(name="accuracy", metric=accuracy) \
        .set_save_best_weights(True) \
        .set_score_metric("accuracy") \
        .set_scheduler(scheduler)
    
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

    print(f"Best epoch = {trainer.best_epoch+1}, with score = {trainer.last_best_score}")

    # Save the model
    torch.save(model.state_dict(),os.path.join(weights_folder,f"{t}.pt"))
    torch.save(trainer.best_weights,os.path.join(best_weights_folder,f"{t}_epoch={trainer.best_epoch+1}.pt"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### General Parameters
    parser.add_argument("--model", type=str, choices=["ABNN","ACMIL","HIPT"], default="ABNN")
    parser.add_argument("--tensors", type=str, required=True)
    parser.add_argument("--weights-folder", type=str, required=True)
    parser.add_argument("--histories-folder", type=str, required=True)
    parser.add_argument('--epochs', type=int, default=DEFAULTS.EPOCHS)
    parser.add_argument('--num-workers', type=int,default=0)
    parser.add_argument('--prefetch-factor', type=int, default=None)
    parser.add_argument('--last-epoch', type=int, default=DEFAULTS.LAST_EPOCH)
    parser.add_argument('--save-weights-every', type=int)

    ### For All models
    parser.add_argument("--dropout", type=float, default=DEFAULTS.DROPOUT)
    parser.add_argument('--learning-rate', type=float, default=DEFAULTS.LEARNING_RATE)
    parser.add_argument('--weight-decay', type=float, default=DEFAULTS.WEIGHT_DEACY)
    parser.add_argument('--sampler', type=str, choices=['random','balanced'], default=DEFAULTS.SAMPLER)
    parser.add_argument("--min-lr", type=float, default=DEFAULTS.MIL_LR)
    parser.add_argument('--decay-alpha', type=float, default=DEFAULTS.DECAY_ALPHA)

    ### ABNN
    parser.add_argument("--filters-in", type=int, default=DEFAULTS.FILTERS_IN)
    parser.add_argument("--filters-out", type=int, default=DEFAULTS.FILTERS_OUT)

    ### ACMIL
    parser.add_argument("--use-lr-decay", type=lambda t : t.lower() == "true", default="true")
    parser.add_argument("--features", type=str, choices=["resnet18","resnet34","resnet50","vit","hipt_4k"], required=False)
    parser.add_argument("--mask-rate", type=float, default=DEFAULTS.MASK_RATE)
    parser.add_argument("--branches-count", type=float, default=DEFAULTS.BRANCHES_COUNT)
    parser.add_argument("--k", type=int, default=DEFAULTS.K)
    parser.add_argument("--d", type=int, default=DEFAULTS.D)

    ### HIPT and ACMIL
    parser.add_argument("--data-augmentation", type=lambda t : t.lower() == "true", default="true")

    args = parser.parse_args()

    main(args)