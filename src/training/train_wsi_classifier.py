import os
import sys
import torch
import numpy as np
import timm
import logging
import dotenv
sys.path.append('../..')
from torch import optim,nn
from argparse import ArgumentParser
from dataclasses import dataclass
from torchvision import transforms as T
from torch.utils.data import DataLoader,WeightedRandomSampler
from typing import Optional,Any
from src.transforms import Transpose,Flip,LeftShift,RightShift,UpShift,DownShift
from src.datasets import TensorDataset,FakeTensorDataset
from src.models.attention import *
from src.models.multi_branch_attention import *
from src.models.hipt import *
from src.utils import seed_everything,load_json
from src.trainer import Trainer
from src.schedulers import CosineScheduler
from src.losses import MBALoss
from torchmetrics import Accuracy,F1Score

env = dotenv.find_dotenv()
logging.basicConfig(level=logging.INFO)

@dataclass
class Config:

    ### SEED
    seed : int = 42

    ### Device
    device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Model
    model_name : str = "abnn" # Model name can be : ABNN,HIPT or ACMIL
    model_params : dict = None

    ### Scheduler
    use_scheduler : bool = True
    decay_alpha : float = 3.0
    min_lr : float = 5 * 1e-6

    ### Optimizer
    learning_rate : float = 1e-3
    weight_decay : float = 1e-3

    ### Data loading
    sampler : str = "random"
    num_workers : int = 0
    prefetch_factor : int = None

    ### Data augmentation & preprocessing
    data_augmentation : bool = False

    ### Dataset
    data_dir : str = ""

    ### Training
    total_epochs : int = 100

    ### Checkpoints
    save_every : Optional[int] = None
    save_on_best : bool = False

@dataclass
class Args:
    experiment : str
    epochs : Optional[int] = None

def create_transforms(config : Config) -> tuple[T.Compose, T.Compose]:
    
    train_transform = []

    if config.data_augmentation:

        train_transform.extend([
            T.RandomChoice([
                T.Lambda(lambd=lambda x : torch.permute(x, dims=(1, 2, 0))),
                T.RandomChoice(transforms=[
                    T.Compose([
                        Transpose(dim0=0,dim1=1),
                        Flip(dims=(1,))
                    ]),
                    Transpose(dim0=0,dim1=1),
                    T.Compose([
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
                T.Lambda(lambd=lambda x : torch.permute(x, dims=(2, 0, 1))),
            ])
        ])

    return T.Compose(train_transform), T.Compose([])

def create_datasets(config : Config) -> tuple[TensorDataset, TensorDataset]:
    
    train_transform, val_transform = create_transforms(config)

    """train_dataset = TensorDataset(
        root=config.data_dir,
        tensor_transform=train_transform,
        split='train',
        output_type='auto'
    )

    val_dataset = TensorDataset(
        root=config.data_dir,
        tensor_transform=val_transform,
        split='val',
        output_type='auto'
    )"""

    train_dataset = FakeTensorDataset(
        shape=(800,512),
        length=384,
        num_classes=3,
        tensor_transform=train_transform,
        label_transform=None
    )

    val_dataset = FakeTensorDataset(
        shape=(800,512),
        length=128,
        num_classes=3,
        tensor_transform=val_transform,
        label_transform=None
    )

    return train_dataset, val_dataset

def create_dataloaders(config : Config) -> tuple[DataLoader, DataLoader]:

    train_dataset, val_dataset = create_datasets(config)

    sampler = None

    if config.sampler == "weighted":

        labels = train_dataset.metadata['type'].values
        values, counts = np.unique(labels, return_counts=True)
        values_counts = dict(zip(values, counts))

        sampler = WeightedRandomSampler(
            weights=[1 / values_counts[label] for label in labels],
            num_samples=len(labels),
            replacement=True
        )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=(config.sampler == "random"),
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        sampler=sampler
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor
    )

    return train_dataloader, val_dataloader

def create_optimizer(model : nn.Module,config : Config) -> optim.Optimizer:
    
    return optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

def create_model(config : Config) -> nn.Module:

    model = timm.create_model(
        model_name=config.model_name,
        num_classes=3,
        pretrained=False,
        kwargs=config.model_params
    ).to(config.device)

    return model

def create_criteria(config : Config) -> nn.Module:

    if config.model_name == "acmil":

        return MBALoss(
            branches_count=config.model_params['branches_count'],
            device=config.device
        )
    
    else:

        return nn.CrossEntropyLoss()

def get_pred(outputs : Any) -> torch.Tensor:

    if isinstance(outputs, tuple):
        return outputs[1]
    else:
        return outputs

def main(args : Args) -> None:

    ### Logger
    logger = logging.getLogger(args.experiment)

    ### Load the configuration file
    logger.info(f"Loading configuration file for experiment {args.experiment}.")

    EXPIREMENETS_DIR = dotenv.get_key(env, 'EXPERIMENTS_FOLDER')
    EXPIREMENET_DIR = os.path.join(EXPIREMENETS_DIR, args.experiment)
    CHECKPOINTS_DIR = os.path.join(EXPIREMENET_DIR, "checkpoints")
    HISTORY_FILE = os.path.join(EXPIREMENET_DIR, "history.csv")

    if not os.path.exists(EXPIREMENET_DIR):
        raise Exception(f"Experiment {args.experiment} does not exist.")

    config_file = os.path.join(EXPIREMENET_DIR, "config.json")

    if not os.path.exists(config_file):
        raise Exception(f"config.json does not exist in {EXPIREMENET_DIR}.")

    config = Config(**load_json(config_file))

    ### Reproducibility
    logger.info(f"Setting seed to {config.seed}.")
    seed_everything(config.seed)

    ### Create the dataloaders
    logger.info("Creating dataloaders.")
    train_dataloader, val_dataloader = create_dataloaders(config)

    ### Load the model
    logger.info("Creating model,loss function and optimizer.")
    os.makedirs(CHECKPOINTS_DIR,exist_ok=True)
    model = create_model(config)

    ### Loss function
    loss_fn = create_criteria(config)

    ### Scheduler
    scheduler = scheduler = CosineScheduler(
        optimizer=optimizer,
        lr=config.learning_rate,
        num_steps_per_epoch=len(train_dataloader),
        last_epoch=config.total_epochs,
        alpha=config.decay_alpha,
        min_lr=config.min_lr
    ) if config.use_scheduler else None

    ### Optimizer
    optimizer = create_optimizer(model, config)

    ### Training
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoints_folder=CHECKPOINTS_DIR,
        history_filename=HISTORY_FILE,
        metrics={
            "accuracy": Accuracy(num_classes=3,task="multiclass"),
            "f1": F1Score(num_classes=3,task="multiclass",average="macro")
        },
        device=config.device,
        save_every=config.save_every,
        save_on_best=config.save_on_best,
        score_metric="loss",
        score_direction="min",
        get_pred=get_pred
    )

    ### Start training
    logger.info("Starting training.")

    trainer.train(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        epochs=(args.epochs or config.total_epochs)
    )

    logger.info("Training finished.")
    logger.info("Goodbye.")

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--epochs", type=int)

    args = parser.parse_args()
    args = Args(**vars(args))

    main(args)