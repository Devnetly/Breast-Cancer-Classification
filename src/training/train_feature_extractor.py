import sys
import os
import torch
import dotenv   
import numpy as np
import timm
import logging
from torch import nn,optim
from torch.optim.lr_scheduler import OneCycleLR
sys.path.append('../..')
from argparse import ArgumentParser
from dataclasses import dataclass
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,WeightedRandomSampler
from typing import Optional
from torchmetrics import Accuracy,F1Score
from torchvision import transforms as T
from src.models.resnet import *
from src.utils import seed_everything,load_json
from src.transforms import LabelMapper
from src.trainer import Trainer

env = dotenv.find_dotenv()
logging.basicConfig(level=logging.INFO)

@dataclass
class Config:

    ### SEED
    seed : int = 42

    ### Device
    device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Model
    model_name : str = "resnet18"
    model_params : Optional[dict] = None

    ### Data augmentation & preprocessing
    preprocessing : str = "nothing"
    data_augmentation : bool = False

    ### Scheduler
    use_scheduler : bool = False
    min_lr : float = 1e-6

    ### Optimizer
    optimizer : str = "adam"
    optimizer_params : Optional[dict] = None
    learning_rate : float = 1e-3
    weight_decay : float = 1e-3

    ### Data loading
    batch_size : int = 32
    sampler : str = "random"
    num_workers : int = 0
    prefetch_factor : int = None

    ### Training
    total_epochs : int = 10

    ### Checkpoints
    save_every : Optional[int] = None
    save_on_best : bool = False


@dataclass
class Args:
    experiment : str
    epochs : Optional[int] = None

def create_transforms(config: Config) -> tuple[T.Compose, T.Compose]:

    data_augmentation = [
        T.RandomChoice([
            T.RandomRotation(degrees=(0,0)),
            T.RandomRotation(degrees=(90,90)),
            T.RandomRotation(degrees=(180,180)),
            T.RandomRotation(degrees=(270,270)),
        ]),
        T.RandomOrder([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]),
    ] if config.data_augmentation else []

    train_transform = T.Compose([
        *data_augmentation,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def create_datasets(config: Config) -> tuple[ImageFolder, ImageFolder, ImageFolder]:

    DATA_FOLDER = dotenv.get_key(env, 'PATCHES_FOLDER')
    
    train_transform, val_transform = create_transforms(config)

    label_mapper = LabelMapper({
        0:0, # 0 is the label for benign (BY)
        1:0, 
        2:0,
        3:1, # 1 is the label for atypical (AT)
        4:1,
        5:2, # 2 is the label for malignant (MT)
        6:2,
    })

    train_dataset = ImageFolder(
        root=os.path.join(DATA_FOLDER,'train'), 
        transform=train_transform,
        target_transform=label_mapper
    )

    val_dataset = ImageFolder(
        root=os.path.join(DATA_FOLDER,'val'), 
        transform=val_transform,
        target_transform=label_mapper
    )

    test_dataset = ImageFolder(
        root=os.path.join(DATA_FOLDER,'test'), 
        transform=val_transform,
        target_transform=label_mapper
    )

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset, val_dataset, test_dataset = create_datasets(config)

    sampler = None

    values,counts = np.unique(train_dataset.targets,return_counts=True)
    values_counts = dict(zip(values,counts))

    if config.sampler == "balanced":

        sampler = WeightedRandomSampler(
            weights=[1 / values_counts[i] for i in train_dataset.targets],
            num_samples=len(train_dataset.targets),
            replacement=True
        )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=(config.sampler == "random"),
        drop_last=True,
        sampler=sampler
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=False,
        drop_last=False
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=False,
        drop_last=False
    )

    return train_dataloader, val_dataloader, test_dataloader

def create_optimizer(config: Config, model: nn.Module) -> optim.Optimizer:

    optimizers = {
        "adam_w": optim.AdamW,
        "adam": optim.Adam,
        "sgd" : optim.SGD,
        "rmsprop": optim.RMSprop,
    }

    optimizer_cls = optimizers[config.optimizer]

    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        **(config.optimizer_params or {})
    )

    return optimizer

def create_model(config: Config) -> nn.Module:
    
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=3,
        kwargs=(config.model_params or {})
    ).to(config.device)

    return model

def main(args : Args):

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
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)

    ### Load the model
    logger.info("Creating model,loss function and optimizer.")
    os.makedirs(CHECKPOINTS_DIR,exist_ok=True)
    model = create_model(config)

    ### Loss function
    loss_fn = nn.CrossEntropyLoss()

    ### Optimizer
    optimizer = create_optimizer(config, model)

    ### Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.total_epochs,
        steps_per_epoch=len(train_dataloader),
    ) if config.use_scheduler else None

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
        score_direction="min"
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