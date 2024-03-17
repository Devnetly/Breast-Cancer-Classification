import torch
import argparse
import dotenv
import os
import sys
sys.path.append("../..")
from torch.utils.data import RandomSampler,Sampler
from torchvision import transforms,datasets
from src.samplers import BalancedSampler
from src.models import ResNet18,ResNet34
from src.utils import load_model_from_folder
from src.transforms import KRandomRotation

class DEFAULTS:
    MODEL = "resnet18"
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 1
    SAMPLER = "random"
    DATA_AUGMENTATION = False

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int,default=DEFAULTS.BATCH_SIZE)
    parser.add_argument("--epochs", type=int,default=DEFAULTS.EPOCHS)
    parser.add_argument("--learning-rate", type=float,default=DEFAULTS.LEARNING_RATE)
    parser.add_argument("--model-type", type=str,default=DEFAULTS.MODEL)
    parser.add_argument("--weights-folder", type=str)
    parser.add_argument("--histories-folder", type=str)
    parser.add_argument("--data-augmentation", type=bool, default=DEFAULTS.DATA_AUGMENTATION)
    parser.add_argument("--sampler", type=str,default=DEFAULTS.SAMPLER)

    return parser.parse_args()

def load_envirement_variables() -> tuple[str, str, str]:

    PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), "HISTORIES_DIR")
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")

    return PATCHES_DIR,HISTORIES_DIR,MODELS_DIR

def load_model(models_dir: str, model_type: str) -> torch.nn.Module:

    model = None

    if model_type == "resnet18":
        model = ResNet18(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)
    elif model_type == "resnet34":
        model = ResNet34(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)

    load_model_from_folder(
        model=model, 
        weights_folder=os.path.join(models_dir, model_type),
        verbose=True,
    )

    return model

def create_sampler(
    type_ : str,
    dataset : datasets.ImageFolder
) -> Sampler:

    sampler = None

    if type_ == "random":
        sampler = RandomSampler(data_source=dataset)
    elif type_ == "balanced":
        sampler = BalancedSampler(dataset=dataset)
    else:
        raise Exception(f"sampler {type_} is not supported.")
    
    return sampler

def create_transforms(data_augmentation : bool) -> tuple[transforms.Compose,transforms.Compose]:

    train_transforms = []   

    if data_augmentation:

        augmentation = transforms.Compose([
            KRandomRotation(probas=[0.4,0.2,0.2,0.2]),
            transforms.RandomResizedCrop(size=(224,224), scale=(0.6, 1.0)),
            transforms.RandomChoice(transforms=[
                transforms.RandomVerticalFlip(p=0.8),
                transforms.RandomHorizontalFlip(p=0.8),
            ],p=[0.5,0.5])
        ]) 

        train_transforms.append(augmentation)

    train_transforms.append(transforms.ToTensor())

    train_transform = transforms.Compose(train_transforms)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform,val_transform