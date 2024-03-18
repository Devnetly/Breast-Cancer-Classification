import torch
import argparse
import dotenv
import os
import sys
sys.path.append("../..")
from torch.utils.data import RandomSampler,Sampler
from torchvision import transforms,datasets
from src.models import ResNet18,ResNet34,ResNet50
from src.utils import load_model_from_folder
from torchsampler import ImbalancedDatasetSampler

class DEFAULTS:
    MODEL = "resnet18"
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 1
    SAMPLER = "random"
    DATA_AUGMENTATION = False
    DROPOUT = 0.0

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int,default=DEFAULTS.BATCH_SIZE)
    parser.add_argument("--epochs", type=int,default=DEFAULTS.EPOCHS)
    parser.add_argument("--learning-rate", type=float,default=DEFAULTS.LEARNING_RATE)
    parser.add_argument("--model-type", type=str,default=DEFAULTS.MODEL,choices=["resnet18","resnet34","resnet50"])
    parser.add_argument("--weights-folder", type=str)
    parser.add_argument("--histories-folder", type=str)
    parser.add_argument("--data-augmentation", type=lambda x : x.lower() == "true", default=DEFAULTS.DATA_AUGMENTATION)
    parser.add_argument("--sampler", type=str,default=DEFAULTS.SAMPLER, choices=["random","balanced"])
    parser.add_argument('--dropout', type=float,default=DEFAULTS.DROPOUT)

    return parser.parse_args()

def load_envirement_variables() -> tuple[str, str, str]:

    PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), "HISTORIES_DIR")
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")

    return PATCHES_DIR,HISTORIES_DIR,MODELS_DIR

def load_model(
    models_dir: str, 
    model_type: str,
    dropout_rate : float
) -> torch.nn.Module:

    model = None

    if model_type == "resnet18":
        model = ResNet18(n_classes=GLOBAL.NUM_CLASSES,dropout_rate=dropout_rate).to(GLOBAL.DEVICE)
    elif model_type == "resnet34":
        model = ResNet34(n_classes=GLOBAL.NUM_CLASSES,dropout_rate=dropout_rate).to(GLOBAL.DEVICE)
    elif model_type == "resnet50":
        model = ResNet50(n_classes=GLOBAL.NUM_CLASSES,dropout_rate=dropout_rate).to(GLOBAL.DEVICE)
    else:
        raise Exception(f'model {model_type} is not supported.')

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
        sampler = ImbalancedDatasetSampler(dataset=dataset)
    else:
        raise Exception(f"sampler {type_} is not supported.")
    
    return sampler

def create_transforms(data_augmentation : bool) -> tuple[transforms.Compose,transforms.Compose]:

    train_transforms = []   
    val_transforms = []

    if data_augmentation:

        train_transforms = [
            transforms.Pad(10, padding_mode='reflect'),
            transforms.RandomRotation(20),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        val_transforms = [
            transforms.Pad(10, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
    else:

        train_transforms = [
            transforms.ToTensor()
        ]

        val_transforms = [
            transforms.ToTensor()
        ]

    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)

    return train_transform,val_transform