import torch
import argparse
import dotenv
import sys
sys.path.append("../..")
from torch.utils.data import RandomSampler,Sampler
from torchvision import transforms,datasets
from src.models import ResNet18,ResNet34,ResNet50
from src.utils import load_model_from_folder
from src.transforms import ReinhardNotmalizer
from torchsampler import ImbalancedDatasetSampler
from torch.optim import SGD,Adam,Optimizer
from randstainna.randstainna import RandStainNA

class DEFAULTS:
    MODEL = "resnet18"
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 1
    SAMPLER = "random"
    PREPROCESSING = "nothing"
    DROPOUT = 0.0
    DECAY_RATE = 0.0
    OPTIMIZER = "adam"
    LAST_EPOCH = -1
    LOSS = "ce"

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int,default=DEFAULTS.BATCH_SIZE)
    parser.add_argument("--epochs", type=int,default=DEFAULTS.EPOCHS)
    parser.add_argument("--learning-rate", type=float,default=DEFAULTS.LEARNING_RATE)
    parser.add_argument("--model-type", type=str,default=DEFAULTS.MODEL,choices=["resnet18","resnet34","resnet50"])
    parser.add_argument("--weights-folder", type=str, required=True)
    parser.add_argument("--histories-folder", type=str, required=True)

    parser.add_argument("--preprocessing", type=str, default=DEFAULTS.PREPROCESSING, choices=[
        'nothing',
        'stain-normalization',
        'augmentation',
        'stain-augmentation'
    ])

    parser.add_argument("--sampler", type=str,default=DEFAULTS.SAMPLER, choices=["random","balanced"])
    parser.add_argument('--dropout', type=float,default=DEFAULTS.DROPOUT)
    parser.add_argument('--decay-rate', type=float, default=DEFAULTS.DECAY_RATE)
    parser.add_argument('--optimizer', type=str, default=DEFAULTS.OPTIMIZER, choices=["adam", "sgd"])
    parser.add_argument('--last-epoch', type=int, default=DEFAULTS.LAST_EPOCH)

    return parser.parse_args()

def load_envirement_variables() -> tuple[str, str, str]:

    PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), "HISTORIES_DIR")
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")
    DATA_DIR = dotenv.get_key(dotenv.find_dotenv(), "DATA_DIR")

    return PATCHES_DIR,HISTORIES_DIR,MODELS_DIR,DATA_DIR

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
        weights_folder=models_dir,
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

def create_transforms(
    type : str,
    template_img_src : str,
    config_file : str
) -> tuple[transforms.Compose,transforms.Compose]:

    train_transforms = []   
    val_transforms = []

    if type == "nothing":

        train_transforms = [
            transforms.ToTensor()
        ]

        val_transforms = [
            transforms.ToTensor()
        ]

    elif type == "stain-normalization":

        train_transforms = [
            ReinhardNotmalizer(template_img_src=template_img_src),
            transforms.ToTensor()
        ]

        val_transforms = [
            ReinhardNotmalizer(template_img_src=template_img_src),
            transforms.ToTensor()
        ]
        
    elif type == "augmentation":

        train_transforms = [
            transforms.Pad(10, padding_mode='reflect'),
            transforms.RandomRotation(20),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        val_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
    elif type == "stain-augmentation":

        train_transforms = [
            RandStainNA(
                yaml_file=config_file,
                std_hyper=-0.3, 
                probability=1.0,
                distribution='normal', 
                is_train=True
            ),
            transforms.ToTensor()
        ]

        val_transforms = [
            transforms.ToTensor()
        ]

    else:
        raise Exception(f"{type} is not supported.")

    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)

    return train_transform,val_transform

def create_optimizer(
    params,
    type : str, 
    lr : float
) -> Optimizer:

    if type == "adam":
        return Adam(params, lr=lr)
    elif type == "sgd":
        return SGD(params, lr=lr)
    else:
        raise Exception(f"optimizer {type} not supported.")